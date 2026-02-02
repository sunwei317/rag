"""
数据驱动文档生成模块
严格基于数据库中的数据生成技术文档，不编造内容

工作流程:
1. 从 Neo4j 获取所有实体和关系
2. 从向量存储获取所有文档 chunks
3. 基于实际数据组织文档结构
4. 使用 LLM 将数据整理成可读文档（不添加新信息）
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
from collections import defaultdict


@dataclass
class DataSection:
    """数据驱动的章节"""
    title: str
    level: int
    content: str
    source_chunks: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        prefix = "#" * (self.level + 1)
        return f"{prefix} {self.title}\n\n{self.content}"


@dataclass 
class DataDrivenDocument:
    """数据驱动生成的文档"""
    title: str
    sections: List[DataSection]
    entity_count: int
    relation_count: int
    chunk_count: int
    
    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        
        for section in self.sections:
            lines.append(section.to_markdown())
            lines.append("")
        
        # 添加数据来源说明
        lines.append("---")
        lines.append("## 数据来源")
        lines.append("")
        lines.append(f"- 实体数量: {self.entity_count}")
        lines.append(f"- 关系数量: {self.relation_count}")
        lines.append(f"- 文档片段数量: {self.chunk_count}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "sections": [{"title": s.title, "level": s.level, "content": s.content} for s in self.sections],
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "chunk_count": self.chunk_count
        }


class DataDrivenWriter:
    """
    数据驱动文档写作器
    
    严格基于数据库数据生成文档:
    - 从 Neo4j 读取实体和关系
    - 从向量存储读取文档内容
    - 使用 LLM 整理格式，但不添加新信息
    
    控制参数:
    - max_entities_per_type: 每种类型最多包含的实体数量
    - max_sections: 最大章节数
    - detail_level: 细节级别 (brief/standard/detailed)
    - include_relations: 是否包含关系章节
    - include_chunks: 是否包含原始文档片段
    - entity_types: 要包含的实体类型列表，None 表示全部
    """
    
    # 细节级别配置
    DETAIL_CONFIGS = {
        "brief": {
            "max_entities_per_type": 5,
            "max_sections": 5,
            "include_description": False,
            "include_aliases": False,
            "include_relations": False,
            "max_chunks_for_overview": 2,
            "entity_name_only": True
        },
        "standard": {
            "max_entities_per_type": 20,
            "max_sections": 10,
            "include_description": True,
            "include_aliases": False,
            "include_relations": True,
            "max_chunks_for_overview": 5,
            "entity_name_only": False
        },
        "detailed": {
            "max_entities_per_type": None,  # 无限制
            "max_sections": None,
            "include_description": True,
            "include_aliases": True,
            "include_relations": True,
            "max_chunks_for_overview": 10,
            "entity_name_only": False
        }
    }
    
    def __init__(
        self,
        graph_store=None,
        vector_store=None,
        llm_client=None,
        model: str = "gpt-4.1-mini",
        # 文档长度控制
        max_entities_per_type: Optional[int] = None,
        max_sections: Optional[int] = None,
        # 内容细节控制
        detail_level: str = "standard",  # brief, standard, detailed
        include_relations: bool = True,
        include_chunks: bool = False,
        entity_types: Optional[List[str]] = None  # 要包含的实体类型
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model = model
        
        # 获取细节级别配置
        self.detail_config = self.DETAIL_CONFIGS.get(detail_level, self.DETAIL_CONFIGS["standard"]).copy()
        
        # 用户自定义参数覆盖默认配置
        if max_entities_per_type is not None:
            self.detail_config["max_entities_per_type"] = max_entities_per_type
        if max_sections is not None:
            self.detail_config["max_sections"] = max_sections
        self.detail_config["include_relations"] = include_relations
        self.detail_config["include_chunks"] = include_chunks
        
        self.entity_types = entity_types
        self.detail_level = detail_level
        
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"Failed to init LLM client: {e}")
    
    def generate_from_database(
        self,
        doc_id: Optional[str] = None,
        doc_type: str = "api_reference",
        title: Optional[str] = None,
        # 运行时覆盖配置
        max_entities_per_type: Optional[int] = None,
        max_sections: Optional[int] = None,
        detail_level: Optional[str] = None,
        entity_types: Optional[List[str]] = None
    ) -> DataDrivenDocument:
        """
        从数据库生成技术文档
        
        Args:
            doc_id: 可选的文档 ID 过滤
            doc_type: 文档类型 (api_reference, user_manual, etc.)
            title: 文档标题，如不提供则自动生成
            max_entities_per_type: 每种类型最多实体数（覆盖初始化配置）
            max_sections: 最大章节数（覆盖初始化配置）
            detail_level: 细节级别 brief/standard/detailed（覆盖初始化配置）
            entity_types: 要包含的实体类型列表（覆盖初始化配置）
            
        Returns:
            DataDrivenDocument: 生成的文档
        """
        # 运行时配置覆盖
        runtime_config = self.detail_config.copy()
        if detail_level and detail_level in self.DETAIL_CONFIGS:
            runtime_config = self.DETAIL_CONFIGS[detail_level].copy()
        if max_entities_per_type is not None:
            runtime_config["max_entities_per_type"] = max_entities_per_type
        if max_sections is not None:
            runtime_config["max_sections"] = max_sections
        
        active_entity_types = entity_types or self.entity_types
        
        logger.info(f"Generating document from database, doc_type={doc_type}, "
                   f"detail_level={detail_level or self.detail_level}, "
                   f"max_entities={runtime_config.get('max_entities_per_type')}, "
                   f"max_sections={runtime_config.get('max_sections')}")
        
        # 1. 获取所有实体和关系
        entities_data = self._get_all_entities(entity_types=active_entity_types)
        relations_data = self._get_all_relations() if runtime_config.get("include_relations", True) else []
        
        logger.info(f"Retrieved {len(entities_data)} entities, {len(relations_data)} relations")
        
        # 2. 获取所有文档 chunks
        chunks_data = self._get_all_chunks(doc_id)
        logger.info(f"Retrieved {len(chunks_data)} chunks")
        
        # 3. 按实体类型组织数据（传入配置参数）
        organized_data = self._organize_data(entities_data, relations_data, chunks_data, runtime_config)
        
        # 4. 根据文档类型生成结构
        if doc_type == "api_reference":
            sections = self._generate_api_reference_sections(organized_data)
        elif doc_type == "user_manual":
            sections = self._generate_user_manual_sections(organized_data)
        else:
            sections = self._generate_general_sections(organized_data)
        
        # 5. 生成文档标题
        if not title:
            title = self._generate_title(organized_data, doc_type)
        
        return DataDrivenDocument(
            title=title,
            sections=sections,
            entity_count=len(entities_data),
            relation_count=len(relations_data),
            chunk_count=len(chunks_data)
        )
    
    def _get_all_entities(self, entity_types: Optional[List[str]] = None) -> List[Dict]:
        """获取所有实体
        
        Args:
            entity_types: 要获取的实体类型列表，None 表示全部
        """
        if not self.graph_store:
            return []
        
        try:
            # 检查是否是 Neo4jStore (有 _driver 属性)
            store_class = self.graph_store.__class__.__name__
            
            if store_class == "Neo4jStore" or hasattr(self.graph_store, '_driver'):
                with self.graph_store._driver.session() as session:
                    if entity_types:
                        # 过滤特定类型
                        result = session.run("""
                            MATCH (e:Entity)
                            WHERE e.entity_type IN $types
                            RETURN e.entity_id as id, e.name as name, 
                                   e.entity_type as type, e.description as description,
                                   e.aliases as aliases
                            ORDER BY e.entity_type, e.name
                        """, types=entity_types)
                    else:
                        result = session.run("""
                            MATCH (e:Entity)
                            RETURN e.entity_id as id, e.name as name, 
                                   e.entity_type as type, e.description as description,
                                   e.aliases as aliases
                            ORDER BY e.entity_type, e.name
                        """)
                    return [dict(record) for record in result]
            elif hasattr(self.graph_store, '_entities'):
                # InMemoryGraphStore
                return [
                    {
                        "id": e.entity_id,
                        "name": e.name,
                        "type": e.entity_type.value,
                        "description": e.description,
                        "aliases": e.aliases
                    }
                    for e in self.graph_store._entities.values()
                ]
            else:
                logger.warning(f"Unknown graph store type: {store_class}")
                return []
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _get_all_relations(self) -> List[Dict]:
        """获取所有关系"""
        if not self.graph_store:
            return []
        
        try:
            store_class = self.graph_store.__class__.__name__
            
            if store_class == "Neo4jStore" or hasattr(self.graph_store, '_driver'):
                with self.graph_store._driver.session() as session:
                    result = session.run("""
                        MATCH (s:Entity)-[r]->(t:Entity)
                        RETURN s.name as source, type(r) as relation_type, 
                               t.name as target, r.description as description
                        ORDER BY type(r), s.name
                    """)
                    return [dict(record) for record in result]
            elif hasattr(self.graph_store, '_relations'):
                # InMemoryGraphStore
                relations = []
                for rel in self.graph_store._relations.values():
                    source = self.graph_store._entities.get(rel.source_id)
                    target = self.graph_store._entities.get(rel.target_id)
                    if source and target:
                        relations.append({
                            "source": source.name,
                            "relation_type": rel.relation_type.value,
                            "target": target.name,
                            "description": rel.description
                        })
                return relations
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get relations: {e}")
            import traceback
            traceback.print_exc()
            return []
            return []
    
    def _get_all_chunks(self, doc_id: Optional[str] = None) -> List[Dict]:
        """获取所有文档 chunks"""
        if not self.vector_store:
            return []
        
        try:
            filter_dict = {"doc_id": doc_id} if doc_id else None
            chunks = self.vector_store.get_all_chunks(filter_dict=filter_dict)
            return chunks if chunks else []
        except Exception as e:
            logger.error(f"Failed to get chunks: {e}")
            return []
    
    def _organize_data(
        self, 
        entities: List[Dict], 
        relations: List[Dict],
        chunks: List[Dict],
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """组织数据
        
        Args:
            entities: 实体列表
            relations: 关系列表
            chunks: 文档片段列表
            config: 配置参数
        """
        config = config or self.detail_config
        max_per_type = config.get("max_entities_per_type")
        
        # 按类型分组实体
        entities_by_type = defaultdict(list)
        for e in entities:
            entity_type = e.get("type", "concept")
            current_list = entities_by_type[entity_type]
            # 应用每种类型的数量限制
            if max_per_type is None or len(current_list) < max_per_type:
                entities_by_type[entity_type].append(e)
        
        # 按关系类型分组
        relations_by_type = defaultdict(list)
        for r in relations:
            relations_by_type[r.get("relation_type", "related_to")].append(r)
        
        # 提取 chunks 中的关键内容
        max_chunks = config.get("max_chunks_for_overview", 5)
        chunk_contents = []
        for c in chunks[:max_chunks * 10]:  # 预取更多以便筛选
            # 确保 c 是字典类型
            if isinstance(c, str):
                content = c
                section = ""
                page = ""
            else:
                content = c.get("content", "")
                section = c.get("metadata", {}).get("section_path", "") if isinstance(c.get("metadata"), dict) else c.get("section_path", "")
                page = c.get("metadata", {}).get("page_number", "") if isinstance(c.get("metadata"), dict) else c.get("page_number", "")
            
            if content and len(content) > 50:
                chunk_contents.append({
                    "content": content,
                    "section": section,
                    "page": page
                })
                if len(chunk_contents) >= max_chunks * 2:
                    break
        
        return {
            "entities_by_type": dict(entities_by_type),
            "relations_by_type": dict(relations_by_type),
            "chunks": chunk_contents,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "config": config
        }
    
    def _generate_api_reference_sections(self, data: Dict) -> List[DataSection]:
        """生成 API 参考文档结构"""
        sections = []
        config = data.get("config", self.detail_config)
        max_sections = config.get("max_sections")
        include_relations = config.get("include_relations", True)
        
        def add_section(section):
            """添加章节，检查数量限制"""
            if max_sections is None or len(sections) < max_sections:
                sections.append(section)
                return True
            return False
        
        # 1. 概述章节
        overview = self._generate_overview_section(data)
        if overview:
            if not add_section(overview):
                return sections
        
        # 2. 产品/组件章节
        products = data["entities_by_type"].get("product", [])
        if products:
            if not add_section(self._generate_entity_section(
                "产品与组件", products, 1, config
            )):
                return sections
        
        # 3. API/接口章节
        apis = data["entities_by_type"].get("api", [])
        if apis:
            if not add_section(self._generate_entity_section(
                "API 接口", apis, 1, config
            )):
                return sections
        
        # 4. 配置项章节
        configs = data["entities_by_type"].get("config", [])
        if configs:
            if not add_section(self._generate_entity_section(
                "配置项", configs, 1, config
            )):
                return sections
        
        # 5. 命令章节
        commands = data["entities_by_type"].get("command", [])
        if commands:
            if not add_section(self._generate_entity_section(
                "命令与操作", commands, 1, config
            )):
                return sections
        
        # 6. 错误处理章节
        errors = data["entities_by_type"].get("error", [])
        if errors:
            if not add_section(self._generate_entity_section(
                "错误码", errors, 1, config
            )):
                return sections
        
        # 7. 实体关系章节
        if include_relations and data["relations_by_type"]:
            if not add_section(self._generate_relations_section(data["relations_by_type"])):
                return sections
        
        # 8. 技术概念章节
        concepts = data["entities_by_type"].get("concept", [])
        if concepts:
            add_section(self._generate_entity_section(
                "技术概念", concepts, 1, config
            ))
        
        return sections
    
    def _generate_user_manual_sections(self, data: Dict) -> List[DataSection]:
        """生成用户手册结构"""
        sections = []
        config = data.get("config", self.detail_config)
        max_sections = config.get("max_sections")
        
        def add_section(section):
            if max_sections is None or len(sections) < max_sections:
                sections.append(section)
                return True
            return False
        
        # 1. 简介
        overview = self._generate_overview_section(data)
        if overview:
            if not add_section(overview):
                return sections
        
        # 2. 产品功能
        products = data["entities_by_type"].get("product", [])
        if products:
            if not add_section(self._generate_entity_section(
                "产品功能", products, 1, config
            )):
                return sections
        
        # 3. 基本操作
        commands = data["entities_by_type"].get("command", [])
        if commands:
            if not add_section(self._generate_entity_section(
                "基本操作", commands, 1, config
            )):
                return sections
        
        # 4. 配置说明
        configs = data["entities_by_type"].get("config", [])
        if configs:
            if not add_section(self._generate_entity_section(
                "配置说明", configs, 1, config
            )):
                return sections
        
        # 5. 故障排除
        errors = data["entities_by_type"].get("error", [])
        if errors:
            add_section(self._generate_entity_section(
                "故障排除", errors, 1, config
            ))
        
        return sections
    
    def _generate_general_sections(self, data: Dict) -> List[DataSection]:
        """生成通用文档结构"""
        sections = []
        config = data.get("config", self.detail_config)
        max_sections = config.get("max_sections")
        include_relations = config.get("include_relations", True)
        
        def add_section(section):
            if max_sections is None or len(sections) < max_sections:
                sections.append(section)
                return True
            return False
        
        # 概述
        overview = self._generate_overview_section(data)
        if overview:
            if not add_section(overview):
                return sections
        
        # 按实体类型生成章节
        type_names = {
            "product": "产品与组件",
            "api": "接口",
            "config": "配置",
            "command": "命令",
            "concept": "概念",
            "error": "错误处理",
            "file": "文件",
            "dependency": "依赖",
            "platform": "平台",
            "version": "版本"
        }
        
        for entity_type, entities in data["entities_by_type"].items():
            if entities:
                title = type_names.get(entity_type, entity_type.title())
                if not add_section(self._generate_entity_section(title, entities, 1, config)):
                    return sections
        
        # 关系章节
        if include_relations and data["relations_by_type"]:
            add_section(self._generate_relations_section(data["relations_by_type"]))
        
        return sections
    
    def _generate_overview_section(self, data: Dict) -> Optional[DataSection]:
        """生成概述章节"""
        # 从 chunks 中提取概述内容
        overview_content = []
        
        for chunk in data["chunks"][:5]:  # 使用前几个 chunks
            content = chunk.get("content", "")
            if "概述" in chunk.get("section", "") or "简介" in chunk.get("section", ""):
                overview_content.append(content)
        
        # 如果没有找到概述，使用统计信息
        if not overview_content:
            entity_count = data["entity_count"]
            relation_count = data["relation_count"]
            
            # 获取主要实体类型
            type_summary = []
            for t, entities in data["entities_by_type"].items():
                if entities:
                    type_summary.append(f"{len(entities)} 个{t}类型实体")
            
            content = f"本文档基于知识图谱中的 {entity_count} 个实体和 {relation_count} 个关系自动生成。\n\n"
            if type_summary:
                content += "包含：" + "、".join(type_summary) + "。"
            
            return DataSection(
                title="概述",
                level=1,
                content=content
            )
        
        return DataSection(
            title="概述",
            level=1,
            content="\n\n".join(overview_content)
        )
    
    def _generate_entity_section(
        self, 
        title: str, 
        entities: List[Dict],
        level: int,
        config: Optional[Dict] = None
    ) -> DataSection:
        """生成实体章节
        
        Args:
            title: 章节标题
            entities: 实体列表
            level: 章节级别
            config: 配置参数
        """
        config = config or self.detail_config
        entity_name_only = config.get("entity_name_only", False)
        include_description = config.get("include_description", True)
        include_aliases = config.get("include_aliases", True)
        
        lines = []
        
        for entity in entities:
            name = entity.get("name", "")
            description = entity.get("description", "")
            aliases = entity.get("aliases", [])
            
            if entity_name_only:
                # brief 模式: 只列出实体名称
                lines.append(f"- {name}")
            else:
                # 实体名称作为小标题
                lines.append(f"### {name}")
                
                if include_description and description:
                    lines.append(f"\n{description}")
                
                if include_aliases and aliases and isinstance(aliases, list) and len(aliases) > 0:
                    lines.append(f"\n**别名**: {', '.join(aliases)}")
                
                lines.append("")
        
        return DataSection(
            title=title,
            level=level,
            content="\n".join(lines),
            entities=[e.get("name", "") for e in entities]
        )
    
    def _generate_relations_section(self, relations_by_type: Dict) -> DataSection:
        """生成关系章节"""
        lines = []
        
        relation_names = {
            "depends_on": "依赖关系",
            "requires": "需求关系",
            "belongs_to": "归属关系",
            "contains": "包含关系",
            "configures": "配置关系",
            "implements": "实现关系",
            "calls": "调用关系",
            "uses": "使用关系",
            "related_to": "关联关系"
        }
        
        for rel_type, relations in relations_by_type.items():
            if not relations:
                continue
                
            type_name = relation_names.get(rel_type, rel_type)
            lines.append(f"### {type_name}")
            lines.append("")
            
            for rel in relations[:20]:  # 限制每种类型最多20个
                source = rel.get("source", "")
                target = rel.get("target", "")
                desc = rel.get("description", "")
                
                if desc:
                    lines.append(f"- **{source}** → **{target}**: {desc}")
                else:
                    lines.append(f"- **{source}** → **{target}**")
            
            lines.append("")
        
        return DataSection(
            title="实体关系",
            level=1,
            content="\n".join(lines)
        )
    
    def _generate_title(self, data: Dict, doc_type: str) -> str:
        """生成文档标题"""
        # 尝试从产品实体获取名称
        products = data["entities_by_type"].get("product", [])
        if products:
            product_name = products[0].get("name", "系统")
        else:
            product_name = "系统"
        
        type_titles = {
            "api_reference": "API 参考手册",
            "user_manual": "用户手册",
            "installation_guide": "安装指南",
            "troubleshooting": "故障排除指南",
            "general": "技术文档"
        }
        
        doc_title = type_titles.get(doc_type, "技术文档")
        
        return f"{product_name} {doc_title}"
    
    def generate_with_llm_formatting(
        self,
        doc_id: Optional[str] = None,
        doc_type: str = "api_reference",
        title: Optional[str] = None
    ) -> DataDrivenDocument:
        """
        使用 LLM 格式化数据库内容（不添加新信息）
        
        LLM 只用于将原始数据整理成更易读的格式，
        严格禁止添加数据库中不存在的信息。
        """
        logger.info("Generating document with LLM formatting")
        
        # 获取原始数据
        entities_data = self._get_all_entities()
        relations_data = self._get_all_relations()
        chunks_data = self._get_all_chunks(doc_id)
        
        if not entities_data and not chunks_data:
            logger.warning("No data found in database")
            return DataDrivenDocument(
                title="空文档",
                sections=[DataSection("提示", 1, "数据库中没有找到数据。请先上传文档。")],
                entity_count=0,
                relation_count=0,
                chunk_count=0
            )
        
        # 准备数据摘要给 LLM
        data_summary = self._prepare_data_summary(entities_data, relations_data, chunks_data)
        
        # 使用 LLM 生成格式化文档
        formatted_content = self._format_with_llm(data_summary, doc_type)
        
        # 解析 LLM 输出
        sections = self._parse_llm_output(formatted_content)
        
        if not title:
            title = self._extract_title_from_data(entities_data, doc_type)
        
        return DataDrivenDocument(
            title=title,
            sections=sections,
            entity_count=len(entities_data),
            relation_count=len(relations_data),
            chunk_count=len(chunks_data)
        )
    
    def _prepare_data_summary(
        self,
        entities: List[Dict],
        relations: List[Dict],
        chunks: List[Dict]
    ) -> str:
        """准备数据摘要"""
        lines = ["# 数据库中的数据", ""]
        
        # 实体数据
        lines.append("## 实体列表")
        entities_by_type = defaultdict(list)
        for e in entities:
            entities_by_type[e.get("type", "concept")].append(e)
        
        for etype, elist in entities_by_type.items():
            lines.append(f"\n### {etype} 类型:")
            for e in elist:
                name = e.get("name", "")
                desc = e.get("description", "")
                if desc:
                    lines.append(f"- {name}: {desc}")
                else:
                    lines.append(f"- {name}")
        
        # 关系数据
        if relations:
            lines.append("\n## 实体关系")
            for r in relations[:50]:  # 限制数量
                lines.append(f"- {r.get('source', '')} --[{r.get('relation_type', '')}]--> {r.get('target', '')}")
        
        # 文档内容摘要
        if chunks:
            lines.append("\n## 文档内容片段")
            for i, chunk in enumerate(chunks[:10]):  # 只取前10个
                content = chunk.get("content", "")[:500]  # 截断长内容
                lines.append(f"\n### 片段 {i+1}")
                lines.append(content)
        
        return "\n".join(lines)
    
    def _format_with_llm(self, data_summary: str, doc_type: str) -> str:
        """使用 LLM 格式化内容"""
        if not self.llm_client:
            logger.warning("LLM client not available, using raw data")
            return data_summary
        
        prompt = f"""你是一个技术文档编辑。请将以下数据整理成一篇专业的{doc_type}技术文档。

重要规则：
1. 只使用提供的数据，绝对不要添加任何数据中没有的信息
2. 不要编造任何示例代码、参数值或功能描述
3. 如果数据不足以形成完整章节，就简短描述已有信息
4. 保持信息的准确性，可以重新组织格式，但不能改变含义
5. 使用 Markdown 格式输出

数据内容：
{data_summary}

请生成技术文档（使用 Markdown 格式，包含适当的章节标题）："""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是严谨的技术文档编辑，只基于提供的数据生成文档，绝不编造内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            return data_summary
    
    def _parse_llm_output(self, content: str) -> List[DataSection]:
        """解析 LLM 输出为章节"""
        sections = []
        current_title = ""
        current_level = 1
        current_content = []
        
        for line in content.split("\n"):
            # 检测标题
            if line.startswith("##"):
                # 保存之前的章节
                if current_title and current_content:
                    sections.append(DataSection(
                        title=current_title,
                        level=current_level,
                        content="\n".join(current_content).strip()
                    ))
                
                # 解析新标题
                level = len(line.split()[0]) - 1  # ## = level 1, ### = level 2
                current_level = min(level, 3)
                current_title = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)
        
        # 保存最后一个章节
        if current_title and current_content:
            sections.append(DataSection(
                title=current_title,
                level=current_level,
                content="\n".join(current_content).strip()
            ))
        
        return sections
    
    def _extract_title_from_data(self, entities: List[Dict], doc_type: str) -> str:
        """从数据中提取标题"""
        # 查找产品名
        product_name = ""
        for e in entities:
            if e.get("type") == "product":
                product_name = e.get('name', '')
                break
        
        type_titles = {
            "api_reference": "API 参考手册",
            "user_manual": "用户手册",
            "installation_guide": "安装指南",
            "general": "技术文档"
        }
        
        doc_title = type_titles.get(doc_type, "技术文档")
        
        if product_name:
            return f"{product_name} {doc_title}"
        return doc_title
