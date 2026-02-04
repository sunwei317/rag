"""
关系构建模块
从实体和文档上下文中抽取实体间的关系

支持的关系类型:
- DEPENDS_ON: 依赖关系 (A 依赖 B)
- REQUIRES: 需求关系 (安装 A 需要 B)
- BELONGS_TO: 归属关系 (API 属于模块)
- REFERENCES: 引用关系 (章节引用另一章节)
- CONFIGURES: 配置关系 (配置项影响组件)
- SUPERSEDES: 版本替代关系
- RELATED_TO: 通用关联关系
"""
import json
import hashlib
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from loguru import logger

from .entity_extractor import Entity, EntityType


class RelationType(Enum):
    """关系类型枚举"""
    DEPENDS_ON = "depends_on"       # A 依赖 B
    REQUIRES = "requires"           # A 需要 B (安装/运行)
    BELONGS_TO = "belongs_to"       # A 属于 B
    CONTAINS = "contains"           # A 包含 B
    REFERENCES = "references"       # A 引用 B
    CONFIGURES = "configures"       # A 配置 B
    AFFECTS = "affects"             # A 影响 B
    SUPERSEDES = "supersedes"       # A 替代 B (版本)
    COMPATIBLE_WITH = "compatible_with"  # A 兼容 B
    RELATED_TO = "related_to"       # A 关联 B (通用)
    IMPLEMENTS = "implements"       # A 实现 B
    EXTENDS = "extends"             # A 扩展 B
    CALLS = "calls"                 # A 调用 B
    USES = "uses"                   # A 使用 B


# 核心关系类型白名单 - 只保留这些高价值关系
CORE_RELATION_TYPES = {
    RelationType.DEPENDS_ON,
    RelationType.REQUIRES,
    RelationType.BELONGS_TO,
    RelationType.CONTAINS,
    RelationType.CONFIGURES,
    RelationType.IMPLEMENTS,
    RelationType.CALLS,
}


@dataclass
class Relation:
    """
    关系数据结构
    
    Attributes:
        relation_id: 关系唯一标识
        source_id: 源实体 ID
        target_id: 目标实体 ID
        relation_type: 关系类型
        properties: 关系属性
        confidence: 置信度
        source_chunk: 来源 chunk ID
        description: 关系描述
    """
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_chunk: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "confidence": self.confidence,
            "source_chunk": self.source_chunk,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            relation_id=data["relation_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source_chunk=data.get("source_chunk", ""),
            description=data.get("description", "")
        )
    
    @staticmethod
    def generate_id(source_id: str, target_id: str, relation_type: RelationType) -> str:
        """生成关系 ID"""
        content = f"{source_id}:{relation_type.value}:{target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# LLM 关系抽取 Prompt
RELATION_EXTRACTION_PROMPT = """你是一个技术文档关系抽取专家。请分析以下实体，结合文档上下文，尽可能多地抽取它们之间的关系。

文档内容:
{content}

已识别的实体:
{entities}

请尽可能多地抽取实体之间的关系，支持的关系类型:
1. depends_on - A 依赖 B (如: MySQL 依赖 JDK, 应用依赖数据库)
2. requires - A 需要 B 才能运行/安装 (如: 功能需要配置项)
3. belongs_to - A 属于 B (如: API 属于模块, 配置项属于组件)
4. contains - A 包含 B (如: 模块包含API, 产品包含功能)
5. references - A 引用 B (如: 文档引用概念, API引用类型)
6. configures - A 配置项影响 B 组件 (如: max_connections配置数据库)
7. affects - A 影响 B 的行为 (如: 版本影响功能, 配置影响性能)
8. supersedes - A 版本替代 B 版本 (如: v2.0替代v1.0)
9. compatible_with - A 兼容 B (如: 版本兼容性)
10. related_to - A 与 B 相关 (通用关系，用于任何相关但不确定类型的关系)
11. implements - A 实现 B (接口) (如: 类实现接口)
12. extends - A 扩展 B (如: 功能扩展基础功能)
13. calls - A 调用 B (如: API调用服务, 函数调用函数)
14. uses - A 使用 B (如: 应用使用数据库, 功能使用API)

请以 JSON 格式输出:
```json
{{
    "relations": [
        {{
            "source": "源实体名称",
            "target": "目标实体名称", 
            "type": "关系类型",
            "description": "关系描述",
            "confidence": 0.9
        }}
    ]
}}
```

重要提示:
- 尽可能多地抽取关系，包括所有明确和隐含的关系
- 如果两个实体在同一段文字中出现，很可能存在关系（如related_to）
- confidence 表示关系的确定程度 (0.0-1.0)，即使不确定也可以设为0.3-0.5
- 关系应该有明确的方向性
- 不要遗漏常见的关系：如产品使用技术、API属于产品、配置影响功能等
- 如果实体A和B在同一句话或段落中，尝试找出它们的关系

输出:"""


class RelationBuilder:
    """
    关系构建器
    
    从实体和文档上下文中抽取关系:
    1. 基于 LLM 的语义关系抽取
    2. 基于规则的结构化关系抽取
    3. 基于共现的关联关系推断
    """
    
    def __init__(
        self,
        llm_client=None,
        model: Optional[str] = None,  # None 时使用配置中的模型
        min_confidence: Optional[float] = None,
        use_cooccurrence: Optional[bool] = None,
        cooccurrence_threshold: Optional[int] = None,
        use_core_relations_only: Optional[bool] = None,
        max_relations_per_chunk: Optional[int] = None
    ):
        self.llm_client = llm_client
        # 如果未指定模型，将在 _init_llm_client 中从配置读取
        self.model = model
        
        # 从 settings 加载 Graph RAG 配置（如果可用）
        try:
            from config.settings import settings
            # 使用传入参数或 settings 配置，都有默认值回退
            self.min_confidence = min_confidence or (settings.graph.get('relation_min_confidence') if settings.graph else None) or 0.3
            self.use_cooccurrence = use_cooccurrence if use_cooccurrence is not None else (settings.graph.get('use_cooccurrence') if settings.graph else None) or False
            self.cooccurrence_threshold = cooccurrence_threshold or (settings.graph.get('cooccurrence_threshold') if settings.graph else None) or 3
            self.use_core_relations_only = use_core_relations_only if use_core_relations_only is not None else (settings.graph.get('use_core_relations_only') if settings.graph else None) or True
            self.max_relations_per_chunk = max_relations_per_chunk or (settings.graph.get('relation_max_per_chunk') if settings.graph else None) or 20
        except (ImportError, AttributeError):
            # 如果 settings 不可用，使用默认值
            self.min_confidence = min_confidence or 0.3
            self.use_cooccurrence = use_cooccurrence if use_cooccurrence is not None else False
            self.cooccurrence_threshold = cooccurrence_threshold or 3
            self.use_core_relations_only = use_core_relations_only if use_core_relations_only is not None else True
            self.max_relations_per_chunk = max_relations_per_chunk or 20
        
        self._init_llm_client()
        
        # 实体名称到 ID 的映射
        self._entity_name_map: Dict[str, str] = {}
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is not None:
            return
        try:
            from openai import OpenAI
            from config.settings import settings
            
            # 优先使用本地 LLM 服务 (OpenAI 兼容 API)
            if settings.llm.writing_provider == "local":
                self.llm_client = OpenAI(
                    api_key="EMPTY",  # 本地服务通常不校验 key
                    base_url=settings.llm.local_api_base
                )
                self.model = settings.llm.local_model  # 使用配置的本地模型
            elif settings.llm.writing_provider == "openai":
                self.llm_client = OpenAI(
                    api_key=settings.llm.openai_api_key
                )
                self.model = settings.llm.writing_model
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm.writing_provider}")
            
            logger.info(f"RelationBuilder initialized LLM client: provider={settings.llm.writing_provider}, model={self.model}")
        except Exception as e:
            logger.error(f"Failed to init LLM client for RelationBuilder: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def build_relations(
        self,
        entities: List[Entity],
        chunks: List[Dict[str, Any]],
        max_concurrent: int = 3,
        max_chunks: int = 15
    ) -> List[Relation]:
        """
        构建实体间的关系（优化版）
        
        Args:
            entities: 实体列表
            chunks: 文档 chunks，用于提供上下文
            max_concurrent: 最大并发数
            max_chunks: 最大处理 chunk 数量
            
        Returns:
            关系列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 构建实体名称映射
        self._build_entity_map(entities)
        
        all_relations = []
        
        # 1. 筛选有实体的 chunks
        chunks_with_entities = []
        for chunk in chunks[:max_chunks]:  # 限制处理数量
            content = chunk.get("content", "")
            chunk_entities = self._find_entities_in_chunk(entities, content)
            if len(chunk_entities) >= 2:
                chunks_with_entities.append({
                    "chunk": chunk,
                    "entities": chunk_entities
                })
        
        logger.info(f"Processing {len(chunks_with_entities)} chunks for relations")
        
        # 2. 并行 LLM 抽取语义关系
        if chunks_with_entities and self.llm_client:
            with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                futures = {}
                for item in chunks_with_entities:
                    chunk = item["chunk"]
                    chunk_entities = item["entities"]
                    future = executor.submit(
                        self._extract_relations_llm,
                        chunk.get("content", ""),
                        chunk_entities,
                        chunk.get("chunk_id", "")
                    )
                    futures[future] = chunk.get("chunk_id", "")
                
                for future in as_completed(futures):
                    try:
                        relations = future.result()
                        all_relations.extend(relations)
                    except Exception as e:
                        logger.error(f"Relation extraction failed: {e}")
        
        # 3. 基于共现的关系推断（快速，不需要 LLM）
        if self.use_cooccurrence:
            cooc_relations = self._build_cooccurrence_relations(entities)
            all_relations.extend(cooc_relations)
        
        # 4. 去重和合并
        merged_relations = self._merge_relations(all_relations)
        
        # 5. 过滤低置信度
        filtered_relations = [
            r for r in merged_relations 
            if r.confidence >= self.min_confidence
        ]
        
        logger.info(f"Built {len(filtered_relations)} relations")
        return filtered_relations
    
    def _build_entity_map(self, entities: List[Entity]):
        """构建实体名称到 ID 的映射"""
        self._entity_name_map.clear()
        
        for entity in entities:
            # 主名称
            self._entity_name_map[entity.name.lower()] = entity.entity_id
            
            # 别名
            for alias in entity.aliases:
                self._entity_name_map[alias.lower()] = entity.entity_id
    
    def _find_entities_in_chunk(
        self, 
        entities: List[Entity], 
        content: str
    ) -> List[Entity]:
        """找到 chunk 中出现的实体"""
        content_lower = content.lower()
        found = []
        
        for entity in entities:
            # 检查实体名称或别名是否出现在内容中
            if entity.name.lower() in content_lower:
                found.append(entity)
            else:
                for alias in entity.aliases:
                    if alias.lower() in content_lower:
                        found.append(entity)
                        break
        
        return found
    
    def _extract_relations_llm(
        self,
        content: str,
        entities: List[Entity],
        chunk_id: str
    ) -> List[Relation]:
        """使用 LLM 抽取关系"""
        if not self.llm_client or len(entities) < 2:
            return []
        
        try:
            # 构建实体描述（增加数量以看到更多实体）
            entity_desc = "\n".join([
                f"- {e.name} ({e.entity_type.value}): {e.description}"
                for e in entities[:50]  # 增加数量（从20提高到50）
            ])
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是技术文档关系抽取专家，擅长识别技术实体之间的关系。"
                    },
                    {
                        "role": "user",
                        "content": RELATION_EXTRACTION_PROMPT.format(
                            content=content[:5000],  # 增加内容长度（从3000提高到5000）
                            entities=entity_desc
                        )
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            if not response or not response.choices:
                logger.warning("Relation extraction returned empty response")
                return []

            message = response.choices[0].message
            raw_content = getattr(message, "content", None)
            if not raw_content:
                logger.warning("Relation extraction returned empty content")
                return []

            try:
                result = json.loads(raw_content)
            except Exception:
                # 尝试从文本中恢复 JSON
                start = raw_content.find("{")
                end = raw_content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        result = json.loads(raw_content[start:end + 1])
                    except Exception as inner_error:
                        logger.error(f"Relation extraction JSON recovery failed: {inner_error}")
                        return []
                else:
                    logger.error("Relation extraction JSON not found in response")
                    return []

            return self._parse_relations(result, chunk_id)
            
        except Exception as e:
            logger.error(f"Relation extraction failed: {e}")
            return []
    
    def _parse_relations(
        self, 
        result: Dict, 
        chunk_id: str
    ) -> List[Relation]:
        """解析 LLM 返回的关系"""
        relations = []
        
        for item in result.get("relations", []):
            # 限制每个 chunk 的关系数量
            if len(relations) >= self.max_relations_per_chunk:
                break
                
            try:
                source_name = item.get("source", "").lower()
                target_name = item.get("target", "").lower()
                type_str = item.get("type", "related_to").lower()
                
                # 查找实体 ID
                source_id = self._entity_name_map.get(source_name)
                target_id = self._entity_name_map.get(target_name)
                
                if not source_id or not target_id:
                    continue
                
                if source_id == target_id:
                    continue  # 跳过自引用
                
                # 映射关系类型
                relation_type = self._map_relation_type(type_str)
                
                # 过滤非核心关系类型
                if self.use_core_relations_only and relation_type not in CORE_RELATION_TYPES:
                    continue
                
                relation = Relation(
                    relation_id=Relation.generate_id(source_id, target_id, relation_type),
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    confidence=item.get("confidence", 0.8),
                    source_chunk=chunk_id,
                    description=item.get("description", "")
                )
                
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to parse relation: {e}")
                continue
        
        return relations
    
    def _map_relation_type(self, type_str: str) -> RelationType:
        """映射关系类型字符串到枚举"""
        type_mapping = {
            "depends_on": RelationType.DEPENDS_ON,
            "requires": RelationType.REQUIRES,
            "belongs_to": RelationType.BELONGS_TO,
            "contains": RelationType.CONTAINS,
            "references": RelationType.REFERENCES,
            "configures": RelationType.CONFIGURES,
            "affects": RelationType.AFFECTS,
            "supersedes": RelationType.SUPERSEDES,
            "compatible_with": RelationType.COMPATIBLE_WITH,
            "related_to": RelationType.RELATED_TO,
            "implements": RelationType.IMPLEMENTS,
            "extends": RelationType.EXTENDS,
            "calls": RelationType.CALLS,
            "uses": RelationType.USES,
        }
        return type_mapping.get(type_str, RelationType.RELATED_TO)
    
    def _build_cooccurrence_relations(
        self, 
        entities: List[Entity]
    ) -> List[Relation]:
        """
        基于共现构建关系
        
        如果两个实体在多个 chunk 中同时出现，建立弱关联关系
        """
        from collections import defaultdict
        
        # 统计共现次数
        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # 构建 chunk -> entities 映射
        chunk_entities: Dict[str, Set[str]] = defaultdict(set)
        for entity in entities:
            for chunk_id in entity.source_chunks:
                chunk_entities[chunk_id].add(entity.entity_id)
        
        # 统计同一 chunk 中的实体对
        for chunk_id, entity_ids in chunk_entities.items():
            entity_list = list(entity_ids)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    pair: Tuple[str, str] = (
                        min(entity_list[i], entity_list[j]),
                        max(entity_list[i], entity_list[j])
                    )
                    cooccurrence[pair] += 1
        
        # 创建关系
        relations = []
        for (e1, e2), count in cooccurrence.items():
            if count >= self.cooccurrence_threshold:
                # 共现次数越多，置信度越高
                confidence = min(0.5 + count * 0.1, 0.9)
                
                relation = Relation(
                    relation_id=Relation.generate_id(e1, e2, RelationType.RELATED_TO),
                    source_id=e1,
                    target_id=e2,
                    relation_type=RelationType.RELATED_TO,
                    confidence=confidence,
                    properties={"cooccurrence_count": count},
                    description=f"共现 {count} 次"
                )
                relations.append(relation)
        
        return relations
    
    def _merge_relations(self, relations: List[Relation]) -> List[Relation]:
        """合并重复关系"""
        merged: Dict[str, Relation] = {}
        
        for relation in relations:
            key = relation.relation_id
            
            if key in merged:
                # 保留置信度更高的
                if relation.confidence > merged[key].confidence:
                    merged[key] = relation
            else:
                merged[key] = relation
        
        return list(merged.values())
    
    def build_structural_relations(
        self,
        entities: List[Entity],
        doc_structure: Dict[str, Any]
    ) -> List[Relation]:
        """
        基于文档结构构建关系
        
        例如：章节包含关系、API 归属关系等
        """
        relations = []
        
        # 按章节路径分组实体
        section_entities: Dict[str, List[Entity]] = {}
        for entity in entities:
            section = entity.properties.get("section_path", "")
            if section:
                if section not in section_entities:
                    section_entities[section] = []
                section_entities[section].append(entity)
        
        # 同一章节的实体建立 BELONGS_TO 关系
        for section, ents in section_entities.items():
            if len(ents) >= 2:
                # 假设第一个是主要实体
                main_entity = ents[0]
                for other in ents[1:]:
                    if main_entity.entity_type == EntityType.PRODUCT:
                        relation = Relation(
                            relation_id=Relation.generate_id(
                                other.entity_id, 
                                main_entity.entity_id, 
                                RelationType.BELONGS_TO
                            ),
                            source_id=other.entity_id,
                            target_id=main_entity.entity_id,
                            relation_type=RelationType.BELONGS_TO,
                            confidence=0.7,
                            description=f"属于 {main_entity.name}"
                        )
                        relations.append(relation)
        
        return relations
