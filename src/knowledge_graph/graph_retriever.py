"""
图检索模块
从知识图谱中检索相关子图，作为 RAG 的额外上下文

检索策略:
1. 实体匹配: 从查询中识别实体，获取相关节点
2. 邻居扩展: 扩展到 N 跳邻居
3. 子图构建: 构建包含相关实体和关系的子图
4. 上下文生成: 将子图转换为文本上下文
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .entity_extractor import Entity, EntityType
from .relation_builder import Relation, RelationType
from .graph_store import GraphStore


@dataclass
class SubGraph:
    """
    子图结构
    
    包含从知识图谱中检索到的相关实体和关系
    """
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    query_entities: List[Entity] = field(default_factory=list)  # 查询命中的实体
    
    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "query_entities": [e.to_dict() for e in self.query_entities]
        }
    
    def is_empty(self) -> bool:
        return len(self.entities) == 0
    
    def to_context_text(self, max_length: int = 2000) -> str:
        """
        将子图转换为文本上下文
        
        格式:
        【知识图谱上下文】
        实体: MySQL (product) - 关系型数据库管理系统
        实体: JDK (dependency) - Java 开发工具包
        关系: MySQL --[requires]--> JDK
        """
        if self.is_empty():
            return ""
        
        lines = ["【知识图谱上下文】"]
        
        # 添加实体信息
        lines.append("\n相关实体:")
        for entity in self.entities[:20]:  # 限制数量
            desc = f" - {entity.description}" if entity.description else ""
            lines.append(f"  • {entity.name} ({entity.entity_type.value}){desc}")
        
        # 添加关系信息
        if self.relations:
            lines.append("\n实体关系:")
            entity_map = {e.entity_id: e.name for e in self.entities}
            
            for rel in self.relations[:30]:  # 限制数量
                source_name = entity_map.get(rel.source_id, rel.source_id)
                target_name = entity_map.get(rel.target_id, rel.target_id)
                lines.append(f"  • {source_name} --[{rel.relation_type.value}]--> {target_name}")
        
        text = "\n".join(lines)
        
        # 截断
        if len(text) > max_length:
            text = text[:max_length] + "\n..."
        
        return text
    
    def to_structured_context(self) -> Dict[str, Any]:
        """
        返回结构化的上下文信息
        
        用于更精确的 LLM prompt 构建
        """
        entity_map = {e.entity_id: e for e in self.entities}
        
        # 构建实体摘要
        entity_summaries = []
        for entity in self.entities:
            summary = {
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "aliases": entity.aliases
            }
            entity_summaries.append(summary)
        
        # 构建关系三元组
        triples = []
        for rel in self.relations:
            source = entity_map.get(rel.source_id)
            target = entity_map.get(rel.target_id)
            if source and target:
                triples.append({
                    "subject": source.name,
                    "predicate": rel.relation_type.value,
                    "object": target.name,
                    "description": rel.description
                })
        
        return {
            "entities": entity_summaries,
            "relations": triples,
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }


class GraphRetriever:
    """
    图检索器
    
    从知识图谱中检索与查询相关的子图:
    1. 实体链接: 将查询中的概念链接到图谱实体
    2. 子图提取: 以命中实体为中心，扩展提取相关子图
    3. 路径发现: 发现实体间的关系路径
    """
    
    def __init__(
        self,
        graph_store: GraphStore,
        llm_client=None,
        model: str = "gpt-4.1-mini",
        max_entities: int = 20,
        max_depth: int = 2,
        min_relevance: float = 0.5
    ):
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.model = model
        self.max_entities = max_entities
        self.max_depth = max_depth
        self.min_relevance = min_relevance
        
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
    
    def retrieve(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        relation_types: Optional[List[RelationType]] = None,
        expand_depth: int = 1
    ) -> SubGraph:
        """
        检索相关子图
        
        Args:
            query: 用户查询
            entity_types: 过滤实体类型
            relation_types: 过滤关系类型
            expand_depth: 邻居扩展深度
            
        Returns:
            SubGraph: 检索到的子图
        """
        logger.debug(f"Graph retrieval for: {query[:50]}...")
        
        # 1. 实体链接 - 从查询中识别实体
        query_entities = self._link_entities(query, entity_types)
        
        if not query_entities:
            logger.debug("No entities found in query")
            return SubGraph()
        
        logger.debug(f"Found {len(query_entities)} entities in query")
        
        # 2. 子图扩展
        all_entities = set()
        all_relations = []
        
        for entity in query_entities:
            neighbors, relations = self.graph_store.get_neighbors(
                entity.entity_id,
                max_depth=expand_depth,
                relation_types=relation_types
            )
            
            all_entities.add(entity.entity_id)
            all_entities.update(e.entity_id for e in neighbors)
            all_relations.extend(relations)
        
        # 3. 获取完整实体信息
        entities = []
        for entity_id in all_entities:
            entity = self.graph_store.get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        # 4. 限制大小
        entities = entities[:self.max_entities]
        
        # 5. 获取实体间的关系
        entity_ids = {e.entity_id for e in entities}
        filtered_relations = [
            r for r in all_relations
            if r.source_id in entity_ids and r.target_id in entity_ids
        ]
        
        # 去重关系
        seen_relations = set()
        unique_relations = []
        for rel in filtered_relations:
            if rel.relation_id not in seen_relations:
                seen_relations.add(rel.relation_id)
                unique_relations.append(rel)
        
        subgraph = SubGraph(
            entities=entities,
            relations=unique_relations,
            query_entities=query_entities
        )
        
        logger.info(
            f"Retrieved subgraph: {len(entities)} entities, "
            f"{len(unique_relations)} relations"
        )
        
        return subgraph
    
    def _link_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Entity]:
        """
        实体链接 - 从查询中识别并链接到图谱实体
        
        策略:
        1. 直接匹配: 查询词直接匹配实体名称
        2. 模糊搜索: 使用图存储的搜索能力
        3. LLM 辅助: 使用 LLM 识别查询中的实体概念
        """
        entities = []
        
        # 1. 分词并直接匹配
        words = self._tokenize(query)
        for word in words:
            if len(word) < 2:
                continue
            found = self.graph_store.search_entities(
                word, 
                entity_types=entity_types,
                limit=3
            )
            entities.extend(found)
        
        # 2. 整体查询搜索
        full_search = self.graph_store.search_entities(
            query,
            entity_types=entity_types,
            limit=5
        )
        entities.extend(full_search)
        
        # 3. LLM 辅助实体识别 (可选)
        if self.llm_client and len(entities) < 3:
            llm_entities = self._llm_entity_recognition(query, entity_types)
            entities.extend(llm_entities)
        
        # 去重
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.entity_id not in seen:
                seen.add(entity.entity_id)
                unique_entities.append(entity)
        
        return unique_entities[:10]  # 限制返回数量
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        # 按非字母数字字符分割
        words = re.split(r'[^\w\u4e00-\u9fff]+', text)
        # 过滤空字符串和纯数字
        return [w for w in words if w and not w.isdigit()]
    
    def _llm_entity_recognition(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None
    ) -> List[Entity]:
        """使用 LLM 识别查询中的实体概念"""
        try:
            type_hint = ""
            if entity_types:
                type_hint = f"关注以下类型: {', '.join(t.value for t in entity_types)}"
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个技术术语识别专家。"
                    },
                    {
                        "role": "user",
                        "content": f"""从以下技术问题中提取关键的技术实体/术语（如产品名、API、配置项等）。
{type_hint}

问题: {query}

请直接输出实体名称列表，每行一个，不需要解释。"""
                    }
                ],
                temperature=0,
                max_tokens=200
            )
            
            # 解析响应
            content = response.choices[0].message.content.strip()
            entity_names = [
                name.strip().strip("•-")
                for name in content.split("\n")
                if name.strip()
            ]
            
            # 在图谱中搜索这些实体
            entities = []
            for name in entity_names:
                found = self.graph_store.search_entities(name, limit=1)
                entities.extend(found)
            
            return entities
            
        except Exception as e:
            logger.warning(f"LLM entity recognition failed: {e}")
            return []
    
    def find_paths(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 3
    ) -> List[List[Relation]]:
        """
        查找两个实体之间的关系路径
        
        用于回答 "A 和 B 有什么关系" 类型的问题
        """
        # 查找源实体
        source_entities = self.graph_store.search_entities(source_name, limit=1)
        if not source_entities:
            return []
        
        # 查找目标实体
        target_entities = self.graph_store.search_entities(target_name, limit=1)
        if not target_entities:
            return []
        
        source_id = source_entities[0].entity_id
        target_id = target_entities[0].entity_id
        
        # 使用图存储的路径查找
        if hasattr(self.graph_store, 'find_path'):
            path = self.graph_store.find_path(source_id, target_id, max_depth)
            if path is not None:
                return [path] if path else []
        
        return []
    
    def get_entity_context(
        self,
        entity_name: str,
        include_neighbors: bool = True
    ) -> Optional[str]:
        """
        获取单个实体的上下文描述
        
        用于增强对特定实体的理解
        """
        entities = self.graph_store.search_entities(entity_name, limit=1)
        if not entities:
            return None
        
        entity = entities[0]
        lines = [
            f"【{entity.name}】",
            f"类型: {entity.entity_type.value}",
        ]
        
        if entity.description:
            lines.append(f"描述: {entity.description}")
        
        if entity.aliases:
            lines.append(f"别名: {', '.join(entity.aliases)}")
        
        if include_neighbors:
            relations = self.graph_store.get_relations(entity.entity_id, "both")
            
            if relations:
                lines.append("\n相关关系:")
                entity_cache = {}
                
                for rel in relations[:10]:
                    # 获取关联实体
                    other_id = rel.target_id if rel.source_id == entity.entity_id else rel.source_id
                    
                    if other_id not in entity_cache:
                        other_entity = self.graph_store.get_entity(other_id)
                        if other_entity:
                            entity_cache[other_id] = other_entity.name
                        else:
                            entity_cache[other_id] = other_id
                    
                    other_name = entity_cache[other_id]
                    
                    if rel.source_id == entity.entity_id:
                        lines.append(f"  → {rel.relation_type.value} → {other_name}")
                    else:
                        lines.append(f"  ← {rel.relation_type.value} ← {other_name}")
        
        return "\n".join(lines)
    
    def retrieve_for_question(
        self,
        question: str,
        question_type: str = "factual"
    ) -> SubGraph:
        """
        根据问题类型智能检索
        
        Args:
            question: 用户问题
            question_type: 问题类型
                - "factual": 事实性问题 (什么是 X)
                - "relational": 关系性问题 (A 和 B 的关系)
                - "procedural": 过程性问题 (如何做 X)
                - "comparative": 比较性问题 (A 和 B 的区别)
        """
        if question_type == "relational":
            # 尝试识别两个实体
            entities = self._link_entities(question)
            if len(entities) >= 2:
                # 查找路径
                paths = []
                if hasattr(self.graph_store, 'find_path'):
                    path = self.graph_store.find_path(
                        entities[0].entity_id,
                        entities[1].entity_id,
                        max_depth=3
                    )
                    if path:
                        paths = path
                
                return SubGraph(
                    entities=entities,
                    relations=paths,
                    query_entities=entities[:2]
                )
        
        elif question_type == "procedural":
            # 过程性问题，关注 REQUIRES, DEPENDS_ON 关系
            return self.retrieve(
                question,
                relation_types=[
                    RelationType.REQUIRES,
                    RelationType.DEPENDS_ON,
                    RelationType.USES
                ],
                expand_depth=2
            )
        
        elif question_type == "comparative":
            # 比较性问题，获取两个实体的完整信息
            entities = self._link_entities(question)
            if len(entities) >= 2:
                all_entities = []
                all_relations = []
                
                for entity in entities[:2]:
                    neighbors, relations = self.graph_store.get_neighbors(
                        entity.entity_id,
                        max_depth=1
                    )
                    all_entities.append(entity)
                    all_entities.extend(neighbors)
                    all_relations.extend(relations)
                
                return SubGraph(
                    entities=all_entities,
                    relations=all_relations,
                    query_entities=entities[:2]
                )
        
        # 默认: 事实性问题
        return self.retrieve(question, expand_depth=1)
