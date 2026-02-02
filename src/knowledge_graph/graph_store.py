"""
图存储模块
支持多种图数据库后端:
- Neo4j: 生产级图数据库
- InMemory: 内存图，适合开发/测试
- NetworkX: 基于 NetworkX 的本地存储
"""
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .entity_extractor import Entity, EntityType
from .relation_builder import Relation, RelationType


@dataclass
class GraphStats:
    """图谱统计信息"""
    node_count: int
    edge_count: int
    entity_types: Dict[str, int]
    relation_types: Dict[str, int]
    
    def to_dict(self) -> Dict:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
            "relation_types": self.relation_types
        }


class GraphStore(ABC):
    """图存储抽象基类"""
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> bool:
        """添加实体节点"""
        pass
    
    @abstractmethod
    def add_relation(self, relation: Relation) -> bool:
        """添加关系边"""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        pass
    
    @abstractmethod
    def get_relations(
        self, 
        entity_id: str, 
        direction: str = "both"
    ) -> List[Relation]:
        """获取实体的关系"""
        pass
    
    @abstractmethod
    def get_neighbors(
        self, 
        entity_id: str, 
        max_depth: int = 1,
        relation_types: Optional[List[RelationType]] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """获取邻居节点"""
        pass
    
    @abstractmethod
    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10
    ) -> List[Entity]:
        """搜索实体"""
        pass
    
    @abstractmethod
    def get_stats(self) -> GraphStats:
        """获取图谱统计"""
        pass
    
    @abstractmethod
    def clear(self):
        """清空图谱"""
        pass


class InMemoryGraphStore(GraphStore):
    """
    内存图存储
    
    适用于:
    - 开发和测试
    - 小规模知识图谱 (<10万节点)
    - 不需要持久化的场景
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        
        # 存储结构
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        
        # 索引结构
        self._outgoing: Dict[str, Set[str]] = {}  # entity_id -> relation_ids (出边)
        self._incoming: Dict[str, Set[str]] = {}  # entity_id -> relation_ids (入边)
        self._name_index: Dict[str, str] = {}     # name.lower() -> entity_id
        
        # 加载已有数据
        if persist_path:
            self._load()
    
    def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        try:
            self._entities[entity.entity_id] = entity
            self._name_index[entity.name.lower()] = entity.entity_id
            
            # 添加别名索引
            for alias in entity.aliases:
                self._name_index[alias.lower()] = entity.entity_id
            
            # 初始化边索引
            if entity.entity_id not in self._outgoing:
                self._outgoing[entity.entity_id] = set()
            if entity.entity_id not in self._incoming:
                self._incoming[entity.entity_id] = set()
            
            return True
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        try:
            # 检查源和目标实体存在
            if relation.source_id not in self._entities:
                logger.warning(f"Source entity not found: {relation.source_id}")
                return False
            if relation.target_id not in self._entities:
                logger.warning(f"Target entity not found: {relation.target_id}")
                return False
            
            self._relations[relation.relation_id] = relation
            
            # 更新索引
            self._outgoing.setdefault(relation.source_id, set()).add(relation.relation_id)
            self._incoming.setdefault(relation.target_id, set()).add(relation.relation_id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            return False
    
    def add_entities_batch(self, entities: List[Entity]) -> int:
        """批量添加实体"""
        count = 0
        for entity in entities:
            if self.add_entity(entity):
                count += 1
        logger.info(f"Added {count} entities")
        return count
    
    def add_relations_batch(self, relations: List[Relation]) -> int:
        """批量添加关系"""
        count = 0
        for relation in relations:
            if self.add_relation(relation):
                count += 1
        logger.info(f"Added {count} relations")
        return count
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """通过名称获取实体"""
        entity_id = self._name_index.get(name.lower())
        if entity_id:
            return self._entities.get(entity_id)
        return None
    
    def get_relations(
        self, 
        entity_id: str, 
        direction: str = "both"
    ) -> List[Relation]:
        """
        获取实体的关系
        
        Args:
            entity_id: 实体 ID
            direction: 方向 ('out', 'in', 'both')
        """
        relation_ids = set()
        
        if direction in ("out", "both"):
            relation_ids.update(self._outgoing.get(entity_id, set()))
        
        if direction in ("in", "both"):
            relation_ids.update(self._incoming.get(entity_id, set()))
        
        return [self._relations[rid] for rid in relation_ids if rid in self._relations]
    
    def get_neighbors(
        self, 
        entity_id: str, 
        max_depth: int = 1,
        relation_types: Optional[List[RelationType]] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        获取邻居节点 (BFS)
        
        Args:
            entity_id: 起始实体 ID
            max_depth: 最大深度
            relation_types: 过滤关系类型
            
        Returns:
            (邻居实体列表, 路径关系列表)
        """
        visited_entities: Set[str] = {entity_id}
        visited_relations: Set[str] = set()
        
        current_level = {entity_id}
        
        for _ in range(max_depth):
            next_level = set()
            
            for eid in current_level:
                # 获取所有关系
                relations = self.get_relations(eid, "both")
                
                for rel in relations:
                    # 过滤关系类型
                    if relation_types and rel.relation_type not in relation_types:
                        continue
                    
                    visited_relations.add(rel.relation_id)
                    
                    # 获取邻居
                    neighbor_id = rel.target_id if rel.source_id == eid else rel.source_id
                    
                    if neighbor_id not in visited_entities:
                        visited_entities.add(neighbor_id)
                        next_level.add(neighbor_id)
            
            current_level = next_level
        
        # 移除起始节点
        visited_entities.discard(entity_id)
        
        entities = [self._entities[eid] for eid in visited_entities if eid in self._entities]
        relations = [self._relations[rid] for rid in visited_relations if rid in self._relations]
        
        return entities, relations
    
    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10
    ) -> List[Entity]:
        """
        搜索实体
        
        简单的模糊匹配实现
        """
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            # 类型过滤
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # 名称匹配
            score = 0
            if query_lower == entity.name.lower():
                score = 1.0
            elif query_lower in entity.name.lower():
                score = 0.8
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                score = 0.6
            elif query_lower in entity.description.lower():
                score = 0.4
            
            if score > 0:
                results.append((entity, score))
        
        # 按分数排序
        results.sort(key=lambda x: (-x[1], x[0].name))
        
        return [entity for entity, _ in results[:limit]]
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 4
    ) -> Optional[List[Relation]]:
        """
        查找两个实体之间的路径 (BFS)
        
        Returns:
            关系路径列表，如果不存在则返回 None
        """
        if source_id == target_id:
            return []
        
        # BFS
        queue = [(source_id, [])]
        visited = {source_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) >= max_depth:
                continue
            
            for rel in self.get_relations(current_id, "both"):
                next_id = rel.target_id if rel.source_id == current_id else rel.source_id
                
                if next_id == target_id:
                    return path + [rel]
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, path + [rel]))
        
        return None
    
    def get_stats(self) -> GraphStats:
        """获取统计信息"""
        entity_types = {}
        for entity in self._entities.values():
            t = entity.entity_type.value
            entity_types[t] = entity_types.get(t, 0) + 1
        
        relation_types = {}
        for relation in self._relations.values():
            t = relation.relation_type.value
            relation_types[t] = relation_types.get(t, 0) + 1
        
        return GraphStats(
            node_count=len(self._entities),
            edge_count=len(self._relations),
            entity_types=entity_types,
            relation_types=relation_types
        )
    
    def clear(self):
        """清空图谱"""
        self._entities.clear()
        self._relations.clear()
        self._outgoing.clear()
        self._incoming.clear()
        self._name_index.clear()
    
    def save(self, path: Optional[str] = None):
        """持久化到文件"""
        save_path = path or self.persist_path
        if not save_path:
            logger.warning("No persist path specified")
            return
        
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations.values()]
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved graph to {save_path}")
    
    def _load(self):
        """从文件加载"""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for e_dict in data.get("entities", []):
                entity = Entity.from_dict(e_dict)
                self.add_entity(entity)
            
            for r_dict in data.get("relations", []):
                relation = Relation.from_dict(r_dict)
                self.add_relation(relation)
            
            logger.info(f"Loaded graph from {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")


class Neo4jStore(GraphStore):
    """
    Neo4j 图存储
    
    适用于:
    - 生产环境
    - 大规模知识图谱
    - 复杂图查询需求
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        self._driver = None
        self._connect()
    
    def _connect(self):
        """建立连接"""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # 测试连接
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except ImportError:
            logger.error("neo4j package not installed. Run: pip install neo4j")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
    
    def add_entity(self, entity: Entity) -> bool:
        """添加实体节点"""
        try:
            with self._driver.session(database=self.database) as session:
                query = """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.name = $name,
                    e.entity_type = $entity_type,
                    e.aliases = $aliases,
                    e.description = $description,
                    e.mentions = $mentions,
                    e.properties = $properties
                """
                session.run(
                    query,
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    aliases=entity.aliases,
                    description=entity.description,
                    mentions=entity.mentions,
                    properties=json.dumps(entity.properties)
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add entity to Neo4j: {e}")
            return False
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系边"""
        try:
            with self._driver.session(database=self.database) as session:
                query = f"""
                MATCH (source:Entity {{entity_id: $source_id}})
                MATCH (target:Entity {{entity_id: $target_id}})
                MERGE (source)-[r:{relation.relation_type.value.upper()}]->(target)
                SET r.relation_id = $relation_id,
                    r.confidence = $confidence,
                    r.description = $description,
                    r.source_chunk = $source_chunk
                """
                session.run(
                    query,
                    source_id=relation.source_id,
                    target_id=relation.target_id,
                    relation_id=relation.relation_id,
                    confidence=relation.confidence,
                    description=relation.description,
                    source_chunk=relation.source_chunk
                )
            return True
        except Exception as e:
            logger.error(f"Failed to add relation to Neo4j: {e}")
            return False
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    "MATCH (e:Entity {entity_id: $entity_id}) RETURN e",
                    entity_id=entity_id
                )
                record = result.single()
                if record:
                    return self._record_to_entity(record["e"])
        except Exception as e:
            logger.error(f"Failed to get entity: {e}")
        return None
    
    def _record_to_entity(self, node) -> Entity:
        """将 Neo4j 节点转换为 Entity"""
        return Entity(
            entity_id=node["entity_id"],
            name=node["name"],
            entity_type=EntityType(node["entity_type"]),
            aliases=list(node.get("aliases", [])),
            description=node.get("description", ""),
            properties=json.loads(node.get("properties", "{}")),
            mentions=node.get("mentions", 1)
        )
    
    def get_relations(
        self, 
        entity_id: str, 
        direction: str = "both"
    ) -> List[Relation]:
        """获取实体的关系"""
        try:
            with self._driver.session(database=self.database) as session:
                if direction == "out":
                    query = """
                    MATCH (e:Entity {entity_id: $entity_id})-[r]->(t)
                    RETURN r, e.entity_id as source, t.entity_id as target, type(r) as rel_type
                    """
                elif direction == "in":
                    query = """
                    MATCH (s)-[r]->(e:Entity {entity_id: $entity_id})
                    RETURN r, s.entity_id as source, e.entity_id as target, type(r) as rel_type
                    """
                else:
                    query = """
                    MATCH (e:Entity {entity_id: $entity_id})-[r]-(t)
                    RETURN r, 
                           CASE WHEN startNode(r) = e THEN e.entity_id ELSE t.entity_id END as source,
                           CASE WHEN endNode(r) = e THEN e.entity_id ELSE t.entity_id END as target,
                           type(r) as rel_type
                    """
                
                result = session.run(query, entity_id=entity_id)
                relations = []
                for record in result:
                    rel = self._record_to_relation(
                        record["r"], 
                        record["source"], 
                        record["target"],
                        record["rel_type"]
                    )
                    relations.append(rel)
                return relations
        except Exception as e:
            logger.error(f"Failed to get relations: {e}")
            return []
    
    def _record_to_relation(
        self, 
        rel, 
        source_id: str, 
        target_id: str,
        rel_type: str
    ) -> Relation:
        """将 Neo4j 关系转换为 Relation"""
        return Relation(
            relation_id=rel.get("relation_id", ""),
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType(rel_type.lower()),
            confidence=rel.get("confidence", 1.0),
            description=rel.get("description", ""),
            source_chunk=rel.get("source_chunk", "")
        )
    
    def get_neighbors(
        self, 
        entity_id: str, 
        max_depth: int = 1,
        relation_types: Optional[List[RelationType]] = None
    ) -> Tuple[List[Entity], List[Relation]]:
        """获取邻居节点"""
        try:
            with self._driver.session(database=self.database) as session:
                # 构建关系类型过滤
                rel_filter = ""
                if relation_types:
                    types = "|".join([rt.value.upper() for rt in relation_types])
                    rel_filter = f":{types}"
                
                query = f"""
                MATCH path = (start:Entity {{entity_id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(end:Entity)
                WHERE start <> end
                UNWIND nodes(path) as n
                UNWIND relationships(path) as rel
                WITH DISTINCT n, rel
                RETURN collect(DISTINCT n) as nodes, collect(DISTINCT rel) as rels
                """
                
                result = session.run(query, entity_id=entity_id)
                record = result.single()
                
                if record:
                    entities = [
                        self._record_to_entity(n) 
                        for n in record["nodes"]
                        if n["entity_id"] != entity_id
                    ]
                    # 简化关系处理
                    relations = []
                    return entities, relations
                
                return [], []
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return [], []
    
    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10
    ) -> List[Entity]:
        """搜索实体"""
        try:
            with self._driver.session(database=self.database) as session:
                type_filter = ""
                if entity_types:
                    types = [f"'{t.value}'" for t in entity_types]
                    type_filter = f"AND e.entity_type IN [{','.join(types)}]"
                
                cypher = f"""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR ANY(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($query))
                   {type_filter}
                RETURN e
                LIMIT $limit
                """
                
                result = session.run(cypher, query=query, limit=limit)
                return [self._record_to_entity(record["e"]) for record in result]
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []
    
    def get_stats(self) -> GraphStats:
        """获取统计信息"""
        try:
            with self._driver.session(database=self.database) as session:
                # 节点数
                node_count = session.run(
                    "MATCH (n:Entity) RETURN count(n) as count"
                ).single()["count"]
                
                # 边数
                edge_count = session.run(
                    "MATCH ()-[r]->() RETURN count(r) as count"
                ).single()["count"]
                
                # 实体类型分布
                entity_types = {}
                result = session.run(
                    "MATCH (e:Entity) RETURN e.entity_type as type, count(*) as count"
                )
                for record in result:
                    entity_types[record["type"]] = record["count"]
                
                # 关系类型分布
                relation_types = {}
                result = session.run(
                    "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
                )
                for record in result:
                    relation_types[record["type"].lower()] = record["count"]
                
                return GraphStats(
                    node_count=node_count,
                    edge_count=edge_count,
                    entity_types=entity_types,
                    relation_types=relation_types
                )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return GraphStats(0, 0, {}, {})
    
    def clear(self):
        """清空图谱"""
        try:
            with self._driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j graph")
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
    
    def create_indexes(self):
        """创建索引以提升查询性能"""
        try:
            with self._driver.session(database=self.database) as session:
                # 实体 ID 索引
                session.run(
                    "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)"
                )
                # 实体名称索引
                session.run(
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )
                # 实体类型索引
                session.run(
                    "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
                )
            logger.info("Created Neo4j indexes")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
