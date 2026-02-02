"""
Knowledge Graph 模块
实现 Graph RAG 增强能力

核心组件:
- EntityExtractor: 从文档中抽取实体
- RelationBuilder: 构建实体间关系
- GraphStore: 图谱存储 (支持 Neo4j/内存)
- GraphRetriever: 图谱检索，提取相关子图
- GraphRAG: 融合图谱上下文的 RAG 系统
- GraphBuildPipeline: 图谱构建流程
"""

from .entity_extractor import EntityExtractor, Entity, EntityType
from .relation_builder import RelationBuilder, Relation, RelationType
from .graph_store import GraphStore, Neo4jStore, InMemoryGraphStore
from .graph_retriever import GraphRetriever, SubGraph
from .graph_rag import GraphRAG
from .pipeline import GraphBuildPipeline, GraphBuildResult, create_graph_pipeline

__all__ = [
    # 实体抽取
    "EntityExtractor",
    "Entity", 
    "EntityType",
    # 关系构建
    "RelationBuilder",
    "Relation",
    "RelationType",
    # 图存储
    "GraphStore",
    "Neo4jStore",
    "InMemoryGraphStore",
    # 图检索
    "GraphRetriever",
    "SubGraph",
    # GraphRAG
    "GraphRAG",
    # Pipeline
    "GraphBuildPipeline",
    "GraphBuildResult",
    "create_graph_pipeline",
]
