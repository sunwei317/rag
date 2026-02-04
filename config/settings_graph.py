"""
Graph RAG 配置类
直接定义在 settings.py 中，避免外部文件
"""
from pydantic import Field 
from pydantic_settings import BaseSettings
from typing import Optional


class GraphSettings(BaseSettings):
    """Graph RAG 配置"""
    
    # 实体抽取配置
    entity_min_mentions: int = Field(default=1, description="实体最少提及次数", env="GRAPH_ENTITY_MIN_MENTIONS")
    entity_batch_size: int = Field(default=3, description="实体抽取批次大小", env="GRAPH_ENTITY_BATCH_SIZE")
    entity_min_content_length: int = Field(default=30, description="实体最小内容长度（字符数）", env="GRAPH_ENTITY_MIN_CONTENT_LENGTH")
    entity_max_tokens: int = Field(default=8000, description="实体抽取最大token数", env="GRAPH_ENTITY_MAX_TOKENS")
    
    # 关系构建配置
    relation_min_confidence: float = Field(default=0.3, description="关系最小置信度", env="GRAPH_RELATION_MIN_CONFIDENCE")
    relation_max_per_chunk: int = Field(default=20, description="每个chunk最大关系数", env="GRAPH_RELATION_MAX_PER_CHUNK")
    relation_max_concurrent: int = Field(default=2, description="关系构建最大并发数", env="GRAPH_RELATION_MAX_CONCURRENT")
    relation_content_length: int = Field(default=4000, description="关系构建最大内容长度", env="GRAPH_RELATION_CONTENT_LENGTH")
    use_cooccurrence: bool = Field(default=True, description="启用共现关系", env="GRAPH_USE_CO_OCCURRENCE")
    
    # 通用配置
    max_entities_per_type: int = Field(default=50, description="每种实体类型最大数量", env="GRAPH_MAX_ENTITIES_PER_TYPE")
    max_relations_total: int = Field(default=500, description="总关系数限制", env="GRAPH_MAX_RELATIONS_TOTAL")
    
    # 性能优化配置
    enable_cache: bool = Field(default=True, description="启用缓存", env="GRAPH_ENABLE_CACHE")
    cache_persist_path: Optional[str] = Field(default="./data/graph_cache", description="缓存持久化路径", env="GRAPH_CACHE_PERSIST_PATH")
    
    # LLM 调用配置
    extract_timeout: int = Field(default=60, description="实体抽取超时时间（秒）", env="GRAPH_EXTRACT_TIMEOUT")
    relation_timeout: int = Field(default=90, description="关系构建超时时间（秒）", env="GRAPH_RELATION_TIMEOUT")
    
    class Config:
        env_prefix = "GRAPH_"  # 使用 GRAPH_ 作为环境变量前缀