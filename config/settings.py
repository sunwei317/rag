"""
RAG 系统配置管理
支持环境变量和配置文件两种方式
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal, Any, Dict
from pathlib import Path
from loguru import logger

# 导入 Graph 配置类
try:
    from .settings_graph import GraphSettings
    GRAPH_SETTINGS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GraphSettings = None
    GRAPH_SETTINGS_AVAILABLE = False


class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置"""
    provider: Literal["openai", "huggingface", "local", "local_api"] = Field(default="local_api", env="EMBEDDING_PROVIDER")
    model_name: str = Field(default="BAAI/bge-m3", env="EMBEDDING_MODEL_NAME")
    dimension: int = Field(default=1024, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    
    # OpenAI Embedding
    openai_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_OPENAI_MODEL")
    
    # 本地 HTTP API Embedding 服务
    local_api_url: str = Field(default="http://localhost:8080/embed", env="EMBEDDING_LOCAL_API_URL")
    
    class Config:
        env_prefix = "EMBEDDING_"


class LLMSettings(BaseSettings):
    """LLM 配置"""
    # 主写作模型
    writing_provider: Literal["openai", "anthropic", "google", "local"] = Field(default="local", env="LLM_WRITING_PROVIDER")
    writing_model: str = Field(default="gpt-oss-20b", env="LLM_WRITING_MODEL")
    
    # 规划模型 (需要强推理能力)
    planning_provider: Literal["openai", "anthropic", "local"] = Field(default="local", env="LLM_PLANNING_PROVIDER")
    planning_model: str = Field(default="gpt-oss-20b", env="LLM_PLANNING_MODEL")
    
    # 多模态模型 (PDF解析/图片理解)
    multimodal_provider: Literal["openai", "google", "local"] = Field(default="local", env="LLM_MULTIMODAL_PROVIDER")
    multimodal_model: str = Field(default="gpt-oss-20b", env="LLM_MULTIMODAL_MODEL")
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # 本地 LLM 服务配置 (OpenAI 兼容 API)
    local_api_base: str = Field(default="http://localhost:8001/v1", env="LLM_LOCAL_API_BASE")
    local_model: str = Field(default="gpt-oss-20b", env="LLM_LOCAL_MODEL")
    
    # 通用参数
    temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")
    
    class Config:
        env_prefix = "LLM_"


class VectorStoreSettings(BaseSettings):
    """向量库配置"""
    provider: Literal["chroma", "milvus", "pgvector"] = "chroma"
    
    # ChromaDB
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "tech_docs"
    
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "tech_docs"
    
    class Config:
        env_prefix = "VECTOR_"


class BM25Settings(BaseSettings):
    """BM25 检索配置"""
    provider: Literal["elasticsearch", "memory"] = "memory"
    
    # Elasticsearch
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_name: str = "tech_docs_bm25"
    
    # 内存 BM25 (适合小规模)
    persist_path: str = "./data/bm25_index.pkl"
    
    class Config:
        env_prefix = "BM25_"


class RerankerSettings(BaseSettings):
    """Reranker 配置"""
    enabled: bool = Field(default=True, env="RERANKER_ENABLED")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3", env="RERANKER_MODEL_NAME")
    top_k: int = Field(default=10, env="RERANKER_TOP_K")
    
    class Config:
        env_prefix = "RERANKER_"


class ChunkingSettings(BaseSettings):
    """切分配置"""
    # 子 Chunk (用于精确检索)
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50
    
    # 父 Chunk (用于上下文补充)
    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    
    # 窗口检索
    window_size: int = 3  # 前后各扩展多少个 chunk
    
    # 表格/代码块特殊处理
    keep_table_intact: bool = True
    keep_code_block_intact: bool = True
    
    class Config:
        env_prefix = "CHUNKING_"


class RetrievalSettings(BaseSettings):
    """检索配置"""
    # 混合检索权重 - 增加 BM25 权重以提升中文关键词匹配
    vector_weight: float = 0.4
    bm25_weight: float = 0.6
    
    # 检索数量
    initial_top_k: int = 30  # 初始召回数量
    final_top_k: int = 10    # 最终保留数量
    
    # 查询转换
    enable_query_expansion: bool = True
    enable_hyde: bool = True
    query_expansion_count: int = 3
    
    class Config:
        env_prefix = "RETRIEVAL_"


class GenerationSettings(BaseSettings):
    """文档生成配置"""
    # 分节生成
    max_section_tokens: int = 2000
    
    # 一致性校验
    enable_consistency_check: bool = True
    enable_terminology_alignment: bool = True
    
    # 引用要求
    require_citations: bool = True
    min_citation_coverage: float = 0.8  # 关键结论至少 80% 需要有引用
    
    # 输出格式
    default_output_format: Literal["markdown", "docx", "html"] = "markdown"
    
    class Config:
        env_prefix = "GENERATION_"


class Settings(BaseSettings):
    """主配置类"""
    # 项目路径
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path("./data"))
    pdf_dir: Path = Field(default_factory=lambda: Path("./data/pdfs"))
    processed_dir: Path = Field(default_factory=lambda: Path("./data/processed"))
    templates_dir: Path = Field(default_factory=lambda: Path("./data/templates"))
    
    # 子配置
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    bm25: BM25Settings = Field(default_factory=BM25Settings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    
    # Graph RAG 配置
    graph: Optional[dict] = None  # 将在初始化时设置
    
    # 日志级别
    log_level: str = "INFO"
    
    # API 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略额外的环境变量
    
    def ensure_dirs(self):
        """确保必要的目录存在"""
        for dir_path in [self.data_dir, self.pdf_dir, self.processed_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()

# Graph RAG 配置定义
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
