"""
RAG 系统配置管理
支持环境变量和配置文件两种方式
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal
from pathlib import Path


class EmbeddingSettings(BaseSettings):
    """Embedding 模型配置"""
    provider: Literal["openai", "huggingface", "local"] = "huggingface"
    model_name: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 32
    
    # OpenAI Embedding
    openai_model: str = "text-embedding-3-large"
    
    class Config:
        env_prefix = "EMBEDDING_"


class LLMSettings(BaseSettings):
    """LLM 配置"""
    # 主写作模型
    writing_provider: Literal["openai", "anthropic", "google"] = "anthropic"
    writing_model: str = "claude-3-5-sonnet-20241022"
    
    # 规划模型 (需要强推理能力)
    planning_provider: Literal["openai", "anthropic"] = "openai"
    planning_model: str = "o3-mini"
    
    # 多模态模型 (PDF解析/图片理解)
    multimodal_provider: Literal["openai", "google"] = "google"
    multimodal_model: str = "gemini-1.5-pro"
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # 通用参数
    temperature: float = 0.3
    max_tokens: int = 4096
    
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
    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 10  # Rerank 后保留的数量
    
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
    # 混合检索权重
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    
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
    
    # 日志级别
    log_level: str = "INFO"
    
    # API 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def ensure_dirs(self):
        """确保必要的目录存在"""
        for dir_path in [self.data_dir, self.pdf_dir, self.processed_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
