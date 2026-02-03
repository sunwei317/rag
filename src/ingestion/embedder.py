"""
向量化模块
支持多种 Embedding 模型
"""
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger
import httpx

from .chunker import Chunk, ChunkingResult


@dataclass
class EmbeddingResult:
    """向量化结果"""
    chunk_id: str
    embedding: np.ndarray
    model: str


class Embedder:
    """
    向量化器
    
    支持:
    - OpenAI text-embedding-3-large
    - HuggingFace BGE-M3
    - 本地 Sentence Transformers
    - 本地 HTTP API 服务 (local_api)
    """
    
    def __init__(
        self,
        provider: str = "local_api",
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        batch_size: int = 32,
        openai_api_key: Optional[str] = None,
        local_api_url: str = "http://localhost:8080/embed"
    ):
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.openai_api_key = openai_api_key
        self.local_api_url = local_api_url
        
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "local_api":
            self._init_local_api()
        else:
            self._init_local()
    
    def _init_openai(self):
        """初始化 OpenAI Embedding"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"Initialized OpenAI Embedding: {self.model_name}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def _init_huggingface(self):
        """初始化 HuggingFace 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized HuggingFace model: {self.model_name}")
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def _init_local(self):
        """初始化本地模型"""
        self._init_huggingface()
    
    def _init_local_api(self):
        """初始化本地 HTTP API 客户端"""
        self._http_client = httpx.Client(timeout=60.0)
        logger.info(f"Initialized Local API Embedding: {self.local_api_url}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """对单个文本进行向量化"""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """对多个文本进行向量化"""
        if self.provider == "openai":
            return self._embed_openai(texts)
        elif self.provider == "local_api":
            return self._embed_local_api(texts)
        else:
            return self._embed_local(texts)
    
    def _embed_openai(self, texts: List[str]) -> List[np.ndarray]:
        """使用 OpenAI API 进行向量化"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            
            for item in response.data:
                embeddings.append(np.array(item.embedding))
        
        return embeddings
    
    def _embed_local(self, texts: List[str]) -> List[np.ndarray]:
        """使用本地模型进行向量化"""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        
        return [embeddings[i] for i in range(len(texts))]
    
    def _embed_local_api(self, texts: List[str]) -> List[np.ndarray]:
        """使用本地 HTTP API 进行向量化"""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                response = self._http_client.post(
                    self.local_api_url,
                    json={"inputs": batch}
                )
                response.raise_for_status()
                
                # 解析响应 - 假设返回格式为 [[...], [...], ...]
                result = response.json()
                
                # 处理不同的响应格式
                if isinstance(result, list):
                    for emb in result:
                        if isinstance(emb, list):
                            embeddings.append(np.array(emb, dtype=np.float32))
                        elif isinstance(emb, dict) and "embedding" in emb:
                            embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                elif isinstance(result, dict):
                    # 可能是 {"embeddings": [[...], [...]]} 格式
                    emb_list = result.get("embeddings", result.get("data", []))
                    for emb in emb_list:
                        if isinstance(emb, list):
                            embeddings.append(np.array(emb, dtype=np.float32))
                        elif isinstance(emb, dict) and "embedding" in emb:
                            embeddings.append(np.array(emb["embedding"], dtype=np.float32))
                            
            except Exception as e:
                logger.error(f"Local API embedding failed: {e}")
                raise
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddingResult]:
        """对 Chunk 列表进行向量化"""
        logger.info(f"开始向量化 {len(chunks)} 个 Chunks")
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append(EmbeddingResult(
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                model=self.model_name
            ))
        
        logger.info(f"向量化完成: {len(results)} 个向量")
        return results
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        对查询进行向量化
        
        注意: 某些模型对查询有特殊处理 (如添加前缀)
        """
        if self.provider == "huggingface" and "bge" in self.model_name.lower():
            # BGE 模型的查询需要添加前缀
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        return self.embed_text(query)


# 便捷函数
def create_embedder(
    provider: str = "huggingface",
    model_name: str = "BAAI/bge-m3"
) -> Embedder:
    """创建向量化器"""
    return Embedder(provider=provider, model_name=model_name)
