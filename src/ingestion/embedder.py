"""
向量化模块
支持多种 Embedding 模型
"""
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger

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
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str = "BAAI/bge-m3",
        dimension: int = 1024,
        batch_size: int = 32,
        openai_api_key: Optional[str] = None
    ):
        self.provider = provider
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.openai_api_key = openai_api_key
        
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "huggingface":
            self._init_huggingface()
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
    
    def embed_text(self, text: str) -> np.ndarray:
        """对单个文本进行向量化"""
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """对多个文本进行向量化"""
        if self.provider == "openai":
            return self._embed_openai(texts)
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
