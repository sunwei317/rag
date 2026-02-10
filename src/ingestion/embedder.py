"""
向量化模块
支持多种 Embedding 模型
"""
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger
import httpx
import os

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
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        openai_api_key: Optional[str] = None,
        local_api_url: Optional[str] = None
    ):
        from config.settings import settings
        self.provider = provider or settings.embedding.provider
        self.model_name = model_name or settings.embedding.model_name
        self.dimension = dimension or settings.embedding.dimension
        self.batch_size = batch_size or settings.embedding.batch_size
        self.openai_api_key = openai_api_key or settings.llm.openai_api_key
        self.local_api_url = local_api_url or settings.embedding.local_api_url
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

        # OCR 文本会很长，按“字符预算”分批可显著降低 413 发生概率
        max_chars_per_request = int(os.getenv("EMBEDDING_MAX_CHARS_PER_REQUEST", "6000"))
        max_chars_per_item = int(os.getenv("EMBEDDING_MAX_CHARS_PER_ITEM", "2000"))
        current_batch: List[str] = []
        current_chars = 0

        for text in texts:
            text = (text or "").strip()
            # 单条过长会触发 413，先截断用于 embedding（不影响原始 chunk 内容）
            if len(text) > max_chars_per_item:
                logger.warning(
                    f"Truncating oversized text for embedding: {len(text)} -> {max_chars_per_item} chars"
                )
                text = text[:max_chars_per_item]
            text_len = len(text)
            if current_batch and (
                len(current_batch) >= self.batch_size
                or current_chars + text_len > max_chars_per_request
            ):
                embeddings.extend(self._embed_local_api_batch(current_batch))
                current_batch = []
                current_chars = 0

            current_batch.append(text)
            current_chars += text_len

        if current_batch:
            embeddings.extend(self._embed_local_api_batch(current_batch))
        
        return embeddings

    def _embed_local_api_batch(self, batch: List[str]) -> List[np.ndarray]:
        """单批请求本地 embedding API，遇到 413 时自动拆分重试"""
        try:
            response = self._http_client.post(
                self.local_api_url,
                json={"inputs": batch}
            )
            response.raise_for_status()

            result = response.json()
            parsed = self._parse_embeddings_result(result)
            if len(parsed) != len(batch):
                raise ValueError(
                    f"Embedding count mismatch: got {len(parsed)}, expected {len(batch)}"
                )
            return parsed
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 413 and len(batch) > 1:
                mid = len(batch) // 2
                logger.warning(
                    f"Embedding payload too large (413), splitting batch {len(batch)} -> {mid}+{len(batch)-mid}"
                )
                return self._embed_local_api_batch(batch[:mid]) + self._embed_local_api_batch(batch[mid:])
            if e.response is not None and e.response.status_code == 413 and len(batch) == 1:
                # 兜底：单条文本仍过大时再截断重试一次
                fallback_single_chars = int(os.getenv("EMBEDDING_FALLBACK_SINGLE_CHARS", "1000"))
                shortened = batch[0][:fallback_single_chars]
                logger.warning(
                    f"Single embedding item still too large (413), retrying with {fallback_single_chars}-char truncation"
                )
                response = self._http_client.post(self.local_api_url, json={"inputs": [shortened]})
                response.raise_for_status()
                parsed = self._parse_embeddings_result(response.json())
                if not parsed:
                    raise ValueError("Embedding API returned empty embedding for truncated item")
                return parsed
            logger.error(f"Local API embedding failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Local API embedding failed: {e}")
            raise

    def _parse_embeddings_result(self, result: Union[List, dict]) -> List[np.ndarray]:
        parsed: List[np.ndarray] = []
        if isinstance(result, list):
            for emb in result:
                if isinstance(emb, list):
                    parsed.append(np.array(emb, dtype=np.float32))
                elif isinstance(emb, dict) and "embedding" in emb:
                    parsed.append(np.array(emb["embedding"], dtype=np.float32))
        elif isinstance(result, dict):
            emb_list = result.get("embeddings", result.get("data", []))
            for emb in emb_list:
                if isinstance(emb, list):
                    parsed.append(np.array(emb, dtype=np.float32))
                elif isinstance(emb, dict) and "embedding" in emb:
                    parsed.append(np.array(emb["embedding"], dtype=np.float32))
        return parsed
    
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
