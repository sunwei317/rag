"""
向量存储模块
支持 ChromaDB 和 Milvus
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加向量"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """删除向量"""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB 向量存储
    
    特点:
    - 本地持久化
    - 支持元数据过滤
    - 开箱即用，无需额外服务
    """
    
    def __init__(
        self,
        collection_name: str = "tech_docs",
        persist_directory: str = "./data/chroma"
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Please install chromadb: pip install chromadb")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 初始化 ChromaDB 客户端
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized: {collection_name} at {persist_directory}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加向量到集合"""
        # ChromaDB 需要 list 格式的 embeddings
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # 清理元数据 (ChromaDB 只支持基本类型)
        cleaned_metadatas = []
        for meta in metadatas:
            cleaned = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned[k] = v
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    cleaned[k] = ",".join(v)  # 列表转字符串
                else:
                    cleaned[k] = str(v)
            cleaned_metadatas.append(cleaned)
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=contents,
            metadatas=cleaned_metadatas
        )
        
        logger.info(f"Added {len(ids)} vectors to ChromaDB")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        # 构建过滤条件
        where = None
        if filter_dict:
            where = {}
            for k, v in filter_dict.items():
                if isinstance(v, list):
                    where[k] = {"$in": v}
                else:
                    where[k] = v
        
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB 返回的是距离，需要转换为相似度
                distance = results["distances"][0][i]
                score = 1 - distance  # 余弦距离转相似度
                
                search_results.append(VectorSearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i]
                ))
        
        return search_results
    
    def delete(self, ids: List[str]):
        """删除向量"""
        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} vectors from ChromaDB")
    
    def get_all_chunks(
        self,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        获取所有 chunks (带可选过滤)
        
        Args:
            filter_dict: 过滤条件
            limit: 最大返回数量
            
        Returns:
            chunks 列表
        """
        # 构建过滤条件
        where = None
        if filter_dict:
            where = {}
            for k, v in filter_dict.items():
                if isinstance(v, list):
                    where[k] = {"$in": v}
                else:
                    where[k] = v
        
        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results and results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunk = {
                        "chunk_id": chunk_id,
                        "content": results["documents"][i] if results["documents"] else "",
                    }
                    if results["metadatas"] and results["metadatas"][i]:
                        chunk.update(results["metadatas"][i])
                    chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks: {e}")
            return []
    
    def add_chunks(self, chunks, embeddings):
        """添加 Chunk 对象"""
        # 去重：避免重复 chunk_id
        seen_ids = set()
        unique_chunks = []
        unique_embeddings = []
        
        for c, e in zip(chunks, embeddings):
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_chunks.append(c)
                unique_embeddings.append(e)
            else:
                logger.warning(f"Skipping duplicate chunk_id: {c.chunk_id}")
        
        if not unique_chunks:
            logger.warning("No unique chunks to add")
            return
        
        ids = [c.chunk_id for c in unique_chunks]
        contents = [c.content for c in unique_chunks]
        metadatas = [c.to_dict() for c in unique_chunks]
        embs = [e.embedding for e in unique_embeddings]
        
        self.add(ids, embs, contents, metadatas)
    
    @property
    def count(self) -> int:
        """返回集合中的向量数量"""
        return self._collection.count()


class MilvusVectorStore(VectorStore):
    """
    Milvus 向量存储
    
    特点:
    - 高性能，支持大规模向量
    - 支持复杂的过滤条件
    - 需要运行 Milvus 服务
    """
    
    def __init__(
        self,
        collection_name: str = "tech_docs",
        host: str = "localhost",
        port: int = 19530,
        dimension: int = 1024
    ):
        try:
            from pymilvus import (
                connections,
                Collection,
                FieldSchema,
                CollectionSchema,
                DataType,
                utility
            )
        except ImportError:
            raise ImportError("Please install pymilvus: pip install pymilvus")
        
        self.collection_name = collection_name
        self.dimension = dimension
        
        # 连接 Milvus
        connections.connect(host=host, port=port)
        
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            self._collection = Collection(collection_name)
        else:
            # 创建集合
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="page", dtype=DataType.INT64)
            ]
            
            schema = CollectionSchema(fields=fields, description="Technical documents")
            self._collection = Collection(name=collection_name, schema=schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
        
        # 加载集合到内存
        self._collection.load()
        
        logger.info(f"Milvus initialized: {collection_name}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加向量"""
        data = [
            ids,
            [emb.tolist() for emb in embeddings],
            contents,
            [m.get("doc_id", "") for m in metadatas],
            [m.get("section_path", "") for m in metadatas],
            [m.get("page_start", 0) for m in metadatas]
        ]
        
        self._collection.insert(data)
        self._collection.flush()
        
        logger.info(f"Added {len(ids)} vectors to Milvus")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """搜索相似向量"""
        # 构建过滤表达式
        expr = None
        if filter_dict:
            conditions = []
            for k, v in filter_dict.items():
                if isinstance(v, str):
                    conditions.append(f'{k} == "{v}"')
                elif isinstance(v, list):
                    values_str = ", ".join(f'"{x}"' for x in v)
                    conditions.append(f'{k} in [{values_str}]')
            if conditions:
                expr = " and ".join(conditions)
        
        results = self._collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            expr=expr,
            output_fields=["content", "doc_id", "section_path", "page"]
        )
        
        search_results = []
        for hit in results[0]:
            search_results.append(VectorSearchResult(
                chunk_id=hit.id,
                score=hit.score,
                content=hit.entity.get("content", ""),
                metadata={
                    "doc_id": hit.entity.get("doc_id", ""),
                    "section_path": hit.entity.get("section_path", ""),
                    "page": hit.entity.get("page", 0)
                }
            ))
        
        return search_results
    
    def delete(self, ids: List[str]):
        """删除向量"""
        expr = f'id in {ids}'
        self._collection.delete(expr)
        logger.info(f"Deleted {len(ids)} vectors from Milvus")
    
    def add_chunks(self, chunks, embeddings):
        """添加 Chunk 对象"""
        ids = [c.chunk_id for c in chunks]
        contents = [c.content for c in chunks]
        metadatas = [c.to_dict() for c in chunks]
        embs = [e.embedding for e in embeddings]
        
        self.add(ids, embs, contents, metadatas)


def create_vector_store(
    provider: str = "chroma",
    **kwargs
) -> VectorStore:
    """创建向量存储实例"""
    if provider == "chroma":
        return ChromaVectorStore(**kwargs)
    elif provider == "milvus":
        return MilvusVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")
