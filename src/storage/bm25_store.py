"""
BM25 检索存储模块
支持内存 BM25 和 Elasticsearch
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from loguru import logger


@dataclass
class BM25SearchResult:
    """BM25 搜索结果"""
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class BM25Store(ABC):
    """BM25 存储抽象基类"""
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加文档"""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[BM25SearchResult]:
        """搜索"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """删除文档"""
        pass


class MemoryBM25Store(BM25Store):
    """
    内存 BM25 存储
    
    特点:
    - 无需额外服务
    - 适合中小规模文档 (<100k)
    - 支持中英文分词
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank-bm25: pip install rank-bm25")
        
        self.persist_path = persist_path
        
        # 文档存储
        self._documents: Dict[str, Dict] = {}  # id -> {content, metadata, tokens}
        self._id_list: List[str] = []
        self._corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        
        # 尝试加载持久化数据
        if persist_path and Path(persist_path).exists():
            self._load()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        支持中英文混合，使用 jieba 进行中文分词
        """
        try:
            import jieba
            # 使用 jieba 精确模式分词
            tokens = list(jieba.cut(text.lower(), cut_all=False))
            # 过滤空白和标点
            tokens = [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]
            return tokens
        except ImportError:
            # 降级到 n-gram 方式
            logger.warning("jieba not installed, falling back to n-gram tokenization")
            return self._tokenize_ngram(text)
    
    def _tokenize_ngram(self, text: str) -> List[str]:
        """
        N-gram 分词（备用方案）
        """
        tokens = []
        
        # 分离中英文
        segments = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_\-\.]+', text.lower())
        
        for seg in segments:
            if re.match(r'^[\u4e00-\u9fff]+$', seg):
                # 中文使用 n-gram
                tokens.extend(list(seg))
                if len(seg) >= 2:
                    for i in range(len(seg) - 1):
                        tokens.append(seg[i:i+2])
                if len(seg) >= 3:
                    for i in range(len(seg) - 2):
                        tokens.append(seg[i:i+3])
            else:
                tokens.append(seg)
        
        return tokens

    def add(
        self,
        ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加文档"""
        for doc_id, content, metadata in zip(ids, contents, metadatas):
            tokens = self._tokenize(content)
            
            self._documents[doc_id] = {
                "content": content,
                "metadata": metadata,
                "tokens": tokens
            }
            
            if doc_id not in self._id_list:
                self._id_list.append(doc_id)
                self._corpus.append(tokens)
        
        # 重建 BM25 索引
        self._rebuild_index()
        
        logger.info(f"Added {len(ids)} documents to BM25 index")
        
        # 持久化
        if self.persist_path:
            self._save()
    
    def _rebuild_index(self):
        """重建 BM25 索引"""
        from rank_bm25 import BM25Okapi
        
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[BM25SearchResult]:
        """搜索"""
        if not self._bm25:
            return []
        
        # 分词
        query_tokens = self._tokenize(query)
        
        # BM25 搜索
        scores = self._bm25.get_scores(query_tokens)
        
        # 获取 top_k 结果
        results = []
        scored_indices = list(enumerate(scores))
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in scored_indices[:top_k * 2]:  # 多取一些用于过滤
            if score <= 0:
                continue
            
            doc_id = self._id_list[idx]
            doc = self._documents[doc_id]
            
            # 元数据过滤
            if filter_dict:
                match = True
                for k, v in filter_dict.items():
                    doc_val = doc["metadata"].get(k)
                    if isinstance(v, list):
                        if doc_val not in v:
                            match = False
                            break
                    elif doc_val != v:
                        match = False
                        break
                
                if not match:
                    continue
            
            results.append(BM25SearchResult(
                chunk_id=doc_id,
                score=float(score),
                content=doc["content"],
                metadata=doc["metadata"]
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete(self, ids: List[str]):
        """删除文档"""
        for doc_id in ids:
            if doc_id in self._documents:
                idx = self._id_list.index(doc_id)
                del self._documents[doc_id]
                del self._id_list[idx]
                del self._corpus[idx]
        
        self._rebuild_index()
        
        if self.persist_path:
            self._save()
        
        logger.info(f"Deleted {len(ids)} documents from BM25 index")
    
    def add_chunks(self, chunks):
        """添加 Chunk 对象"""
        ids = [c.chunk_id for c in chunks]
        contents = [c.content for c in chunks]
        metadatas = [c.to_dict() for c in chunks]
        
        self.add(ids, contents, metadatas)
    
    def _save(self):
        """保存到文件"""
        data = {
            "documents": self._documents,
            "id_list": self._id_list,
            "corpus": self._corpus
        }
        
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.debug(f"BM25 index saved to {self.persist_path}")
    
    def _load(self):
        """从文件加载"""
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            
            self._documents = data["documents"]
            self._id_list = data["id_list"]
            self._corpus = data["corpus"]
            
            self._rebuild_index()
            
            logger.info(f"BM25 index loaded from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
    
    @property
    def count(self) -> int:
        """返回文档数量"""
        return len(self._documents)


class ElasticsearchBM25Store(BM25Store):
    """
    Elasticsearch BM25 存储
    
    特点:
    - 高性能，支持大规模文档
    - 原生 BM25
    - 丰富的查询语法
    """
    
    def __init__(
        self,
        index_name: str = "tech_docs_bm25",
        host: str = "localhost",
        port: int = 9200,
        **kwargs
    ):
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError("Please install elasticsearch: pip install elasticsearch")
        
        self.index_name = index_name
        self._client = Elasticsearch(
            hosts=[{"host": host, "port": port, "scheme": "http"}],
            **kwargs
        )
        
        # 创建索引 (如果不存在)
        if not self._client.indices.exists(index=index_name):
            self._create_index()
        
        logger.info(f"Elasticsearch initialized: {index_name}")
    
    def _create_index(self):
        """创建索引"""
        mappings = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",  # 中文分词
                        "search_analyzer": "ik_smart"
                    },
                    "doc_id": {"type": "keyword"},
                    "section_path": {"type": "keyword"},
                    "section_title": {"type": "text"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "product": {"type": "keyword"},
                    "version": {"type": "keyword"},
                    "keywords": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"
                        }
                    }
                }
            }
        }
        
        try:
            self._client.indices.create(index=self.index_name, body=mappings)
        except Exception as e:
            # 如果 ik 分词器不存在，使用默认分词
            mappings["mappings"]["properties"]["content"] = {"type": "text"}
            self._client.indices.create(index=self.index_name, body=mappings)
        
        logger.info(f"Created Elasticsearch index: {self.index_name}")
    
    def add(
        self,
        ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """添加文档"""
        from elasticsearch.helpers import bulk
        
        actions = []
        for doc_id, content, metadata in zip(ids, contents, metadatas):
            doc = {
                "_index": self.index_name,
                "_id": doc_id,
                "_source": {
                    "content": content,
                    **{k: v for k, v in metadata.items() if isinstance(v, (str, int, float, list))}
                }
            }
            actions.append(doc)
        
        bulk(self._client, actions)
        self._client.indices.refresh(index=self.index_name)
        
        logger.info(f"Added {len(ids)} documents to Elasticsearch")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[BM25SearchResult]:
        """搜索"""
        # 构建查询
        must = [{"match": {"content": query}}]
        filter_clauses = []
        
        if filter_dict:
            for k, v in filter_dict.items():
                if isinstance(v, list):
                    filter_clauses.append({"terms": {k: v}})
                else:
                    filter_clauses.append({"term": {k: v}})
        
        body = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses
                }
            },
            "size": top_k
        }
        
        response = self._client.search(index=self.index_name, body=body)
        
        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            results.append(BM25SearchResult(
                chunk_id=hit["_id"],
                score=hit["_score"],
                content=source.get("content", ""),
                metadata={k: v for k, v in source.items() if k != "content"}
            ))
        
        return results
    
    def delete(self, ids: List[str]):
        """删除文档"""
        from elasticsearch.helpers import bulk
        
        actions = [
            {"_op_type": "delete", "_index": self.index_name, "_id": doc_id}
            for doc_id in ids
        ]
        
        bulk(self._client, actions, raise_on_error=False)
        
        logger.info(f"Deleted {len(ids)} documents from Elasticsearch")
    
    def add_chunks(self, chunks):
        """添加 Chunk 对象"""
        ids = [c.chunk_id for c in chunks]
        contents = [c.content for c in chunks]
        metadatas = [c.to_dict() for c in chunks]
        
        self.add(ids, contents, metadatas)


def create_bm25_store(
    provider: str = "memory",
    **kwargs
) -> BM25Store:
    """创建 BM25 存储实例"""
    if provider == "memory":
        return MemoryBM25Store(**kwargs)
    elif provider == "elasticsearch":
        return ElasticsearchBM25Store(**kwargs)
    else:
        raise ValueError(f"Unknown BM25 store provider: {provider}")
