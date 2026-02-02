"""
混合检索模块
结合向量检索和 BM25 检索，使用 RRF (Reciprocal Rank Fusion) 融合
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.vector_store import VectorStore, VectorSearchResult
from src.storage.bm25_store import BM25Store, BM25SearchResult
from src.ingestion.embedder import Embedder


@dataclass
class RetrievalResult:
    """检索结果"""
    chunk_id: str
    content: str
    score: float
    vector_score: float
    bm25_score: float
    rrf_score: float
    metadata: Dict[str, Any]
    source: str  # 'vector', 'bm25', 'both'
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "rrf_score": self.rrf_score,
            "metadata": self.metadata,
            "source": self.source
        }


class HybridSearcher:
    """
    混合检索器
    
    结合两种检索方式:
    1. 向量检索: 语义理解，捕获概念相似性
    2. BM25 检索: 关键词匹配，精确定位技术术语
    
    融合策略:
    - Reciprocal Rank Fusion (RRF)
    - 加权融合
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        embedder: Embedder,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60  # RRF 参数
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf"  # 'rrf' or 'weighted'
    ) -> List[RetrievalResult]:
        """
        混合检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filter_dict: 过滤条件 (如 product, version)
            fusion_method: 融合方法 ('rrf' 或 'weighted')
        
        Returns:
            List[RetrievalResult]: 排序后的检索结果
        """
        logger.debug(f"Hybrid search: {query[:50]}...")
        
        # 向量检索
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # 多取一些用于融合
            filter_dict=filter_dict
        )
        
        # BM25 检索
        bm25_results = self.bm25_store.search(
            query=query,
            top_k=top_k * 2,
            filter_dict=filter_dict
        )
        
        # 融合结果
        if fusion_method == "rrf":
            fused_results = self._rrf_fusion(vector_results, bm25_results)
        else:
            fused_results = self._weighted_fusion(vector_results, bm25_results)
        
        # 截取 top_k
        return fused_results[:top_k]
    
    def _rrf_fusion(
        self,
        vector_results: List[VectorSearchResult],
        bm25_results: List[BM25SearchResult]
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF)
        
        RRF 分数 = sum(1 / (k + rank_i))
        其中 k 是一个常数 (通常为 60)，rank_i 是在第 i 个排序列表中的排名
        """
        # 构建 chunk_id 到排名的映射
        vector_ranks = {r.chunk_id: i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {r.chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        
        # 收集所有 chunk_id
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # 构建内容和元数据映射
        content_map = {}
        metadata_map = {}
        vector_score_map = {}
        bm25_score_map = {}
        
        for r in vector_results:
            content_map[r.chunk_id] = r.content
            metadata_map[r.chunk_id] = r.metadata
            vector_score_map[r.chunk_id] = r.score
        
        for r in bm25_results:
            if r.chunk_id not in content_map:
                content_map[r.chunk_id] = r.content
                metadata_map[r.chunk_id] = r.metadata
            bm25_score_map[r.chunk_id] = r.score
        
        # 计算 RRF 分数
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            if chunk_id in vector_ranks:
                rrf_score += self.vector_weight / (self.rrf_k + vector_ranks[chunk_id])
            
            if chunk_id in bm25_ranks:
                rrf_score += self.bm25_weight / (self.rrf_k + bm25_ranks[chunk_id])
            
            rrf_scores[chunk_id] = rrf_score
        
        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 构建结果
        results = []
        for chunk_id in sorted_ids:
            # 确定来源
            in_vector = chunk_id in vector_ranks
            in_bm25 = chunk_id in bm25_ranks
            
            if in_vector and in_bm25:
                source = "both"
            elif in_vector:
                source = "vector"
            else:
                source = "bm25"
            
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content_map.get(chunk_id, ""),
                score=rrf_scores[chunk_id],
                vector_score=vector_score_map.get(chunk_id, 0.0),
                bm25_score=bm25_score_map.get(chunk_id, 0.0),
                rrf_score=rrf_scores[chunk_id],
                metadata=metadata_map.get(chunk_id, {}),
                source=source
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        vector_results: List[VectorSearchResult],
        bm25_results: List[BM25SearchResult]
    ) -> List[RetrievalResult]:
        """
        加权融合
        
        归一化各自的分数后加权求和
        """
        # 归一化向量分数
        vector_scores = {}
        if vector_results:
            max_v = max(r.score for r in vector_results)
            min_v = min(r.score for r in vector_results)
            range_v = max_v - min_v if max_v != min_v else 1.0
            
            for r in vector_results:
                vector_scores[r.chunk_id] = (r.score - min_v) / range_v
        
        # 归一化 BM25 分数
        bm25_scores = {}
        if bm25_results:
            max_b = max(r.score for r in bm25_results)
            min_b = min(r.score for r in bm25_results)
            range_b = max_b - min_b if max_b != min_b else 1.0
            
            for r in bm25_results:
                bm25_scores[r.chunk_id] = (r.score - min_b) / range_b
        
        # 收集所有 chunk_id 和内容
        all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        content_map = {}
        metadata_map = {}
        raw_vector_scores = {}
        raw_bm25_scores = {}
        
        for r in vector_results:
            content_map[r.chunk_id] = r.content
            metadata_map[r.chunk_id] = r.metadata
            raw_vector_scores[r.chunk_id] = r.score
        
        for r in bm25_results:
            if r.chunk_id not in content_map:
                content_map[r.chunk_id] = r.content
                metadata_map[r.chunk_id] = r.metadata
            raw_bm25_scores[r.chunk_id] = r.score
        
        # 计算加权分数
        weighted_scores = {}
        for chunk_id in all_chunk_ids:
            v_score = vector_scores.get(chunk_id, 0.0)
            b_score = bm25_scores.get(chunk_id, 0.0)
            weighted_scores[chunk_id] = (
                self.vector_weight * v_score + 
                self.bm25_weight * b_score
            )
        
        # 排序
        sorted_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)
        
        # 构建结果
        results = []
        for chunk_id in sorted_ids:
            in_vector = chunk_id in vector_scores
            in_bm25 = chunk_id in bm25_scores
            
            if in_vector and in_bm25:
                source = "both"
            elif in_vector:
                source = "vector"
            else:
                source = "bm25"
            
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=content_map.get(chunk_id, ""),
                score=weighted_scores[chunk_id],
                vector_score=raw_vector_scores.get(chunk_id, 0.0),
                bm25_score=raw_bm25_scores.get(chunk_id, 0.0),
                rrf_score=0.0,
                metadata=metadata_map.get(chunk_id, {}),
                source=source
            ))
        
        return results
    
    def vector_only_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """仅向量检索"""
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                vector_score=r.score,
                bm25_score=0.0,
                rrf_score=0.0,
                metadata=r.metadata,
                source="vector"
            )
            for r in vector_results
        ]
    
    def bm25_only_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """仅 BM25 检索"""
        bm25_results = self.bm25_store.search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                vector_score=0.0,
                bm25_score=r.score,
                rrf_score=0.0,
                metadata=r.metadata,
                source="bm25"
            )
            for r in bm25_results
        ]
