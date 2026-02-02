"""
多路召回模块
并行执行多个查询，合并结果提高召回率
"""
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import time

from src.retrieval.hybrid_search import HybridSearcher, RetrievalResult
from src.retrieval.query_transformer import QueryTransformer, TransformedQuery


@dataclass
class MultiQueryResult:
    """多路召回结果"""
    results: List[RetrievalResult]
    queries_used: List[str]
    query_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "queries_used": self.queries_used,
            "query_stats": self.query_stats
        }


class MultiQueryRetriever:
    """
    多路召回检索器
    
    策略:
    1. 使用 QueryTransformer 生成多个查询版本
    2. 并行执行所有查询
    3. 使用 RRF 融合多路结果
    4. 去重并排序返回
    
    优势:
    - 提高召回率：不同查询可能命中不同的相关文档
    - 降低延迟：并行执行减少总时间
    - 鲁棒性：单个查询失败不影响整体结果
    """
    
    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        query_transformer: Optional[QueryTransformer] = None,
        max_parallel_queries: int = 5,
        rrf_k: int = 60,
        include_hyde: bool = True
    ):
        self.hybrid_searcher = hybrid_searcher
        self.query_transformer = query_transformer or QueryTransformer()
        self.max_parallel_queries = max_parallel_queries
        self.rrf_k = rrf_k
        self.include_hyde = include_hyde
        
        # 线程池用于并行检索
        self._executor = ThreadPoolExecutor(max_workers=max_parallel_queries)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> MultiQueryResult:
        """
        多路并行检索
        
        Args:
            query: 原始查询
            top_k: 最终返回数量
            filter_dict: 过滤条件
            context: 上下文信息 (用于查询扩展)
        
        Returns:
            MultiQueryResult: 合并后的检索结果
        """
        start_time = time.time()
        
        # 1. 生成多个查询版本
        transformed = self.query_transformer.transform(query, context)
        queries = self._build_query_list(transformed)
        
        logger.info(f"Multi-query retrieval with {len(queries)} queries")
        
        # 2. 并行执行检索
        all_results = self._parallel_search(queries, top_k * 2, filter_dict)
        
        # 3. 多路 RRF 融合
        fused_results = self._multi_rrf_fusion(all_results, queries)
        
        # 4. 截取 top_k
        final_results = fused_results[:top_k]
        
        elapsed = time.time() - start_time
        
        return MultiQueryResult(
            results=final_results,
            queries_used=queries,
            query_stats={
                "original_query": query,
                "num_queries": len(queries),
                "total_candidates": sum(len(r) for r in all_results.values()),
                "unique_results": len(fused_results),
                "elapsed_seconds": round(elapsed, 3)
            }
        )
    
    async def async_search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> MultiQueryResult:
        """异步多路并行检索"""
        start_time = time.time()
        
        # 1. 生成多个查询版本
        transformed = self.query_transformer.transform(query, context)
        queries = self._build_query_list(transformed)
        
        logger.info(f"Async multi-query retrieval with {len(queries)} queries")
        
        # 2. 异步并行执行检索
        all_results = await self._async_parallel_search(queries, top_k * 2, filter_dict)
        
        # 3. 多路 RRF 融合
        fused_results = self._multi_rrf_fusion(all_results, queries)
        
        # 4. 截取 top_k
        final_results = fused_results[:top_k]
        
        elapsed = time.time() - start_time
        
        return MultiQueryResult(
            results=final_results,
            queries_used=queries,
            query_stats={
                "original_query": query,
                "num_queries": len(queries),
                "total_candidates": sum(len(r) for r in all_results.values()),
                "unique_results": len(fused_results),
                "elapsed_seconds": round(elapsed, 3)
            }
        )
    
    def _build_query_list(self, transformed: TransformedQuery) -> List[str]:
        """构建查询列表"""
        queries = list(transformed.expanded)
        
        # 添加 HyDE 假设答案作为额外查询
        if self.include_hyde and transformed.hyde_answer:
            # 截取 HyDE 答案的前 500 字符作为查询
            hyde_query = transformed.hyde_answer[:500]
            queries.append(hyde_query)
        
        # 限制查询数量
        return queries[:self.max_parallel_queries]
    
    def _parallel_search(
        self,
        queries: List[str],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, List[RetrievalResult]]:
        """并行执行多个查询"""
        results = {}
        
        def search_single(q: str) -> tuple:
            try:
                r = self.hybrid_searcher.search(q, top_k, filter_dict)
                return q, r
            except Exception as e:
                logger.warning(f"Search failed for query '{q[:50]}...': {e}")
                return q, []
        
        # 并行执行
        futures = [
            self._executor.submit(search_single, q)
            for q in queries
        ]
        
        for future in futures:
            try:
                q, r = future.result(timeout=30)
                results[q] = r
            except Exception as e:
                logger.warning(f"Query execution failed: {e}")
        
        return results
    
    async def _async_parallel_search(
        self,
        queries: List[str],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, List[RetrievalResult]]:
        """异步并行执行多个查询"""
        
        async def search_single(q: str) -> tuple:
            try:
                # 在线程池中执行同步搜索
                loop = asyncio.get_event_loop()
                r = await loop.run_in_executor(
                    self._executor,
                    lambda: self.hybrid_searcher.search(q, top_k, filter_dict)
                )
                return q, r
            except Exception as e:
                logger.warning(f"Async search failed for query '{q[:50]}...': {e}")
                return q, []
        
        # 并行执行所有查询
        tasks = [search_single(q) for q in queries]
        results_list = await asyncio.gather(*tasks)
        
        return {q: r for q, r in results_list}
    
    def _multi_rrf_fusion(
        self,
        all_results: Dict[str, List[RetrievalResult]],
        queries: List[str]
    ) -> List[RetrievalResult]:
        """
        多路 RRF 融合
        
        对每个查询的结果计算 RRF 分数，然后合并
        """
        # chunk_id -> 累计 RRF 分数
        rrf_scores: Dict[str, float] = {}
        
        # chunk_id -> 最佳结果对象 (保留元数据)
        best_results: Dict[str, RetrievalResult] = {}
        
        # chunk_id -> 命中的查询数
        hit_counts: Dict[str, int] = {}
        
        for query_idx, query in enumerate(queries):
            if query not in all_results:
                continue
            
            results = all_results[query]
            
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id
                
                # 计算该查询下的 RRF 分数
                rrf_score = 1.0 / (self.rrf_k + rank)
                
                # 累加分数
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    hit_counts[chunk_id] = 0
                
                rrf_scores[chunk_id] += rrf_score
                hit_counts[chunk_id] += 1
                
                # 保留分数最高的结果对象
                if chunk_id not in best_results or result.score > best_results[chunk_id].score:
                    best_results[chunk_id] = result
        
        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # 构建最终结果
        final_results = []
        for chunk_id in sorted_ids:
            original = best_results[chunk_id]
            
            final_results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=original.content,
                score=rrf_scores[chunk_id],
                vector_score=original.vector_score,
                bm25_score=original.bm25_score,
                rrf_score=rrf_scores[chunk_id],
                metadata={
                    **original.metadata,
                    "multi_query_hits": hit_counts[chunk_id]
                },
                source=original.source
            ))
        
        return final_results
    
    def search_with_fallback(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        min_results: int = 3
    ) -> MultiQueryResult:
        """
        带降级策略的检索
        
        如果多路召回结果不足，逐步放宽条件
        """
        # 第一次尝试：完整多路召回
        result = self.search(query, top_k, filter_dict)
        
        if len(result.results) >= min_results:
            return result
        
        logger.info(f"Only {len(result.results)} results, trying fallback...")
        
        # 降级 1：移除过滤条件
        if filter_dict:
            result = self.search(query, top_k, None)
            if len(result.results) >= min_results:
                return result
        
        # 降级 2：仅使用原始查询的向量检索
        vector_results = self.hybrid_searcher.vector_only_search(query, top_k)
        
        if len(vector_results) > len(result.results):
            return MultiQueryResult(
                results=vector_results,
                queries_used=[query],
                query_stats={
                    "original_query": query,
                    "fallback": "vector_only",
                    "num_results": len(vector_results)
                }
            )
        
        return result
    
    def close(self):
        """关闭资源"""
        self._executor.shutdown(wait=False)
