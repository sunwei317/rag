"""
Reranker 模块
对初召回的结果进行二次排序，提高技术细节命中率
"""
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger

from .hybrid_search import RetrievalResult


@dataclass
class RerankResult:
    """重排结果"""
    chunk_id: str
    content: str
    rerank_score: float
    original_score: float
    metadata: dict
    
    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "rerank_score": self.rerank_score,
            "original_score": self.original_score,
            "metadata": self.metadata
        }


class Reranker:
    """
    重排序器
    
    使用专用的 Reranker 模型对初召回结果进行二次排序
    对于技术文档场景，Reranker 能显著提高:
    - 参数、步骤、错误码等精确匹配
    - 长尾查询的相关性
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        
        self._model = None
        self._init_model()
    
    def _init_model(self):
        """初始化 Reranker 模型"""
        try:
            from FlagEmbedding import FlagReranker
            
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )
            
            logger.info(f"Initialized Reranker: {self.model_name}")
        except ImportError:
            logger.warning("FlagEmbedding not installed. Using fallback reranker.")
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load Reranker model: {e}")
            self._model = None
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 初召回结果
            top_k: 返回数量 (None 表示返回所有)
        
        Returns:
            List[RerankResult]: 重排后的结果
        """
        if not results:
            return []
        
        if self._model is None:
            # 使用 fallback 方法
            return self._fallback_rerank(query, results, top_k)
        
        logger.debug(f"Reranking {len(results)} results for: {query[:50]}...")
        
        # 构建 query-passage 对
        pairs = [[query, r.content] for r in results]
        
        # 计算 rerank 分数
        scores = self._model.compute_score(pairs, normalize=True)
        
        # 如果只有一个结果，scores 可能不是列表
        if not isinstance(scores, list):
            scores = [scores]
        
        # 构建重排结果
        rerank_results = []
        for result, score in zip(results, scores):
            rerank_results.append(RerankResult(
                chunk_id=result.chunk_id,
                content=result.content,
                rerank_score=float(score),
                original_score=result.score,
                metadata=result.metadata
            ))
        
        # 按 rerank 分数排序
        rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            rerank_results = rerank_results[:top_k]
        
        return rerank_results
    
    def _fallback_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Fallback 重排方法
        使用简单的关键词匹配作为辅助排序
        """
        query_terms = set(query.lower().split())
        
        rerank_results = []
        for result in results:
            content_lower = result.content.lower()
            
            # 计算关键词匹配分数
            matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = matches / len(query_terms) if query_terms else 0
            
            # 组合原始分数和关键词分数
            rerank_score = 0.7 * result.score + 0.3 * keyword_score
            
            rerank_results.append(RerankResult(
                chunk_id=result.chunk_id,
                content=result.content,
                rerank_score=rerank_score,
                original_score=result.score,
                metadata=result.metadata
            ))
        
        rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            rerank_results = rerank_results[:top_k]
        
        return rerank_results


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker
    
    使用 sentence-transformers 的 Cross-Encoder
    适合不想依赖 FlagEmbedding 的场景
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            self._model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            
            logger.info(f"Initialized CrossEncoder: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            self._model = None
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}")
            self._model = None
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """重排序"""
        if not results or self._model is None:
            return [
                RerankResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    rerank_score=r.score,
                    original_score=r.score,
                    metadata=r.metadata
                )
                for r in results
            ][:top_k] if top_k else []
        
        # 构建输入对
        pairs = [(query, r.content) for r in results]
        
        # 计算分数
        scores = self._model.predict(pairs)
        
        # 构建结果
        rerank_results = []
        for result, score in zip(results, scores):
            rerank_results.append(RerankResult(
                chunk_id=result.chunk_id,
                content=result.content,
                rerank_score=float(score),
                original_score=result.score,
                metadata=result.metadata
            ))
        
        rerank_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        if top_k:
            rerank_results = rerank_results[:top_k]
        
        return rerank_results


def create_reranker(
    provider: str = "flag",
    model_name: Optional[str] = None
) -> Reranker:
    """创建 Reranker"""
    if provider == "flag":
        return Reranker(model_name=model_name or "BAAI/bge-reranker-v2-m3")
    elif provider == "cross-encoder":
        return CrossEncoderReranker(model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    else:
        return Reranker()
