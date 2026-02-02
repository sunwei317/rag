"""
质量评估模块
使用 RAGAS 框架评估 RAG 系统质量
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class EvaluationResult:
    """评估结果"""
    faithfulness: float      # 忠实度: 生成内容是否基于检索结果
    answer_relevance: float  # 相关度: 是否回答了用户问题
    context_precision: float # 检索精度: 检索结果是否有用
    context_recall: float    # 召回率: 是否检索到了相关信息
    overall_score: float     # 综合得分
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score
        }


@dataclass
class TestCase:
    """测试用例"""
    question: str
    ground_truth: str  # 标准答案
    context: Optional[List[str]] = None  # 检索到的上下文
    answer: Optional[str] = None  # 生成的答案


class RAGEvaluator:
    """
    RAG 质量评估器
    
    基于 RAGAS 框架评估:
    - Faithfulness: 生成的内容是否全都在检索到的 Chunk 里
    - Answer Relevance: 是否真的回答了用户的问题
    - Context Precision: 搜出来的东西是不是真的有用
    - Context Recall: 是否检索到了关键信息
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._ragas_available = False
        
        self._check_ragas()
    
    def _check_ragas(self):
        """检查 RAGAS 是否可用"""
        try:
            import ragas
            self._ragas_available = True
            logger.info("RAGAS is available")
        except ImportError:
            logger.warning("RAGAS not installed. Using simplified evaluation.")
            self._ragas_available = False
    
    def evaluate(
        self,
        test_cases: List[TestCase]
    ) -> List[EvaluationResult]:
        """
        评估测试用例
        
        Args:
            test_cases: 测试用例列表
        
        Returns:
            评估结果列表
        """
        if self._ragas_available:
            return self._evaluate_with_ragas(test_cases)
        else:
            return self._evaluate_simplified(test_cases)
    
    def _evaluate_with_ragas(
        self,
        test_cases: List[TestCase]
    ) -> List[EvaluationResult]:
        """使用 RAGAS 评估"""
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset
        
        # 准备数据
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for tc in test_cases:
            data["question"].append(tc.question)
            data["answer"].append(tc.answer or "")
            data["contexts"].append(tc.context or [])
            data["ground_truth"].append(tc.ground_truth)
        
        dataset = Dataset.from_dict(data)
        
        # 评估
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        # 转换结果
        results = []
        df = result.to_pandas()
        
        for _, row in df.iterrows():
            faith = row.get("faithfulness", 0.0) or 0.0
            relevance = row.get("answer_relevancy", 0.0) or 0.0
            precision = row.get("context_precision", 0.0) or 0.0
            recall = row.get("context_recall", 0.0) or 0.0
            
            overall = (faith + relevance + precision + recall) / 4
            
            results.append(EvaluationResult(
                faithfulness=faith,
                answer_relevance=relevance,
                context_precision=precision,
                context_recall=recall,
                overall_score=overall
            ))
        
        return results
    
    def _evaluate_simplified(
        self,
        test_cases: List[TestCase]
    ) -> List[EvaluationResult]:
        """简化版评估 (不依赖 RAGAS)"""
        results = []
        
        for tc in test_cases:
            # 简单的关键词匹配评估
            faithfulness = self._calc_faithfulness(tc.answer, tc.context)
            relevance = self._calc_relevance(tc.question, tc.answer)
            precision = self._calc_context_precision(tc.question, tc.context)
            recall = self._calc_context_recall(tc.ground_truth, tc.context)
            
            overall = (faithfulness + relevance + precision + recall) / 4
            
            results.append(EvaluationResult(
                faithfulness=faithfulness,
                answer_relevance=relevance,
                context_precision=precision,
                context_recall=recall,
                overall_score=overall
            ))
        
        return results
    
    def _calc_faithfulness(
        self,
        answer: Optional[str],
        context: Optional[List[str]]
    ) -> float:
        """计算忠实度 (简化版)"""
        if not answer or not context:
            return 0.0
        
        # 检查答案中的关键词是否出现在上下文中
        answer_words = set(answer.lower().split())
        context_text = " ".join(context).lower()
        
        matches = sum(1 for word in answer_words if word in context_text)
        return matches / len(answer_words) if answer_words else 0.0
    
    def _calc_relevance(
        self,
        question: str,
        answer: Optional[str]
    ) -> float:
        """计算相关度 (简化版)"""
        if not answer:
            return 0.0
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = question_words & answer_words
        return len(overlap) / len(question_words) if question_words else 0.0
    
    def _calc_context_precision(
        self,
        question: str,
        context: Optional[List[str]]
    ) -> float:
        """计算上下文精度 (简化版)"""
        if not context:
            return 0.0
        
        question_words = set(question.lower().split())
        
        relevant_count = 0
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            if question_words & ctx_words:
                relevant_count += 1
        
        return relevant_count / len(context) if context else 0.0
    
    def _calc_context_recall(
        self,
        ground_truth: str,
        context: Optional[List[str]]
    ) -> float:
        """计算上下文召回 (简化版)"""
        if not context:
            return 0.0
        
        gt_words = set(ground_truth.lower().split())
        context_text = " ".join(context).lower()
        
        matches = sum(1 for word in gt_words if word in context_text)
        return matches / len(gt_words) if gt_words else 0.0
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """评估单个问答"""
        tc = TestCase(
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth or ""
        )
        
        results = self.evaluate([tc])
        return results[0] if results else None


class HitRateEvaluator:
    """
    Hit Rate 评估器
    
    评估检索的 Hit Rate@K:
    在前 K 个搜出来的片段里，是否包含了正确答案所在的片段
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        k_values: List[int] = [1, 3, 5, 10, 20]
    ) -> Dict[str, float]:
        """
        评估 Hit Rate
        
        Args:
            test_cases: 测试用例 [{question, relevant_chunk_ids}]
            k_values: 评估的 K 值列表
        
        Returns:
            {hit_rate@1, hit_rate@3, ...}
        """
        results = {f"hit_rate@{k}": 0.0 for k in k_values}
        
        for tc in test_cases:
            question = tc["question"]
            relevant_ids = set(tc["relevant_chunk_ids"])
            
            # 检索
            max_k = max(k_values)
            search_results = self.retriever.search(question, top_k=max_k)
            retrieved_ids = [r.chunk_id for r in search_results]
            
            # 计算各 K 值的 hit
            for k in k_values:
                top_k_ids = set(retrieved_ids[:k])
                if relevant_ids & top_k_ids:
                    results[f"hit_rate@{k}"] += 1
        
        # 计算比例
        n = len(test_cases)
        for k in k_values:
            results[f"hit_rate@{k}"] /= n if n > 0 else 1
        
        return results


def create_evaluator() -> RAGEvaluator:
    """创建评估器"""
    return RAGEvaluator()
