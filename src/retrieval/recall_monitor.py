"""
召回率监控模块
生产环境的检索质量监控和日志记录
"""
import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import threading
from loguru import logger

from src.retrieval.hybrid_search import RetrievalResult


@dataclass
class RetrievalMetrics:
    """单次检索的指标"""
    query_id: str
    query: str
    timestamp: str
    
    # 检索结果统计
    num_results: int
    num_vector_hits: int
    num_bm25_hits: int
    num_both_hits: int  # 两种方式都命中
    
    # 分数统计
    avg_score: float
    max_score: float
    min_score: float
    avg_vector_score: float
    avg_bm25_score: float
    
    # 性能指标
    latency_ms: float
    
    # 额外信息
    filters_used: Optional[Dict[str, Any]] = None
    multi_query_count: int = 1
    reranker_used: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RecallReport:
    """召回率报告"""
    period_start: str
    period_end: str
    total_queries: int
    
    # 召回统计
    avg_results_per_query: float
    queries_with_zero_results: int
    zero_result_rate: float
    
    # 来源分布
    vector_only_rate: float
    bm25_only_rate: float
    hybrid_hit_rate: float
    
    # 分数分布
    avg_score: float
    score_p50: float
    score_p90: float
    
    # 性能
    avg_latency_ms: float
    p99_latency_ms: float
    
    # 低召回查询
    low_recall_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class RecallMonitor:
    """
    召回率监控器
    
    功能:
    1. 记录每次检索的详细指标
    2. 定期生成召回率报告
    3. 检测异常模式 (如零结果率过高)
    4. 支持日志输出和文件持久化
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        alert_threshold: float = 0.1,  # 零结果率超过 10% 告警
        buffer_size: int = 1000,
        enable_file_logging: bool = True,
        alert_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        self.log_dir = Path(log_dir) if log_dir else Path("data/recall_logs")
        self.alert_threshold = alert_threshold
        self.buffer_size = buffer_size
        self.enable_file_logging = enable_file_logging
        self.alert_callback = alert_callback
        
        # 指标缓冲区
        self._metrics_buffer: List[RetrievalMetrics] = []
        self._buffer_lock = threading.Lock()
        
        # 统计计数器
        self._total_queries = 0
        self._zero_result_queries = 0
        self._source_counts = defaultdict(int)
        self._latencies: List[float] = []
        self._scores: List[float] = []
        
        # 确保日志目录存在
        if enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RecallMonitor initialized, log_dir={self.log_dir}")
    
    def record(
        self,
        query: str,
        results: List[RetrievalResult],
        latency_ms: float,
        query_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        multi_query_count: int = 1,
        reranker_used: bool = False
    ):
        """
        记录一次检索
        
        Args:
            query: 查询文本
            results: 检索结果
            latency_ms: 延迟 (毫秒)
            query_id: 查询ID (可选)
            filters: 使用的过滤条件
            multi_query_count: 多路查询数量
            reranker_used: 是否使用了重排序
        """
        timestamp = datetime.now().isoformat()
        query_id = query_id or f"q_{int(time.time() * 1000)}"
        
        # 计算指标
        num_results = len(results)
        num_vector = sum(1 for r in results if r.source in ("vector", "both"))
        num_bm25 = sum(1 for r in results if r.source in ("bm25", "both"))
        num_both = sum(1 for r in results if r.source == "both")
        
        scores = [r.score for r in results] if results else [0.0]
        vector_scores = [r.vector_score for r in results if r.vector_score > 0]
        bm25_scores = [r.bm25_score for r in results if r.bm25_score > 0]
        
        metrics = RetrievalMetrics(
            query_id=query_id,
            query=query[:200],  # 截断长查询
            timestamp=timestamp,
            num_results=num_results,
            num_vector_hits=num_vector,
            num_bm25_hits=num_bm25,
            num_both_hits=num_both,
            avg_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            avg_vector_score=sum(vector_scores) / len(vector_scores) if vector_scores else 0.0,
            avg_bm25_score=sum(bm25_scores) / len(bm25_scores) if bm25_scores else 0.0,
            latency_ms=latency_ms,
            filters_used=filters,
            multi_query_count=multi_query_count,
            reranker_used=reranker_used
        )
        
        # 更新统计
        self._update_stats(metrics)
        
        # 添加到缓冲区
        with self._buffer_lock:
            self._metrics_buffer.append(metrics)
            
            # 缓冲区满时刷新
            if len(self._metrics_buffer) >= self.buffer_size:
                self._flush_buffer()
        
        # 记录日志
        self._log_metrics(metrics)
        
        # 检查异常
        self._check_alerts(metrics)
    
    def _update_stats(self, metrics: RetrievalMetrics):
        """更新统计计数器"""
        self._total_queries += 1
        
        if metrics.num_results == 0:
            self._zero_result_queries += 1
        
        # 更新来源统计
        if metrics.num_both_hits > 0:
            self._source_counts["hybrid"] += 1
        elif metrics.num_vector_hits > 0:
            self._source_counts["vector_only"] += 1
        elif metrics.num_bm25_hits > 0:
            self._source_counts["bm25_only"] += 1
        
        # 保留最近的延迟和分数用于百分位计算
        self._latencies.append(metrics.latency_ms)
        self._scores.append(metrics.avg_score)
        
        # 限制列表大小
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]
            self._scores = self._scores[-5000:]
    
    def _log_metrics(self, metrics: RetrievalMetrics):
        """记录指标到日志"""
        # 基本信息
        log_msg = (
            f"[Retrieval] query_id={metrics.query_id} "
            f"results={metrics.num_results} "
            f"vector={metrics.num_vector_hits} "
            f"bm25={metrics.num_bm25_hits} "
            f"hybrid={metrics.num_both_hits} "
            f"avg_score={metrics.avg_score:.4f} "
            f"latency={metrics.latency_ms:.1f}ms"
        )
        
        if metrics.num_results == 0:
            logger.warning(log_msg + " [ZERO_RESULTS]")
        elif metrics.num_results < 3:
            logger.info(log_msg + " [LOW_RECALL]")
        else:
            logger.debug(log_msg)
    
    def _check_alerts(self, metrics: RetrievalMetrics):
        """检查是否需要告警"""
        # 零结果率告警
        if self._total_queries >= 100:
            zero_rate = self._zero_result_queries / self._total_queries
            
            if zero_rate > self.alert_threshold:
                alert_msg = (
                    f"High zero-result rate: {zero_rate:.2%} "
                    f"({self._zero_result_queries}/{self._total_queries})"
                )
                logger.warning(f"[ALERT] {alert_msg}")
                
                if self.alert_callback:
                    self.alert_callback("high_zero_result_rate", {
                        "rate": zero_rate,
                        "threshold": self.alert_threshold,
                        "total_queries": self._total_queries
                    })
        
        # 高延迟告警
        if metrics.latency_ms > 5000:  # 5秒
            logger.warning(
                f"[ALERT] High latency: {metrics.latency_ms:.0f}ms "
                f"for query: {metrics.query[:50]}..."
            )
            
            if self.alert_callback:
                self.alert_callback("high_latency", {
                    "latency_ms": metrics.latency_ms,
                    "query": metrics.query
                })
    
    def _flush_buffer(self):
        """刷新缓冲区到文件"""
        if not self.enable_file_logging or not self._metrics_buffer:
            return
        
        # 按日期分文件
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"recall_metrics_{date_str}.jsonl"
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                for metrics in self._metrics_buffer:
                    f.write(json.dumps(metrics.to_dict(), ensure_ascii=False) + '\n')
            
            logger.debug(f"Flushed {len(self._metrics_buffer)} metrics to {log_file}")
            self._metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    def generate_report(
        self,
        hours: int = 24,
        include_low_recall_queries: bool = True
    ) -> RecallReport:
        """
        生成召回率报告
        
        Args:
            hours: 报告时间范围 (小时)
            include_low_recall_queries: 是否包含低召回查询样例
        
        Returns:
            RecallReport
        """
        now = datetime.now()
        period_start = now - timedelta(hours=hours)
        
        # 从文件加载指标
        all_metrics = self._load_metrics_from_files(period_start, now)
        
        if not all_metrics:
            return RecallReport(
                period_start=period_start.isoformat(),
                period_end=now.isoformat(),
                total_queries=0,
                avg_results_per_query=0,
                queries_with_zero_results=0,
                zero_result_rate=0,
                vector_only_rate=0,
                bm25_only_rate=0,
                hybrid_hit_rate=0,
                avg_score=0,
                score_p50=0,
                score_p90=0,
                avg_latency_ms=0,
                p99_latency_ms=0,
                low_recall_queries=[]
            )
        
        # 计算统计
        total = len(all_metrics)
        zero_results = sum(1 for m in all_metrics if m.num_results == 0)
        
        results_counts = [m.num_results for m in all_metrics]
        scores = [m.avg_score for m in all_metrics]
        latencies = [m.latency_ms for m in all_metrics]
        
        # 来源统计
        vector_only = sum(1 for m in all_metrics if m.num_vector_hits > 0 and m.num_bm25_hits == 0)
        bm25_only = sum(1 for m in all_metrics if m.num_bm25_hits > 0 and m.num_vector_hits == 0)
        hybrid = sum(1 for m in all_metrics if m.num_both_hits > 0)
        
        # 百分位计算
        sorted_scores = sorted(scores)
        sorted_latencies = sorted(latencies)
        
        def percentile(arr, p):
            if not arr:
                return 0
            idx = int(len(arr) * p / 100)
            return arr[min(idx, len(arr) - 1)]
        
        # 低召回查询
        low_recall = []
        if include_low_recall_queries:
            for m in all_metrics:
                if m.num_results < 3:
                    low_recall.append({
                        "query": m.query,
                        "num_results": m.num_results,
                        "timestamp": m.timestamp
                    })
            low_recall = low_recall[:20]  # 最多 20 条
        
        return RecallReport(
            period_start=period_start.isoformat(),
            period_end=now.isoformat(),
            total_queries=total,
            avg_results_per_query=sum(results_counts) / total,
            queries_with_zero_results=zero_results,
            zero_result_rate=zero_results / total if total > 0 else 0,
            vector_only_rate=vector_only / total if total > 0 else 0,
            bm25_only_rate=bm25_only / total if total > 0 else 0,
            hybrid_hit_rate=hybrid / total if total > 0 else 0,
            avg_score=sum(scores) / total if total > 0 else 0,
            score_p50=percentile(sorted_scores, 50),
            score_p90=percentile(sorted_scores, 90),
            avg_latency_ms=sum(latencies) / total if total > 0 else 0,
            p99_latency_ms=percentile(sorted_latencies, 99),
            low_recall_queries=low_recall
        )
    
    def _load_metrics_from_files(
        self,
        start: datetime,
        end: datetime
    ) -> List[RetrievalMetrics]:
        """从日志文件加载指标"""
        metrics = []
        
        if not self.enable_file_logging:
            return metrics
        
        # 遍历日期范围内的文件
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"recall_metrics_{date_str}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            ts = datetime.fromisoformat(data["timestamp"])
                            
                            if start <= ts <= end:
                                metrics.append(RetrievalMetrics(**data))
                except Exception as e:
                    logger.warning(f"Failed to load {log_file}: {e}")
            
            current += timedelta(days=1)
        
        return metrics
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取当前统计摘要"""
        total = self._total_queries
        
        if total == 0:
            return {"status": "no_data"}
        
        return {
            "total_queries": total,
            "zero_result_rate": self._zero_result_queries / total,
            "source_distribution": {
                "hybrid": self._source_counts["hybrid"] / total,
                "vector_only": self._source_counts["vector_only"] / total,
                "bm25_only": self._source_counts["bm25_only"] / total
            },
            "avg_latency_ms": sum(self._latencies) / len(self._latencies) if self._latencies else 0,
            "avg_score": sum(self._scores) / len(self._scores) if self._scores else 0
        }
    
    def reset_stats(self):
        """重置统计计数器"""
        self._total_queries = 0
        self._zero_result_queries = 0
        self._source_counts.clear()
        self._latencies.clear()
        self._scores.clear()
        logger.info("RecallMonitor stats reset")
    
    def close(self):
        """关闭监控器，刷新缓冲区"""
        with self._buffer_lock:
            self._flush_buffer()
        logger.info("RecallMonitor closed")


class MonitoredHybridSearcher:
    """
    带监控的混合检索器
    
    包装 HybridSearcher，自动记录检索指标
    """
    
    def __init__(
        self,
        hybrid_searcher,
        monitor: Optional[RecallMonitor] = None
    ):
        self.hybrid_searcher = hybrid_searcher
        self.monitor = monitor or RecallMonitor()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        fusion_method: str = "rrf",
        query_id: Optional[str] = None
    ) -> List[RetrievalResult]:
        """带监控的检索"""
        start_time = time.time()
        
        # 执行检索
        results = self.hybrid_searcher.search(
            query=query,
            top_k=top_k,
            filter_dict=filter_dict,
            fusion_method=fusion_method
        )
        
        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        # 记录指标
        self.monitor.record(
            query=query,
            results=results,
            latency_ms=latency_ms,
            query_id=query_id,
            filters=filter_dict
        )
        
        return results
    
    def get_report(self, hours: int = 24) -> RecallReport:
        """获取召回率报告"""
        return self.monitor.generate_report(hours)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return self.monitor.get_stats_summary()


# 全局监控器实例
_global_monitor: Optional[RecallMonitor] = None


def get_recall_monitor() -> RecallMonitor:
    """获取全局监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RecallMonitor()
    return _global_monitor


def record_retrieval(
    query: str,
    results: List[RetrievalResult],
    latency_ms: float,
    **kwargs
):
    """便捷函数：记录检索指标"""
    get_recall_monitor().record(
        query=query,
        results=results,
        latency_ms=latency_ms,
        **kwargs
    )
