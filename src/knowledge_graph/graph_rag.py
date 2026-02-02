"""
Graph RAG 融合模块
将知识图谱检索与传统向量检索融合，提供增强的 RAG 能力

融合策略:
1. 并行检索: 同时进行向量检索和图谱检索
2. 上下文合并: 将 Chunk 上下文与图谱上下文合并
3. 智能排序: 根据图谱关系调整检索结果排序
4. 引用增强: 在回答中引用图谱知识
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from .graph_retriever import GraphRetriever, SubGraph
from .graph_store import GraphStore
from .entity_extractor import EntityType


@dataclass
class GraphRAGContext:
    """
    Graph RAG 融合上下文
    
    包含向量检索和图谱检索的合并结果
    """
    # 向量检索结果
    chunk_context: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    
    # 图谱检索结果
    graph_context: str = ""
    subgraph: Optional[SubGraph] = None
    
    # 融合后的上下文
    merged_context: str = ""
    
    # 元信息
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "chunk_context": self.chunk_context,
            "chunks": self.chunks,
            "graph_context": self.graph_context,
            "subgraph": self.subgraph.to_dict() if self.subgraph else None,
            "merged_context": self.merged_context,
            "retrieval_stats": self.retrieval_stats
        }


@dataclass 
class GraphRAGResponse:
    """Graph RAG 响应"""
    answer: str
    context: GraphRAGContext
    references: List[Dict[str, Any]] = field(default_factory=list)
    graph_insights: List[str] = field(default_factory=list)  # 从图谱获得的洞察
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "context": self.context.to_dict(),
            "references": self.references,
            "graph_insights": self.graph_insights
        }


class GraphRAG:
    """
    Graph RAG 融合系统
    
    融合传统 RAG 检索和知识图谱检索:
    1. 向量检索: 获取相关文档片段
    2. 图谱检索: 获取相关实体和关系
    3. 上下文融合: 合并两种上下文
    4. 增强生成: 使用融合上下文生成回答
    
    优势:
    - 跨文档关联: 图谱捕获跨文档的实体关系
    - 多跳推理: 支持复杂的关系推理问题
    - 上下文完整性: 提供结构化的知识背景
    """
    
    def __init__(
        self,
        # 向量检索组件 (现有)
        hybrid_searcher=None,
        reranker=None,
        
        # 图谱检索组件 (新增)
        graph_retriever: Optional[GraphRetriever] = None,
        graph_store: Optional[GraphStore] = None,
        
        # LLM
        llm_client=None,
        model: str = "gpt-4.1",
        
        # 融合参数
        chunk_weight: float = 0.6,      # Chunk 上下文权重
        graph_weight: float = 0.4,      # 图谱上下文权重
        max_chunk_context: int = 3000,  # 最大 Chunk 上下文长度
        max_graph_context: int = 1500,  # 最大图谱上下文长度
        
        # 行为配置
        use_graph_rerank: bool = True,  # 使用图谱信息重排序
        include_graph_insights: bool = True  # 在回答中包含图谱洞察
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.graph_retriever = graph_retriever
        self.graph_store = graph_store
        self.llm_client = llm_client
        self.model = model
        
        self.chunk_weight = chunk_weight
        self.graph_weight = graph_weight
        self.max_chunk_context = max_chunk_context
        self.max_graph_context = max_graph_context
        
        self.use_graph_rerank = use_graph_rerank
        self.include_graph_insights = include_graph_insights
        
        # 初始化图谱检索器
        if graph_store and not graph_retriever:
            self.graph_retriever = GraphRetriever(graph_store)
        
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> GraphRAGContext:
        """
        融合检索
        
        同时进行向量检索和图谱检索，返回融合后的上下文
        
        Args:
            query: 用户查询
            top_k: 返回的 Chunk 数量
            filter_dict: 过滤条件
            
        Returns:
            GraphRAGContext: 融合后的上下文
        """
        context = GraphRAGContext()
        stats: Dict[str, Any] = {"query": query}
        
        # 1. 向量检索 (传统 RAG)
        chunks = []
        chunk_context = ""
        
        if self.hybrid_searcher:
            try:
                results = self.hybrid_searcher.search(
                    query=query,
                    top_k=top_k * 2,  # 多检索一些用于重排
                    filter_dict=filter_dict
                )
                
                # 重排序
                if self.reranker and results:
                    results = self.reranker.rerank(query, results, top_k=top_k)
                else:
                    results = results[:top_k]
                
                # 构建 Chunk 上下文
                chunks = [
                    {
                        "chunk_id": r.chunk_id,
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
                
                chunk_context = self._build_chunk_context(chunks)
                stats["chunk_count"] = len(chunks)
                
            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")
                stats["chunk_error"] = str(e)
        
        # 2. 图谱检索
        subgraph = SubGraph()
        graph_context = ""
        
        if self.graph_retriever:
            try:
                subgraph = self.graph_retriever.retrieve(
                    query=query,
                    expand_depth=2
                )
                
                graph_context = subgraph.to_context_text(
                    max_length=self.max_graph_context
                )
                
                stats["graph_entities"] = len(subgraph.entities)
                stats["graph_relations"] = len(subgraph.relations)
                
            except Exception as e:
                logger.error(f"Graph retrieval failed: {e}")
                stats["graph_error"] = str(e)
        
        # 3. 使用图谱信息重排序 Chunks
        if self.use_graph_rerank and chunks and not subgraph.is_empty():
            chunks = self._graph_rerank(chunks, subgraph)
        
        # 4. 融合上下文
        merged_context = self._merge_context(chunk_context, graph_context)
        
        context.chunk_context = chunk_context
        context.chunks = chunks
        context.graph_context = graph_context
        context.subgraph = subgraph
        context.merged_context = merged_context
        context.retrieval_stats = stats
        
        return context
    
    def _build_chunk_context(self, chunks: List[Dict]) -> str:
        """构建 Chunk 上下文"""
        lines = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            
            # 截断检查
            if total_length + len(content) > self.max_chunk_context:
                remaining = self.max_chunk_context - total_length
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            
            lines.append(f"[文档片段 {i+1}]\n{content}\n")
            total_length += len(content)
        
        return "\n".join(lines)
    
    def _merge_context(self, chunk_context: str, graph_context: str) -> str:
        """合并 Chunk 和图谱上下文"""
        parts = []
        
        if graph_context:
            parts.append(graph_context)
        
        if chunk_context:
            parts.append("【相关文档片段】\n" + chunk_context)
        
        return "\n\n".join(parts)
    
    def _graph_rerank(
        self, 
        chunks: List[Dict], 
        subgraph: SubGraph
    ) -> List[Dict]:
        """
        使用图谱信息重排序 Chunks
        
        策略:
        - 包含图谱实体的 Chunk 提升排名
        - 包含多个相关实体的 Chunk 更优先
        """
        # 获取图谱实体名称
        entity_names = set()
        for entity in subgraph.entities:
            entity_names.add(entity.name.lower())
            for alias in entity.aliases:
                entity_names.add(alias.lower())
        
        # 计算每个 Chunk 的图谱相关性
        for chunk in chunks:
            content_lower = chunk["content"].lower()
            
            # 统计命中的实体数量
            entity_hits = sum(
                1 for name in entity_names 
                if name in content_lower
            )
            
            # 调整分数
            graph_boost = min(entity_hits * 0.05, 0.2)  # 最多提升 20%
            chunk["graph_score"] = entity_hits
            chunk["adjusted_score"] = chunk["score"] * (1 + graph_boost)
        
        # 按调整后的分数排序
        chunks.sort(key=lambda x: x.get("adjusted_score", x["score"]), reverse=True)
        
        return chunks
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> GraphRAGResponse:
        """
        融合查询
        
        执行检索并生成回答
        
        Args:
            query: 用户问题
            top_k: 检索数量
            filter_dict: 过滤条件
            stream: 是否流式输出
            
        Returns:
            GraphRAGResponse: 包含回答和上下文的响应
        """
        # 1. 融合检索
        context = self.retrieve(query, top_k, filter_dict)
        
        # 2. 生成回答
        answer = self._generate_answer(query, context)
        
        # 3. 提取图谱洞察
        graph_insights = []
        if self.include_graph_insights and context.subgraph:
            graph_insights = self._extract_insights(context.subgraph)
        
        # 4. 构建引用
        references = self._build_references(context.chunks)
        
        return GraphRAGResponse(
            answer=answer,
            context=context,
            references=references,
            graph_insights=graph_insights
        )
    
    def _generate_answer(self, query: str, context: GraphRAGContext) -> str:
        """生成回答"""
        if not self.llm_client:
            return "LLM 客户端未初始化"
        
        system_prompt = """你是一个技术文档问答助手。请基于提供的上下文回答用户问题。

回答要求:
1. 准确: 只使用上下文中的信息，不要编造
2. 完整: 尽可能提供完整的答案
3. 引用: 如果可能，指出信息来源
4. 如果上下文不包含答案，请明确说明"""

        user_prompt = f"""上下文信息:
{context.merged_context}

用户问题: {query}

请回答:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"生成回答时出错: {e}"
    
    def _extract_insights(self, subgraph: SubGraph) -> List[str]:
        """从图谱中提取洞察"""
        insights = []
        
        # 关键实体
        key_entities = subgraph.query_entities[:3]
        if key_entities:
            entity_names = [e.name for e in key_entities]
            insights.append(f"识别到关键实体: {', '.join(entity_names)}")
        
        # 重要关系
        important_relations = [
            r for r in subgraph.relations 
            if r.relation_type.value in ['depends_on', 'requires', 'contains']
        ][:3]
        
        entity_map = {e.entity_id: e.name for e in subgraph.entities}
        
        for rel in important_relations:
            source = entity_map.get(rel.source_id, "?")
            target = entity_map.get(rel.target_id, "?")
            insights.append(f"{source} {rel.relation_type.value} {target}")
        
        return insights
    
    def _build_references(self, chunks: List[Dict]) -> List[Dict]:
        """构建引用列表"""
        references = []
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            references.append({
                "chunk_id": chunk.get("chunk_id"),
                "doc_title": metadata.get("doc_title", ""),
                "section_title": metadata.get("section_title", ""),
                "page": metadata.get("page", 0),
                "score": chunk.get("score", 0)
            })
        
        return references
    
    def analyze_query_type(self, query: str) -> str:
        """
        分析查询类型以优化检索策略
        
        Returns:
            查询类型: factual, relational, procedural, comparative
        """
        query_lower = query.lower()
        
        # 关系性问题
        if any(kw in query_lower for kw in ["和", "与", "关系", "区别", "不同"]):
            if any(kw in query_lower for kw in ["区别", "不同", "比较"]):
                return "comparative"
            return "relational"
        
        # 过程性问题
        if any(kw in query_lower for kw in ["如何", "怎么", "步骤", "安装", "配置", "部署"]):
            return "procedural"
        
        # 默认事实性问题
        return "factual"
    
    def smart_query(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> GraphRAGResponse:
        """
        智能查询
        
        根据查询类型自动选择最佳检索策略
        """
        query_type = self.analyze_query_type(query)
        logger.info(f"Query type: {query_type}")
        
        # 根据查询类型调整图谱检索策略
        context = GraphRAGContext()
        
        if self.graph_retriever:
            subgraph = self.graph_retriever.retrieve_for_question(
                query, 
                question_type=query_type
            )
            context.subgraph = subgraph
            context.graph_context = subgraph.to_context_text()
        
        # 向量检索
        if self.hybrid_searcher:
            results = self.hybrid_searcher.search(query, top_k=top_k, filter_dict=filter_dict)
            chunks = [{"chunk_id": r.chunk_id, "content": r.content, "score": r.score, "metadata": r.metadata} for r in results]
            context.chunks = chunks
            context.chunk_context = self._build_chunk_context(chunks)
        
        context.merged_context = self._merge_context(
            context.chunk_context, 
            context.graph_context
        )
        
        # 生成回答
        answer = self._generate_answer(query, context)
        
        return GraphRAGResponse(
            answer=answer,
            context=context,
            references=self._build_references(context.chunks),
            graph_insights=self._extract_insights(context.subgraph) if context.subgraph else []
        )
