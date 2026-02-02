"""
RAG Chat 模块
基于检索的问答系统
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class Reference:
    """引用来源"""
    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: str
    section_title: str
    page: int
    content_preview: str  # 内容预览 (前 200 字符)
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_path": self.section_path,
            "section_title": self.section_title,
            "page": self.page,
            "content_preview": self.content_preview
        }
    
    def format_citation(self) -> str:
        """格式化引用"""
        return f"[{self.doc_title} - {self.section_title}, 第{self.page}页]"


@dataclass
class ChatResponse:
    """问答响应"""
    answer: str
    references: List[Reference]
    query: str
    context_used: List[str]
    model: str
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "references": [r.to_dict() for r in self.references],
            "query": self.query,
            "context_used": self.context_used,
            "model": self.model
        }


class RAGChat:
    """
    RAG 问答系统
    
    特性:
    1. 混合检索 (向量 + BM25)
    2. 查询转换 (扩展/HyDE)
    3. 重排序
    4. 父子索引上下文扩展
    5. 引用追溯
    6. [NEW] Graph RAG 增强 - 知识图谱上下文融合
    """
    
    def __init__(
        self,
        hybrid_searcher=None,
        reranker=None,
        query_transformer=None,
        llm_client=None,
        model: str = "gpt-4.1",
        parent_chunk_store=None,  # 用于获取父 Chunk
        window_size: int = 2,  # 上下文窗口大小
        # Graph RAG 相关参数
        graph_retriever=None,  # 图谱检索器
        use_graph_context: bool = True,  # 是否使用图谱上下文
        graph_context_weight: float = 0.3  # 图谱上下文权重
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.llm_client = llm_client
        self.model = model
        self.parent_chunk_store = parent_chunk_store
        self.window_size = window_size
        
        # Graph RAG 组件
        self.graph_retriever = graph_retriever
        self.use_graph_context = use_graph_context
        self.graph_context_weight = graph_context_weight
        
        self._init_llm_client()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.llm_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to init LLM client: {e}")
    
    def _is_short_query(self, query: str, min_chars: int = 8, min_words: int = 2) -> bool:
        """
        检查查询是否太短
        
        Args:
            query: 查询文本
            min_chars: 最小字符数（中文）
            min_words: 最小词数（适用于分词后）
        
        Returns:
            bool: 是否为短查询
        """
        # 去除空白
        query = query.strip()
        
        # 字符数检查
        if len(query) < min_chars:
            return True
        
        # 使用 jieba 分词检查词数
        try:
            import jieba
            words = [w for w in jieba.cut(query) if w.strip()]
            if len(words) < min_words:
                return True
        except ImportError:
            pass
        
        return False
    
    def _enhance_short_query(self, query: str) -> Optional[str]:
        """
        使用 LLM 增强短查询
        
        Args:
            query: 原始短查询
        
        Returns:
            Optional[str]: 增强后的查询，失败返回 None
        """
        if not self.llm_client:
            return None
        
        try:
            prompt = f"""你是一个技术文档检索助手。用户输入了一个非常简短的查询，请帮助扩展成一个更完整、更具体的问题。

用户查询: {query}

要求:
1. 保持用户的原始意图
2. 扩展成一个完整的问题句
3. 如果是技术术语，可以添加"是什么"、"如何使用"、"有什么功能"等
4. 只输出扩展后的问题，不要其他解释
5. 控制在 30 字以内

扩展后的问题:"""

            response = self.llm_client.chat.completions.create(
                model="gpt-4.1-mini",  # 使用轻量模型节省成本
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # 确保增强后的查询有效
            if enhanced and len(enhanced) > len(query):
                return enhanced
            
        except Exception as e:
            logger.warning(f"Failed to enhance short query: {e}")
        
        return None

    def ask(
        self,
        question: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_query_transform: bool = True,
        use_rerank: bool = True,
        expand_context: bool = True,
        use_graph: bool = True  # 是否使用图谱增强
    ) -> ChatResponse:
        """
        回答问题
        
        Args:
            question: 用户问题
            filter_dict: 过滤条件 (如 product, version)
            top_k: 最终使用的上下文数量
            use_query_transform: 是否使用查询转换
            use_rerank: 是否使用重排序
            expand_context: 是否扩展到父 Chunk
            use_graph: 是否使用 Graph RAG 增强
        
        Returns:
            ChatResponse: 包含答案和引用
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        # 0. 短查询增强 - 如果查询太短，使用 LLM 扩展
        original_question = question
        if self._is_short_query(question):
            enhanced = self._enhance_short_query(question)
            if enhanced:
                logger.info(f"Short query enhanced: '{question}' -> '{enhanced}'")
                question = enhanced
        
        # 1. 图谱检索 (Graph RAG)
        graph_context = ""
        graph_entities = []
        if use_graph and self.use_graph_context and self.graph_retriever:
            try:
                subgraph = self.graph_retriever.retrieve(
                    query=question,
                    expand_depth=2
                )
                if not subgraph.is_empty():
                    graph_context = subgraph.to_context_text(max_length=1500)
                    graph_entities = [e.name for e in subgraph.query_entities]
                    logger.info(f"Graph context: {len(subgraph.entities)} entities, {len(subgraph.relations)} relations")
            except Exception as e:
                logger.warning(f"Graph retrieval failed: {e}")
        
        # 1. 查询转换
        queries = [question]
        if use_query_transform and self.query_transformer:
            transformed = self.query_transformer.transform(question)
            queries = transformed.expanded
            
            # 如果有 HyDE 答案，也用于检索
            if transformed.hyde_answer:
                queries.append(transformed.hyde_answer)
        
        # 2. 混合检索 (多查询)
        all_results = []
        seen_ids = set()
        
        for q in queries:
            results = self.hybrid_searcher.search(
                query=q,
                top_k=top_k * 2,
                filter_dict=filter_dict
            )
            
            for r in results:
                if r.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk_id)
        
        # 3. 重排序
        if use_rerank and self.reranker and all_results:
            rerank_results = self.reranker.rerank(
                query=question,
                results=all_results,
                top_k=top_k * 2
            )
            # 转换回 RetrievalResult 格式
            from ..retrieval.hybrid_search import RetrievalResult
            all_results = [
                RetrievalResult(
                    chunk_id=r.chunk_id,
                    content=r.content,
                    score=r.rerank_score,
                    vector_score=0,
                    bm25_score=0,
                    rrf_score=0,
                    metadata=r.metadata,
                    source="reranked"
                )
                for r in rerank_results
            ]
        
        # 4. 上下文扩展 (获取父 Chunk 或窗口)
        contexts = []
        for r in all_results[:top_k]:
            if expand_context and self.parent_chunk_store:
                # 尝试获取父 Chunk
                parent_content = self._get_parent_content(r.chunk_id)
                if parent_content:
                    contexts.append(parent_content)
                else:
                    contexts.append(r.content)
            else:
                contexts.append(r.content)
        
        # 5. 构建引用
        references = []
        for r in all_results[:top_k]:
            meta = r.metadata
            references.append(Reference(
                chunk_id=r.chunk_id,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                section_path=meta.get("section_path", ""),
                section_title=meta.get("section_title", ""),
                page=meta.get("page_start", 0),
                content_preview=r.content[:200]
            ))
        
        # 6. 生成答案 (包含图谱上下文)
        answer = self._generate_answer(question, contexts, references, graph_context)
        
        return ChatResponse(
            answer=answer,
            references=references,
            query=question,
            context_used=contexts,
            model=self.model
        )
    
    def _get_parent_content(self, chunk_id: str) -> Optional[str]:
        """获取父 Chunk 内容"""
        if not self.parent_chunk_store:
            return None
        
        try:
            parent_id = self.parent_chunk_store.get_parent_chunk_id(chunk_id)
            if parent_id:
                parent_info = self.parent_chunk_store.get_chunk_info(parent_id)
                if parent_info:
                    return parent_info.get("content")
        except Exception as e:
            logger.warning(f"Failed to get parent chunk: {e}")
        
        return None
    
    def _generate_answer(
        self,
        question: str,
        contexts: List[str],
        references: List[Reference],
        graph_context: str = ""
    ) -> str:
        """生成答案"""
        if not self.llm_client:
            return "LLM client not configured."
        
        # 构建上下文
        context_text = "\n\n---\n\n".join([
            f"[来源 {i+1}: {ref.format_citation()}]\n{ctx}"
            for i, (ctx, ref) in enumerate(zip(contexts, references))
        ])
        
        # 添加图谱上下文
        if graph_context:
            context_text = f"{graph_context}\n\n---\n\n{context_text}"
        
        prompt = f"""你是一个严格的技术文档问答助手。你必须严格根据提供的参考资料回答问题。

【重要规则】
1. **严禁编造**: 只能使用参考资料中明确存在的信息，绝对不能添加、推测或编造任何内容
2. **找不到就说明**: 如果参考资料中没有相关信息，必须明确回答"根据现有文档，未找到关于此问题的相关信息"
3. **引用来源**: 每个回答点都必须标注来源，格式为 [来源 X]
4. **原文优先**: 尽量使用参考资料中的原文表述，避免改写或总结
5. **不做延伸**: 不要提供参考资料之外的建议、解释或背景知识

参考资料:
{context_text}

用户问题: {question}

请严格根据上述参考资料回答:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"生成答案时出错: {str(e)}"
    
    def ask_stream(
        self,
        question: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ):
        """流式问答 (返回生成器)"""
        # 简化的流式实现
        response = self.ask(question, filter_dict, top_k)
        
        # 模拟流式输出
        for char in response.answer:
            yield char
