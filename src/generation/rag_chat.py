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
    """
    
    def __init__(
        self,
        hybrid_searcher=None,
        reranker=None,
        query_transformer=None,
        llm_client=None,
        model: str = "gpt-4.1",
        parent_chunk_store=None,  # 用于获取父 Chunk
        window_size: int = 2  # 上下文窗口大小
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.query_transformer = query_transformer
        self.llm_client = llm_client
        self.model = model
        self.parent_chunk_store = parent_chunk_store
        self.window_size = window_size
        
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
    
    def ask(
        self,
        question: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        use_query_transform: bool = True,
        use_rerank: bool = True,
        expand_context: bool = True
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
        
        Returns:
            ChatResponse: 包含答案和引用
        """
        logger.info(f"Processing question: {question[:50]}...")
        
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
        
        # 6. 生成答案
        answer = self._generate_answer(question, contexts, references)
        
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
        references: List[Reference]
    ) -> str:
        """生成答案"""
        if not self.llm_client:
            return "LLM client not configured."
        
        # 构建上下文
        context_text = "\n\n---\n\n".join([
            f"[来源 {i+1}: {ref.format_citation()}]\n{ctx}"
            for i, (ctx, ref) in enumerate(zip(contexts, references))
        ])
        
        prompt = f"""你是一个技术文档助手。请根据以下参考资料回答用户的问题。

要求:
1. 只基于提供的参考资料回答，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 在回答中引用来源，格式为 [来源 X]
4. 使用清晰、专业的技术文档风格

参考资料:
{context_text}

用户问题: {question}

请回答:"""

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
