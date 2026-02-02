"""
查询转换模块
实现 Query Expansion 和 HyDE 策略提高召回率
"""
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TransformedQuery:
    """转换后的查询"""
    original: str
    expanded: List[str]  # 扩展后的多个查询
    hyde_answer: Optional[str] = None  # HyDE 生成的假设答案


class QueryTransformer:
    """
    查询转换器
    
    策略:
    1. Query Expansion (多路径查询): 将用户问题拆解/改写成多个版本
    2. HyDE (Hypothetical Document Embeddings): 先生成"伪答案"再检索
    3. 术语词典扩展: 使用领域术语词典进行同义词和缩写扩展
    """
    
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4.1-mini",
        enable_expansion: bool = True,
        enable_hyde: bool = True,
        enable_terminology: bool = True,
        expansion_count: int = 3,
        terminology_dict=None
    ):
        self.llm_client = llm_client
        self.model = model
        self.enable_expansion = enable_expansion
        self.enable_hyde = enable_hyde
        self.enable_terminology = enable_terminology
        self.expansion_count = expansion_count
        self.terminology_dict = terminology_dict
        
        self._init_llm_client()
        self._init_terminology()
    
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
    
    def _init_terminology(self):
        """初始化术语词典"""
        if self.terminology_dict is None and self.enable_terminology:
            try:
                from src.retrieval.terminology_dict import get_terminology_dict
                self.terminology_dict = get_terminology_dict()
                logger.info("Terminology dictionary loaded")
            except Exception as e:
                logger.warning(f"Failed to load terminology dict: {e}")
                self.terminology_dict = None
    
    def transform(
        self,
        query: str,
        context: Optional[str] = None
    ) -> TransformedQuery:
        """
        转换查询
        
        Args:
            query: 原始查询
            context: 可选的上下文 (如产品名称、对话历史)
        
        Returns:
            TransformedQuery: 包含扩展查询和 HyDE 答案
        """
        expanded = [query]  # 始终包含原始查询
        hyde_answer = None
        
        # 1. 术语词典扩展 (快速, 无 LLM 调用)
        if self.enable_terminology and self.terminology_dict:
            try:
                term_expanded = self.terminology_dict.expand_query(query)
                for q in term_expanded:
                    if q not in expanded:
                        expanded.append(q)
            except Exception as e:
                logger.warning(f"Terminology expansion failed: {e}")
        
        # 2. LLM 驱动的查询扩展
        if self.enable_expansion and self.llm_client:
            try:
                llm_expanded = self._expand_query(query, context)
                for q in llm_expanded:
                    if q not in expanded:
                        expanded.append(q)
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")
        
        # 3. HyDE 假设文档生成
        if self.enable_hyde and self.llm_client:
            try:
                hyde_answer = self._generate_hyde(query, context)
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")
        
        return TransformedQuery(
            original=query,
            expanded=expanded,
            hyde_answer=hyde_answer
        )
    
    def _expand_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> List[str]:
        """
        Query Expansion
        
        将用户问题改写成多个版本:
        - 含缩写版本
        - 含全称版本
        - 功能描述版本
        """
        context_str = f"\n上下文: {context}" if context else ""
        
        prompt = f"""你是一个技术文档检索助手。请将以下用户问题改写成 {self.expansion_count} 个不同的搜索查询，以提高检索召回率。

用户问题: {query}{context_str}

要求:
1. 保持原意，但使用不同的表达方式
2. 如果问题中有缩写，展开一个版本使用全称
3. 如果问题中有全称，缩写一个版本
4. 可以添加同义词或相关术语
5. 每个查询一行，不要编号

只输出查询，不要其他解释。"""

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        expanded_text = response.choices[0].message.content.strip()
        expanded_queries = [q.strip() for q in expanded_text.split("\n") if q.strip()]
        
        # 确保包含原始查询
        if query not in expanded_queries:
            expanded_queries.insert(0, query)
        
        return expanded_queries[:self.expansion_count + 1]
    
    def _generate_hyde(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        HyDE (Hypothetical Document Embeddings)
        
        生成一个假设的答案文档，用于语义检索
        因为答案与文档的文风更接近，通常能提高召回率
        """
        context_str = f"\n上下文: {context}" if context else ""
        
        prompt = f"""你是一个技术文档写作专家。请假设你是在写一份技术文档来回答以下问题。
直接写出可能出现在技术文档中的段落，不要写"根据文档"这样的前缀。

问题: {query}{context_str}

请用技术文档的风格写 2-3 段可能包含答案的内容:"""

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def simple_expand(self, query: str) -> List[str]:
        """
        简单的规则化扩展 (不需要 LLM)
        
        适用于离线或快速场景
        """
        expanded = [query]
        
        # 常见缩写展开
        abbreviations = {
            "API": "Application Programming Interface",
            "SDK": "Software Development Kit",
            "URL": "Uniform Resource Locator",
            "HTTP": "HyperText Transfer Protocol",
            "HTTPS": "HTTP Secure",
            "JSON": "JavaScript Object Notation",
            "XML": "Extensible Markup Language",
            "SQL": "Structured Query Language",
            "DB": "Database",
            "UI": "User Interface",
            "UX": "User Experience",
            "CLI": "Command Line Interface",
            "GUI": "Graphical User Interface",
            "TCP": "Transmission Control Protocol",
            "UDP": "User Datagram Protocol",
            "IP": "Internet Protocol",
        }
        
        query_upper = query.upper()
        for abbr, full in abbreviations.items():
            if abbr in query_upper:
                # 添加含全称的版本
                expanded.append(query.replace(abbr, full).replace(abbr.lower(), full))
                break
            if full.upper() in query_upper:
                # 添加含缩写的版本
                expanded.append(query.replace(full, abbr).replace(full.lower(), abbr))
                break
        
        # 添加同义词版本
        synonyms = {
            "配置": ["设置", "参数"],
            "安装": ["部署", "setup"],
            "错误": ["异常", "故障", "问题"],
            "方法": ["函数", "接口", "操作"],
        }
        
        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    expanded.append(query.replace(word, syn))
                break
        
        return expanded[:self.expansion_count + 1]
