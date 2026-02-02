"""
术语库管理模块
确保技术文档中的术语一致性
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re
from loguru import logger

from ..storage.metadata_store import MetadataStore


@dataclass
class TermInfo:
    """术语信息"""
    term: str           # 标准术语
    aliases: List[str]  # 别名/同义词
    definition: str     # 定义
    category: str       # 分类
    product: str        # 关联产品


class TerminologyManager:
    """
    术语库管理器
    
    功能:
    1. 术语标准化: 将文本中的非标准术语替换为标准术语
    2. 术语检测: 检测文本中使用的术语
    3. 一致性检查: 确保同一文档中术语使用一致
    """
    
    def __init__(self, metadata_store: Optional[MetadataStore] = None):
        self.metadata_store = metadata_store
        
        # 内置的常见术语映射
        self._builtin_terms: Dict[str, TermInfo] = {}
        self._alias_to_term: Dict[str, str] = {}
        
        self._load_builtin_terms()
        
        if metadata_store:
            self._load_terms_from_db()
    
    def _load_builtin_terms(self):
        """加载内置术语"""
        builtin = [
            TermInfo(
                term="Application Programming Interface",
                aliases=["API", "接口", "应用程序接口"],
                definition="应用程序编程接口",
                category="技术术语",
                product=""
            ),
            TermInfo(
                term="Configuration",
                aliases=["Config", "配置", "设置", "参数设置"],
                definition="系统或应用的配置",
                category="技术术语",
                product=""
            ),
            TermInfo(
                term="Database",
                aliases=["DB", "数据库", "数据存储"],
                definition="数据库",
                category="技术术语",
                product=""
            ),
            TermInfo(
                term="User Interface",
                aliases=["UI", "用户界面", "界面"],
                definition="用户界面",
                category="技术术语",
                product=""
            ),
            TermInfo(
                term="Command Line Interface",
                aliases=["CLI", "命令行", "命令行接口"],
                definition="命令行接口",
                category="技术术语",
                product=""
            ),
        ]
        
        for term_info in builtin:
            self._add_term_to_cache(term_info)
    
    def _load_terms_from_db(self):
        """从数据库加载术语"""
        if not self.metadata_store:
            return
        
        try:
            terms = self.metadata_store.list_terms()
            for t in terms:
                term_info = TermInfo(
                    term=t["term"],
                    aliases=t.get("aliases", []),
                    definition=t.get("definition", ""),
                    category=t.get("category", ""),
                    product=t.get("product", "")
                )
                self._add_term_to_cache(term_info)
            
            logger.info(f"Loaded {len(terms)} terms from database")
        except Exception as e:
            logger.warning(f"Failed to load terms from database: {e}")
    
    def _add_term_to_cache(self, term_info: TermInfo):
        """添加术语到缓存"""
        self._builtin_terms[term_info.term] = term_info
        
        # 建立别名映射
        for alias in term_info.aliases:
            self._alias_to_term[alias.lower()] = term_info.term
        self._alias_to_term[term_info.term.lower()] = term_info.term
    
    def add_term(
        self,
        term: str,
        aliases: List[str] = None,
        definition: str = "",
        category: str = "",
        product: str = ""
    ):
        """添加术语"""
        term_info = TermInfo(
            term=term,
            aliases=aliases or [],
            definition=definition,
            category=category,
            product=product
        )
        
        self._add_term_to_cache(term_info)
        
        # 持久化到数据库
        if self.metadata_store:
            self.metadata_store.add_term(
                term=term,
                aliases=aliases,
                definition=definition,
                category=category,
                product=product
            )
    
    def get_standard_term(self, text: str) -> Optional[str]:
        """获取标准术语"""
        return self._alias_to_term.get(text.lower())
    
    def normalize_text(
        self,
        text: str,
        mark_terms: bool = False
    ) -> str:
        """
        标准化文本中的术语
        
        Args:
            text: 输入文本
            mark_terms: 是否标记术语 (如 [术语: XXX])
        
        Returns:
            标准化后的文本
        """
        result = text
        
        # 按别名长度排序，优先匹配长的
        sorted_aliases = sorted(
            self._alias_to_term.keys(),
            key=len,
            reverse=True
        )
        
        for alias in sorted_aliases:
            if len(alias) < 2:  # 跳过太短的
                continue
            
            standard_term = self._alias_to_term[alias]
            
            # 构建正则模式 (忽略大小写，单词边界)
            pattern = r'\b' + re.escape(alias) + r'\b'
            
            if mark_terms:
                replacement = f"[术语: {standard_term}]"
            else:
                replacement = standard_term
            
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def detect_terms(self, text: str) -> List[Dict]:
        """
        检测文本中的术语
        
        Returns:
            List of {term, position, original_text}
        """
        detected = []
        
        for alias, standard_term in self._alias_to_term.items():
            if len(alias) < 2:
                continue
            
            pattern = r'\b' + re.escape(alias) + r'\b'
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detected.append({
                    "term": standard_term,
                    "position": match.start(),
                    "original_text": match.group()
                })
        
        # 按位置排序
        detected.sort(key=lambda x: x["position"])
        
        return detected
    
    def check_consistency(self, text: str) -> List[Dict]:
        """
        检查术语一致性
        
        返回不一致的术语使用
        """
        issues = []
        
        # 检测所有术语
        detected = self.detect_terms(text)
        
        # 按标准术语分组
        term_usages: Dict[str, Set[str]] = {}
        for d in detected:
            term = d["term"]
            original = d["original_text"]
            
            if term not in term_usages:
                term_usages[term] = set()
            term_usages[term].add(original)
        
        # 检查不一致
        for term, usages in term_usages.items():
            if len(usages) > 1:
                issues.append({
                    "term": term,
                    "variants": list(usages),
                    "suggestion": f"统一使用 '{term}' 或选择一个一致的表达"
                })
        
        return issues


def create_terminology_manager(
    metadata_store: Optional[MetadataStore] = None
) -> TerminologyManager:
    """创建术语库管理器"""
    return TerminologyManager(metadata_store)
