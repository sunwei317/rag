"""
领域术语词典模块
提供技术领域的同义词、缩写、术语扩展
"""
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from loguru import logger


@dataclass
class TermEntry:
    """术语条目"""
    term: str
    canonical: str  # 规范形式
    aliases: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    domain: str = "general"  # 所属领域
    
    def all_forms(self) -> Set[str]:
        """获取术语的所有形式"""
        forms = {self.term, self.canonical}
        forms.update(self.aliases)
        forms.update(self.abbreviations)
        return forms


class DomainTerminologyDict:
    """
    领域术语词典
    
    功能:
    1. 术语规范化：将不同表述映射到标准形式
    2. 同义词扩展：查询时扩展同义词提高召回
    3. 缩写处理：自动展开/收缩缩写
    4. 领域分类：按领域组织术语
    """
    
    def __init__(self, custom_dict_path: Optional[str] = None):
        # 术语 -> 规范形式
        self._term_to_canonical: Dict[str, str] = {}
        
        # 规范形式 -> TermEntry
        self._canonical_to_entry: Dict[str, TermEntry] = {}
        
        # 缩写 -> 全称
        self._abbr_to_full: Dict[str, str] = {}
        
        # 全称 -> 缩写
        self._full_to_abbr: Dict[str, str] = {}
        
        # 领域 -> 术语列表
        self._domain_terms: Dict[str, List[str]] = {}
        
        # 加载内置词典
        self._load_builtin_dict()
        
        # 加载自定义词典
        if custom_dict_path and Path(custom_dict_path).exists():
            self._load_custom_dict(custom_dict_path)
    
    def _load_builtin_dict(self):
        """加载内置技术词典"""
        
        # ==================== 编程语言 ====================
        self._add_term(TermEntry(
            term="Python",
            canonical="Python",
            aliases=["python", "py"],
            abbreviations=["py"],
            related_terms=["CPython", "PyPy", "Jython"],
            domain="programming"
        ))
        
        self._add_term(TermEntry(
            term="JavaScript",
            canonical="JavaScript",
            aliases=["javascript", "js", "JS"],
            abbreviations=["JS"],
            related_terms=["ECMAScript", "ES6", "Node.js"],
            domain="programming"
        ))
        
        self._add_term(TermEntry(
            term="TypeScript",
            canonical="TypeScript",
            aliases=["typescript", "ts", "TS"],
            abbreviations=["TS"],
            related_terms=["JavaScript", "tsc"],
            domain="programming"
        ))
        
        # ==================== API/协议 ====================
        self._add_term(TermEntry(
            term="API",
            canonical="Application Programming Interface",
            aliases=["api", "接口", "应用程序接口", "应用编程接口"],
            abbreviations=["API"],
            related_terms=["REST API", "GraphQL", "SDK"],
            domain="api"
        ))
        
        self._add_term(TermEntry(
            term="REST",
            canonical="Representational State Transfer",
            aliases=["rest", "RESTful", "restful"],
            abbreviations=["REST"],
            related_terms=["HTTP", "API", "CRUD"],
            domain="api"
        ))
        
        self._add_term(TermEntry(
            term="SDK",
            canonical="Software Development Kit",
            aliases=["sdk", "软件开发包", "开发工具包"],
            abbreviations=["SDK"],
            related_terms=["API", "Library"],
            domain="api"
        ))
        
        self._add_term(TermEntry(
            term="HTTP",
            canonical="HyperText Transfer Protocol",
            aliases=["http", "超文本传输协议"],
            abbreviations=["HTTP"],
            related_terms=["HTTPS", "REST", "TCP"],
            domain="protocol"
        ))
        
        self._add_term(TermEntry(
            term="HTTPS",
            canonical="HyperText Transfer Protocol Secure",
            aliases=["https", "HTTP Secure"],
            abbreviations=["HTTPS"],
            related_terms=["HTTP", "TLS", "SSL"],
            domain="protocol"
        ))
        
        # ==================== 数据格式 ====================
        self._add_term(TermEntry(
            term="JSON",
            canonical="JavaScript Object Notation",
            aliases=["json", "Json"],
            abbreviations=["JSON"],
            related_terms=["XML", "YAML", "BSON"],
            domain="data_format"
        ))
        
        self._add_term(TermEntry(
            term="XML",
            canonical="Extensible Markup Language",
            aliases=["xml", "Xml"],
            abbreviations=["XML"],
            related_terms=["JSON", "HTML", "XSLT"],
            domain="data_format"
        ))
        
        self._add_term(TermEntry(
            term="YAML",
            canonical="YAML Ain't Markup Language",
            aliases=["yaml", "yml"],
            abbreviations=["YAML", "YML"],
            related_terms=["JSON", "TOML", "INI"],
            domain="data_format"
        ))
        
        # ==================== 数据库 ====================
        self._add_term(TermEntry(
            term="SQL",
            canonical="Structured Query Language",
            aliases=["sql", "结构化查询语言"],
            abbreviations=["SQL"],
            related_terms=["MySQL", "PostgreSQL", "NoSQL"],
            domain="database"
        ))
        
        self._add_term(TermEntry(
            term="NoSQL",
            canonical="Not Only SQL",
            aliases=["nosql", "非关系型数据库"],
            abbreviations=["NoSQL"],
            related_terms=["MongoDB", "Redis", "Cassandra"],
            domain="database"
        ))
        
        self._add_term(TermEntry(
            term="数据库",
            canonical="数据库",
            aliases=["database", "DB", "db", "资料库"],
            abbreviations=["DB"],
            related_terms=["SQL", "NoSQL", "RDBMS"],
            domain="database"
        ))
        
        # ==================== AI/ML ====================
        self._add_term(TermEntry(
            term="AI",
            canonical="Artificial Intelligence",
            aliases=["ai", "人工智能"],
            abbreviations=["AI"],
            related_terms=["ML", "DL", "NLP"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="ML",
            canonical="Machine Learning",
            aliases=["ml", "机器学习"],
            abbreviations=["ML"],
            related_terms=["AI", "DL", "模型训练"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="DL",
            canonical="Deep Learning",
            aliases=["dl", "深度学习"],
            abbreviations=["DL"],
            related_terms=["神经网络", "CNN", "RNN", "Transformer"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="NLP",
            canonical="Natural Language Processing",
            aliases=["nlp", "自然语言处理"],
            abbreviations=["NLP"],
            related_terms=["LLM", "文本分析", "分词"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="LLM",
            canonical="Large Language Model",
            aliases=["llm", "大语言模型", "大型语言模型"],
            abbreviations=["LLM"],
            related_terms=["GPT", "Claude", "Gemini", "Transformer"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="RAG",
            canonical="Retrieval Augmented Generation",
            aliases=["rag", "检索增强生成"],
            abbreviations=["RAG"],
            related_terms=["LLM", "向量检索", "知识库"],
            domain="ai"
        ))
        
        self._add_term(TermEntry(
            term="Embedding",
            canonical="Embedding",
            aliases=["embedding", "嵌入", "向量嵌入", "词嵌入"],
            abbreviations=[],
            related_terms=["向量", "Vector", "语义表示"],
            domain="ai"
        ))
        
        # ==================== 云计算/DevOps ====================
        self._add_term(TermEntry(
            term="Docker",
            canonical="Docker",
            aliases=["docker", "容器"],
            abbreviations=[],
            related_terms=["Kubernetes", "Container", "镜像"],
            domain="devops"
        ))
        
        self._add_term(TermEntry(
            term="Kubernetes",
            canonical="Kubernetes",
            aliases=["kubernetes", "k8s", "K8s"],
            abbreviations=["K8s"],
            related_terms=["Docker", "Pod", "容器编排"],
            domain="devops"
        ))
        
        self._add_term(TermEntry(
            term="CI/CD",
            canonical="Continuous Integration and Continuous Deployment",
            aliases=["cicd", "CI CD", "持续集成", "持续部署"],
            abbreviations=["CI/CD", "CICD"],
            related_terms=["Jenkins", "GitHub Actions", "DevOps"],
            domain="devops"
        ))
        
        # ==================== 网络/安全 ====================
        self._add_term(TermEntry(
            term="TCP",
            canonical="Transmission Control Protocol",
            aliases=["tcp", "传输控制协议"],
            abbreviations=["TCP"],
            related_terms=["UDP", "IP", "Socket"],
            domain="network"
        ))
        
        self._add_term(TermEntry(
            term="UDP",
            canonical="User Datagram Protocol",
            aliases=["udp", "用户数据报协议"],
            abbreviations=["UDP"],
            related_terms=["TCP", "IP", "Socket"],
            domain="network"
        ))
        
        self._add_term(TermEntry(
            term="IP",
            canonical="Internet Protocol",
            aliases=["ip", "网际协议", "IP地址"],
            abbreviations=["IP"],
            related_terms=["TCP", "IPv4", "IPv6"],
            domain="network"
        ))
        
        self._add_term(TermEntry(
            term="SSL",
            canonical="Secure Sockets Layer",
            aliases=["ssl", "安全套接层"],
            abbreviations=["SSL"],
            related_terms=["TLS", "HTTPS", "证书"],
            domain="security"
        ))
        
        self._add_term(TermEntry(
            term="TLS",
            canonical="Transport Layer Security",
            aliases=["tls", "传输层安全"],
            abbreviations=["TLS"],
            related_terms=["SSL", "HTTPS", "加密"],
            domain="security"
        ))
        
        self._add_term(TermEntry(
            term="OAuth",
            canonical="Open Authorization",
            aliases=["oauth", "OAuth2", "OAuth 2.0"],
            abbreviations=["OAuth"],
            related_terms=["认证", "授权", "JWT", "Token"],
            domain="security"
        ))
        
        self._add_term(TermEntry(
            term="JWT",
            canonical="JSON Web Token",
            aliases=["jwt", "Json Web Token"],
            abbreviations=["JWT"],
            related_terms=["Token", "OAuth", "认证"],
            domain="security"
        ))
        
        # ==================== 前端 ====================
        self._add_term(TermEntry(
            term="HTML",
            canonical="HyperText Markup Language",
            aliases=["html", "超文本标记语言"],
            abbreviations=["HTML"],
            related_terms=["CSS", "DOM", "网页"],
            domain="frontend"
        ))
        
        self._add_term(TermEntry(
            term="CSS",
            canonical="Cascading Style Sheets",
            aliases=["css", "层叠样式表", "样式表"],
            abbreviations=["CSS"],
            related_terms=["HTML", "SCSS", "SASS"],
            domain="frontend"
        ))
        
        self._add_term(TermEntry(
            term="DOM",
            canonical="Document Object Model",
            aliases=["dom", "文档对象模型"],
            abbreviations=["DOM"],
            related_terms=["HTML", "JavaScript", "Virtual DOM"],
            domain="frontend"
        ))
        
        self._add_term(TermEntry(
            term="UI",
            canonical="User Interface",
            aliases=["ui", "用户界面", "界面"],
            abbreviations=["UI"],
            related_terms=["UX", "GUI", "前端"],
            domain="frontend"
        ))
        
        self._add_term(TermEntry(
            term="UX",
            canonical="User Experience",
            aliases=["ux", "用户体验"],
            abbreviations=["UX"],
            related_terms=["UI", "交互设计", "可用性"],
            domain="frontend"
        ))
        
        # ==================== 通用开发术语 ====================
        self._add_term(TermEntry(
            term="配置",
            canonical="配置",
            aliases=["config", "configuration", "设置", "参数", "设定"],
            abbreviations=["config", "cfg"],
            related_terms=["环境变量", "配置文件"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="安装",
            canonical="安装",
            aliases=["install", "installation", "部署", "setup", "装设"],
            abbreviations=[],
            related_terms=["配置", "初始化"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="错误",
            canonical="错误",
            aliases=["error", "异常", "exception", "故障", "问题", "bug", "Bug"],
            abbreviations=[],
            related_terms=["调试", "日志", "排错"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="函数",
            canonical="函数",
            aliases=["function", "方法", "method", "操作", "过程", "procedure"],
            abbreviations=["func", "fn"],
            related_terms=["参数", "返回值", "调用"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="变量",
            canonical="变量",
            aliases=["variable", "var", "参数", "属性", "字段"],
            abbreviations=["var"],
            related_terms=["常量", "类型"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="类",
            canonical="类",
            aliases=["class", "对象", "object", "实例", "instance"],
            abbreviations=[],
            related_terms=["方法", "属性", "继承"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="异步",
            canonical="异步",
            aliases=["async", "asynchronous", "非同步", "异步编程"],
            abbreviations=["async"],
            related_terms=["同步", "并发", "await", "Promise"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="同步",
            canonical="同步",
            aliases=["sync", "synchronous", "同步编程"],
            abbreviations=["sync"],
            related_terms=["异步", "阻塞"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="缓存",
            canonical="缓存",
            aliases=["cache", "caching", "快取"],
            abbreviations=[],
            related_terms=["Redis", "Memcached", "内存"],
            domain="general"
        ))
        
        self._add_term(TermEntry(
            term="日志",
            canonical="日志",
            aliases=["log", "logging", "logs", "记录"],
            abbreviations=[],
            related_terms=["调试", "监控", "追踪"],
            domain="general"
        ))
        
        logger.info(f"Loaded {len(self._canonical_to_entry)} built-in terms")
    
    def _add_term(self, entry: TermEntry):
        """添加术语条目"""
        canonical = entry.canonical
        
        # 注册到规范形式映射
        self._canonical_to_entry[canonical] = entry
        
        # 注册所有形式到规范形式的映射
        for form in entry.all_forms():
            self._term_to_canonical[form.lower()] = canonical
        
        # 注册缩写
        for abbr in entry.abbreviations:
            self._abbr_to_full[abbr.upper()] = canonical
            self._full_to_abbr[canonical] = abbr.upper()
        
        # 注册到领域
        if entry.domain not in self._domain_terms:
            self._domain_terms[entry.domain] = []
        self._domain_terms[entry.domain].append(canonical)
    
    def _load_custom_dict(self, path: str):
        """加载自定义词典 (JSON 格式)"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data.get("terms", []):
                entry = TermEntry(
                    term=item["term"],
                    canonical=item.get("canonical", item["term"]),
                    aliases=item.get("aliases", []),
                    abbreviations=item.get("abbreviations", []),
                    related_terms=item.get("related_terms", []),
                    domain=item.get("domain", "custom")
                )
                self._add_term(entry)
            
            logger.info(f"Loaded custom dictionary from {path}")
        except Exception as e:
            logger.warning(f"Failed to load custom dictionary: {e}")
    
    def normalize(self, term: str) -> str:
        """
        规范化术语
        
        将术语转换为标准形式
        """
        return self._term_to_canonical.get(term.lower(), term)
    
    def expand(self, term: str) -> List[str]:
        """
        扩展术语
        
        返回术语的所有同义形式
        """
        normalized = term.lower()
        
        if normalized not in self._term_to_canonical:
            return [term]
        
        canonical = self._term_to_canonical[normalized]
        entry = self._canonical_to_entry.get(canonical)
        
        if not entry:
            return [term]
        
        return list(entry.all_forms())
    
    def expand_abbreviation(self, abbr: str) -> Optional[str]:
        """展开缩写为全称"""
        return self._abbr_to_full.get(abbr.upper())
    
    def abbreviate(self, term: str) -> Optional[str]:
        """将全称转换为缩写"""
        canonical = self._term_to_canonical.get(term.lower(), term)
        return self._full_to_abbr.get(canonical)
    
    def get_related_terms(self, term: str) -> List[str]:
        """获取相关术语"""
        normalized = term.lower()
        
        if normalized not in self._term_to_canonical:
            return []
        
        canonical = self._term_to_canonical[normalized]
        entry = self._canonical_to_entry.get(canonical)
        
        if not entry:
            return []
        
        return entry.related_terms
    
    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询
        
        识别查询中的术语并生成扩展版本
        """
        expanded_queries = [query]
        
        # 查找查询中的术语
        words = re.findall(r'\b[\w/]+\b', query, re.UNICODE)
        
        for word in words:
            # 尝试展开缩写
            full_form = self.expand_abbreviation(word)
            if full_form and full_form != word:
                expanded_queries.append(query.replace(word, full_form))
            
            # 尝试获取同义词
            synonyms = self.expand(word)
            for syn in synonyms:
                if syn.lower() != word.lower():
                    expanded_queries.append(query.replace(word, syn))
                    break  # 只添加一个同义词版本
        
        # 去重
        seen = set()
        unique = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        
        return unique[:5]  # 限制数量
    
    def get_domain_terms(self, domain: str) -> List[str]:
        """获取某领域的所有术语"""
        return self._domain_terms.get(domain, [])
    
    def list_domains(self) -> List[str]:
        """列出所有领域"""
        return list(self._domain_terms.keys())
    
    def add_custom_term(
        self,
        term: str,
        canonical: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        abbreviations: Optional[List[str]] = None,
        related_terms: Optional[List[str]] = None,
        domain: str = "custom"
    ):
        """动态添加自定义术语"""
        entry = TermEntry(
            term=term,
            canonical=canonical or term,
            aliases=aliases or [],
            abbreviations=abbreviations or [],
            related_terms=related_terms or [],
            domain=domain
        )
        self._add_term(entry)
    
    def export_dict(self, path: str):
        """导出词典为 JSON"""
        data = {
            "terms": []
        }
        
        for canonical, entry in self._canonical_to_entry.items():
            data["terms"].append({
                "term": entry.term,
                "canonical": entry.canonical,
                "aliases": entry.aliases,
                "abbreviations": entry.abbreviations,
                "related_terms": entry.related_terms,
                "domain": entry.domain
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported {len(data['terms'])} terms to {path}")


# 全局单例
_default_dict: Optional[DomainTerminologyDict] = None


def get_terminology_dict() -> DomainTerminologyDict:
    """获取术语词典单例"""
    global _default_dict
    if _default_dict is None:
        _default_dict = DomainTerminologyDict()
    return _default_dict
