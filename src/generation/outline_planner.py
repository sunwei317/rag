"""
大纲规划模块
根据需求生成文档大纲并规划每节所需的证据
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SectionPlan:
    """章节规划"""
    title: str
    level: int  # 1-6
    description: str  # 该章节的内容描述
    required_topics: List[str]  # 该章节需要检索的主题
    word_count_estimate: int  # 预估字数
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "level": self.level,
            "description": self.description,
            "required_topics": self.required_topics,
            "word_count_estimate": self.word_count_estimate
        }


@dataclass
class DocumentOutline:
    """文档大纲"""
    title: str
    doc_type: str
    target_audience: str
    sections: List[SectionPlan]
    template_source: Optional[str] = None  # 参考的模板文档
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "doc_type": self.doc_type,
            "target_audience": self.target_audience,
            "sections": [s.to_dict() for s in self.sections],
            "template_source": self.template_source
        }
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式的大纲"""
        lines = [f"# {self.title}", ""]
        
        for section in self.sections:
            prefix = "#" * (section.level + 1)
            lines.append(f"{prefix} {section.title}")
            lines.append(f"*{section.description}*")
            lines.append("")
        
        return "\n".join(lines)


class OutlinePlanner:
    """
    大纲规划器
    
    功能:
    1. 根据文档类型和目标读者生成合适的大纲
    2. 检索相似文档的结构作为参考
    3. 规划每个章节需要检索的主题
    """
    
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-oss-20b",
        hybrid_searcher=None,
        local_api_base: str = None,
        local_model: str = None
    ):
        self.llm_client = llm_client
        self.model = model
        self.hybrid_searcher = hybrid_searcher
        self.local_api_base = local_api_base
        self.local_model = local_model
        
        self._init_llm_client()
        
        # 预定义的文档类型模板
        self._doc_templates = self._load_templates()
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                
                # 优先使用本地 LLM 服务
                if self.local_api_base:
                    self.llm_client = OpenAI(
                        base_url=self.local_api_base,
                        api_key="not-needed"
                    )
                    if self.local_model:
                        self.model = self.local_model
                    logger.info(f"OutlinePlanner: Initialized Local LLM: {self.local_api_base}")
                else:
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        self.llm_client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to init LLM client: {e}")
    
    def _load_templates(self) -> Dict[str, List[SectionPlan]]:
        """加载预定义的文档模板"""
        return {
            "installation_guide": [
                SectionPlan("概述", 1, "介绍产品和安装目的", ["产品简介", "系统要求"], 200),
                SectionPlan("系统要求", 1, "列出硬件和软件要求", ["硬件要求", "软件依赖", "操作系统"], 300),
                SectionPlan("安装前准备", 1, "安装前需要完成的准备工作", ["准备工作", "环境配置"], 400),
                SectionPlan("安装步骤", 1, "详细的安装步骤", ["安装流程", "配置步骤"], 800),
                SectionPlan("安装验证", 1, "验证安装是否成功", ["验证方法", "测试命令"], 300),
                SectionPlan("常见问题", 1, "安装过程中的常见问题及解决方案", ["错误处理", "故障排除"], 500),
            ],
            "api_reference": [
                SectionPlan("概述", 1, "API 概述和认证方式", ["API简介", "认证授权"], 300),
                SectionPlan("快速开始", 1, "快速开始使用 API", ["示例代码", "快速入门"], 400),
                SectionPlan("API 端点", 1, "所有 API 端点的详细说明", ["接口列表", "参数说明"], 1500),
                SectionPlan("请求与响应", 1, "请求格式和响应结构", ["请求格式", "响应结构"], 500),
                SectionPlan("错误码", 1, "错误码列表及说明", ["错误码", "异常处理"], 400),
                SectionPlan("SDK", 1, "各语言 SDK 使用说明", ["SDK", "客户端库"], 500),
            ],
            "user_manual": [
                SectionPlan("简介", 1, "产品简介和功能概述", ["产品概述", "功能特性"], 300),
                SectionPlan("快速入门", 1, "快速开始使用产品", ["入门指南", "基本操作"], 500),
                SectionPlan("基本功能", 1, "核心功能使用说明", ["功能说明", "操作步骤"], 1000),
                SectionPlan("高级功能", 1, "高级功能和配置", ["高级配置", "进阶功能"], 800),
                SectionPlan("最佳实践", 1, "使用最佳实践和建议", ["最佳实践", "使用建议"], 500),
                SectionPlan("故障排除", 1, "常见问题和解决方案", ["故障排除", "问题解决"], 500),
            ],
            "release_notes": [
                SectionPlan("版本概述", 1, "本次发布的主要内容", ["版本信息", "发布说明"], 200),
                SectionPlan("新增功能", 1, "新增的功能特性", ["新功能", "新特性"], 500),
                SectionPlan("改进优化", 1, "改进和优化的内容", ["优化", "改进"], 400),
                SectionPlan("问题修复", 1, "修复的问题列表", ["Bug修复", "问题修复"], 400),
                SectionPlan("已知问题", 1, "已知问题和限制", ["已知问题", "限制"], 200),
                SectionPlan("升级说明", 1, "从旧版本升级的说明", ["升级", "迁移"], 300),
            ],
            "troubleshooting": [
                SectionPlan("概述", 1, "故障排除指南介绍", ["故障排除概述"], 150),
                SectionPlan("常见问题", 1, "常见问题及解决方案", ["常见问题", "FAQ"], 800),
                SectionPlan("错误信息", 1, "错误信息及处理方法", ["错误码", "异常信息"], 600),
                SectionPlan("诊断工具", 1, "诊断和调试工具使用", ["诊断", "调试工具"], 400),
                SectionPlan("日志分析", 1, "日志分析方法", ["日志", "日志分析"], 400),
                SectionPlan("联系支持", 1, "获取技术支持的方式", ["技术支持", "联系方式"], 150),
            ],
        }
    
    def plan(
        self,
        product: str,
        doc_type: str,
        target_audience: str = "developer",
        additional_requirements: Optional[str] = None,
        use_template: bool = True,
        search_similar: bool = True
    ) -> DocumentOutline:
        """
        生成文档大纲
        
        Args:
            product: 产品名称
            doc_type: 文档类型 (installation_guide, api_reference, user_manual, etc.)
            target_audience: 目标读者 (developer, admin, end_user)
            additional_requirements: 额外需求
            use_template: 是否使用预定义模板
            search_similar: 是否检索相似文档作为参考
        
        Returns:
            DocumentOutline: 文档大纲
        """
        logger.info(f"Planning outline for {product} - {doc_type}")
        
        # 1. 获取基础模板
        base_sections = []
        template_source = None
        
        if use_template and doc_type in self._doc_templates:
            base_sections = self._doc_templates[doc_type].copy()
            template_source = f"内置模板: {doc_type}"
        
        # 2. 检索相似文档结构
        similar_structure = None
        if search_similar and self.hybrid_searcher:
            similar_structure = self._search_similar_structure(product, doc_type)
            if similar_structure:
                template_source = similar_structure.get("source", template_source)
        
        # 3. 使用 LLM 优化大纲
        if self.llm_client:
            sections = self._refine_with_llm(
                product=product,
                doc_type=doc_type,
                target_audience=target_audience,
                base_sections=base_sections,
                similar_structure=similar_structure,
                additional_requirements=additional_requirements
            )
        else:
            sections = base_sections
        
        # 4. 生成标题
        title = self._generate_title(product, doc_type)
        
        return DocumentOutline(
            title=title,
            doc_type=doc_type,
            target_audience=target_audience,
            sections=sections,
            template_source=template_source
        )
    
    def _search_similar_structure(
        self,
        product: str,
        doc_type: str
    ) -> Optional[Dict[str, Any]]:
        """检索相似文档的结构"""
        if not self.hybrid_searcher:
            return None
        
        try:
            # 搜索同类型文档的目录/结构
            query = f"{product} {doc_type} 目录 章节结构"
            results = self.hybrid_searcher.search(
                query=query,
                top_k=3,
                filter_dict={"doc_type": doc_type} if doc_type else None
            )
            
            if results:
                return {
                    "source": results[0].metadata.get("doc_title", ""),
                    "structure": results[0].content
                }
        except Exception as e:
            logger.warning(f"Failed to search similar structure: {e}")
        
        return None
    
    def _refine_with_llm(
        self,
        product: str,
        doc_type: str,
        target_audience: str,
        base_sections: List[SectionPlan],
        similar_structure: Optional[Dict],
        additional_requirements: Optional[str]
    ) -> List[SectionPlan]:
        """使用 LLM 优化大纲"""
        # 构建基础结构描述
        base_structure = ""
        if base_sections:
            base_structure = "\n".join([
                f"- {s.title}: {s.description}"
                for s in base_sections
            ])
        
        similar_ref = ""
        if similar_structure:
            similar_ref = f"\n参考文档结构:\n{similar_structure.get('structure', '')[:500]}"
        
        additional = ""
        if additional_requirements:
            additional = f"\n额外要求:\n{additional_requirements}"
        
        prompt = f"""你是一个技术文档架构师。请为以下文档设计详细的大纲结构。

产品: {product}
文档类型: {doc_type}
目标读者: {target_audience}

基础结构:
{base_structure}
{similar_ref}
{additional}

请输出优化后的大纲，格式如下 (每行一个章节):
标题 | 级别(1-3) | 描述 | 需要检索的主题(逗号分隔) | 预估字数

示例:
安装概述 | 1 | 介绍安装目的和系统要求 | 产品简介,系统要求 | 200

只输出大纲内容，不要其他解释。"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_outline(content)
        except Exception as e:
            logger.warning(f"LLM outline generation failed: {e}")
            return base_sections
    
    def _parse_outline(self, content: str) -> List[SectionPlan]:
        """解析 LLM 生成的大纲"""
        sections = []
        
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                try:
                    sections.append(SectionPlan(
                        title=parts[0],
                        level=int(parts[1]) if parts[1].isdigit() else 1,
                        description=parts[2],
                        required_topics=[t.strip() for t in parts[3].split(",")],
                        word_count_estimate=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 300
                    ))
                except Exception:
                    continue
        
        return sections
    
    def _generate_title(self, product: str, doc_type: str) -> str:
        """生成文档标题"""
        type_names = {
            "installation_guide": "安装指南",
            "api_reference": "API 参考文档",
            "user_manual": "用户手册",
            "release_notes": "发布说明",
            "troubleshooting": "故障排除指南",
            "configuration": "配置指南",
            "quick_start": "快速入门指南",
        }
        
        type_name = type_names.get(doc_type, doc_type)
        return f"{product} {type_name}"
