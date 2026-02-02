"""
文档生成代理
整合大纲规划、分节写作、一致性校验的完整流程
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from .outline_planner import OutlinePlanner, DocumentOutline
from .section_writer import SectionWriter, GeneratedDocument
from .consistency_checker import ConsistencyChecker, ConsistencyReport

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class GenerationConfig:
    """生成配置"""
    product: str
    doc_type: str
    target_audience: str = "developer"
    version: Optional[str] = None
    additional_requirements: Optional[str] = None
    output_format: str = "markdown"  # markdown, docx, html
    enable_consistency_check: bool = True
    auto_fix_issues: bool = True


@dataclass
class GenerationResult:
    """生成结果"""
    document: GeneratedDocument
    outline: DocumentOutline
    consistency_report: Optional[ConsistencyReport]
    output_path: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            "document": self.document.to_dict(),
            "outline": self.outline.to_dict(),
            "consistency_report": self.consistency_report.to_dict() if self.consistency_report else None,
            "output_path": self.output_path
        }


class DocAgent:
    """
    文档生成代理
    
    完整的文档生成流程:
    1. 需求分析 → 确定文档类型和目标读者
    2. 大纲规划 → 检索模板 + 规划章节结构
    3. 分节检索 → 每节独立检索所需信息
    4. 分节写作 → 基于检索结果生成内容
    5. 一致性校验 → 术语/矛盾/引用检查
    6. 格式化输出 → Markdown/Docx/HTML
    """
    
    def __init__(
        self,
        hybrid_searcher=None,
        reranker=None,
        terminology_manager=None,
        outline_planner: Optional[OutlinePlanner] = None,
        section_writer: Optional[SectionWriter] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        output_dir: str = "./output"
    ):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.terminology_manager = terminology_manager
        
        # 初始化子模块
        self.outline_planner = outline_planner or OutlinePlanner(
            hybrid_searcher=hybrid_searcher
        )
        
        self.section_writer = section_writer or SectionWriter(
            hybrid_searcher=hybrid_searcher,
            reranker=reranker,
            terminology_manager=terminology_manager
        )
        
        self.consistency_checker = consistency_checker or ConsistencyChecker(
            terminology_manager=terminology_manager
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        config: GenerationConfig,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        生成文档
        
        Args:
            config: 生成配置
            filter_dict: 检索过滤条件
        
        Returns:
            GenerationResult: 生成结果
        """
        logger.info(f"Starting document generation: {config.product} - {config.doc_type}")
        
        # 构建过滤条件
        if filter_dict is None:
            filter_dict = {}
        
        if config.version:
            filter_dict["version"] = config.version
        
        # 1. 大纲规划
        logger.info("Step 1: Planning outline...")
        outline = self.outline_planner.plan(
            product=config.product,
            doc_type=config.doc_type,
            target_audience=config.target_audience,
            additional_requirements=config.additional_requirements
        )
        
        logger.info(f"Outline planned with {len(outline.sections)} sections")
        
        # 2. 分节写作
        logger.info("Step 2: Writing sections...")
        document = self.section_writer.write_document(
            outline=outline,
            product=config.product,
            filter_dict=filter_dict
        )
        
        logger.info(f"Document generated with {len(document.sections)} sections")
        
        # 3. 一致性校验
        consistency_report = None
        if config.enable_consistency_check:
            logger.info("Step 3: Checking consistency...")
            consistency_report = self.consistency_checker.check(document)
            
            logger.info(f"Consistency check: {consistency_report.total_issues} issues found")
            
            # 自动修复
            if config.auto_fix_issues and not consistency_report.passed:
                logger.info("Attempting to fix issues...")
                document = self.consistency_checker.fix_issues(
                    document,
                    consistency_report.issues
                )
        
        # 4. 导出文档
        logger.info("Step 4: Exporting document...")
        output_path = self._export_document(document, config)
        
        logger.info(f"Document exported to: {output_path}")
        
        return GenerationResult(
            document=document,
            outline=outline,
            consistency_report=consistency_report,
            output_path=str(output_path) if output_path else None
        )
    
    def generate_from_prompt(
        self,
        prompt: str,
        product: Optional[str] = None
    ) -> GenerationResult:
        """
        从自然语言提示生成文档
        
        Args:
            prompt: 用户需求描述
            product: 产品名称 (可选)
        
        Returns:
            GenerationResult: 生成结果
        """
        # 解析用户需求
        config = self._parse_prompt(prompt, product)
        
        return self.generate(config)
    
    def _parse_prompt(self, prompt: str, product: Optional[str]) -> GenerationConfig:
        """解析用户需求"""
        import re
        
        # 文档类型关键词映射
        doc_type_keywords = {
            "installation_guide": ["安装", "部署", "install", "deploy", "setup"],
            "api_reference": ["api", "接口", "reference"],
            "user_manual": ["用户手册", "使用指南", "user manual", "user guide"],
            "release_notes": ["发布说明", "release notes", "changelog", "更新日志"],
            "troubleshooting": ["故障", "排错", "troubleshoot", "问题"],
            "configuration": ["配置", "config", "设置"],
            "quick_start": ["快速入门", "quick start", "入门"],
        }
        
        # 识别文档类型
        doc_type = "user_manual"  # 默认
        prompt_lower = prompt.lower()
        
        for dtype, keywords in doc_type_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                doc_type = dtype
                break
        
        # 识别目标读者
        target_audience = "developer"
        if any(kw in prompt_lower for kw in ["管理员", "admin", "运维"]):
            target_audience = "admin"
        elif any(kw in prompt_lower for kw in ["用户", "user", "客户"]):
            target_audience = "end_user"
        
        # 识别产品名称
        if not product:
            # 尝试从 prompt 中提取产品名称
            product_match = re.search(r'["\']([^"\']+)["\']|产品[：:]\s*(\S+)', prompt)
            if product_match:
                product = product_match.group(1) or product_match.group(2)
            else:
                product = "Product"
        
        return GenerationConfig(
            product=product,
            doc_type=doc_type,
            target_audience=target_audience,
            additional_requirements=prompt
        )
    
    def _export_document(
        self,
        document: GeneratedDocument,
        config: GenerationConfig
    ) -> Optional[Path]:
        """导出文档"""
        # 生成文件名
        filename = f"{config.product}_{config.doc_type}".replace(" ", "_")
        
        if config.output_format == "markdown":
            output_path = self.output_dir / f"{filename}.md"
            document.export(str(output_path), "markdown")
        elif config.output_format == "docx":
            output_path = self.output_dir / f"{filename}.docx"
            document.export(str(output_path), "docx")
        elif config.output_format == "html":
            output_path = self.output_dir / f"{filename}.html"
            document.export(str(output_path), "html")
        else:
            logger.warning(f"Unknown format: {config.output_format}")
            return None
        
        return output_path
    
    def preview_outline(
        self,
        product: str,
        doc_type: str,
        target_audience: str = "developer"
    ) -> str:
        """预览大纲 (Markdown 格式)"""
        outline = self.outline_planner.plan(
            product=product,
            doc_type=doc_type,
            target_audience=target_audience
        )
        
        return outline.to_markdown()
    
    def list_doc_types(self) -> List[str]:
        """列出支持的文档类型"""
        return [
            "installation_guide",
            "api_reference",
            "user_manual",
            "release_notes",
            "troubleshooting",
            "configuration",
            "quick_start"
        ]


# 便捷函数
def generate_document(
    product: str,
    doc_type: str,
    target_audience: str = "developer",
    output_format: str = "markdown"
) -> GenerationResult:
    """生成文档的便捷函数"""
    agent = DocAgent()
    config = GenerationConfig(
        product=product,
        doc_type=doc_type,
        target_audience=target_audience,
        output_format=output_format
    )
    return agent.generate(config)
