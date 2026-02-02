"""
一致性校验模块
确保生成的文档内容一致性
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import re

from .section_writer import GeneratedDocument, SectionContent


@dataclass
class ConsistencyIssue:
    """一致性问题"""
    issue_type: str  # 'terminology', 'contradiction', 'missing_citation', 'format'
    severity: str    # 'error', 'warning', 'info'
    location: str    # 章节位置
    description: str
    suggestion: str
    
    def to_dict(self) -> Dict:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "location": self.location,
            "description": self.description,
            "suggestion": self.suggestion
        }


@dataclass
class ConsistencyReport:
    """一致性检查报告"""
    document_title: str
    total_issues: int
    errors: int
    warnings: int
    issues: List[ConsistencyIssue]
    passed: bool
    
    def to_dict(self) -> Dict:
        return {
            "document_title": self.document_title,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "issues": [i.to_dict() for i in self.issues],
            "passed": self.passed
        }
    
    def summary(self) -> str:
        """生成摘要"""
        status = "✓ 通过" if self.passed else "✗ 未通过"
        return f"""一致性检查报告: {self.document_title}
状态: {status}
总问题数: {self.total_issues} (错误: {self.errors}, 警告: {self.warnings})
"""


class ConsistencyChecker:
    """
    一致性检查器
    
    检查项目:
    1. 术语一致性: 同一术语的不同表述
    2. 矛盾检测: 前后文内容矛盾
    3. 引用覆盖率: 关键结论是否有来源
    4. 格式一致性: 列表、表格、代码块格式
    """
    
    def __init__(
        self,
        terminology_manager=None,
        llm_client=None,
        model: str = "gpt-4.1-mini",
        min_citation_coverage: float = 0.8
    ):
        self.terminology_manager = terminology_manager
        self.llm_client = llm_client
        self.model = model
        self.min_citation_coverage = min_citation_coverage
        
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
    
    def check(self, document: GeneratedDocument) -> ConsistencyReport:
        """
        执行一致性检查
        
        Args:
            document: 生成的文档
        
        Returns:
            ConsistencyReport: 检查报告
        """
        logger.info(f"Checking consistency for: {document.title}")
        
        issues = []
        
        # 1. 术语一致性检查
        terminology_issues = self._check_terminology(document)
        issues.extend(terminology_issues)
        
        # 2. 矛盾检测
        contradiction_issues = self._check_contradictions(document)
        issues.extend(contradiction_issues)
        
        # 3. 引用覆盖率检查
        citation_issues = self._check_citation_coverage(document)
        issues.extend(citation_issues)
        
        # 4. 格式一致性检查
        format_issues = self._check_format_consistency(document)
        issues.extend(format_issues)
        
        # 统计
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        
        # 判断是否通过
        passed = errors == 0
        
        return ConsistencyReport(
            document_title=document.title,
            total_issues=len(issues),
            errors=errors,
            warnings=warnings,
            issues=issues,
            passed=passed
        )
    
    def _check_terminology(self, document: GeneratedDocument) -> List[ConsistencyIssue]:
        """检查术语一致性"""
        issues = []
        
        if not self.terminology_manager:
            return issues
        
        # 合并所有内容
        full_text = "\n\n".join([s.content for s in document.sections])
        
        # 检查术语一致性
        term_issues = self.terminology_manager.check_consistency(full_text)
        
        for ti in term_issues:
            issues.append(ConsistencyIssue(
                issue_type="terminology",
                severity="warning",
                location="全文",
                description=f"术语 '{ti['term']}' 存在多种表述: {', '.join(ti['variants'])}",
                suggestion=ti['suggestion']
            ))
        
        return issues
    
    def _check_contradictions(self, document: GeneratedDocument) -> List[ConsistencyIssue]:
        """检查矛盾内容"""
        issues = []
        
        if not self.llm_client:
            return issues
        
        # 提取可能存在矛盾的内容类型
        # 主要关注: 版本号、默认值、端口、参数值等
        
        patterns_to_check = [
            (r'版本[：:]\s*[\d\.]+', "版本信息"),
            (r'默认[值]?[：:为是]\s*\S+', "默认值"),
            (r'端口[：:为是]\s*\d+', "端口号"),
            (r'(\d+)\s*(MB|GB|KB|ms|秒|分钟)', "数值参数"),
        ]
        
        found_values = {}  # {category: [(value, location)]}
        
        for section in document.sections:
            for pattern, category in patterns_to_check:
                matches = re.findall(pattern, section.content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = "".join(match)
                    
                    if category not in found_values:
                        found_values[category] = []
                    found_values[category].append((match, section.title))
        
        # 检查每个类别的值是否一致
        for category, values in found_values.items():
            unique_values = set(v[0] for v in values)
            if len(unique_values) > 1:
                locations = ", ".join(set(v[1] for v in values))
                issues.append(ConsistencyIssue(
                    issue_type="contradiction",
                    severity="error",
                    location=locations,
                    description=f"{category}存在矛盾: {', '.join(unique_values)}",
                    suggestion=f"请检查并统一{category}的描述"
                ))
        
        # 使用 LLM 进行深度矛盾检测
        if len(document.sections) >= 2:
            llm_issues = self._llm_contradiction_check(document)
            issues.extend(llm_issues)
        
        return issues
    
    def _llm_contradiction_check(self, document: GeneratedDocument) -> List[ConsistencyIssue]:
        """使用 LLM 检测矛盾"""
        issues = []
        
        # 准备检查内容
        sections_text = "\n\n---\n\n".join([
            f"【{s.title}】\n{s.content[:1000]}"  # 限制长度
            for s in document.sections
        ])
        
        prompt = f"""你是一个技术文档审核专家。请检查以下文档片段是否存在内容矛盾。

文档内容:
{sections_text}

请检查是否存在以下类型的矛盾:
1. 同一参数在不同位置有不同的值
2. 步骤顺序或操作说明相互矛盾
3. 功能描述前后不一致

如果发现矛盾，请按以下格式输出 (每行一个问题):
位置1 | 位置2 | 矛盾描述

如果没有发现矛盾，请输出: 无矛盾"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            if result != "无矛盾" and "|" in result:
                for line in result.split("\n"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        issues.append(ConsistencyIssue(
                            issue_type="contradiction",
                            severity="error",
                            location=f"{parts[0]} / {parts[1]}",
                            description=parts[2],
                            suggestion="请核实并修正矛盾内容"
                        ))
        except Exception as e:
            logger.warning(f"LLM contradiction check failed: {e}")
        
        return issues
    
    def _check_citation_coverage(self, document: GeneratedDocument) -> List[ConsistencyIssue]:
        """检查引用覆盖率"""
        issues = []
        
        for section in document.sections:
            # 检测可能需要引用的内容
            # 启发式: 包含数字、具体参数、"必须"/"应该" 等的句子
            sentences = re.split(r'[。.!?！？]', section.content)
            
            sentences_needing_citation = []
            for sent in sentences:
                if len(sent) < 10:
                    continue
                
                # 检查是否包含需要引用的内容
                needs_citation = (
                    re.search(r'\d+', sent) or  # 包含数字
                    re.search(r'必须|应该|需要|要求|规定', sent) or  # 规范性语言
                    re.search(r'默认|配置|参数|设置', sent)  # 配置相关
                )
                
                if needs_citation:
                    has_citation = re.search(r'\[资料\d+\]|\[\d+\]', sent)
                    if not has_citation:
                        sentences_needing_citation.append(sent[:50])
            
            # 计算引用覆盖率
            if sentences_needing_citation:
                total_citations = len(section.citations)
                coverage = total_citations / len(sentences_needing_citation) if sentences_needing_citation else 1.0
                
                if coverage < self.min_citation_coverage:
                    issues.append(ConsistencyIssue(
                        issue_type="missing_citation",
                        severity="warning",
                        location=section.title,
                        description=f"引用覆盖率不足 ({coverage:.0%})，部分内容缺少来源标注",
                        suggestion="请为关键结论添加引用来源"
                    ))
        
        return issues
    
    def _check_format_consistency(self, document: GeneratedDocument) -> List[ConsistencyIssue]:
        """检查格式一致性"""
        issues = []
        
        list_styles = set()  # 列表样式 (-, *, 1.)
        code_block_styles = set()  # 代码块样式
        
        for section in document.sections:
            content = section.content
            
            # 检查列表样式
            if re.search(r'^[-*•]\s', content, re.MULTILINE):
                list_styles.add("bullet")
            if re.search(r'^\d+[.)\)]\s', content, re.MULTILINE):
                list_styles.add("numbered")
            
            # 检查代码块
            if "```" in content:
                code_block_styles.add("fenced")
            if re.search(r'^    \S', content, re.MULTILINE):
                code_block_styles.add("indented")
        
        # 检查是否混用了不同样式
        if len(code_block_styles) > 1:
            issues.append(ConsistencyIssue(
                issue_type="format",
                severity="info",
                location="全文",
                description="代码块使用了多种格式 (围栏式和缩进式)",
                suggestion="建议统一使用围栏式代码块 (```)"
            ))
        
        return issues
    
    def fix_issues(
        self,
        document: GeneratedDocument,
        issues: List[ConsistencyIssue]
    ) -> GeneratedDocument:
        """
        尝试自动修复一致性问题
        
        目前支持修复:
        - 术语不一致
        
        Args:
            document: 原文档
            issues: 问题列表
        
        Returns:
            修复后的文档
        """
        if not self.terminology_manager:
            return document
        
        # 修复术语一致性
        fixed_sections = []
        for section in document.sections:
            fixed_content = self.terminology_manager.normalize_text(section.content)
            
            fixed_sections.append(SectionContent(
                title=section.title,
                level=section.level,
                content=fixed_content,
                citations=section.citations,
                word_count=len(fixed_content)
            ))
        
        return GeneratedDocument(
            title=document.title,
            sections=fixed_sections,
            metadata=document.metadata,
            all_citations=document.all_citations
        )
