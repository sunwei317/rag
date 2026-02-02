"""
PDF 结构化解析模块
支持 Layout-aware 解析，保留标题层级、段落、列表、表格、图注等结构
支持扫描版 PDF 的 OCR 处理
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from loguru import logger
import base64


class ElementType(Enum):
    """文档元素类型"""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    IMAGE = "image"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"


@dataclass
class BBox:
    """边界框"""
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    def to_dict(self) -> Dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class DocumentElement:
    """文档元素"""
    element_type: ElementType
    content: str
    page: int
    bbox: Optional[BBox] = None
    level: int = 0  # 标题级别 (1-6)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.element_type.value,
            "content": self.content,
            "page": self.page,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "level": self.level,
            "metadata": self.metadata
        }


@dataclass
class Section:
    """文档章节"""
    title: str
    level: int
    section_path: str  # e.g., "1.2.3"
    elements: List[DocumentElement] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "level": self.level,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "elements": [e.to_dict() for e in self.elements],
            "children": [c.to_dict() for c in self.children]
        }
    
    def get_text(self) -> str:
        """获取章节的纯文本内容"""
        texts = []
        for elem in self.elements:
            texts.append(elem.content)
        return "\n".join(texts)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    filename: str
    title: str
    sections: List[Section]
    metadata: Dict[str, Any]
    total_pages: int
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
            "total_pages": self.total_pages
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save(self, path: Path):
        """保存解析结果"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


class PDFParser:
    """
    PDF 结构化解析器
    
    核心功能:
    1. 提取文档结构 (标题层级、段落、列表)
    2. 表格解析 (转换为 Markdown 格式)
    3. 图片提取与描述
    4. 代码块识别
    5. 扫描版 PDF 的 OCR 处理
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        detect_headers_footers: bool = True,
        min_heading_font_size: float = 12.0,
        enable_ocr: bool = True,
        ocr_provider: str = "paddleocr",
        ocr_lang: str = "ch",
        ocr_confidence_threshold: float = 0.6,
    ):
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.detect_headers_footers = detect_headers_footers
        self.min_heading_font_size = min_heading_font_size
        self.enable_ocr = enable_ocr
        self.ocr_provider = ocr_provider
        self.ocr_lang = ocr_lang
        self.ocr_confidence_threshold = ocr_confidence_threshold
        
        # OCR 处理器 (懒加载)
        self._ocr_processor = None
        
        # 标题模式匹配
        self.heading_patterns = [
            r"^第[一二三四五六七八九十\d]+章\s+",  # 中文章节
            r"^第[一二三四五六七八九十\d]+节\s+",  # 中文节
            r"^\d+\.\s+",  # 1. Title
            r"^\d+\.\d+\s+",  # 1.1 Title
            r"^\d+\.\d+\.\d+\s+",  # 1.1.1 Title
            r"^[A-Z]\.\s+",  # A. Title
            r"^附录\s*[A-Z]?\s*",  # 附录
        ]
    
    def _get_ocr_processor(self):
        """懒加载 OCR 处理器"""
        if self._ocr_processor is None and self.enable_ocr:
            try:
                from src.ingestion.ocr_processor import OCRProcessor
                self._ocr_processor = OCRProcessor(
                    provider=self.ocr_provider,
                    lang=self.ocr_lang,
                    confidence_threshold=self.ocr_confidence_threshold,
                    enable_table_recognition=self.extract_tables,
                    enable_layout_analysis=True
                )
                logger.info(f"OCR processor initialized: {self.ocr_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize OCR processor: {e}")
                self._ocr_processor = None
        return self._ocr_processor
    
    def _is_scanned_pdf(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """检测是否为扫描版 PDF"""
        try:
            doc = fitz.open(pdf_path)
            total_text_length = 0
            pages_checked = min(sample_pages, len(doc))
            
            for i in range(pages_checked):
                page = doc[i]
                text = page.get_text()
                total_text_length += len(text.strip())
            
            doc.close()
            
            # 如果平均每页文本少于 100 字符，判定为扫描版
            avg_text_per_page = total_text_length / pages_checked if pages_checked > 0 else 0
            is_scanned = avg_text_per_page < 100
            
            logger.info(f"PDF scan detection: avg_text={avg_text_per_page:.0f} chars/page, is_scanned={is_scanned}")
            return is_scanned
        except Exception as e:
            logger.warning(f"Scan detection failed: {e}")
            return False
    
    def parse(
        self, 
        pdf_path: Path, 
        metadata: Optional[Dict[str, Any]] = None,
        force_ocr: bool = False
    ) -> ParsedDocument:
        """
        解析 PDF 文档
        
        Args:
            pdf_path: PDF 文件路径
            metadata: 额外的元数据 (产品、版本、密级等)
            force_ocr: 强制使用 OCR (即使不是扫描版)
        
        Returns:
            ParsedDocument: 结构化的文档对象
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"开始解析 PDF: {pdf_path.name}")
        
        # 检测是否为扫描版
        is_scanned = self._is_scanned_pdf(pdf_path)
        use_ocr = force_ocr or (is_scanned and self.enable_ocr)
        
        if use_ocr:
            logger.info("检测到扫描版 PDF，启用 OCR 处理...")
            return self._parse_with_ocr(pdf_path, metadata)
        
        # 使用 PyMuPDF 提取基本内容 (非扫描版)
        return self._parse_native(pdf_path, metadata)
    
    def _parse_native(
        self,
        pdf_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """原生解析 (非扫描版)"""
        doc = fitz.open(pdf_path)
        
        # 提取文档元数据
        doc_metadata = self._extract_metadata(doc, metadata or {})
        doc_metadata["is_scanned"] = False
        doc_metadata["ocr_used"] = False
        
        # 提取所有元素
        elements = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_elements = self._extract_page_elements(doc, page, page_num + 1)
            elements.extend(page_elements)
        
        # 提取表格 (使用 pdfplumber)
        if self.extract_tables:
            table_elements = self._extract_tables(pdf_path)
            elements = self._merge_table_elements(elements, table_elements)
        
        # 构建章节结构
        sections = self._build_section_tree(elements)
        
        # 生成文档 ID
        import hashlib
        doc_id = hashlib.md5(f"{pdf_path.name}_{doc_metadata.get('version', '')}".encode()).hexdigest()[:12]
        
        # 提取文档标题
        title = self._extract_title(doc) or pdf_path.stem
        
        total_pages = len(doc)
        doc.close()
        
        parsed_doc = ParsedDocument(
            doc_id=doc_id,
            filename=pdf_path.name,
            title=title,
            sections=sections,
            metadata=doc_metadata,
            total_pages=total_pages
        )
        
        logger.info(f"解析完成: {len(sections)} 个顶级章节, {sum(len(s.elements) for s in sections)} 个元素")
        
        return parsed_doc
    
    def _parse_with_ocr(
        self,
        pdf_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """使用 OCR 解析扫描版 PDF"""
        ocr = self._get_ocr_processor()
        
        if ocr is None:
            logger.warning("OCR processor not available, falling back to native parsing")
            return self._parse_native(pdf_path, metadata)
        
        # 执行 OCR
        ocr_result = ocr.process_pdf(str(pdf_path))
        
        # 将 OCR 结果转换为文档元素
        elements = []
        
        for page_result in ocr_result.pages:
            page_num = page_result.page_num
            
            # 处理文本块
            for text_block in page_result.text_blocks:
                element_type = (
                    ElementType.HEADING 
                    if text_block.block_type == "title" 
                    else ElementType.PARAGRAPH
                )
                
                # 尝试识别标题层级
                level = 0
                if element_type == ElementType.HEADING:
                    level = self._detect_heading_level_from_text(text_block.text)
                
                bbox = BBox(
                    x0=text_block.bbox[0],
                    y0=text_block.bbox[1],
                    x1=text_block.bbox[2],
                    y1=text_block.bbox[3]
                )
                
                element = DocumentElement(
                    element_type=element_type,
                    content=text_block.text,
                    page=page_num,
                    bbox=bbox,
                    level=level,
                    metadata={
                        "confidence": text_block.confidence,
                        "ocr": True
                    }
                )
                elements.append(element)
            
            # 处理表格
            for table in page_result.tables:
                element = DocumentElement(
                    element_type=ElementType.TABLE,
                    content=table.markdown or table.to_markdown(),
                    page=page_num,
                    bbox=BBox(
                        x0=table.bbox[0],
                        y0=table.bbox[1],
                        x1=table.bbox[2],
                        y1=table.bbox[3]
                    ),
                    metadata={
                        "confidence": table.confidence,
                        "cells": table.cells,
                        "html": table.html,
                        "ocr": True
                    }
                )
                elements.append(element)
            
            # 处理图片
            for figure in page_result.figures:
                element = DocumentElement(
                    element_type=ElementType.IMAGE,
                    content=figure.caption or "[图片]",
                    page=page_num,
                    bbox=BBox(
                        x0=figure.bbox[0],
                        y0=figure.bbox[1],
                        x1=figure.bbox[2],
                        y1=figure.bbox[3]
                    ),
                    metadata={
                        "image_path": figure.image_path,
                        "description": figure.description,
                        "ocr": True
                    }
                )
                elements.append(element)
        
        # 按页面和位置排序
        elements.sort(key=lambda e: (e.page, e.bbox.y0 if e.bbox else 0))
        
        # 构建章节结构
        sections = self._build_section_tree(elements)
        
        # 生成文档 ID
        import hashlib
        doc_id = hashlib.md5(f"{pdf_path.name}_ocr".encode()).hexdigest()[:12]
        
        # 构建元数据
        doc_metadata = metadata or {}
        doc_metadata.update({
            "is_scanned": True,
            "ocr_used": True,
            "ocr_provider": ocr_result.ocr_provider,
            "ocr_confidence": ocr_result.avg_confidence,
            "total_pages": ocr_result.total_pages
        })
        
        # 尝试从第一个标题元素提取文档标题
        title = pdf_path.stem
        for elem in elements:
            if elem.element_type == ElementType.HEADING and elem.level <= 1:
                title = elem.content[:100]
                break
        
        parsed_doc = ParsedDocument(
            doc_id=doc_id,
            filename=pdf_path.name,
            title=title,
            sections=sections,
            metadata=doc_metadata,
            total_pages=ocr_result.total_pages
        )
        
        logger.info(
            f"OCR 解析完成: {len(sections)} 个顶级章节, "
            f"{sum(len(s.elements) for s in sections)} 个元素, "
            f"平均置信度: {ocr_result.avg_confidence:.2f}"
        )
        
        return parsed_doc
    
    def _detect_heading_level_from_text(self, text: str) -> int:
        """从文本内容推断标题层级"""
        text = text.strip()
        
        # 中文章节
        if re.match(r"^第[一二三四五六七八九十\d]+章", text):
            return 1
        if re.match(r"^第[一二三四五六七八九十\d]+节", text):
            return 2
        
        # 数字编号
        if re.match(r"^\d+\.\d+\.\d+\.\d+\s+", text):
            return 4
        if re.match(r"^\d+\.\d+\.\d+\s+", text):
            return 3
        if re.match(r"^\d+\.\d+\s+", text):
            return 2
        if re.match(r"^\d+\.\s+", text):
            return 1
        
        # 附录
        if re.match(r"^附录\s*[A-Z]?", text):
            return 1
        
        return 0
    
    def _extract_metadata(self, doc: fitz.Document, extra_metadata: Dict) -> Dict[str, Any]:
        """提取文档元数据"""
        pdf_metadata = doc.metadata or {}
        
        return {
            "title": pdf_metadata.get("title", ""),
            "author": pdf_metadata.get("author", ""),
            "subject": pdf_metadata.get("subject", ""),
            "keywords": pdf_metadata.get("keywords", ""),
            "creator": pdf_metadata.get("creator", ""),
            "producer": pdf_metadata.get("producer", ""),
            "creation_date": pdf_metadata.get("creationDate", ""),
            "mod_date": pdf_metadata.get("modDate", ""),
            "total_pages": len(doc),
            **extra_metadata
        }
    
    def _extract_title(self, doc: fitz.Document) -> Optional[str]:
        """提取文档标题"""
        # 尝试从元数据获取
        if doc.metadata and doc.metadata.get("title"):
            return doc.metadata["title"]
        
        # 尝试从第一页提取最大字体的文本
        if len(doc) > 0:
            first_page = doc[0]
            blocks = first_page.get_text("dict")["blocks"]
            
            max_font_size = 0
            title_text = ""
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                title_text = span["text"].strip()
            
            if title_text:
                return title_text
        
        return None
    
    def _extract_page_elements(
        self, 
        doc: fitz.Document,
        page: fitz.Page, 
        page_num: int
    ) -> List[DocumentElement]:
        """提取页面元素"""
        elements = []
        
        # 获取页面字典数据
        blocks = page.get_text("dict")["blocks"]
        
        # 检测页眉页脚区域
        header_y, footer_y = self._detect_header_footer_bounds(page)
        
        for block in blocks:
            # 跳过图片块
            if block["type"] == 1:
                if self.extract_images:
                    img_elem = self._extract_image(doc, page, block, page_num)
                    if img_elem:
                        elements.append(img_elem)
                continue
            
            # 文本块
            if "lines" not in block:
                continue
            
            bbox = BBox(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])
            
            # 跳过页眉页脚
            if self.detect_headers_footers:
                if bbox.y0 < header_y or bbox.y1 > footer_y:
                    continue
            
            # 合并行文本
            block_text = ""
            block_font_size = 0
            is_bold = False
            
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                    block_font_size = max(block_font_size, span["size"])
                    if "bold" in span["font"].lower():
                        is_bold = True
                
                block_text += line_text.strip() + " "
            
            block_text = block_text.strip()
            if not block_text:
                continue
            
            # 判断元素类型
            element_type, level = self._classify_element(block_text, block_font_size, is_bold)
            
            elements.append(DocumentElement(
                element_type=element_type,
                content=block_text,
                page=page_num,
                bbox=bbox,
                level=level,
                metadata={"font_size": block_font_size, "is_bold": is_bold}
            ))
        
        return elements
    
    def _classify_element(
        self, 
        text: str, 
        font_size: float, 
        is_bold: bool
    ) -> Tuple[ElementType, int]:
        """分类文档元素"""
        text_stripped = text.strip()
        
        # 检查是否是列表项
        if re.match(r"^[\•\-\*\◦\▪]\s+", text_stripped) or re.match(r"^\d+[\.\)]\s+\w", text_stripped):
            return ElementType.LIST, 0
        
        # 检查是否是代码块 (启发式: 等宽字体特征)
        if self._is_code_block(text_stripped):
            return ElementType.CODE, 0
        
        # 检查是否是标题
        for pattern in self.heading_patterns:
            if re.match(pattern, text_stripped):
                level = self._determine_heading_level(text_stripped, font_size)
                return ElementType.HEADING, level
        
        # 基于字体大小判断标题
        if font_size >= self.min_heading_font_size and is_bold and len(text_stripped) < 100:
            level = self._determine_heading_level(text_stripped, font_size)
            return ElementType.HEADING, level
        
        # 默认为段落
        return ElementType.PARAGRAPH, 0
    
    def _determine_heading_level(self, text: str, font_size: float) -> int:
        """确定标题级别"""
        # 基于编号模式
        if re.match(r"^第[一二三四五六七八九十\d]+章", text):
            return 1
        if re.match(r"^\d+\.\d+\.\d+\s+", text):
            return 3
        if re.match(r"^\d+\.\d+\s+", text):
            return 2
        if re.match(r"^\d+\.\s+", text):
            return 1
        
        # 基于字体大小
        if font_size >= 18:
            return 1
        elif font_size >= 16:
            return 2
        elif font_size >= 14:
            return 3
        else:
            return 4
    
    def _is_code_block(self, text: str) -> bool:
        """判断是否是代码块"""
        # 简单启发式规则
        code_indicators = [
            r"^\s*def\s+\w+",  # Python 函数
            r"^\s*class\s+\w+",  # Python 类
            r"^\s*import\s+",  # import 语句
            r"^\s*#include",  # C/C++ 头文件
            r"^\s*\$\s+",  # Shell 命令
            r"^\s*>\s+",  # 命令提示符
            r"^\s*\{.*\}\s*$",  # JSON/代码块
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        return False
    
    def _detect_header_footer_bounds(self, page: fitz.Page) -> Tuple[float, float]:
        """检测页眉页脚边界"""
        rect = page.rect
        # 默认上下各留 5% 作为页眉页脚区域
        header_y = rect.height * 0.05
        footer_y = rect.height * 0.95
        return header_y, footer_y
    
    def _extract_image(
        self, 
        doc: fitz.Document,
        page: fitz.Page, 
        block: Dict, 
        page_num: int
    ) -> Optional[DocumentElement]:
        """提取图片元素"""
        try:
            bbox = BBox(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])
            
            # 获取图片数据
            xref = block.get("xref", 0)
            if xref:
                base_image = doc.extract_image(xref)
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_b64 = base64.b64encode(image_bytes).decode()
                    
                    # 尝试提取附近的图注
                    caption = self._find_nearby_caption(page, bbox)
                    
                    return DocumentElement(
                        element_type=ElementType.IMAGE,
                        content=caption or f"[Image on page {page_num}]",
                        page=page_num,
                        bbox=bbox,
                        metadata={
                            "image_data": image_b64[:100] + "...",  # 截断以节省空间
                            "image_ext": image_ext,
                            "has_caption": bool(caption)
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to extract image: {e}")
        
        return None
    
    def _find_nearby_caption(self, page: fitz.Page, image_bbox: BBox) -> Optional[str]:
        """查找图片附近的图注"""
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 1 or "lines" not in block:
                continue
            
            block_bbox = BBox(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])
            
            # 检查是否在图片下方附近
            if abs(block_bbox.y0 - image_bbox.y1) < 30:
                text = " ".join(
                    span["text"] 
                    for line in block["lines"] 
                    for span in line["spans"]
                ).strip()
                
                # 检查是否像图注
                if re.match(r"^(图|Figure|Fig\.?|表|Table)\s*[\d\-\.]+", text, re.IGNORECASE):
                    return text
        
        return None
    
    def _extract_tables(self, pdf_path: Path) -> List[DocumentElement]:
        """使用 pdfplumber 提取表格"""
        table_elements = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        
                        # 转换为 Markdown 格式
                        markdown_table = self._table_to_markdown(table)
                        
                        if markdown_table:
                            table_elements.append(DocumentElement(
                                element_type=ElementType.TABLE,
                                content=markdown_table,
                                page=page_num,
                                metadata={"rows": len(table), "cols": len(table[0]) if table else 0}
                            ))
        except Exception as e:
            logger.warning(f"Failed to extract tables with pdfplumber: {e}")
        
        return table_elements
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """将表格转换为 Markdown 格式"""
        if not table:
            return ""
        
        # 清理单元格
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                cell_text = str(cell) if cell else ""
                cell_text = cell_text.replace("\n", " ").replace("|", "\\|").strip()
                cleaned_row.append(cell_text)
            cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""
        
        # 确保所有行有相同的列数
        max_cols = max(len(row) for row in cleaned_table)
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # 构建 Markdown 表格
        lines = []
        
        # 表头
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # 数据行
        for row in cleaned_table[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def _merge_table_elements(
        self, 
        elements: List[DocumentElement], 
        table_elements: List[DocumentElement]
    ) -> List[DocumentElement]:
        """合并表格元素到元素列表"""
        # 按页面分组表格
        tables_by_page = {}
        for te in table_elements:
            tables_by_page.setdefault(te.page, []).append(te)
        
        # 将表格插入到合适位置
        merged = []
        current_page = 0
        
        for elem in elements:
            if elem.page != current_page:
                # 插入上一页的表格
                if current_page in tables_by_page:
                    merged.extend(tables_by_page[current_page])
                current_page = elem.page
            
            merged.append(elem)
        
        # 插入最后一页的表格
        if current_page in tables_by_page:
            merged.extend(tables_by_page[current_page])
        
        return merged
    
    def _build_section_tree(self, elements: List[DocumentElement]) -> List[Section]:
        """构建章节树结构"""
        sections = []
        section_stack = []  # 用于跟踪当前章节层级
        section_counters = [0] * 6  # 用于生成 section_path
        
        current_section = None
        
        for elem in elements:
            if elem.element_type == ElementType.HEADING and elem.level > 0:
                # 更新章节计数器
                level = elem.level
                section_counters[level - 1] += 1
                for i in range(level, 6):
                    section_counters[i] = 0
                
                # 生成 section_path
                section_path = ".".join(str(c) for c in section_counters[:level] if c > 0)
                
                # 创建新章节
                new_section = Section(
                    title=elem.content,
                    level=level,
                    section_path=section_path,
                    page_start=elem.page
                )
                
                # 确定父章节
                while section_stack and section_stack[-1].level >= level:
                    popped = section_stack.pop()
                    popped.page_end = elem.page - 1
                
                if section_stack:
                    section_stack[-1].children.append(new_section)
                else:
                    sections.append(new_section)
                
                section_stack.append(new_section)
                current_section = new_section
            
            elif current_section:
                current_section.elements.append(elem)
            else:
                # 没有章节时,创建一个默认章节
                if not sections:
                    default_section = Section(
                        title="Introduction",
                        level=1,
                        section_path="0",
                        page_start=elem.page
                    )
                    sections.append(default_section)
                    section_stack.append(default_section)
                    current_section = default_section
                
                current_section.elements.append(elem)
        
        # 设置最后一个章节的结束页
        for section in section_stack:
            if section.page_end == 0:
                section.page_end = elements[-1].page if elements else 1
        
        return sections


# 便捷函数
def parse_pdf(pdf_path: str | Path, metadata: Optional[Dict] = None) -> ParsedDocument:
    """解析 PDF 文档的便捷函数"""
    parser = PDFParser()
    return parser.parse(Path(pdf_path), metadata)
