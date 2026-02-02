"""
OCR 处理模块
处理扫描版 PDF，支持文字、表格、图片识别
"""
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not installed, some preprocessing features disabled")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class OCRProvider(Enum):
    """OCR 提供商"""
    PADDLEOCR = "paddleocr"       # PaddleOCR + PPStructure
    AZURE = "azure"               # Azure Document Intelligence
    GOOGLE = "google"             # Google Document AI
    TESSERACT = "tesseract"       # Tesseract OCR (fallback)


@dataclass
class OCRTextBlock:
    """OCR 识别的文本块"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    block_type: str = "text"  # text, title, table, figure
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "block_type": self.block_type
        }


@dataclass
class OCRTable:
    """OCR 识别的表格"""
    cells: List[List[str]]
    confidence: float
    bbox: Tuple[int, int, int, int]
    markdown: str = ""
    html: str = ""
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        if self.markdown:
            return self.markdown
        
        if not self.cells:
            return ""
        
        lines = []
        # 表头
        if self.cells:
            header = "| " + " | ".join(self.cells[0]) + " |"
            lines.append(header)
            lines.append("|" + "|".join(["---"] * len(self.cells[0])) + "|")
        
        # 数据行
        for row in self.cells[1:]:
            line = "| " + " | ".join(row) + " |"
            lines.append(line)
        
        self.markdown = "\n".join(lines)
        return self.markdown


@dataclass
class OCRFigure:
    """OCR 识别的图片"""
    image_path: str
    caption: str
    bbox: Tuple[int, int, int, int]
    description: str = ""  # 多模态模型生成的描述


@dataclass
class OCRPageResult:
    """单页 OCR 结果"""
    page_num: int
    text_blocks: List[OCRTextBlock] = field(default_factory=list)
    tables: List[OCRTable] = field(default_factory=list)
    figures: List[OCRFigure] = field(default_factory=list)
    full_text: str = ""
    is_scanned: bool = False
    confidence: float = 0.0


@dataclass
class OCRDocumentResult:
    """整个文档的 OCR 结果"""
    pages: List[OCRPageResult]
    total_pages: int
    avg_confidence: float
    is_scanned: bool
    ocr_provider: str


class OCRProcessor:
    """
    OCR 处理器
    
    支持多种 OCR 引擎：
    1. PaddleOCR + PPStructure: 开源，中英文支持好，表格识别强
    2. Azure Document Intelligence: 商业级准确率
    3. Google Document AI: 商业级准确率
    4. Tesseract: 开源兜底方案
    
    功能：
    - 自动检测是否为扫描版 PDF
    - 文字识别 (中英文)
    - 表格结构识别
    - 图片提取和描述
    - 版面分析
    """
    
    def __init__(
        self,
        provider: str = "paddleocr",
        lang: str = "ch",  # ch, en, ch+en
        use_gpu: bool = False,
        confidence_threshold: float = 0.6,
        enable_table_recognition: bool = True,
        enable_layout_analysis: bool = True,
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None
    ):
        self.provider = OCRProvider(provider)
        self.lang = lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.enable_table_recognition = enable_table_recognition
        self.enable_layout_analysis = enable_layout_analysis
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_key = azure_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        self._ocr_engine = None
        self._table_engine = None
        self._layout_engine = None
        
        self._init_engine()
    
    def _init_engine(self):
        """初始化 OCR 引擎"""
        if self.provider == OCRProvider.PADDLEOCR:
            self._init_paddleocr()
        elif self.provider == OCRProvider.AZURE:
            self._init_azure()
        elif self.provider == OCRProvider.TESSERACT:
            self._init_tesseract()
        else:
            logger.warning(f"Provider {self.provider} not fully implemented, falling back to PaddleOCR")
            self._init_paddleocr()
    
    def _init_paddleocr(self):
        """初始化 PaddleOCR"""
        try:
            from paddleocr import PaddleOCR, PPStructure
            
            # 基础 OCR 引擎
            self._ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
            
            # 表格和版面分析引擎
            if self.enable_table_recognition or self.enable_layout_analysis:
                self._layout_engine = PPStructure(
                    table=self.enable_table_recognition,
                    ocr=True,
                    show_log=False,
                    use_gpu=self.use_gpu
                )
            
            logger.info("PaddleOCR initialized successfully")
        except ImportError as e:
            logger.error(f"PaddleOCR not installed: {e}")
            logger.info("Install with: pip install paddleocr paddlepaddle")
            raise
    
    def _init_azure(self):
        """初始化 Azure Document Intelligence"""
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            if not self.azure_endpoint or not self.azure_key:
                raise ValueError("Azure credentials not configured")
            
            self._ocr_engine = DocumentAnalysisClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_key)
            )
            
            logger.info("Azure Document Intelligence initialized successfully")
        except ImportError:
            logger.error("Azure SDK not installed")
            logger.info("Install with: pip install azure-ai-formrecognizer")
            raise
    
    def _init_tesseract(self):
        """初始化 Tesseract OCR"""
        try:
            import pytesseract
            
            # 测试 Tesseract 是否可用
            pytesseract.get_tesseract_version()
            self._ocr_engine = pytesseract
            
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
            raise
    
    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        检测 PDF 是否为扫描版
        
        策略：
        1. 尝试直接提取文本
        2. 如果文本很少或为空，判定为扫描版
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            total_text_length = 0
            pages_checked = min(sample_pages, len(doc))
            
            for i in range(pages_checked):
                page = doc[i]
                text = page.get_text()
                total_text_length += len(text.strip())
            
            doc.close()
            
            # 如果平均每页文本少于 100 字符，判定为扫描版
            avg_text_per_page = total_text_length / pages_checked
            is_scanned = avg_text_per_page < 100
            
            logger.info(f"PDF scan detection: avg_text={avg_text_per_page:.0f}, is_scanned={is_scanned}")
            return is_scanned
        
        except Exception as e:
            logger.warning(f"Scan detection failed: {e}, assuming scanned")
            return True
    
    def process_pdf(self, pdf_path: str) -> OCRDocumentResult:
        """
        处理 PDF 文档
        
        Args:
            pdf_path: PDF 文件路径
        
        Returns:
            OCRDocumentResult: OCR 结果
        """
        logger.info(f"Processing PDF with OCR: {pdf_path}")
        
        # 检测是否为扫描版
        is_scanned = self.is_scanned_pdf(pdf_path)
        
        if self.provider == OCRProvider.PADDLEOCR:
            return self._process_with_paddleocr(pdf_path, is_scanned)
        elif self.provider == OCRProvider.AZURE:
            return self._process_with_azure(pdf_path, is_scanned)
        elif self.provider == OCRProvider.TESSERACT:
            return self._process_with_tesseract(pdf_path, is_scanned)
        else:
            return self._process_with_paddleocr(pdf_path, is_scanned)
    
    def _process_with_paddleocr(self, pdf_path: str, is_scanned: bool) -> OCRDocumentResult:
        """使用 PaddleOCR 处理"""
        import fitz
        
        doc = fitz.open(pdf_path)
        pages = []
        total_confidence = 0.0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 将页面转换为图像
            pix = page.get_pixmap(dpi=300)  # 高 DPI 提升识别准确率
            img_data = pix.tobytes("png")
            
            # 保存临时图像
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            
            try:
                page_result = self._ocr_page_paddleocr(tmp_path, page_num + 1)
                page_result.is_scanned = is_scanned
                pages.append(page_result)
                total_confidence += page_result.confidence
            finally:
                os.unlink(tmp_path)
        
        doc.close()
        
        avg_confidence = total_confidence / len(pages) if pages else 0.0
        
        return OCRDocumentResult(
            pages=pages,
            total_pages=len(pages),
            avg_confidence=avg_confidence,
            is_scanned=is_scanned,
            ocr_provider="paddleocr"
        )
    
    def _ocr_page_paddleocr(self, image_path: str, page_num: int) -> OCRPageResult:
        """使用 PaddleOCR 处理单页"""
        text_blocks = []
        tables = []
        figures = []
        full_text_parts = []
        confidences = []
        
        # 使用版面分析
        if self._layout_engine:
            try:
                result = self._layout_engine(image_path)
                
                for item in result:
                    item_type = item.get("type", "text")
                    bbox = item.get("bbox", [0, 0, 0, 0])
                    bbox_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    
                    if item_type == "table":
                        # 表格识别结果
                        table_html = item.get("res", {}).get("html", "")
                        table_cells = self._parse_table_html(table_html)
                        
                        table = OCRTable(
                            cells=table_cells,
                            confidence=0.9,
                            bbox=bbox_tuple,
                            html=table_html
                        )
                        table.to_markdown()
                        tables.append(table)
                        full_text_parts.append(f"\n[表格]\n{table.markdown}\n")
                    
                    elif item_type == "figure":
                        # 图片
                        figure = OCRFigure(
                            image_path="",
                            caption="",
                            bbox=bbox_tuple
                        )
                        figures.append(figure)
                        full_text_parts.append("\n[图片]\n")
                    
                    elif item_type in ("text", "title", "reference"):
                        # 文本内容
                        res = item.get("res", [])
                        for line in res:
                            if isinstance(line, dict):
                                text = line.get("text", "")
                                conf = line.get("confidence", 0.0)
                            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                                conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.8
                            else:
                                continue
                            
                            if conf >= self.confidence_threshold:
                                block = OCRTextBlock(
                                    text=text,
                                    confidence=conf,
                                    bbox=bbox_tuple,
                                    block_type="title" if item_type == "title" else "text"
                                )
                                text_blocks.append(block)
                                full_text_parts.append(text)
                                confidences.append(conf)
            
            except Exception as e:
                logger.warning(f"Layout analysis failed: {e}, falling back to basic OCR")
                self._basic_ocr_paddleocr(image_path, text_blocks, full_text_parts, confidences)
        else:
            self._basic_ocr_paddleocr(image_path, text_blocks, full_text_parts, confidences)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return OCRPageResult(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text="\n".join(full_text_parts),
            confidence=avg_confidence
        )
    
    def _basic_ocr_paddleocr(
        self,
        image_path: str,
        text_blocks: List[OCRTextBlock],
        full_text_parts: List[str],
        confidences: List[float]
    ):
        """基础 OCR (不带版面分析)"""
        result = self._ocr_engine.ocr(image_path, cls=True)
        
        if result and result[0]:
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                if conf >= self.confidence_threshold:
                    # 转换 bbox 格式
                    x1 = min(p[0] for p in bbox)
                    y1 = min(p[1] for p in bbox)
                    x2 = max(p[0] for p in bbox)
                    y2 = max(p[1] for p in bbox)
                    
                    block = OCRTextBlock(
                        text=text,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    )
                    text_blocks.append(block)
                    full_text_parts.append(text)
                    confidences.append(conf)
    
    def _parse_table_html(self, html: str) -> List[List[str]]:
        """解析表格 HTML 为二维数组"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, "html.parser")
            rows = []
            
            for tr in soup.find_all("tr"):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    cells.append(td.get_text(strip=True))
                if cells:
                    rows.append(cells)
            
            return rows
        except Exception as e:
            logger.warning(f"Table HTML parsing failed: {e}")
            return []
    
    def _process_with_azure(self, pdf_path: str, is_scanned: bool) -> OCRDocumentResult:
        """使用 Azure Document Intelligence 处理"""
        from azure.ai.formrecognizer import AnalyzeResult
        
        with open(pdf_path, "rb") as f:
            poller = self._ocr_engine.begin_analyze_document(
                "prebuilt-layout",  # 使用预构建的布局模型
                document=f
            )
        
        result: AnalyzeResult = poller.result()
        
        pages = []
        for page in result.pages:
            text_blocks = []
            tables_on_page = []
            figures_on_page = []
            full_text_parts = []
            confidences = []
            
            # 处理文本行
            for line in page.lines:
                bbox = (
                    int(line.polygon[0].x),
                    int(line.polygon[0].y),
                    int(line.polygon[2].x),
                    int(line.polygon[2].y)
                )
                
                block = OCRTextBlock(
                    text=line.content,
                    confidence=0.95,  # Azure 不返回逐行置信度
                    bbox=bbox,
                    block_type="text"
                )
                text_blocks.append(block)
                full_text_parts.append(line.content)
                confidences.append(0.95)
            
            # 处理表格
            for table in result.tables:
                if table.bounding_regions and table.bounding_regions[0].page_number == page.page_number:
                    cells = []
                    max_row = max(cell.row_index for cell in table.cells)
                    max_col = max(cell.column_index for cell in table.cells)
                    
                    # 初始化二维数组
                    for _ in range(max_row + 1):
                        cells.append([""] * (max_col + 1))
                    
                    # 填充单元格
                    for cell in table.cells:
                        cells[cell.row_index][cell.column_index] = cell.content or ""
                    
                    ocr_table = OCRTable(
                        cells=cells,
                        confidence=0.95,
                        bbox=(0, 0, 0, 0)  # Azure 返回的是区域
                    )
                    ocr_table.to_markdown()
                    tables_on_page.append(ocr_table)
                    full_text_parts.append(f"\n[表格]\n{ocr_table.markdown}\n")
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            pages.append(OCRPageResult(
                page_num=page.page_number,
                text_blocks=text_blocks,
                tables=tables_on_page,
                figures=figures_on_page,
                full_text="\n".join(full_text_parts),
                is_scanned=is_scanned,
                confidence=avg_conf
            ))
        
        avg_confidence = sum(p.confidence for p in pages) / len(pages) if pages else 0.0
        
        return OCRDocumentResult(
            pages=pages,
            total_pages=len(pages),
            avg_confidence=avg_confidence,
            is_scanned=is_scanned,
            ocr_provider="azure"
        )
    
    def _process_with_tesseract(self, pdf_path: str, is_scanned: bool) -> OCRDocumentResult:
        """使用 Tesseract OCR 处理"""
        import fitz
        import pytesseract
        
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            
            # 转换为 PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR
            ocr_data = pytesseract.image_to_data(
                img,
                lang="chi_sim+eng" if "ch" in self.lang else "eng",
                output_type=pytesseract.Output.DICT
            )
            
            text_blocks = []
            full_text_parts = []
            confidences = []
            
            n_boxes = len(ocr_data["text"])
            for i in range(n_boxes):
                text = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i])
                
                if text and conf >= self.confidence_threshold * 100:
                    x = ocr_data["left"][i]
                    y = ocr_data["top"][i]
                    w = ocr_data["width"][i]
                    h = ocr_data["height"][i]
                    
                    block = OCRTextBlock(
                        text=text,
                        confidence=conf / 100,
                        bbox=(x, y, x + w, y + h)
                    )
                    text_blocks.append(block)
                    full_text_parts.append(text)
                    confidences.append(conf / 100)
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            pages.append(OCRPageResult(
                page_num=page_num + 1,
                text_blocks=text_blocks,
                tables=[],  # Tesseract 不支持表格识别
                figures=[],
                full_text=" ".join(full_text_parts),
                is_scanned=is_scanned,
                confidence=avg_conf
            ))
        
        doc.close()
        
        avg_confidence = sum(p.confidence for p in pages) / len(pages) if pages else 0.0
        
        return OCRDocumentResult(
            pages=pages,
            total_pages=len(pages),
            avg_confidence=avg_confidence,
            is_scanned=is_scanned,
            ocr_provider="tesseract"
        )
    
    def preprocess_image(self, image_path: str) -> str:
        """
        图像预处理，提升 OCR 准确率
        
        处理：
        - 灰度化
        - 二值化
        - 去噪
        - 倾斜矫正
        """
        if not HAS_CV2:
            return image_path
        
        try:
            img = cv2.imread(image_path)
            
            # 灰度化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 去噪
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # 自适应二值化
            binary = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # 倾斜矫正
            corrected = self._deskew(binary)
            
            # 保存预处理后的图像
            output_path = image_path.replace(".png", "_preprocessed.png")
            cv2.imwrite(output_path, corrected)
            
            return output_path
        
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image_path
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """倾斜矫正"""
        try:
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5:  # 只有倾斜角度大于 0.5 度才矫正
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
            
            return image
        except Exception:
            return image
    
    def enhance_table_recognition(
        self,
        image_path: str,
        existing_tables: List[OCRTable]
    ) -> List[OCRTable]:
        """
        增强表格识别
        
        使用多种策略提升表格识别准确率
        """
        if not HAS_CV2 or not existing_tables:
            return existing_tables
        
        enhanced_tables = []
        
        for table in existing_tables:
            # 如果置信度较低，尝试重新识别
            if table.confidence < 0.8:
                try:
                    # 裁剪表格区域
                    img = cv2.imread(image_path)
                    x1, y1, x2, y2 = table.bbox
                    table_img = img[y1:y2, x1:x2]
                    
                    # 保存裁剪的表格图像
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        cv2.imwrite(tmp.name, table_img)
                        
                        # 单独对表格区域进行 OCR
                        if self._layout_engine:
                            result = self._layout_engine(tmp.name)
                            for item in result:
                                if item.get("type") == "table":
                                    table_html = item.get("res", {}).get("html", "")
                                    table.cells = self._parse_table_html(table_html)
                                    table.html = table_html
                                    table.confidence = 0.9
                                    table.to_markdown()
                                    break
                        
                        os.unlink(tmp.name)
                
                except Exception as e:
                    logger.warning(f"Table enhancement failed: {e}")
            
            enhanced_tables.append(table)
        
        return enhanced_tables


def create_ocr_processor(
    provider: str = "paddleocr",
    **kwargs
) -> OCRProcessor:
    """创建 OCR 处理器"""
    return OCRProcessor(provider=provider, **kwargs)
