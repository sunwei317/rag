"""
OCR 处理模块
处理扫描版 PDF，支持文字、表格、图片识别
"""
import os
import tempfile
import json
import re
from datetime import datetime
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
    DOTS_OCR = "dots_ocr"         # Dots.ocr via vLLM OpenAI-compatible API
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
    temp_output_path: Optional[str] = None


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
        azure_key: Optional[str] = None,
        dots_ocr_api_base: Optional[str] = None,
        dots_ocr_model_name: Optional[str] = None,
        dots_ocr_figure_cv_fallback: Optional[bool] = None
    ):
        self.provider = OCRProvider(provider)
        self.lang = lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.enable_table_recognition = enable_table_recognition
        self.enable_layout_analysis = enable_layout_analysis
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_key = azure_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self.dots_ocr_api_base = dots_ocr_api_base or os.getenv("DOTS_OCR_API_BASE", "http://dots-ocr-vllm:8000/v1")
        self.dots_ocr_model_name = dots_ocr_model_name or os.getenv(
            "DOTS_OCR_MODEL_NAME", "rednote-hilab/dots.ocr"
        )
        if dots_ocr_figure_cv_fallback is None:
            self.dots_ocr_figure_cv_fallback = os.getenv(
                "DOTS_OCR_FIGURE_CV_FALLBACK", "true"
            ).strip().lower() in ("1", "true", "yes", "on")
        else:
            self.dots_ocr_figure_cv_fallback = bool(dots_ocr_figure_cv_fallback)
        
        self._ocr_engine = None
        self._table_engine = None
        self._layout_engine = None
        
        self._init_engine()
    
    def _init_engine(self):
        """初始化 OCR 引擎"""
        if self.provider == OCRProvider.PADDLEOCR:
            self._init_paddleocr()
        elif self.provider == OCRProvider.DOTS_OCR:
            self._init_dots_ocr()
        elif self.provider == OCRProvider.AZURE:
            self._init_azure()
        elif self.provider == OCRProvider.TESSERACT:
            self._init_tesseract()
        else:
            logger.warning(f"Provider {self.provider} not fully implemented, falling back to PaddleOCR")
            self._init_paddleocr()

    def _init_dots_ocr(self):
        """初始化 Dots.ocr (vLLM OpenAI 兼容接口)"""
        try:
            from openai import OpenAI
            self._ocr_engine = OpenAI(
                api_key=os.getenv("DOTS_OCR_API_KEY", "EMPTY"),
                base_url=self.dots_ocr_api_base
            )
            logger.info(
                f"Dots.ocr initialized: base={self.dots_ocr_api_base}, model={self.dots_ocr_model_name}, "
                f"figure_cv_fallback={self.dots_ocr_figure_cv_fallback}"
            )
        except ImportError as e:
            logger.error(f"openai package not installed for dots_ocr: {e}")
            raise
    
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
            result = self._process_with_paddleocr(pdf_path, is_scanned)
        elif self.provider == OCRProvider.DOTS_OCR:
            result = self._process_with_dots_ocr(pdf_path, is_scanned)
        elif self.provider == OCRProvider.AZURE:
            result = self._process_with_azure(pdf_path, is_scanned)
        elif self.provider == OCRProvider.TESSERACT:
            result = self._process_with_tesseract(pdf_path, is_scanned)
        else:
            result = self._process_with_paddleocr(pdf_path, is_scanned)

        temp_path = self._save_ocr_result_temp(pdf_path, result)
        if temp_path:
            result.temp_output_path = temp_path
        return result

    def _save_ocr_result_temp(self, pdf_path: str, result: OCRDocumentResult) -> Optional[str]:
        """将 OCR 输出保存到宿主机可见目录中的 JSON 文件，便于排查问题"""
        try:
            output = {
                "pdf_path": pdf_path,
                "ocr_provider": result.ocr_provider,
                "total_pages": result.total_pages,
                "avg_confidence": result.avg_confidence,
                "is_scanned": result.is_scanned,
                "pages": []
            }

            for page in result.pages:
                output["pages"].append({
                    "page_num": page.page_num,
                    "is_scanned": page.is_scanned,
                    "confidence": page.confidence,
                    "full_text": page.full_text,
                    "text_blocks": [
                        {
                            "text": tb.text,
                            "confidence": tb.confidence,
                            "bbox": list(tb.bbox),
                            "block_type": tb.block_type
                        }
                        for tb in page.text_blocks
                    ],
                    "tables": [
                        {
                            "cells": t.cells,
                            "confidence": t.confidence,
                            "bbox": list(t.bbox),
                            "markdown": t.markdown,
                            "html": t.html
                        }
                        for t in page.tables
                    ],
                    "figures": [
                        {
                            "image_path": f.image_path,
                            "caption": f.caption,
                            "bbox": list(f.bbox),
                            "description": f.description
                        }
                        for f in page.figures
                    ]
                })

            # 默认写入 /app/data，docker-compose 已映射到宿主机 ./data
            output_dir = Path(os.getenv("OCR_OUTPUT_DIR", "/app/data/ocr_outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)

            base_name = Path(pdf_path).stem
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name)[:80]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_path = output_dir / f"ocr_output_{safe_name}_{ts}.json"

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            temp_path = str(out_path)

            logger.info(f"OCR output saved to temp file: {temp_path}")
            return temp_path
        except Exception as e:
            logger.warning(f"Failed to save OCR temp output: {e}")
            return None

    def _process_with_dots_ocr(self, pdf_path: str, is_scanned: bool) -> OCRDocumentResult:
        """使用 Dots.ocr (vLLM) 处理"""
        import fitz

        doc = fitz.open(pdf_path)
        pages = []
        total_confidence = 0.0

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name

            try:
                try:
                    page_result = self._ocr_page_dots_ocr(tmp_path, page_num + 1)
                except Exception as page_err:
                    logger.error(f"Dots OCR page {page_num + 1} failed: {page_err}")
                    page_result = OCRPageResult(
                        page_num=page_num + 1,
                        text_blocks=[],
                        tables=[],
                        figures=[],
                        full_text="",
                        confidence=0.0
                    )
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
            ocr_provider="dots_ocr"
        )

    def _ocr_page_dots_ocr(self, image_path: str, page_num: int) -> OCRPageResult:
        """使用 Dots.ocr 处理单页（返回结构化版面结果）"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        import base64
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        if HAS_PIL:
            with Image.open(image_path) as img:
                page_w, page_h = img.size
        else:
            page_w, page_h = 1000, 1000

        prompt = f"""
Perform OCR for this page and return STRICT JSON only.
Page size: width={page_w}, height={page_h}.

JSON schema:
{{
  "full_text": "string",
  "text_blocks": [
    {{
      "text": "string",
      "confidence": 0.0,
      "bbox": [x1, y1, x2, y2],
      "block_type": "text"
    }}
  ],
  "tables": [
    {{
      "cells": [["h1","h2"],["v1","v2"]],
      "markdown": "|...|",
      "confidence": 0.0,
      "bbox": [x1, y1, x2, y2]
    }}
  ],
  "figures": [
    {{
      "caption": "string",
      "description": "string",
      "bbox": [x1, y1, x2, y2]
    }}
  ]
}}

Rules:
- bbox must use absolute pixel coordinates.
- Keep reading order in text_blocks.
- If no table/figure, use empty list.
- Return JSON only, no markdown/code fence.
"""

        response = self._ocr_engine.chat.completions.create(
            model=self.dots_ocr_model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=8192
        )

        content = ""
        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""

        payload: Dict[str, Any] = {}
        try:
            loaded = json.loads(content)
            payload = self._coerce_payload(loaded)
        except Exception:
            payload = self._extract_json_from_text(content) or {}

        # vLLM xgrammar 在部分请求会失败导致空响应；若结构化失败，退回纯文本 OCR
        if not payload and not content.strip():
            logger.warning("Dots OCR returned empty structured response, retrying with plain-text prompt")
            try:
                fallback_prompt = "Perform OCR and return plain text only, in reading order."
                fallback_resp = self._ocr_engine.chat.completions.create(
                    model=self.dots_ocr_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fallback_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                                }
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=4096
                )
                if fallback_resp and fallback_resp.choices and fallback_resp.choices[0].message:
                    fallback_text = (fallback_resp.choices[0].message.content or "").strip()
                    if fallback_text:
                        payload = {
                            "full_text": fallback_text,
                            "text_blocks": [
                                {
                                    "text": fallback_text,
                                    "confidence": 0.8,
                                    "bbox": [0, 0, int(page_w), int(page_h)],
                                    "block_type": "text"
                                }
                            ],
                            "tables": [],
                            "figures": []
                        }
            except Exception as fallback_err:
                logger.warning(f"Dots OCR fallback request failed: {fallback_err}")

        default_bbox = (0, 0, int(page_w), int(page_h))

        text_blocks: List[OCRTextBlock] = []
        for item in payload.get("text_blocks", []):
            text = (item.get("text", "") or "").strip()
            if not text:
                continue
            bbox = self._normalize_bbox(item.get("bbox"), page_w, page_h, default_bbox)
            conf = float(item.get("confidence", 0.85) or 0.85)
            block_type = item.get("block_type", "text") or "text"
            text_blocks.append(
                OCRTextBlock(
                    text=text,
                    confidence=conf,
                    bbox=bbox,
                    block_type=block_type
                )
            )

        tables: List[OCRTable] = []
        for item in payload.get("tables", []):
            cells = item.get("cells", []) or []
            markdown = (item.get("markdown", "") or "").strip()
            bbox = self._normalize_bbox(item.get("bbox"), page_w, page_h, default_bbox)
            conf = float(item.get("confidence", 0.8) or 0.8)
            table = OCRTable(cells=cells, confidence=conf, bbox=bbox, markdown=markdown)
            if not table.markdown and table.cells:
                table.to_markdown()
            tables.append(table)

        detected_regions: List[Tuple[int, int, int, int]] = []
        if self.dots_ocr_figure_cv_fallback:
            detected_regions = self._detect_figure_regions_cv(
                image_path=image_path,
                page_w=page_w,
                page_h=page_h,
                text_blocks=text_blocks,
                tables=tables,
            )

        figure_items = payload.get("figures", []) or payload.get("images", []) or payload.get("pictures", [])
        if not isinstance(figure_items, list):
            figure_items = []

        figures: List[OCRFigure] = []
        region_idx = 0
        for item in figure_items:
            if not isinstance(item, dict):
                continue
            bbox = self._normalize_bbox(item.get("bbox"), page_w, page_h, default_bbox)
            if bbox == default_bbox and region_idx < len(detected_regions):
                bbox = detected_regions[region_idx]
                region_idx += 1
            figures.append(
                OCRFigure(
                    image_path=item.get("image_path", ""),
                    caption=(item.get("caption", "") or "").strip(),
                    description=(item.get("description", "") or "").strip(),
                    bbox=bbox
                )
            )

        # 仅在开启兜底时启用图题推断 / CV 区域补偿
        if not figures and self.dots_ocr_figure_cv_fallback:
            figures = self._infer_figures_from_text_blocks(text_blocks)
            if figures and detected_regions:
                used: set[int] = set()
                for fig in figures:
                    match_idx = self._match_region_for_caption(fig.bbox, detected_regions, used)
                    if match_idx is not None:
                        fig.bbox = detected_regions[match_idx]
                        used.add(match_idx)
            elif detected_regions:
                figures = [
                    OCRFigure(
                        image_path="",
                        caption="Detected figure region",
                        description="Detected by CV layout analysis.",
                        bbox=b,
                    )
                    for b in detected_regions
                ]

        full_text = (payload.get("full_text", "") or "").strip()
        if not full_text:
            full_text = "\n".join([b.text for b in text_blocks]).strip()

        # 兜底：至少返回一个文本块，且 bbox 非 0
        if not text_blocks and full_text:
            text_blocks.append(
                OCRTextBlock(
                    text=full_text,
                    confidence=0.85,
                    bbox=default_bbox,
                    block_type="text"
                )
            )

        conf_values = [b.confidence for b in text_blocks] + [t.confidence for t in tables]
        avg_conf = sum(conf_values) / len(conf_values) if conf_values else (0.85 if full_text else 0.0)

        return OCRPageResult(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            full_text=full_text,
            confidence=avg_conf
        )

    def _coerce_payload(self, loaded: Any) -> Dict[str, Any]:
        """将模型返回统一为 dict 格式，兼容 list/string 结构"""
        if isinstance(loaded, dict):
            figures = loaded.get("figures")
            if not figures:
                for alias in ("images", "pictures", "illustrations", "charts", "diagrams"):
                    alias_items = loaded.get(alias)
                    if isinstance(alias_items, list) and alias_items:
                        loaded["figures"] = alias_items
                        break
            return loaded

        if isinstance(loaded, list):
            text_parts: List[str] = []
            text_blocks: List[Dict[str, Any]] = []
            tables: List[Dict[str, Any]] = []
            figures: List[Dict[str, Any]] = []
            for item in loaded:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or ""
                    if txt:
                        text_parts.append(str(txt))
                        text_blocks.append({
                            "text": str(txt),
                            "confidence": item.get("confidence", 0.8),
                            "bbox": item.get("bbox", [0, 0, 0, 0]),
                            "block_type": item.get("block_type", "text"),
                        })
                    if item.get("cells") or item.get("markdown"):
                        tables.append(item)
                    if item.get("caption") or item.get("description") or item.get("image_path"):
                        figures.append(item)
                elif isinstance(item, str):
                    text_parts.append(item)
                    text_blocks.append({
                        "text": item,
                        "confidence": 0.8,
                        "bbox": [0, 0, 0, 0],
                        "block_type": "text",
                    })
            return {
                "full_text": "\n".join(text_parts).strip(),
                "text_blocks": text_blocks,
                "tables": tables,
                "figures": figures,
            }

        if isinstance(loaded, str):
            return {
                "full_text": loaded,
                "text_blocks": [{"text": loaded, "confidence": 0.8, "bbox": [0, 0, 0, 0], "block_type": "text"}],
                "tables": [],
                "figures": [],
            }

        return {}

    def _looks_like_figure_caption(self, text: str) -> bool:
        if not text:
            return False
        t = text.strip()
        if re.match(r"^(图|图表)\s*\d+([\-—\.]\d+)*", t):
            return True
        if re.match(r"^figure\s*\d+([\-—\.]\d+)*", t, flags=re.IGNORECASE):
            return True
        return False

    def _infer_figures_from_text_blocks(self, text_blocks: List[OCRTextBlock]) -> List[OCRFigure]:
        figures: List[OCRFigure] = []
        for block in text_blocks:
            if self._looks_like_figure_caption(block.text):
                figures.append(
                    OCRFigure(
                        image_path="",
                        caption=block.text.strip(),
                        description="Inferred from figure caption text block.",
                        bbox=block.bbox
                    )
                )
        return figures

    def _match_region_for_caption(
        self,
        caption_bbox: Tuple[int, int, int, int],
        regions: List[Tuple[int, int, int, int]],
        used: set
    ) -> Optional[int]:
        """给图题框匹配最近的图片区域（优先图题上方且有水平重叠）"""
        cx1, cy1, cx2, cy2 = caption_bbox
        best_idx = None
        best_score = float("inf")
        for i, (rx1, ry1, rx2, ry2) in enumerate(regions):
            if i in used:
                continue
            horizontal_overlap = max(0, min(cx2, rx2) - max(cx1, rx1))
            min_width = max(1, min(cx2 - cx1, rx2 - rx1))
            overlap_ratio = horizontal_overlap / min_width
            if overlap_ratio < 0.15:
                continue

            # 优先图题上方区域；否则允许下方近邻
            if ry2 <= cy1:
                distance = cy1 - ry2
            else:
                distance = (ry1 - cy2) + 2000

            if distance < best_score:
                best_score = distance
                best_idx = i
        return best_idx

    def _detect_figure_regions_cv(
        self,
        image_path: str,
        page_w: int,
        page_h: int,
        text_blocks: List[OCRTextBlock],
        tables: List[OCRTable]
    ) -> List[Tuple[int, int, int, int]]:
        """基于版面分析检测图片区域，排除文本和表格区域"""
        if not HAS_CV2:
            return []

        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 近似去除文字：先构建文本/表格掩码，避免把文字连通域当成图片
        exclusion = np.zeros_like(gray)
        for block in text_blocks:
            x1, y1, x2, y2 = block.bbox
            pad = 6
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(page_w, x2 + pad)
            y2 = min(page_h, y2 + pad)
            cv2.rectangle(exclusion, (x1, y1), (x2, y2), 255, thickness=-1)

        for table in tables:
            x1, y1, x2, y2 = table.bbox
            cv2.rectangle(exclusion, (x1, y1), (x2, y2), 255, thickness=-1)

        # 边缘 + 闭运算获取候选图块
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        merged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 用 exclusion 去除文本密集区
        mask = cv2.bitwise_and(merged, cv2.bitwise_not(exclusion))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = int(page_w * page_h * 0.01)   # 至少占页面 1%
        max_area = int(page_w * page_h * 0.75)   # 最大不超过 75%

        regions: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area or area > max_area:
                continue
            if w < 80 or h < 80:
                continue

            # 图片区域通常不是超细长条，放宽但过滤极端噪声
            ratio = w / max(h, 1)
            if ratio < 0.15 or ratio > 6.5:
                continue

            # 候选区域内若仍为高文字密度，过滤掉
            roi_excl = exclusion[y:y + h, x:x + w]
            text_cover = float(np.count_nonzero(roi_excl)) / float(max(1, roi_excl.size))
            if text_cover > 0.45:
                continue

            regions.append((x, y, x + w, y + h))

        # 去重：按面积降序保留，去掉高度重叠候选
        regions = sorted(regions, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        deduped: List[Tuple[int, int, int, int]] = []
        for box in regions:
            if any(self._bbox_iou(box, keep) > 0.6 for keep in deduped):
                continue
            deduped.append(box)

        # 阅读顺序输出
        deduped.sort(key=lambda b: (b[1], b[0]))
        return deduped[:8]

    def _bbox_iou(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """从混合文本中提取 JSON 对象"""
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            loaded = json.loads(text[start:end + 1])
            return self._coerce_payload(loaded)
        except Exception:
            return None

    def _normalize_bbox(
        self,
        bbox_raw: Any,
        page_w: int,
        page_h: int,
        default_bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """规范化 bbox，确保落在页面范围内且非零"""
        try:
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                return default_bbox
            x1, y1, x2, y2 = [int(float(v)) for v in bbox_raw]
            x1 = max(0, min(x1, page_w))
            x2 = max(0, min(x2, page_w))
            y1 = max(0, min(y1, page_h))
            y2 = max(0, min(y2, page_h))
            if x2 <= x1 or y2 <= y1:
                return default_bbox
            return (x1, y1, x2, y2)
        except Exception:
            return default_bbox
    
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
