"""
Ingestion 模块
负责 PDF 解析、切分、向量化、OCR 处理
"""
from .pdf_parser import PDFParser, ParsedDocument, parse_pdf
from .chunker import SmartChunker, Chunk, ChunkingResult, chunk_document
from .embedder import Embedder, create_embedder
from .pipeline import IngestionPipeline, ingest_pdf
from .ocr_processor import (
    OCRProcessor,
    OCRTextBlock,
    OCRTable,
    OCRFigure,
    OCRPageResult,
    OCRDocumentResult
)
from .image_processor import (
    ImageProcessor,
    ImageRetriever,
    ExtractedImage,
    ImageType,
    process_pdf_images,
    create_image_processor
)
from .table_processor import (
    TableProcessor,
    TableChunker,
    StructuredTable,
    TableCell,
    TableColumn,
    TableRow,
    TableType,
    ColumnType,
    extract_tables,
    table_to_chunks
)

__all__ = [
    # PDF 解析
    "PDFParser",
    "ParsedDocument", 
    "parse_pdf",
    # 切分
    "SmartChunker",
    "Chunk",
    "ChunkingResult",
    "chunk_document",
    # 向量化
    "Embedder",
    "create_embedder",
    # 流水线
    "IngestionPipeline",
    "ingest_pdf",
    # OCR 处理
    "OCRProcessor",
    "OCRTextBlock",
    "OCRTable",
    "OCRFigure",
    "OCRPageResult",
    "OCRDocumentResult",
    # 图片处理
    "ImageProcessor",
    "ImageRetriever",
    "ExtractedImage",
    "ImageType",
    "process_pdf_images",
    "create_image_processor",
    # 表格处理
    "TableProcessor",
    "TableChunker",
    "StructuredTable",
    "TableCell",
    "TableColumn",
    "TableRow",
    "TableType",
    "ColumnType",
    "extract_tables",
    "table_to_chunks"
]
