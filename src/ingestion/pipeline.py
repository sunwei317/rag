"""
数据摄取流水线
整合 PDF 解析、切分、向量化、索引构建
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
from dataclasses import dataclass
import os

from .pdf_parser import PDFParser, ParsedDocument
from .chunker import SmartChunker, ChunkingResult, Chunk
from .embedder import Embedder

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class IngestionStats:
    """摄取统计信息"""
    doc_id: str
    filename: str
    total_pages: int
    total_sections: int
    parent_chunks: int
    child_chunks: int
    total_tokens: int


class IngestionPipeline:
    """
    数据摄取流水线
    
    流程:
    1. PDF 解析 → 结构化文档
    2. 智能切分 → 父子 Chunks
    3. 向量化 → Embeddings
    4. 索引构建 → Vector DB + BM25
    """
    
    def __init__(
        self,
        vector_store=None,
        bm25_store=None,
        embedder: Optional[Embedder] = None,
        save_intermediate: bool = True
    ):
        # 解析器
        ocr_provider = os.getenv("OCR_PROVIDER", "dots_ocr")
        force_ocr_all = os.getenv("OCR_FORCE_ALL_PDF", "true").lower() in ("1", "true", "yes", "on")
        self.parser = PDFParser(
            extract_images=True,
            extract_tables=True,
            enable_ocr=True,
            ocr_provider=ocr_provider
        )
        self.force_ocr_all = force_ocr_all
        logger.info(f"Ingestion OCR config: provider={ocr_provider}, force_all_pdf={self.force_ocr_all}")
        
        # 切分器
        self.chunker = SmartChunker(
            child_chunk_size=settings.chunking.child_chunk_size,
            child_chunk_overlap=settings.chunking.child_chunk_overlap,
            parent_chunk_size=settings.chunking.parent_chunk_size,
            parent_chunk_overlap=settings.chunking.parent_chunk_overlap
        )
        
        # 向量化器
        self.embedder = embedder or Embedder(
            provider=settings.embedding.provider,
            model_name=settings.embedding.model_name,
            batch_size=settings.embedding.batch_size
        )
        
        # 存储
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        
        self.save_intermediate = save_intermediate
        
        # 确保目录存在
        settings.ensure_dirs()
    
    def ingest_pdf(
        self,
        pdf_path: str | Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestionStats:
        """
        摄取单个 PDF 文档
        
        Args:
            pdf_path: PDF 文件路径
            metadata: 额外元数据
                - product: 产品名称
                - version: 版本号
                - doc_type: 文档类型 (user_manual, api_reference, installation_guide, etc.)
                - security_level: 安全等级
                - department: 部门
        
        Returns:
            IngestionStats: 摄取统计信息
        """
        pdf_path = Path(pdf_path)
        logger.info(f"开始摄取文档: {pdf_path.name}")
        
        # 1. 解析 PDF
        logger.info("Step 1: 解析 PDF...")
        parsed_doc = self.parser.parse(pdf_path, metadata, force_ocr=self.force_ocr_all)
        
        if self.save_intermediate:
            self._save_parsed_doc(parsed_doc)
        
        # 2. 智能切分
        logger.info("Step 2: 智能切分...")
        chunking_result = self.chunker.chunk_document(parsed_doc)
        
        if self.save_intermediate:
            self._save_chunks(chunking_result)
        
        # 3. 向量化 (只对子 Chunks 进行向量化用于检索)
        logger.info("Step 3: 向量化...")
        embeddings = self.embedder.embed_chunks(chunking_result.child_chunks)
        
        # 4. 构建索引
        logger.info("Step 4: 构建索引...")
        self._build_indexes(chunking_result, embeddings)
        
        # 统计信息
        stats = IngestionStats(
            doc_id=parsed_doc.doc_id,
            filename=pdf_path.name,
            total_pages=parsed_doc.total_pages,
            total_sections=len(parsed_doc.sections),
            parent_chunks=len(chunking_result.parent_chunks),
            child_chunks=len(chunking_result.child_chunks),
            total_tokens=sum(c.token_count for c in chunking_result.all_chunks)
        )
        
        logger.info(f"摄取完成: {stats}")
        return stats
    
    def ingest_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*.pdf",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[IngestionStats]:
        """批量摄取目录下的 PDF"""
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob(pattern))
        
        logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")
        
        results = []
        for pdf_path in pdf_files:
            try:
                stats = self.ingest_pdf(pdf_path, metadata)
                results.append(stats)
            except Exception as e:
                logger.error(f"摄取失败 {pdf_path.name}: {e}")
        
        return results
    
    def _save_parsed_doc(self, doc: ParsedDocument):
        """保存解析结果"""
        output_path = settings.processed_dir / f"{doc.doc_id}_parsed.json"
        doc.save(output_path)
        logger.debug(f"保存解析结果: {output_path}")
    
    def _save_chunks(self, result: ChunkingResult):
        """保存切分结果"""
        import json
        
        output_path = settings.processed_dir / f"{result.doc_id}_chunks.json"
        
        data = {
            "doc_id": result.doc_id,
            "parent_chunks": [c.to_dict() for c in result.parent_chunks],
            "child_chunks": [c.to_dict() for c in result.child_chunks]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"保存切分结果: {output_path}")
    
    def _build_indexes(self, chunking_result: ChunkingResult, embeddings):
        """构建向量和 BM25 索引"""
        # 向量索引
        if self.vector_store:
            self.vector_store.add_chunks(
                chunking_result.child_chunks,
                embeddings
            )
        
        # BM25 索引 (使用所有 chunks)
        if self.bm25_store:
            self.bm25_store.add_chunks(chunking_result.all_chunks)
        
        logger.info("索引构建完成")


# 便捷函数
def ingest_pdf(pdf_path: str | Path, metadata: Optional[Dict] = None) -> IngestionStats:
    """摄取 PDF 的便捷函数"""
    pipeline = IngestionPipeline()
    return pipeline.ingest_pdf(pdf_path, metadata)
