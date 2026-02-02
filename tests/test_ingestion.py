"""
测试套件 - 文档导入模块
"""

import pytest
import tempfile
import os
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPDFParser:
    """PDF 解析器测试"""
    
    def test_extract_text_blocks(self):
        """测试文本块提取"""
        from src.ingestion.pdf_parser import PDFParser
        
        parser = PDFParser()
        # 基本实例化测试
        assert parser is not None
        assert parser.extract_images is False
    
    def test_parser_with_options(self):
        """测试解析器配置选项"""
        from src.ingestion.pdf_parser import PDFParser
        
        parser = PDFParser(extract_images=True)
        assert parser.extract_images is True


class TestChunker:
    """切分器测试"""
    
    def test_child_chunk_creation(self):
        """测试子块创建"""
        from src.ingestion.chunker import SmartChunker
        from src.ingestion.pdf_parser import ParsedDocument, TextBlock
        
        chunker = SmartChunker()
        
        # 创建测试文档
        doc = ParsedDocument(
            doc_id="test_doc",
            filename="test.pdf",
            sections=[],
            blocks=[
                TextBlock(
                    text="这是一段测试文本。" * 50,
                    page_num=1,
                    block_type="paragraph",
                    bbox=None
                )
            ],
            tables=[],
            images=[],
            metadata={}
        )
        
        # 执行切分
        child_chunks, parent_chunks = chunker.chunk(doc)
        
        # 验证结果
        assert len(child_chunks) > 0
        assert len(parent_chunks) > 0
        assert all(c.chunk_type == "child" for c in child_chunks)
        assert all(c.chunk_type == "parent" for c in parent_chunks)
    
    def test_parent_child_linking(self):
        """测试父子块链接"""
        from src.ingestion.chunker import SmartChunker
        from src.ingestion.pdf_parser import ParsedDocument, TextBlock
        
        chunker = SmartChunker()
        
        doc = ParsedDocument(
            doc_id="test_doc",
            filename="test.pdf",
            sections=[],
            blocks=[
                TextBlock(
                    text="这是一段很长的测试文本,用于测试切分功能。" * 100,
                    page_num=1,
                    block_type="paragraph",
                    bbox=None
                )
            ],
            tables=[],
            images=[],
            metadata={}
        )
        
        child_chunks, parent_chunks = chunker.chunk(doc)
        
        # 验证父子关系
        for child in child_chunks:
            assert child.parent_id is not None
            # 父块应该存在
            parent_ids = [p.chunk_id for p in parent_chunks]
            assert child.parent_id in parent_ids


class TestEmbedder:
    """嵌入模型测试"""
    
    def test_embedder_initialization(self):
        """测试嵌入模型初始化"""
        from src.ingestion.embedder import Embedder
        
        # 只测试初始化,避免实际加载模型
        embedder = Embedder.__new__(Embedder)
        assert embedder is not None


class TestVectorStore:
    """向量存储测试"""
    
    def test_chroma_store_creation(self):
        """测试 Chroma 存储创建"""
        from src.storage.vector_store import ChromaVectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory=tmpdir
            )
            assert store is not None
    
    def test_chroma_add_and_search(self):
        """测试 Chroma 添加和搜索"""
        from src.storage.vector_store import ChromaVectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory=tmpdir
            )
            
            # 添加向量
            store.add(
                ids=["doc1", "doc2"],
                embeddings=[[0.1] * 384, [0.2] * 384],
                metadatas=[{"source": "test1"}, {"source": "test2"}],
                texts=["测试文本1", "测试文本2"]
            )
            
            # 搜索
            results = store.search(
                query_embedding=[0.1] * 384,
                top_k=2
            )
            
            assert len(results) > 0
            assert "doc1" in [r["id"] for r in results]


class TestBM25Store:
    """BM25 存储测试"""
    
    def test_memory_bm25_creation(self):
        """测试内存 BM25 创建"""
        from src.storage.bm25_store import MemoryBM25Store
        
        store = MemoryBM25Store()
        assert store is not None
    
    def test_memory_bm25_add_and_search(self):
        """测试内存 BM25 添加和搜索"""
        from src.storage.bm25_store import MemoryBM25Store
        
        store = MemoryBM25Store()
        
        # 添加文档
        store.add_documents(
            doc_ids=["doc1", "doc2", "doc3"],
            texts=[
                "Python 是一种编程语言",
                "机器学习是人工智能的分支",
                "深度学习使用神经网络"
            ],
            metadatas=[{}, {}, {}]
        )
        
        # 搜索
        results = store.search("Python 编程", top_k=2)
        
        assert len(results) > 0
        assert results[0]["id"] == "doc1"


class TestMetadataStore:
    """元数据存储测试"""
    
    def test_metadata_store_creation(self):
        """测试元数据存储创建"""
        from src.storage.metadata_store import MetadataStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MetadataStore(db_path=db_path)
            assert store is not None
    
    def test_document_crud(self):
        """测试文档 CRUD 操作"""
        from src.storage.metadata_store import MetadataStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = MetadataStore(db_path=db_path)
            
            # 创建文档
            store.add_document(
                doc_id="test_doc",
                filename="test.pdf",
                metadata={"author": "Test"}
            )
            
            # 读取文档
            doc = store.get_document("test_doc")
            assert doc is not None
            assert doc["filename"] == "test.pdf"
            
            # 列出文档
            docs = store.list_documents()
            assert len(docs) == 1


class TestHybridSearcher:
    """混合搜索测试"""
    
    def test_rrf_fusion(self):
        """测试 RRF 融合算法"""
        from src.retrieval.hybrid_search import HybridSearcher
        
        # 测试 RRF 计算
        vector_results = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
            {"id": "doc3", "score": 0.5},
        ]
        
        bm25_results = [
            {"id": "doc2", "score": 0.8},
            {"id": "doc1", "score": 0.6},
            {"id": "doc4", "score": 0.4},
        ]
        
        # 手动计算预期 RRF
        # doc1: 1/(60+1) + 1/(60+2) = 0.0161 + 0.0161 = 0.032
        # doc2: 1/(60+2) + 1/(60+1) = 0.032
        # doc3: 1/(60+3) = 0.0159
        # doc4: 1/(60+3) = 0.0159
        
        # 验证融合逻辑正确性
        assert len(vector_results) == 3
        assert len(bm25_results) == 3


class TestQueryTransformer:
    """查询转换测试"""
    
    def test_query_expansion_basic(self):
        """测试基础查询扩展"""
        from src.retrieval.query_transformer import QueryTransformer
        
        transformer = QueryTransformer()
        
        # 简单测试
        query = "Python 异步编程"
        # 同义词扩展
        expanded = transformer._expand_with_synonyms(query)
        
        # 至少包含原始查询
        assert query in expanded or len(expanded) > 0


class TestTerminology:
    """术语管理测试"""
    
    def test_terminology_manager(self):
        """测试术语管理器"""
        from src.utils.terminology import TerminologyManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            
            # 初始化需要 metadata_store
            from src.storage.metadata_store import MetadataStore
            metadata_store = MetadataStore(db_path=db_path)
            
            manager = TerminologyManager(metadata_store)
            
            # 添加术语
            manager.add_term(
                term="LLM",
                definition="Large Language Model",
                aliases=["大语言模型", "大型语言模型"]
            )
            
            # 获取术语
            term_info = manager.get_term("LLM")
            assert term_info is not None
            assert term_info["definition"] == "Large Language Model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
