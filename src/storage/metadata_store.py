"""
元数据存储模块
管理文档元数据和 Chunk 索引
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import sqlite3
from datetime import datetime
from loguru import logger


@dataclass
class DocumentMetadata:
    """文档元数据"""
    doc_id: str
    filename: str
    title: str
    product: str = ""
    version: str = ""
    doc_type: str = ""
    security_level: str = "internal"
    department: str = ""
    author: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataStore:
    """
    元数据存储
    
    使用 SQLite 存储文档元数据和 Chunk 索引
    支持:
    - 文档管理 (CRUD)
    - Chunk 关联查询
    - 版本追踪
    - 权限控制
    """
    
    def __init__(self, db_path: str = "./data/metadata.db"):
        self.db_path = db_path
        
        # 确保目录存在
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        logger.info(f"MetadataStore initialized: {db_path}")
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 文档表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                title TEXT,
                product TEXT,
                version TEXT,
                doc_type TEXT,
                security_level TEXT DEFAULT 'internal',
                department TEXT,
                author TEXT,
                total_pages INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                status TEXT DEFAULT 'active',
                metadata_json TEXT
            )
        """)
        
        # Chunk 索引表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_type TEXT,
                section_path TEXT,
                section_title TEXT,
                page_start INTEGER,
                page_end INTEGER,
                parent_id TEXT,
                token_count INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        """)
        
        # 术语表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terminology (
                term_id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL UNIQUE,
                aliases TEXT,
                definition TEXT,
                category TEXT,
                product TEXT,
                created_at TEXT
            )
        """)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks (parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_product ON documents (product)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON documents (status)")
        
        conn.commit()
        conn.close()
    
    def add_document(self, metadata: DocumentMetadata) -> bool:
        """添加文档元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_id, filename, title, product, version, doc_type, security_level,
                 department, author, total_pages, total_chunks, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.doc_id,
                metadata.filename,
                metadata.title,
                metadata.product,
                metadata.version,
                metadata.doc_type,
                metadata.security_level,
                metadata.department,
                metadata.author,
                metadata.total_pages,
                metadata.total_chunks,
                metadata.created_at or now,
                now,
                metadata.status
            ))
            
            conn.commit()
            logger.info(f"Added document metadata: {metadata.doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
        finally:
            conn.close()
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return DocumentMetadata(**dict(row))
        return None
    
    def list_documents(
        self,
        product: Optional[str] = None,
        doc_type: Optional[str] = None,
        status: str = "active",
        limit: int = 100
    ) -> List[DocumentMetadata]:
        """列出文档"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM documents WHERE status = ?"
        params = [status]
        
        if product:
            query += " AND product = ?"
            params.append(product)
        
        if doc_type:
            query += " AND doc_type = ?"
            params.append(doc_type)
        
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return [DocumentMetadata(**dict(row)) for row in rows]
    
    def delete_document(self, doc_id: str, hard_delete: bool = False) -> bool:
        """删除文档"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if hard_delete:
                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            else:
                # 软删除
                cursor.execute(
                    "UPDATE documents SET status = 'deleted', updated_at = ? WHERE doc_id = ?",
                    (datetime.now().isoformat(), doc_id)
                )
            
            conn.commit()
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
        finally:
            conn.close()
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """批量添加 Chunk 索引"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, doc_id, chunk_type, section_path, section_title,
                     page_start, page_end, parent_id, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk["chunk_id"],
                    chunk["doc_id"],
                    chunk.get("chunk_type", ""),
                    chunk.get("section_path", ""),
                    chunk.get("section_title", ""),
                    chunk.get("page_start", 0),
                    chunk.get("page_end", 0),
                    chunk.get("parent_id"),
                    chunk.get("token_count", 0)
                ))
            
            conn.commit()
            logger.info(f"Added {len(chunks)} chunk indices")
            return True
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return False
        finally:
            conn.close()
    
    def get_parent_chunk_id(self, child_chunk_id: str) -> Optional[str]:
        """获取父 Chunk ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT parent_id FROM chunks WHERE chunk_id = ?",
            (child_chunk_id,)
        )
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else None
    
    def get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取 Chunk 信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        return dict(row) if row else None
    
    # 术语管理
    def add_term(
        self,
        term: str,
        aliases: List[str] = None,
        definition: str = "",
        category: str = "",
        product: str = ""
    ) -> bool:
        """添加术语"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO terminology 
                (term, aliases, definition, category, product, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                term,
                ",".join(aliases) if aliases else "",
                definition,
                category,
                product,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to add term: {e}")
            return False
        finally:
            conn.close()
    
    def get_term(self, term: str) -> Optional[Dict[str, Any]]:
        """获取术语"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM terminology WHERE term = ? OR aliases LIKE ?",
            (term, f"%{term}%")
        )
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            data = dict(row)
            data["aliases"] = data["aliases"].split(",") if data["aliases"] else []
            return data
        return None
    
    def list_terms(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出术语"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT * FROM terminology WHERE category = ? ORDER BY term",
                (category,)
            )
        else:
            cursor.execute("SELECT * FROM terminology ORDER BY term")
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            data = dict(row)
            data["aliases"] = data["aliases"].split(",") if data["aliases"] else []
            result.append(data)
        
        return result
