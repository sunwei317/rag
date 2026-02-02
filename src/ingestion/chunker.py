"""
智能切分模块
实现父子索引 (Parent-Child Indexing) 策略

核心思路:
- 子 Chunk (200-500 tokens): 用于高精度的向量检索匹配
- 父 Chunk (完整章节/段落): 保证 LLM 获得完整上下文
- 检索时匹配子 Chunk，但实际返回父 Chunk
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import hashlib
import re
import tiktoken
from loguru import logger

from .pdf_parser import ParsedDocument, Section, DocumentElement, ElementType


class ChunkType(Enum):
    """Chunk 类型"""
    CHILD = "child"     # 细粒度，用于精确检索
    PARENT = "parent"   # 粗粒度，用于上下文


@dataclass
class Chunk:
    """
    Chunk 数据结构
    
    包含完整的元数据以支持:
    - 精确检索 (section_path, page, keywords)
    - 来源追溯 (doc_id, version, security_level)
    - 父子关联 (parent_id, child_ids)
    """
    chunk_id: str
    chunk_type: ChunkType
    content: str
    
    # 来源信息
    doc_id: str
    doc_title: str
    section_path: str
    section_title: str
    page_start: int
    page_end: int
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 父子关系
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # Token 信息
    token_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section_path": self.section_path,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "token_count": self.token_count
        }


@dataclass
class ChunkingResult:
    """切分结果"""
    doc_id: str
    parent_chunks: List[Chunk]
    child_chunks: List[Chunk]
    chunk_index: Dict[str, Chunk]  # chunk_id -> Chunk
    
    @property
    def all_chunks(self) -> List[Chunk]:
        return self.parent_chunks + self.child_chunks
    
    def get_parent(self, child_id: str) -> Optional[Chunk]:
        """获取子 Chunk 的父 Chunk"""
        child = self.chunk_index.get(child_id)
        if child and child.parent_id:
            return self.chunk_index.get(child.parent_id)
        return None
    
    def get_children(self, parent_id: str) -> List[Chunk]:
        """获取父 Chunk 的所有子 Chunk"""
        parent = self.chunk_index.get(parent_id)
        if parent:
            return [self.chunk_index[cid] for cid in parent.child_ids if cid in self.chunk_index]
        return []


class SmartChunker:
    """
    智能切分器
    
    特性:
    1. 父子索引: 子 Chunk 用于检索，父 Chunk 提供上下文
    2. 结构感知: 按章节/段落切分，保持逻辑完整性
    3. 特殊处理: 表格和代码块不切分
    4. 关键词提取: 为每个 Chunk 提取关键词用于 BM25
    """
    
    def __init__(
        self,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 200,
        keep_table_intact: bool = True,
        keep_code_intact: bool = True,
        tokenizer_name: str = "cl100k_base"
    ):
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.keep_table_intact = keep_table_intact
        self.keep_code_intact = keep_code_intact
        
        # 初始化 tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """计算 token 数量"""
        return len(self.tokenizer.encode(text))
    
    def chunk_document(self, doc: ParsedDocument) -> ChunkingResult:
        """
        对文档进行切分
        
        流程:
        1. 遍历所有章节
        2. 为每个章节创建父 Chunk
        3. 将父 Chunk 切分为子 Chunk
        4. 建立父子关联
        """
        logger.info(f"开始切分文档: {doc.doc_id}")
        
        parent_chunks = []
        child_chunks = []
        chunk_index = {}
        
        # 递归处理所有章节
        def process_section(section: Section, parent_path: str = ""):
            # 构建章节内容
            section_content = self._build_section_content(section)
            
            if not section_content.strip():
                # 处理子章节
                for child_section in section.children:
                    process_section(child_section, section.section_path)
                return
            
            # 创建父 Chunk (可能需要进一步切分大章节)
            parent_chunk_contents = self._split_into_parents(section_content)
            
            for idx, parent_content in enumerate(parent_chunk_contents):
                parent_id = self._generate_chunk_id(doc.doc_id, section.section_path, f"p{idx}")
                
                parent_chunk = Chunk(
                    chunk_id=parent_id,
                    chunk_type=ChunkType.PARENT,
                    content=parent_content,
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    section_path=section.section_path,
                    section_title=section.title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    metadata={
                        **doc.metadata,
                        "parent_index": idx,
                        "keywords": self._extract_keywords(parent_content)
                    },
                    token_count=self.count_tokens(parent_content)
                )
                
                parent_chunks.append(parent_chunk)
                chunk_index[parent_id] = parent_chunk
                
                # 创建子 Chunks
                child_contents = self._split_into_children(parent_content)
                
                for cidx, child_content in enumerate(child_contents):
                    child_id = self._generate_chunk_id(doc.doc_id, section.section_path, f"p{idx}c{cidx}")
                    
                    child_chunk = Chunk(
                        chunk_id=child_id,
                        chunk_type=ChunkType.CHILD,
                        content=child_content,
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        section_path=section.section_path,
                        section_title=section.title,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        metadata={
                            **doc.metadata,
                            "child_index": cidx,
                            "keywords": self._extract_keywords(child_content)
                        },
                        parent_id=parent_id,
                        token_count=self.count_tokens(child_content)
                    )
                    
                    child_chunks.append(child_chunk)
                    chunk_index[child_id] = child_chunk
                    parent_chunk.child_ids.append(child_id)
            
            # 递归处理子章节
            for child_section in section.children:
                process_section(child_section, section.section_path)
        
        # 处理所有顶级章节
        for section in doc.sections:
            process_section(section)
        
        logger.info(f"切分完成: {len(parent_chunks)} 个父 Chunk, {len(child_chunks)} 个子 Chunk")
        
        return ChunkingResult(
            doc_id=doc.doc_id,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
            chunk_index=chunk_index
        )
    
    def _build_section_content(self, section: Section) -> str:
        """构建章节内容"""
        parts = []
        
        # 添加标题
        parts.append(f"{'#' * section.level} {section.title}")
        parts.append("")
        
        # 添加元素内容
        for elem in section.elements:
            if elem.element_type == ElementType.TABLE:
                # 表格保持完整
                parts.append(elem.content)
                parts.append("")
            elif elem.element_type == ElementType.CODE:
                # 代码块保持完整
                parts.append("```")
                parts.append(elem.content)
                parts.append("```")
                parts.append("")
            elif elem.element_type == ElementType.LIST:
                parts.append(f"• {elem.content}")
            elif elem.element_type == ElementType.IMAGE:
                parts.append(f"[图片: {elem.content}]")
                parts.append("")
            else:
                parts.append(elem.content)
                parts.append("")
        
        return "\n".join(parts)
    
    def _split_into_parents(self, content: str) -> List[str]:
        """
        将内容切分为父 Chunks
        
        策略:
        1. 如果内容小于 parent_chunk_size，直接返回
        2. 否则按段落切分，累积到接近 parent_chunk_size
        """
        tokens = self.count_tokens(content)
        
        if tokens <= self.parent_chunk_size:
            return [content]
        
        # 按段落切分
        paragraphs = self._split_by_paragraphs(content)
        
        parent_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # 特殊处理: 表格和代码块不切分
            if self._is_special_block(para):
                if current_chunk:
                    parent_chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                parent_chunks.append(para)
                continue
            
            if current_tokens + para_tokens > self.parent_chunk_size and current_chunk:
                parent_chunks.append("\n\n".join(current_chunk))
                
                # 添加重叠
                overlap_text = self._get_overlap_text(current_chunk, self.parent_chunk_overlap)
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = self.count_tokens(overlap_text) if overlap_text else 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        if current_chunk:
            parent_chunks.append("\n\n".join(current_chunk))
        
        return parent_chunks
    
    def _split_into_children(self, content: str) -> List[str]:
        """
        将父 Chunk 切分为子 Chunks
        
        策略:
        1. 按句子切分
        2. 累积到接近 child_chunk_size
        3. 保持一定重叠
        """
        tokens = self.count_tokens(content)
        
        if tokens <= self.child_chunk_size:
            return [content]
        
        # 先检查是否是表格或代码块
        if self._is_special_block(content):
            # 特殊块不切分，作为单独的子 chunk
            return [content]
        
        # 按句子切分
        sentences = self._split_by_sentences(content)
        
        child_chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.child_chunk_size and current_chunk:
                child_chunks.append(" ".join(current_chunk))
                
                # 添加重叠
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.child_chunk_overlap)
                current_chunk = overlap_sentences
                current_tokens = sum(self.count_tokens(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        if current_chunk:
            child_chunks.append(" ".join(current_chunk))
        
        return child_chunks
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """按段落切分"""
        # 按双换行符切分
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, content: str) -> List[str]:
        """按句子切分"""
        # 支持中英文句子
        # 按句号、问号、感叹号、分号切分
        sentences = re.split(r'(?<=[。！？；.!?;])\s*', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_special_block(self, content: str) -> bool:
        """判断是否是表格或代码块"""
        content_stripped = content.strip()
        
        # 表格检测 (Markdown 格式)
        if content_stripped.startswith("|") and "---" in content_stripped:
            return True
        
        # 代码块检测
        if content_stripped.startswith("```") or content_stripped.startswith("~~~"):
            return True
        
        return False
    
    def _get_overlap_text(self, chunks: List[str], max_tokens: int) -> str:
        """获取重叠文本"""
        if not chunks:
            return ""
        
        overlap_parts = []
        total_tokens = 0
        
        for chunk in reversed(chunks):
            chunk_tokens = self.count_tokens(chunk)
            if total_tokens + chunk_tokens > max_tokens:
                break
            overlap_parts.insert(0, chunk)
            total_tokens += chunk_tokens
        
        return "\n\n".join(overlap_parts)
    
    def _get_overlap_sentences(self, sentences: List[str], max_tokens: int) -> List[str]:
        """获取重叠句子"""
        if not sentences:
            return []
        
        overlap = []
        total_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if total_tokens + sentence_tokens > max_tokens:
                break
            overlap.insert(0, sentence)
            total_tokens += sentence_tokens
        
        return overlap
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词 (用于 BM25)"""
        keywords = []
        
        # 提取可能的技术术语
        # 1. 大写缩写 (e.g., API, HTTP, TCP)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        keywords.extend(acronyms)
        
        # 2. 驼峰命名 (e.g., getUserName)
        camel_case = re.findall(r'\b[a-z]+(?:[A-Z][a-z]+)+\b', content)
        keywords.extend(camel_case)
        
        # 3. 下划线命名 (e.g., user_name)
        snake_case = re.findall(r'\b[a-z]+(?:_[a-z]+)+\b', content)
        keywords.extend(snake_case)
        
        # 4. 版本号 (e.g., v1.2.0, 2.0.0)
        versions = re.findall(r'\bv?\d+\.\d+(?:\.\d+)?(?:-\w+)?\b', content)
        keywords.extend(versions)
        
        # 5. 错误码 (e.g., 0x0001, ERROR_001)
        error_codes = re.findall(r'\b(?:0x[0-9A-Fa-f]+|ERROR_\w+|ERR_\w+)\b', content)
        keywords.extend(error_codes)
        
        # 6. 命令/路径 (e.g., /usr/bin, --config)
        paths_cmds = re.findall(r'(?:/[\w/]+|--?\w+)', content)
        keywords.extend(paths_cmds)
        
        # 去重并返回
        return list(set(keywords))
    
    def _generate_chunk_id(self, doc_id: str, section_path: str, suffix: str) -> str:
        """生成 Chunk ID"""
        raw = f"{doc_id}_{section_path}_{suffix}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# 便捷函数
def chunk_document(
    doc: ParsedDocument,
    child_size: int = 400,
    parent_size: int = 1500
) -> ChunkingResult:
    """切分文档的便捷函数"""
    chunker = SmartChunker(
        child_chunk_size=child_size,
        parent_chunk_size=parent_size
    )
    return chunker.chunk_document(doc)
