"""
表格处理模块

功能：
1. 精确提取表格结构 (行/列/合并单元格)
2. 语义化表格理解 (表头识别、列类型推断)
3. 多种输出格式 (Markdown, JSON, 自然语言)
4. 表格完整性校验
5. 针对 RAG 的表格优化切分

设计原则：
- 保持表格数据关系的完整性
- 检索时能匹配表格中的具体数据
- 生成时能准确引用表格内容
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import re
import json
from pathlib import Path

from loguru import logger


class ColumnType(Enum):
    """列类型"""
    TEXT = "text"           # 文本
    NUMBER = "number"       # 数值
    PERCENTAGE = "percentage"  # 百分比
    DATE = "date"           # 日期
    BOOLEAN = "boolean"     # 布尔值
    CODE = "code"           # 代码/命令
    MIXED = "mixed"         # 混合类型


class TableType(Enum):
    """表格类型"""
    DATA = "data"           # 数据表格 (如参数配置表)
    COMPARISON = "comparison"  # 对比表格 (如功能对比)
    SPECIFICATION = "specification"  # 规格表格 (如硬件规格)
    MAPPING = "mapping"     # 映射表格 (如错误码对照)
    PROCEDURE = "procedure"  # 步骤表格 (如操作步骤)
    MATRIX = "matrix"       # 矩阵表格 (如兼容性矩阵)
    UNKNOWN = "unknown"


@dataclass
class TableCell:
    """表格单元格"""
    value: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    cell_type: ColumnType = ColumnType.TEXT
    
    def __str__(self):
        return self.value


@dataclass
class TableColumn:
    """表格列"""
    index: int
    header: str
    column_type: ColumnType
    values: List[str] = field(default_factory=list)
    
    def get_unique_values(self) -> List[str]:
        return list(set(self.values))


@dataclass
class TableRow:
    """表格行"""
    index: int
    cells: List[TableCell] = field(default_factory=list)
    is_header: bool = False
    
    def to_dict(self, headers: List[str]) -> Dict[str, str]:
        """转换为字典格式"""
        result = {}
        for i, cell in enumerate(self.cells):
            key = headers[i] if i < len(headers) else f"col_{i}"
            result[key] = cell.value
        return result


@dataclass
class StructuredTable:
    """
    结构化表格
    
    包含完整的表格语义信息：
    - 原始单元格数据
    - 表头信息
    - 列类型推断
    - 表格类型分类
    - 多种输出格式
    """
    table_id: str
    source_doc: str
    page_num: int
    
    # 结构数据
    cells: List[List[TableCell]] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    columns: List[TableColumn] = field(default_factory=list)
    
    # 元数据
    num_rows: int = 0
    num_cols: int = 0
    has_header: bool = True
    table_type: TableType = TableType.UNKNOWN
    title: str = ""  # 表格标题 (如 "表1: 配置参数说明")
    caption: str = ""  # 表格说明
    
    # 原始内容
    raw_data: List[List[str]] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        if not self.raw_data:
            return ""
        
        lines = []
        
        # 添加标题
        if self.title:
            lines.append(f"**{self.title}**")
            lines.append("")
        
        # 表头
        header_row = self.raw_data[0] if self.has_header else [f"列{i+1}" for i in range(self.num_cols)]
        lines.append("| " + " | ".join(self._escape_cell(c) for c in header_row) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
        
        # 数据行
        start_row = 1 if self.has_header else 0
        for row in self.raw_data[start_row:]:
            # 确保列数一致
            padded_row = row + [""] * (len(header_row) - len(row))
            lines.append("| " + " | ".join(self._escape_cell(c) for c in padded_row[:len(header_row)]) + " |")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """转换为 JSON 格式"""
        records = []
        start_row = 1 if self.has_header else 0
        
        for row in self.raw_data[start_row:]:
            record = {}
            for i, value in enumerate(row):
                key = self.headers[i] if i < len(self.headers) else f"col_{i}"
                record[key] = value
            records.append(record)
        
        return json.dumps(records, ensure_ascii=False, indent=2)
    
    def to_natural_language(self) -> str:
        """
        转换为自然语言描述
        
        适用于 LLM 理解和检索匹配
        """
        lines = []
        
        # 表格概述
        if self.title:
            lines.append(f"这是一个关于「{self.title}」的表格。")
        
        lines.append(f"表格共有 {self.num_rows} 行 {self.num_cols} 列。")
        
        if self.headers:
            lines.append(f"列包括：{', '.join(self.headers)}。")
        
        lines.append("")
        
        # 逐行描述
        start_row = 1 if self.has_header else 0
        for i, row in enumerate(self.raw_data[start_row:], 1):
            row_desc = self._describe_row(row)
            if row_desc:
                lines.append(f"第{i}行：{row_desc}")
        
        return "\n".join(lines)
    
    def to_row_chunks(self) -> List[Dict[str, Any]]:
        """
        按行切分为独立的可检索单元
        
        每行生成一个 chunk，包含：
        - 完整的行数据
        - 表头信息作为上下文
        - 表格标题
        
        适用于检索具体数据项
        """
        chunks = []
        start_row = 1 if self.has_header else 0
        
        for i, row in enumerate(self.raw_data[start_row:]):
            # 构建行内容
            row_content = self._format_row_for_retrieval(row)
            
            chunk = {
                "chunk_id": f"{self.table_id}_row_{i}",
                "content": row_content,
                "content_type": "table_row",
                "metadata": {
                    "table_id": self.table_id,
                    "table_title": self.title,
                    "row_index": i,
                    "headers": self.headers,
                    "row_data": dict(zip(self.headers, row)) if self.headers else row,
                    "source_doc": self.source_doc,
                    "page_num": self.page_num
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def query_by_column(self, column_name: str, value: str) -> List[Dict[str, str]]:
        """
        按列查询表格数据
        
        示例：query_by_column("参数名", "timeout") 
        返回所有参数名为 timeout 的行
        """
        results = []
        
        if column_name not in self.headers:
            return results
        
        col_idx = self.headers.index(column_name)
        start_row = 1 if self.has_header else 0
        
        for row in self.raw_data[start_row:]:
            if col_idx < len(row) and value.lower() in row[col_idx].lower():
                results.append(dict(zip(self.headers, row)))
        
        return results
    
    def _escape_cell(self, value: str) -> str:
        """转义单元格内容"""
        if not value:
            return ""
        return value.replace("|", "\\|").replace("\n", "<br>").strip()
    
    def _describe_row(self, row: List[str]) -> str:
        """描述一行数据"""
        if not row or not self.headers:
            return ""
        
        parts = []
        for i, value in enumerate(row):
            if value and i < len(self.headers):
                parts.append(f"{self.headers[i]}是「{value}」")
        
        return "，".join(parts)
    
    def _format_row_for_retrieval(self, row: List[str]) -> str:
        """格式化行数据用于检索"""
        parts = []
        
        if self.title:
            parts.append(f"[{self.title}]")
        
        for i, value in enumerate(row):
            if value:
                header = self.headers[i] if i < len(self.headers) else f"列{i+1}"
                parts.append(f"{header}: {value}")
        
        return " | ".join(parts)


class TableProcessor:
    """
    表格处理器
    
    核心功能：
    1. 从 PDF 提取表格 (支持多种方法)
    2. 识别表格结构 (表头、合并单元格)
    3. 推断列类型和表格类型
    4. 验证数据完整性
    5. 生成适合 RAG 的表格表示
    """
    
    def __init__(
        self,
        detect_header: bool = True,
        infer_column_types: bool = True,
        validate_structure: bool = True,
        min_rows: int = 2,
        min_cols: int = 2
    ):
        self.detect_header = detect_header
        self.infer_column_types = infer_column_types
        self.validate_structure = validate_structure
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def extract_tables_from_pdf(
        self,
        pdf_path: str,
        method: str = "pdfplumber"  # pdfplumber, camelot, tabula
    ) -> List[StructuredTable]:
        """
        从 PDF 提取表格
        
        Args:
            pdf_path: PDF 文件路径
            method: 提取方法
                - pdfplumber: 适合简单表格，速度快
                - camelot: 适合复杂表格，需要 Ghostscript
                - tabula: 适合有线表格
        """
        if method == "pdfplumber":
            return self._extract_with_pdfplumber(pdf_path)
        elif method == "camelot":
            return self._extract_with_camelot(pdf_path)
        elif method == "tabula":
            return self._extract_with_tabula(pdf_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[StructuredTable]:
        """使用 pdfplumber 提取表格"""
        import pdfplumber
        
        tables = []
        pdf_path = Path(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 10,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1
                    }
                )
                
                for idx, raw_table in enumerate(page_tables):
                    if not self._is_valid_table(raw_table):
                        continue
                    
                    # 清理和处理表格
                    cleaned = self._clean_table(raw_table)
                    
                    # 创建结构化表格
                    table = self._create_structured_table(
                        raw_data=cleaned,
                        table_id=f"{pdf_path.stem}_p{page_num}_t{idx}",
                        source_doc=pdf_path.name,
                        page_num=page_num
                    )
                    
                    # 尝试提取表格标题
                    table.title = self._find_table_title(page, raw_table)
                    
                    tables.append(table)
        
        logger.info(f"从 {pdf_path.name} 提取了 {len(tables)} 个表格")
        return tables
    
    def _extract_with_camelot(self, pdf_path: str) -> List[StructuredTable]:
        """使用 Camelot 提取表格 (更适合复杂表格)"""
        try:
            import camelot
        except ImportError:
            logger.warning("Camelot not installed, falling back to pdfplumber")
            return self._extract_with_pdfplumber(pdf_path)
        
        tables = []
        pdf_path = Path(pdf_path)
        
        # 先尝试 lattice (有线表格)
        lattice_tables = camelot.read_pdf(
            str(pdf_path), 
            pages='all',
            flavor='lattice'
        )
        
        for idx, camelot_table in enumerate(lattice_tables):
            if camelot_table.accuracy < 50:
                continue
            
            raw_data = camelot_table.df.values.tolist()
            
            if not self._is_valid_table(raw_data):
                continue
            
            table = self._create_structured_table(
                raw_data=raw_data,
                table_id=f"{pdf_path.stem}_t{idx}",
                source_doc=pdf_path.name,
                page_num=camelot_table.page
            )
            
            table.metadata = {
                "accuracy": camelot_table.accuracy,
                "whitespace": camelot_table.whitespace,
                "extraction_method": "camelot_lattice"
            }
            
            tables.append(table)
        
        return tables
    
    def _extract_with_tabula(self, pdf_path: str) -> List[StructuredTable]:
        """使用 Tabula 提取表格"""
        try:
            import tabula
        except ImportError:
            logger.warning("Tabula not installed, falling back to pdfplumber")
            return self._extract_with_pdfplumber(pdf_path)
        
        tables = []
        pdf_path = Path(pdf_path)
        
        dfs = tabula.read_pdf(str(pdf_path), pages='all')
        
        for idx, df in enumerate(dfs):
            raw_data = [df.columns.tolist()] + df.values.tolist()
            
            if not self._is_valid_table(raw_data):
                continue
            
            table = self._create_structured_table(
                raw_data=raw_data,
                table_id=f"{pdf_path.stem}_t{idx}",
                source_doc=pdf_path.name,
                page_num=1  # tabula 不返回页码
            )
            
            tables.append(table)
        
        return tables
    
    def _create_structured_table(
        self,
        raw_data: List[List[str]],
        table_id: str,
        source_doc: str,
        page_num: int
    ) -> StructuredTable:
        """创建结构化表格对象"""
        
        # 检测表头
        has_header = self._detect_header_row(raw_data) if self.detect_header else True
        
        # 提取表头
        headers = raw_data[0] if has_header and raw_data else []
        headers = [str(h).strip() if h else f"列{i+1}" for i, h in enumerate(headers)]
        
        # 推断列类型
        columns = []
        if self.infer_column_types and headers:
            for i, header in enumerate(headers):
                col_values = [row[i] for row in raw_data[1:] if i < len(row)]
                col_type = self._infer_column_type(col_values)
                columns.append(TableColumn(
                    index=i,
                    header=header,
                    column_type=col_type,
                    values=col_values
                ))
        
        # 推断表格类型
        table_type = self._classify_table_type(headers, raw_data)
        
        # 构建单元格矩阵
        cells = []
        for i, row in enumerate(raw_data):
            row_cells = []
            for j, value in enumerate(row):
                cell = TableCell(
                    value=str(value).strip() if value else "",
                    row=i,
                    col=j,
                    is_header=(i == 0 and has_header),
                    cell_type=columns[j].column_type if j < len(columns) else ColumnType.TEXT
                )
                row_cells.append(cell)
            cells.append(row_cells)
        
        return StructuredTable(
            table_id=table_id,
            source_doc=source_doc,
            page_num=page_num,
            cells=cells,
            headers=headers,
            columns=columns,
            num_rows=len(raw_data),
            num_cols=len(headers) if headers else (len(raw_data[0]) if raw_data else 0),
            has_header=has_header,
            table_type=table_type,
            raw_data=raw_data
        )
    
    def _clean_table(self, raw_table: List[List[Any]]) -> List[List[str]]:
        """清理表格数据"""
        cleaned = []
        
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cell_text = ""
                else:
                    cell_text = str(cell).strip()
                    # 清理多余空白
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    # 转义特殊字符
                    cell_text = cell_text.replace('\n', ' ').replace('\r', '')
                cleaned_row.append(cell_text)
            cleaned.append(cleaned_row)
        
        return cleaned
    
    def _is_valid_table(self, table: List[List[Any]]) -> bool:
        """验证表格是否有效"""
        if not table:
            return False
        
        # 检查行数
        if len(table) < self.min_rows:
            return False
        
        # 检查列数
        if not table[0] or len(table[0]) < self.min_cols:
            return False
        
        # 检查是否有有效内容
        non_empty_cells = sum(
            1 for row in table for cell in row 
            if cell and str(cell).strip()
        )
        
        total_cells = len(table) * len(table[0])
        if non_empty_cells < total_cells * 0.3:
            return False
        
        return True
    
    def _detect_header_row(self, table: List[List[str]]) -> bool:
        """检测是否有表头行"""
        if not table or len(table) < 2:
            return False
        
        first_row = table[0]
        second_row = table[1]
        
        # 检查第一行是否看起来像表头
        header_indicators = 0
        
        for cell in first_row:
            cell_str = str(cell).strip() if cell else ""
            
            # 表头通常更短
            if len(cell_str) < 30:
                header_indicators += 1
            
            # 表头通常不是纯数字
            if cell_str and not re.match(r'^[\d,.\-+%]+$', cell_str):
                header_indicators += 1
            
            # 表头常见关键词
            header_keywords = ['名称', '参数', '说明', '类型', '值', '描述', 
                             'name', 'type', 'value', 'description', 'id']
            if any(kw in cell_str.lower() for kw in header_keywords):
                header_indicators += 2
        
        return header_indicators >= len(first_row)
    
    def _infer_column_type(self, values: List[str]) -> ColumnType:
        """推断列类型"""
        if not values:
            return ColumnType.TEXT
        
        non_empty = [v for v in values if v and str(v).strip()]
        if not non_empty:
            return ColumnType.TEXT
        
        # 检查是否是数值
        numeric_count = sum(1 for v in non_empty if self._is_numeric(v))
        if numeric_count > len(non_empty) * 0.8:
            # 检查是否是百分比
            if any('%' in str(v) for v in non_empty):
                return ColumnType.PERCENTAGE
            return ColumnType.NUMBER
        
        # 检查是否是日期
        date_count = sum(1 for v in non_empty if self._is_date(v))
        if date_count > len(non_empty) * 0.8:
            return ColumnType.DATE
        
        # 检查是否是布尔值
        bool_values = {'是', '否', 'yes', 'no', 'true', 'false', '√', '×', '✓', '✗'}
        bool_count = sum(1 for v in non_empty if str(v).lower().strip() in bool_values)
        if bool_count > len(non_empty) * 0.8:
            return ColumnType.BOOLEAN
        
        # 检查是否是代码
        code_count = sum(1 for v in non_empty if self._looks_like_code(v))
        if code_count > len(non_empty) * 0.5:
            return ColumnType.CODE
        
        return ColumnType.TEXT
    
    def _is_numeric(self, value: str) -> bool:
        """判断是否是数值"""
        value = str(value).strip().replace(',', '').replace(' ', '')
        value = re.sub(r'[%$€¥]', '', value)
        try:
            float(value)
            return True
        except:
            return False
    
    def _is_date(self, value: str) -> bool:
        """判断是否是日期"""
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 2024-01-15
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # 01/15/2024
            r'\d{4}年\d{1,2}月\d{1,2}日',      # 2024年1月15日
        ]
        for pattern in date_patterns:
            if re.search(pattern, str(value)):
                return True
        return False
    
    def _looks_like_code(self, value: str) -> bool:
        """判断是否像代码"""
        code_indicators = [
            r'^[A-Za-z_][A-Za-z0-9_]*\s*\(',  # 函数调用
            r'^[A-Za-z_][A-Za-z0-9_]*\s*=',   # 赋值
            r'^--\w+',                         # 命令行参数
            r'^\$\w+',                         # 变量
            r'^[A-Z_]+$',                      # 常量
        ]
        for pattern in code_indicators:
            if re.search(pattern, str(value)):
                return True
        return False
    
    def _classify_table_type(
        self, 
        headers: List[str], 
        data: List[List[str]]
    ) -> TableType:
        """分类表格类型"""
        headers_lower = [h.lower() for h in headers]
        
        # 规格表 (参数/配置)
        spec_keywords = ['参数', '配置', '规格', 'spec', 'config', 'parameter', 
                        '项目', '属性', 'property', 'setting']
        if any(kw in h for h in headers_lower for kw in spec_keywords):
            return TableType.SPECIFICATION
        
        # 对比表
        if len(headers) >= 3:
            # 第一列是特性，其他列是不同产品/版本
            first_col_headers = [row[0] for row in data[1:] if row]
            if all(not self._is_numeric(h) for h in first_col_headers):
                return TableType.COMPARISON
        
        # 映射表 (错误码、状态码)
        mapping_keywords = ['错误码', '代码', '状态', 'code', 'error', 'status', 'id']
        if any(kw in h for h in headers_lower for kw in mapping_keywords):
            return TableType.MAPPING
        
        # 步骤表
        step_keywords = ['步骤', '序号', 'step', 'no', '#', '操作']
        if any(kw in h for h in headers_lower for kw in step_keywords):
            return TableType.PROCEDURE
        
        return TableType.DATA
    
    def _find_table_title(self, page, table_data) -> str:
        """查找表格标题"""
        # 这需要根据具体的 PDF 结构来实现
        # 通常表格标题在表格上方，格式如 "表1: xxx" 或 "Table 1. xxx"
        try:
            text = page.extract_text() or ""
            
            # 查找表格标题模式
            patterns = [
                r'表\s*\d+[.:：]\s*([^\n]+)',
                r'Table\s*\d+[.:]\s*([^\n]+)',
                r'图表\s*\d+[.:：]\s*([^\n]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(0).strip()
        except:
            pass
        
        return ""
    
    def validate_table_structure(self, table: StructuredTable) -> Dict[str, Any]:
        """
        验证表格结构完整性
        
        检查：
        1. 列数一致性
        2. 空单元格比例
        3. 数据类型一致性
        4. 重复行检测
        """
        issues = []
        
        # 1. 检查列数一致性
        col_counts = [len(row) for row in table.raw_data]
        if len(set(col_counts)) > 1:
            issues.append({
                "type": "inconsistent_columns",
                "message": f"列数不一致: {set(col_counts)}",
                "severity": "warning"
            })
        
        # 2. 检查空单元格比例
        total_cells = table.num_rows * table.num_cols
        empty_cells = sum(
            1 for row in table.raw_data 
            for cell in row if not cell or not str(cell).strip()
        )
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0
        
        if empty_ratio > 0.5:
            issues.append({
                "type": "high_empty_ratio",
                "message": f"空单元格比例过高: {empty_ratio:.1%}",
                "severity": "warning"
            })
        
        # 3. 检查数据类型一致性
        for col in table.columns:
            if col.column_type == ColumnType.MIXED:
                issues.append({
                    "type": "mixed_column_type",
                    "message": f"列 '{col.header}' 包含混合数据类型",
                    "severity": "info"
                })
        
        # 4. 检查重复行
        row_strings = ["|".join(str(c) for c in row) for row in table.raw_data[1:]]
        duplicates = len(row_strings) - len(set(row_strings))
        if duplicates > 0:
            issues.append({
                "type": "duplicate_rows",
                "message": f"发现 {duplicates} 个重复行",
                "severity": "warning"
            })
        
        return {
            "is_valid": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues,
            "stats": {
                "rows": table.num_rows,
                "cols": table.num_cols,
                "empty_ratio": empty_ratio,
                "has_header": table.has_header,
                "table_type": table.table_type.value
            }
        }


class TableChunker:
    """
    表格切分器
    
    策略：
    1. 小表格 (< 20 行): 作为整体保存
    2. 大表格: 按逻辑分组切分，每组保留表头
    3. 每行也生成独立的可检索 chunk
    """
    
    def __init__(
        self,
        max_rows_per_chunk: int = 20,
        always_include_header: bool = True,
        generate_row_chunks: bool = True,
        generate_summary: bool = True
    ):
        self.max_rows_per_chunk = max_rows_per_chunk
        self.always_include_header = always_include_header
        self.generate_row_chunks = generate_row_chunks
        self.generate_summary = generate_summary
    
    def chunk_table(self, table: StructuredTable) -> List[Dict[str, Any]]:
        """
        切分表格为可检索的 chunks
        
        生成三类 chunk：
        1. 表格整体/分片 (Markdown 格式)
        2. 行级 chunks (用于精确匹配)
        3. 表格摘要 (自然语言描述)
        """
        chunks = []
        
        # 1. 表格 Markdown chunks
        table_chunks = self._split_table_content(table)
        chunks.extend(table_chunks)
        
        # 2. 行级 chunks
        if self.generate_row_chunks:
            row_chunks = table.to_row_chunks()
            chunks.extend(row_chunks)
        
        # 3. 表格摘要
        if self.generate_summary:
            summary_chunk = self._generate_summary_chunk(table)
            chunks.append(summary_chunk)
        
        return chunks
    
    def _split_table_content(self, table: StructuredTable) -> List[Dict[str, Any]]:
        """切分表格内容"""
        chunks = []
        
        data_rows = table.raw_data[1:] if table.has_header else table.raw_data
        
        if len(data_rows) <= self.max_rows_per_chunk:
            # 小表格，整体作为一个 chunk
            chunks.append({
                "chunk_id": f"{table.table_id}_full",
                "content": table.to_markdown(),
                "content_type": "table",
                "metadata": {
                    "table_id": table.table_id,
                    "table_title": table.title,
                    "table_type": table.table_type.value,
                    "num_rows": table.num_rows,
                    "num_cols": table.num_cols,
                    "headers": table.headers,
                    "source_doc": table.source_doc,
                    "page_num": table.page_num,
                    "is_complete": True
                }
            })
        else:
            # 大表格，分片
            for i in range(0, len(data_rows), self.max_rows_per_chunk):
                slice_rows = data_rows[i:i + self.max_rows_per_chunk]
                
                # 重建带表头的表格
                slice_data = [table.raw_data[0]] + slice_rows if table.has_header else slice_rows
                
                slice_table = StructuredTable(
                    table_id=f"{table.table_id}_part{i // self.max_rows_per_chunk}",
                    source_doc=table.source_doc,
                    page_num=table.page_num,
                    headers=table.headers,
                    num_rows=len(slice_data),
                    num_cols=table.num_cols,
                    has_header=table.has_header,
                    table_type=table.table_type,
                    title=table.title,
                    raw_data=slice_data
                )
                
                chunks.append({
                    "chunk_id": slice_table.table_id,
                    "content": slice_table.to_markdown(),
                    "content_type": "table",
                    "metadata": {
                        "table_id": table.table_id,
                        "table_title": table.title,
                        "part_index": i // self.max_rows_per_chunk,
                        "total_parts": (len(data_rows) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk,
                        "headers": table.headers,
                        "source_doc": table.source_doc,
                        "page_num": table.page_num,
                        "is_complete": False
                    }
                })
        
        return chunks
    
    def _generate_summary_chunk(self, table: StructuredTable) -> Dict[str, Any]:
        """生成表格摘要 chunk"""
        summary = table.to_natural_language()
        
        return {
            "chunk_id": f"{table.table_id}_summary",
            "content": summary,
            "content_type": "table_summary",
            "metadata": {
                "table_id": table.table_id,
                "table_title": table.title,
                "table_type": table.table_type.value,
                "headers": table.headers,
                "source_doc": table.source_doc,
                "page_num": table.page_num
            }
        }


# ==================== 便捷函数 ====================

def extract_tables(pdf_path: str, method: str = "pdfplumber") -> List[StructuredTable]:
    """提取 PDF 中的表格"""
    processor = TableProcessor()
    return processor.extract_tables_from_pdf(pdf_path, method)


def table_to_chunks(table: StructuredTable) -> List[Dict[str, Any]]:
    """将表格转换为可检索的 chunks"""
    chunker = TableChunker()
    return chunker.chunk_table(table)
