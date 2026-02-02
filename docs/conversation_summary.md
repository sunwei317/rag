# RAG 项目开发完整对话记录

> 本文档完整记录了技术文档 RAG 系统的开发对话过程

## 📅 时间：2026年2月1日

---

# 对话记录

---

## 对话 1: 用户提供项目需求

**用户：**

根据提供的下面的思路，创建一个完整可以运行的项目。

```
# 技术文档 RAG 系统

## 1. 项目概述

### 1.1 背景
在软件研发过程中，技术文档是知识传递的核心载体。当前企业面临以下挑战：
- 大量 PDF 格式技术文档难以检索和问答
- 新文档编写时需要人工翻阅大量历史文档，效率低下
- 知识复用困难，导致内容不一致或冗余

### 1.2 方案价值
构建基于 RAG（Retrieval-Augmented Generation）的技术文档智能系统：
- **检索问答（Chat）**：对已有 PDF 文档提问，获取精确答案并引用来源
- **文档生成（Generate）**：基于用户需求和已有知识库，自动生成符合规范的技术文档

### 1.3 系统边界

| 功能 | 范围 | 说明 |
|------|------|------|
| 输入文档类型 | PDF | 支持文字版和扫描版（OCR） |
| 输出格式 | Markdown / Word | 可导出为标准文档格式 |
| 问答模式 | 单轮问答 + 多轮对话 | 支持追问和上下文理解 |
| 支持语言 | 中文 / 英文 | 混合语言文档 |
| 部署方式 | 本地部署 / 私有化 | 数据不出企业内网 |

---

## 2. 技术架构

### 2.1 整体架构图
┌─────────────────────────────────────────────────────────────────────────────┐
│                           技术文档 RAG 系统架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     1. 数据摄取与知识库构建 (Offline)                  │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │   │
│  │  │ PDF 获取  │ → │ 结构解析  │ → │ 智能切分  │ → │ 双索引(Vector+BM25)│  │   │
│  │  │ 版本管理  │   │ Layout   │   │ 父子索引  │   │ + Metadata        │  │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        2. 在线推理链路                                │   │
│  │                                                                       │   │
│  │  A. 检索问答链 (RAG Chat)                                            │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│  │  │ Query  │→ │ 混合检索│→ │Rerank │→ │Context │→ │ LLM    │        │   │
│  │  │Rewrite │  │Vec+BM25│  │ 重排序 │  │Builder │  │ 生成   │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘        │   │
│  │                                                                       │   │
│  │  B. 文档生成链 (Doc Agent)                                           │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│  │  │ 需求   │→ │ 大纲   │→ │ 分节   │→ │一致性  │→ │ 格式化 │        │   │
│  │  │ 输入   │  │ 规划   │  │ 写作   │  │ 校验   │  │ 输出   │        │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

### 2.2 核心模块说明

#### 2.2.1 PDF 解析模块
目标：将 PDF 转换为结构化数据，保留原有的层级关系（章节、段落、列表）

方法：
1. **Layout-Aware 解析**
   - 使用 PyMuPDF / pdfplumber / Unstructured.io 提取布局信息
   - 识别标题、段落、表格、图片区域
   - 保留页码、字体大小、位置信息用于后续标题判别

2. **表格处理**
   - 转换为 Markdown 格式保留表格结构
   - 表格不切分，作为完整 Chunk 存储

3. **元数据自动抽取**
   - 从首页/目录提取：产品名称、版本号、发布日期
   - 标注文档类型（安装指南、API手册、故障排除等）
   - 识别密级标识（如"内部资料"）

#### 2.2.2 智能切分策略
核心思想：父子索引 (Parent-Child Indexing)

- **子 Chunk (200-500 tokens)**：用于高精度的向量检索匹配
- **父 Chunk (完整章节/段落)**：保证 LLM 获得完整上下文

实现：
子 Chunk → 向量检索命中 → 取回对应的父 Chunk → 送入 LLM

优势：
- 检索精度高（细粒度匹配）
- 上下文完整（粗粒度输出）
- 避免语义截断

#### 2.2.3 双索引机制

| 索引类型 | 技术选型 | 用途 | 优势 |
|----------|----------|------|------|
| 向量索引 | Milvus / ChromaDB | 语义检索 | 理解同义词、相似表述 |
| BM25 索引 | Elasticsearch / 内存 | 关键词检索 | 精确匹配专有术语 |

混合检索策略：
1. 向量检索取 Top-20
2. BM25 检索取 Top-20
3. RRF (Reciprocal Rank Fusion) 融合：score = Σ 1/(k + rank)，k=60

#### 2.2.4 重排序层 (Reranker)

使用 Cross-Encoder 模型对检索结果重新打分：
- 推荐模型：`bge-reranker-v2-m3` / `cohere-rerank`
- 输入：(Query, Chunk) pairs
- 输出：相关性分数
- 保留 Top-5 进入 LLM 上下文

---

## 3. 检索问答链 (RAG Chat)

### 3.1 Query 理解与改写

| 技术 | 说明 | 实现 |
|------|------|------|
| 查询扩展 | 将用户简短问题扩展为更完整的表述 | LLM 改写 |
| HyDE | 先让 LLM 生成假设性答案，再用答案检索 | 提升召回 |
| 术语对齐 | 将口语化表达映射到文档中的专业术语 | 同义词表/实体链接 |

### 3.2 上下文构建

模板：
相关文档内容：

【来源：{doc_title} - {section_path}，页码：{page}】
{chunk_content}
---

要求：
1. 优先使用最相关的 3-5 个 Chunk
2. 按相关度排序
3. 包含来源信息用于引用

### 3.3 生成与引用

- 使用 LLM 生成答案
- 输出格式包含引用标记 [1][2]
- 返回结构：{ answer, citations: [{doc, page, text}] }

---

## 4. 文档生成链 (Doc Agent)

### 4.1 工作流程

用户需求 → 大纲规划 → 分节写作 → 一致性校验 → 格式输出

### 4.2 大纲规划

基于需求和知识库检索，确定文档结构：
1. 从知识库检索相似文档的目录结构
2. LLM 结合需求生成初始大纲
3. 用户可交互式调整

大纲模板库：
- 安装部署指南
- API 参考手册
- 故障排除手册
- 发行说明
- 用户操作手册
- 产品规格说明书
- 集成开发指南

### 4.3 分节写作

对每个章节独立调用 RAG：
1. 根据章节主题检索相关知识
2. 结合检索内容生成章节内容
3. 保持术语一致性（使用术语表约束）

### 4.4 一致性校验

检查项：
- 术语统一（同一概念使用相同名称）
- 版本号一致
- 交叉引用正确（"参见第X章"）
- 格式规范（标题层级、编号风格）

### 4.5 格式输出

支持导出：
- Markdown → 直接渲染
- Word (docx) → python-docx 库
- 可扩展至 PDF、Confluence

---

## 5. 技术选型

### 5.1 核心组件

| 组件 | 选型 | 备选 |
|------|------|------|
| PDF 解析 | PyMuPDF + pdfplumber | Unstructured.io |
| 向量数据库 | ChromaDB (轻量) | Milvus (生产级) |
| BM25 | rank-bm25 (内存) | Elasticsearch |
| Embedding | BAAI/bge-m3 | OpenAI text-embedding-3-large |
| Reranker | BAAI/bge-reranker-v2-m3 | Cohere Rerank |
| LLM | OpenAI GPT-4 / Claude | 本地 Qwen2.5-72B |

### 5.2 Embedding 模型对比

| 模型 | 维度 | 中文效果 | 推理速度 | 许可 |
|------|------|----------|----------|------|
| bge-m3 | 1024 | ★★★★★ | 快 | MIT |
| bge-large-zh | 1024 | ★★★★☆ | 较快 | MIT |
| text-embedding-3-large | 3072 | ★★★★☆ | 需API | 商业 |

推荐：`bge-m3`（多语言、多粒度、多功能）

---

## 6. 元数据设计

### 6.1 Chunk Metadata Schema

{
  "doc_id": "uuid",
  "doc_title": "ProductX 安装指南",
  "doc_type": "installation_guide",
  "product": "ProductX",
  "version": "3.2.1",
  "security_level": "internal",
  "section_path": "第2章.2.1节",
  "section_title": "系统要求",
  "page_start": 12,
  "page_end": 14,
  "chunk_index": 5,
  "parent_chunk_id": "xxx",
  "keywords": ["CPU", "内存", "操作系统"],
  "last_updated": "2024-03-15"
}

### 6.2 元数据过滤用法

- 按产品过滤：filter = {"product": "ProductX"}
- 按版本过滤：filter = {"version": {"$gte": "3.0"}}
- 按文档类型：filter = {"doc_type": "API"}

---

## 7. 评估方案

### 7.1 检索质量评估

| 指标 | 定义 | 目标值 |
|------|------|--------|
| Recall@10 | 正确答案在Top10的比例 | > 85% |
| MRR | 正确答案排名的倒数平均 | > 0.7 |
| 端到端准确率 | 最终答案正确的比例 | > 80% |

### 7.2 生成质量评估

使用 LLM-as-Judge：
- 准确性：内容是否符合知识库
- 完整性：是否覆盖所有必要点
- 一致性：术语、格式是否统一
- 可读性：表达是否清晰

### 7.3 自动评估工具

推荐使用 RAGAS 框架：
- Context Relevancy
- Faithfulness
- Answer Relevancy

---

## 8. 部署架构

### 8.1 服务拆分

前端 (React/Vue)
    ↓
API Gateway
    ↓
┌─────────────────────────────────────┐
│  RAG Service (FastAPI)              │
│  - /chat     问答接口               │
│  - /generate 文档生成接口            │
│  - /ingest   文档摄取接口            │
└─────────────────────────────────────┘
    ↓
┌──────────┬──────────┬──────────┐
│ Vector DB│ BM25 Index│ LLM API │
└──────────┴──────────┴──────────┘

### 8.2 资源估算

| 组件 | 规格 | 说明 |
|------|------|------|
| RAG Service | 4C8G × 2 | 无状态，可水平扩展 |
| Vector DB | 8C32G | 取决于文档量 |
| GPU (可选) | A10 / L4 | 本地推理 Embedding / LLM |

---

## 9. 实施路线图

### Phase 1: MVP (4周)
- PDF 解析 + 切分
- 基础向量检索
- 简单问答功能

### Phase 2: 增强 (4周)
- 混合检索 + Reranker
- 元数据过滤
- 引用追溯

### Phase 3: 文档生成 (4周)
- 大纲规划
- 分节写作
- 一致性校验

### Phase 4: 生产化 (持续)
- 性能优化
- 监控告警
- 增量更新

---

## 10. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| PDF 解析质量差 | 检索不准 | 增加 OCR、人工校验 |
| 向量检索召回低 | 漏掉正确答案 | 混合检索、Query扩展 |
| LLM 幻觉 | 生成错误内容 | 引用验证、置信度标注 |
| 术语不一致 | 文档质量差 | 术语库约束 |
```

**助手回应：**

创建了完整的项目结构，包含 8 个主要任务：

1. ✅ 创建项目结构和配置文件
2. ✅ PDF 解析模块 (Layout-Aware)
3. ✅ 智能切分 + 父子索引
4. ✅ 混合检索 (Vector + BM25 + RRF)
5. ✅ Reranker 重排序
6. ✅ 文档生成代理 (大纲规划、分节写作、一致性校验)
7. ✅ FastAPI 服务 (20+ 端点)
8. ✅ 配置文件和启动脚本

---

## 对话 2: 召回率优化讨论

**用户：**
召回，你有考虑到吗？

**助手回应：**

确认了已实现的召回优化策略：
1. ✅ 混合检索 (Vector + BM25 + RRF融合)
2. ✅ 查询扩展 (HyDE + 同义词)
3. ✅ 父子索引 (检索子Chunk，返回父Chunk)

识别了 3 个未实现的增强策略：
1. ❌ 多查询并行检索
2. ❌ 领域术语词典
3. ❌ 召回监控与日志

---

## 对话 3: 实现缺失的召回优化策略

**用户：**
把缺失的3个策略补充实现

**助手回应：**

创建了三个新模块：

### 1. `multi_query_retriever.py` - 多路并行召回
- 同时执行多种查询策略
- RRF 融合去重
- 支持原始查询、HyDE、同义词扩展

### 2. `terminology_dict.py` - 领域术语词典
- 50+ 内置技术术语
- 缩写 ↔ 全称双向映射
- 支持 JSON/YAML 导入导出

### 3. `recall_monitor.py` - 召回监控
- 实时质量评估
- 低召回告警
- 日志记录和报告生成

---

## 对话 4: 更新 README 文档

**用户：**
写一个详细的readme.md，怎样使用，技术实现，等等

**助手回应：**

完全重写了 README.md，包含约 1000 行详细文档：
- 系统架构图
- 项目结构说明
- 快速开始指南
- 详细使用指南 (API 示例、Python SDK)
- 技术实现细节
- 配置说明
- API 端点列表

---

## 对话 5: OCR 预处理

**用户：**
扫描的文档可能需要OCR预处理，选用合适的工具，尤其是表格，图片等

**助手回应：**

创建了 `ocr_processor.py` 模块：

**OCR 提供者对比：**
| 提供者 | 准确率 | 表格识别 | 成本 |
|--------|--------|----------|------|
| PaddleOCR ⭐ | 高 | PPStructure | 免费 |
| Azure Doc Intelligence | 极高 | 最强 | 付费 |
| Tesseract | 中 | ❌ | 免费 |

**功能：**
- 自动检测扫描版 PDF
- 图像预处理 (二值化、去噪、倾斜校正)
- PPStructure 表格识别
- 集成到 PDFParser

---

## 对话 6: 图片处理与多模态 RAG

**用户：**
对于扫描的文档中的图片应该怎样处理？RAG怎样处理图片，保存，embedding,查询？

**助手回应：**

创建了 `image_processor.py` 模块：

**完整流程：**
```
PDF 提取 → AI 描述生成 → 双向量化 → 存储 → 检索
```

**核心类：**
- `ImageProcessor` - 图片提取、AI 描述、向量化
- `ImageRetriever` - 文搜图、图搜图

**双向量策略：**
| 向量 | 模型 | 用途 |
|-----|------|-----|
| 文本向量 | BGE-M3 | 用户输入文字 → 搜索相关图片 |
| 图片向量 | CLIP | 上传图片 → 搜索相似图片 |

**Vision 提供者：**
- OpenAI GPT-4o
- Anthropic Claude
- Google Gemini

---

## 对话 7: 表格处理

**用户：**
文档中的表格是怎样处理保证里面的数据关系是正确的？

**助手回应：**

创建了 `table_processor.py` 模块：

**核心策略：**
1. 结构化存储 (行/列/合并单元格)
2. 表格不切分，作为整体保存
3. 三级索引：表格整体 + 行级 + 摘要
4. 多格式输出：Markdown / JSON / 自然语言

**核心类：**
- `StructuredTable` - 结构化表格
- `TableProcessor` - 表格提取、类型推断
- `TableChunker` - 智能切分

**表格类型自动分类：**
- specification (规格表)
- comparison (对比表)
- mapping (映射表)
- procedure (步骤表)

**输出格式：**
```python
table.to_markdown()      # Markdown 格式
table.to_json()          # JSON 格式
table.to_natural_language()  # 自然语言描述
table.to_row_chunks()    # 行级 chunks
```

---

## 对话 8: 导出对话

**用户：**
怎样可以完整地下载这个对话的历史

**助手回应：**

提供了几种方法：
1. 手动复制粘贴
2. 创建摘要文档 `docs/conversation_summary.md`

**用户：**
没有看到 Export Chat 或导出聊天

**助手回应：**

确认 VS Code Copilot Chat 没有内置导出功能，建议手动复制。

**用户：**
聊天的所有内容都保存到这个文档中

**助手回应：**

（当前操作）将完整对话记录保存到此文档。

---

# 项目文件清单

## 创建的所有文件

### 配置文件
- `config/__init__.py`
- `config/settings.py`
- `.env.example`
- `.gitignore`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `pytest.ini`
- `run_server.py`
- `README.md`

### 数据摄取模块 (`src/ingestion/`)
- `__init__.py`
- `pdf_parser.py` - PDF 结构化解析
- `ocr_processor.py` - OCR 处理
- `image_processor.py` - 图片处理
- `table_processor.py` - 表格处理
- `chunker.py` - 智能切分
- `embedder.py` - 向量化
- `pipeline.py` - 摄取管道

### 检索模块 (`src/retrieval/`)
- `__init__.py`
- `hybrid_search.py` - 混合检索
- `query_transformer.py` - 查询转换
- `reranker.py` - 重排序
- `multi_query_retriever.py` - 多路召回
- `terminology_dict.py` - 术语词典
- `recall_monitor.py` - 召回监控

### 生成模块 (`src/generation/`)
- `__init__.py`
- `rag_chat.py` - 检索问答
- `doc_agent.py` - 文档生成代理
- `outline_planner.py` - 大纲规划
- `section_writer.py` - 分节写作
- `consistency_checker.py` - 一致性校验

### 存储模块 (`src/storage/`)
- `__init__.py`
- `vector_store.py` - 向量库
- `bm25_store.py` - BM25 索引
- `metadata_store.py` - 元数据存储

### 工具模块 (`src/utils/`)
- `__init__.py`
- `terminology.py` - 术语库
- `evaluator.py` - 评估工具

### API 服务 (`src/api/`)
- `__init__.py`
- `main.py` - FastAPI 服务

### 测试 (`tests/`)
- `__init__.py`
- `test_ingestion.py`
- `test_api.py`

### 示例 (`examples/`)
- `demo.py`

### 文档 (`docs/`)
- `conversation_summary.md` (本文件)

---

# 关键技术实现细节

## 1. PDF 解析 (Layout-Aware)

```python
class PDFParser:
    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        detect_headers_footers: bool = True,
        min_heading_font_size: float = 12.0,
        enable_ocr: bool = True,
        ocr_provider: str = "paddleocr",
        ocr_lang: str = "ch",
        ocr_confidence_threshold: float = 0.6
    ):
        ...
```

**功能：**
- 识别标题层级 (H1-H6)
- 提取表格转 Markdown
- 提取图片和图注
- 自动检测扫描版并启用 OCR

## 2. 智能切分 (父子索引)

```python
class SmartChunker:
    def __init__(
        self,
        child_chunk_size: int = 400,    # 子 Chunk
        child_chunk_overlap: int = 50,
        parent_chunk_size: int = 1500,  # 父 Chunk
        parent_chunk_overlap: int = 200,
        keep_table_intact: bool = True,  # 表格不切分
        keep_code_intact: bool = True    # 代码不切分
    ):
        ...
```

**原理：**
- 子 Chunk 用于精确向量检索
- 命中后返回父 Chunk 保证上下文完整

## 3. 混合检索 (RRF 融合)

```python
def hybrid_search(query, top_k=20, weights=(0.5, 0.5)):
    # 1. 向量检索
    vector_results = vector_store.search(query_embedding, top_k)
    
    # 2. BM25 检索
    bm25_results = bm25_store.search(query, top_k)
    
    # 3. RRF 融合
    # score = Σ 1/(k + rank), k=60
    return rrf_fusion(vector_results, bm25_results)
```

## 4. 多路并行召回

```python
class MultiQueryRetriever:
    async def retrieve(self, query: str) -> List[Chunk]:
        queries = self.generate_queries(query)
        # queries = [原始查询, HyDE假设答案, 同义词扩展, ...]
        
        # 并行检索
        all_results = await asyncio.gather(*[
            self.search(q) for q in queries
        ])
        
        # RRF 融合去重
        return self.rrf_fusion(all_results)
```

## 5. OCR 处理流程

```python
class OCRProcessor:
    def process_pdf(self, pdf_path: str) -> OCRDocumentResult:
        # 1. 检测是否为扫描版
        if not self.is_scanned_pdf(pdf_path):
            return None
        
        # 2. 逐页处理
        for page in pdf_pages:
            # 图像预处理
            image = self.preprocess_image(page)
            
            # OCR 识别
            text_blocks = self.ocr_engine.ocr(image)
            
            # 表格识别 (PPStructure)
            tables = self.structure_engine(image)
            
        return result
```

## 6. 图片处理流程

```python
class ImageProcessor:
    def process_document_images(self, pdf_path: str):
        # 1. 提取图片
        images = self.extract_images_from_pdf(pdf_path)
        
        # 2. AI 生成描述 (GPT-4V/Claude)
        for img in images:
            img.description = self.generate_image_description(img)
        
        # 3. 生成向量
        for img in images:
            img.text_embedding = self.embed_text(img.description)
            img.image_embedding = self.embed_image_with_clip(img)
        
        return images
```

## 7. 表格处理策略

```python
class TableProcessor:
    def create_structured_table(self, raw_data):
        # 1. 检测表头
        has_header = self.detect_header_row(raw_data)
        
        # 2. 推断列类型
        columns = []
        for i, header in enumerate(headers):
            col_type = self.infer_column_type(values)
            columns.append(TableColumn(header, col_type))
        
        # 3. 分类表格类型
        table_type = self.classify_table_type(headers, data)
        
        return StructuredTable(...)

class TableChunker:
    def chunk_table(self, table):
        chunks = []
        
        # 1. 表格 Markdown (整体或分片)
        chunks.extend(self.split_table_content(table))
        
        # 2. 行级 chunks (精确匹配)
        chunks.extend(table.to_row_chunks())
        
        # 3. 表格摘要 (自然语言)
        chunks.append(self.generate_summary_chunk(table))
        
        return chunks
```

---

# 依赖列表 (requirements.txt)

```txt
# Core Framework
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6

# PDF Processing
pymupdf>=1.23.0
pdfplumber>=0.10.0
unstructured[pdf]>=0.12.0
pdf2image>=1.16.0
pytesseract>=0.3.10

# OCR Processing
paddlepaddle>=2.6.0
paddleocr>=2.7.0
opencv-python>=4.9.0
beautifulsoup4>=4.12.0
lxml>=5.1.0

# Vector Store
chromadb>=0.4.22
pymilvus>=2.3.0

# Embedding & Reranking
sentence-transformers>=2.3.0
FlagEmbedding>=1.2.0

# Multimodal
Pillow>=10.2.0
transformers>=4.38.0
torch>=2.2.0

# BM25 Search
rank-bm25>=0.2.2
elasticsearch>=8.12.0

# LLM Clients
openai>=1.12.0
anthropic>=0.18.0
google-generativeai>=0.4.0

# Document Generation
python-docx>=1.1.0
markdown>=3.5.0
jinja2>=3.1.0

# Data Processing
pandas>=2.2.0
numpy>=1.26.0
tiktoken>=0.6.0

# Evaluation
ragas>=0.1.0
datasets>=2.17.0

# Utilities
loguru>=0.7.0
python-dotenv>=1.0.0
tenacity>=8.2.0
aiohttp>=3.9.0
httpx>=0.26.0
tqdm>=4.66.0
rich>=13.7.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

---

# 启动和使用

## 启动服务

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Keys

# 3. 启动服务
python run_server.py

# 4. 访问 API 文档
open http://localhost:8000/docs
```

## 使用示例

### 文档摄取
```python
from src.ingestion import IngestionPipeline

pipeline = IngestionPipeline(vector_store, bm25_store, embedder)
stats = pipeline.ingest_pdf(
    "manual.pdf",
    metadata={"product": "ProductX", "version": "1.0"}
)
```

### 问答
```python
from src.generation import RAGChat

response = await rag_chat.chat(
    query="如何配置网络参数？",
    filter_dict={"product": "ProductX"}
)
print(response["answer"])
print(response["citations"])
```

### OCR 处理
```python
from src.ingestion import PDFParser

parser = PDFParser(enable_ocr=True, ocr_provider="paddleocr")
result = parser.parse("scanned_doc.pdf")
```

### 图片处理
```python
from src.ingestion import process_pdf_images

images = process_pdf_images("manual.pdf", vision_provider="openai")
for img in images:
    print(f"{img.image_id}: {img.description}")
```

### 表格提取
```python
from src.ingestion import extract_tables

tables = extract_tables("config.pdf")
for table in tables:
    print(table.to_markdown())
    print(table.to_natural_language())
```

---

## 📅 2026年2月2日 更新

### 对话 12: Graph RAG 实现与 Neo4j 集成

**用户：** 对于这个文档 RAG 系统，使用 Graph RAG 有什么优势吗？

**Copilot：** 分析了 Graph RAG 在技术文档场景的优势，包括：
- 跨文档实体关联（如 MySQL 和 Redis 的依赖关系）
- 多跳推理能力（A → B → C 的间接依赖）
- 实体消歧（区分同名但不同上下文的概念）
- 版本兼容性分析

**用户：** 实现这个推荐的 Graph RAG 集成方案

**Copilot：** 创建了完整的 `src/knowledge_graph/` 模块：
- `entity_extractor.py` - LLM + 规则混合实体抽取
- `relation_builder.py` - 语义关系构建
- `graph_store.py` - 支持内存和 Neo4j 双存储
- `graph_retriever.py` - 子图检索和路径查找
- `graph_rag.py` - Graph RAG 与向量检索融合
- `pipeline.py` - 图谱构建流水线

### 对话 13: Docker 部署与 Neo4j 配置

**用户：** 在 Docker 里面运行，使用 Neo4j

**Copilot：** 更新了 `docker-compose.yml`，添加 Neo4j 服务：
```yaml
neo4j:
  image: neo4j:5.17-community
  profiles: ["graphrag"]
  ports:
    - "7474:7474"  # Web UI
    - "7687:7687"  # Bolt
  environment:
    NEO4J_AUTH: neo4j/graphrag123
```

启动命令：`docker-compose --profile graphrag up -d`

### 对话 14-16: 性能优化（3轮）

**用户：** 这个过程非常慢，优化这个过程

**Copilot：** 实施了三轮优化：

**第一轮优化：**
- 批量处理 LLM 请求 (`batch_size=10`)
- 多线程并行实体抽取 (`ThreadPoolExecutor`)
- Neo4j 批量写入优化

**第二轮优化：**
- 增加并行度 (`max_workers=5`)
- 减少 LLM 调用次数
- 优化图谱构建流程

**第三轮优化（用户选择 1, 3, 4）：**
1. ✅ 异步后台处理 - 图谱构建改为后台任务
2. ❌ 增量更新（未实现）
3. ✅ 实体抽取缓存 - Pickle 持久化到 `/tmp/entity_cache/`
4. ✅ 减少关系类型 - 核心关系白名单（DEPENDS_ON, REQUIRES 等）

**结果：** 文档上传从 ~2m43s 优化到 ~14s（图谱后台构建 ~52s）

### 对话 17-18: 数据驱动文档生成

**用户：** 修改代码，根据数据库中的数据，撰写一篇类似的技术文档。如果数据没有在数据库中提供，就不要提及，不可自己生成

**Copilot：** 创建了 `DataDrivenWriter` 模块：

**新增文件：**
- `src/generation/data_driven_writer.py`

**新增 API 端点：**
- `POST /api/generate/from-database` - 从数据库生成文档
- `GET /api/generate/preview-data` - 预览可用数据

**核心功能：**
- 从 Neo4j 读取所有实体和关系
- 从 ChromaDB 读取文档 chunks
- 按实体类型组织文档结构
- 可选 LLM 格式化（美化输出）

### 对话 19: 文档长度和内容细节控制

**用户：** 如何控制生成文档的长度和内容细节？

**Copilot：** 添加了完整的控制参数：

**细节级别预设 (`detail_level`)：**
| 级别 | 每类实体数 | 章节数 | 包含描述 | 包含关系 |
|------|-----------|--------|----------|----------|
| `brief` | 5 | 5 | ❌ | ❌ |
| `standard` | 20 | 10 | ✅ | ✅ |
| `detailed` | 无限 | 无限 | ✅ | ✅ |

**自定义参数：**
- `max_entities_per_type` - 每种类型最多实体数
- `max_sections` - 最大章节数
- `include_relations` - 是否包含关系章节
- `entity_types` - 指定要包含的实体类型列表

**API 示例：**
```bash
# 简略文档
curl -X POST /api/generate/from-database \
  -d '{"detail_level": "brief", "max_sections": 3}'

# 只含 API 和错误码
curl -X POST /api/generate/from-database \
  -d '{"entity_types": ["api", "error"], "detail_level": "detailed"}'
```

---

## 代码示例汇总

### Graph RAG 使用
```python
from src.knowledge_graph import GraphRAG, Neo4jStore

graph_store = Neo4jStore(uri="bolt://localhost:7687", user="neo4j", password="graphrag123")
graph_rag = GraphRAG(hybrid_searcher=hybrid_searcher, graph_store=graph_store)

response = graph_rag.smart_query("MySQL 需要什么依赖？")
print(response.answer)
print(response.graph_insights)
```

### 数据驱动文档生成
```python
from src.generation import DataDrivenWriter

writer = DataDrivenWriter(
    graph_store=graph_store,
    vector_store=vector_store,
    detail_level="standard",
    max_entities_per_type=10,
    entity_types=["api", "config", "error"]
)

document = writer.generate_with_llm_formatting(
    doc_type="api_reference",
    title="SmartHome Pro API 文档"
)

print(f"实体数: {document.entity_count}")
print(f"章节数: {len(document.sections)}")
print(document.to_markdown())
```

---

*完整对话记录 - 更新时间: 2026-02-02*
