# 技术文档 RAG 系统

> 一个专为技术文档设计的企业级 RAG（检索增强生成）系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 核心能力

| 功能 | 描述 |
|------|------|
| 🔍 **RAG 问答** | 基于技术文档的智能检索问答，支持引用溯源 |
| 📝 **文档生成** | 自动生成符合企业规范的技术文档 (安装指南、API 文档等) |
| 🔄 **混合检索** | Vector + BM25 双路检索，RRF 融合，提升召回率 |
| 🎯 **多路召回** | 查询扩展 + 并行检索，解决词汇鸿沟问题 |
| 📊 **召回监控** | 生产级检索质量监控与告警 |
| 🕸️ **Graph RAG** | 🆕 知识图谱增强，支持跨文档关联与多跳推理 |
| 📤 **数据驱动文档生成** | 🆕 严格基于数据库数据生成文档，不编造内容 |
| 📥 **文档导出** | 🆕 支持导出为 Markdown、Word (DOCX)、PDF 格式 |
| 💬 **智能对话理解** | 🆕 多轮对话上下文理解，指代词解析，短查询增强 |
| 🔤 **中文分词优化** | 🆕 jieba 分词 + BM25，提升中文检索准确率 |

---

## 📐 系统架构

```
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
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        3. 召回率优化策略                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ 查询扩展    │  │ 术语词典    │  │ 多路并行    │  │ 召回监控    │    │   │
│  │  │ HyDE+同义词 │  │ 缩写/全称   │  │ RRF融合    │  │ 实时告警    │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     4. 🆕 Graph RAG 知识图谱增强                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ 实体抽取    │  │ 关系构建    │  │ 图谱检索    │  │ 上下文融合  │    │   │
│  │  │ LLM+规则   │  │ 语义+共现   │  │ 子图提取    │  │ Chunk+Graph │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
rag/
├── src/
│   ├── ingestion/              # 数据摄取模块
│   │   ├── pdf_parser.py       # PDF 结构化解析 (Layout-Aware)
│   │   ├── ocr_processor.py    # ⭐ 扫描版 PDF OCR 处理
│   │   ├── image_processor.py  # ⭐ 多模态图片处理 (提取/描述/向量化)
│   │   ├── table_processor.py  # ⭐ 表格结构提取与语义理解
│   │   ├── chunker.py          # 智能切分 + 父子索引
│   │   ├── embedder.py         # 多模型向量化
│   │   └── pipeline.py         # 摄取管道
│   │
│   ├── retrieval/              # 检索模块
│   │   ├── hybrid_search.py    # 混合检索 (Vector + BM25 + RRF)
│   │   ├── query_transformer.py # 查询转换 (HyDE, Expansion)
│   │   ├── reranker.py         # 重排序 (BGE-Reranker)
│   │   ├── multi_query_retriever.py  # ⭐ 多路并行召回
│   │   ├── terminology_dict.py       # ⭐ 领域术语词典
│   │   └── recall_monitor.py         # ⭐ 召回率监控
│   │
│   ├── knowledge_graph/        # 🆕 知识图谱模块 (Graph RAG)
│   │   ├── entity_extractor.py # 实体抽取 (LLM + 规则)
│   │   ├── relation_builder.py # 关系构建
│   │   ├── graph_store.py      # 图存储 (Neo4j/内存)
│   │   ├── graph_retriever.py  # 图谱检索
│   │   ├── graph_rag.py        # Graph RAG 融合
│   │   └── pipeline.py         # 图谱构建 Pipeline
│   │
│   ├── generation/             # 生成模块
│   │   ├── rag_chat.py         # 检索问答 (支持 Graph RAG)
│   │   ├── doc_agent.py        # 文档生成代理
│   │   ├── outline_planner.py  # 大纲规划 (7种模板)
│   │   ├── section_writer.py   # 分节写作
│   │   ├── consistency_checker.py # 一致性校验
│   │   └── data_driven_writer.py  # 🆕 数据驱动文档生成
│   │
│   ├── storage/                # 存储模块
│   │   ├── vector_store.py     # 向量库 (Chroma/Milvus)
│   │   ├── bm25_store.py       # BM25 索引 (Memory/ES)
│   │   └── metadata_store.py   # 元数据存储 (SQLite)
│   │
│   ├── utils/                  # 工具模块
│   │   ├── terminology.py      # 术语库管理
│   │   └── evaluator.py        # 质量评估 (RAGAS)
│   │
│   └── api/                    # API 服务
│       └── main.py             # FastAPI 服务 (20+ 端点)
│
├── config/
│   ├── __init__.py
│   └── settings.py             # Pydantic 配置管理
│
├── examples/
│   └── demo.py                 # 演示脚本
│
├── tests/
│   ├── test_ingestion.py       # 摄取模块测试
│   └── test_api.py             # API 测试
│
├── data/                       # 数据目录 (git ignored)
│   ├── chroma/                 # Chroma 持久化
│   ├── uploads/                # 上传文件
│   └── recall_logs/            # 召回监控日志
│
├── output/                     # 生成文档输出
├── .env.example                # 环境变量模板
├── .gitignore
├── Dockerfile
├── docker-compose.yml          # Docker 编排
├── pytest.ini
├── requirements.txt
├── run_server.py               # 启动脚本
└── README.md
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo_url>
cd rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入必要的 API Keys
```

**必需配置项：**

```bash
# 至少配置一个 LLM 提供商
OPENAI_API_KEY=sk-your-openai-key
# 或
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

### 3. 启动服务

**方式一：直接启动**
```bash
python run_server.py
# 服务地址: http://localhost:8000
# API 文档: http://localhost:8000/docs
```

**方式二：使用 Docker**
```bash
# 仅启动应用
docker-compose up -d rag-app

# 🆕 启动完整服务 (含 Neo4j Graph RAG)
docker-compose --profile graphrag up -d
# Neo4j Web UI: http://localhost:7474 (用户: neo4j, 密码: graphrag123)

# 启动完整服务 (含 Milvus)
docker-compose --profile milvus up -d

# 启动完整服务 (含 Elasticsearch)
docker-compose --profile elasticsearch up -d
```

### 4. 验证安装

```bash
# 健康检查
curl http://localhost:8000/health

# 查看 API 文档
open http://localhost:8000/docs
```

---

## 📖 详细使用指南

### 一、文档摄取

#### 1.1 通过 API 上传文档

```bash
# 上传 PDF 文件
curl -X POST "http://localhost:8000/api/ingest/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf" \
  -F "product=ProductX" \
  -F "version=1.0.0" \
  -F "doc_type=user_manual"
```

#### 1.2 通过 Python SDK

```python
from src.ingestion.pipeline import IngestionPipeline
from src.storage import create_vector_store, create_bm25_store
from src.ingestion import Embedder

# 初始化存储
vector_store = create_vector_store()
bm25_store = create_bm25_store()
embedder = Embedder()

# 创建摄取管道
pipeline = IngestionPipeline(
    vector_store=vector_store,
    bm25_store=bm25_store,
    embedder=embedder
)

# 摄取文档
stats = pipeline.ingest_pdf(
    pdf_path="path/to/document.pdf",
    metadata={
        "product": "ProductX",
        "version": "1.0.0",
        "doc_type": "user_manual"
    }
)

print(f"文档ID: {stats.doc_id}")
print(f"父块数: {stats.parent_chunks}")
print(f"子块数: {stats.child_chunks}")
```

---

### 二、🆕 Graph RAG 知识图谱

Graph RAG 为传统 RAG 增加了知识图谱能力，支持跨文档实体关联和多跳推理。

#### 2.1 构建知识图谱

```python
from src.knowledge_graph import (
    GraphBuildPipeline,
    InMemoryGraphStore,
    create_graph_pipeline
)

# 方式一：快速创建 Pipeline
pipeline = create_graph_pipeline(
    persist_path="./data/knowledge_graph.json",
    llm_model="gpt-4.1-mini"
)

# 从 chunks 构建图谱
chunks = [
    {"chunk_id": "c1", "content": "MySQL 8.0 需要 JDK 11..."},
    {"chunk_id": "c2", "content": "Redis 可作为 MySQL 缓存..."},
]
result = pipeline.build(chunks)

print(f"实体数: {result.entity_count}")
print(f"关系数: {result.relation_count}")
```

#### 2.2 使用 Graph RAG 查询

```python
from src.knowledge_graph import GraphRAG, GraphRetriever, InMemoryGraphStore

# 加载图谱
graph_store = InMemoryGraphStore(persist_path="./data/knowledge_graph.json")

# 创建 Graph RAG
graph_rag = GraphRAG(
    hybrid_searcher=hybrid_searcher,  # 现有的向量检索器
    graph_store=graph_store,
    model="gpt-4.1"
)

# 智能查询 (自动识别问题类型)
response = graph_rag.smart_query("MySQL 需要什么依赖？")

print(f"答案: {response.answer}")
print(f"图谱洞察: {response.graph_insights}")
```

#### 2.3 图谱检索

```python
from src.knowledge_graph import GraphRetriever

retriever = GraphRetriever(graph_store)

# 检索相关子图
subgraph = retriever.retrieve(
    query="如何优化 MySQL 性能",
    expand_depth=2  # 扩展2跳邻居
)

# 转换为文本上下文
context = subgraph.to_context_text()
print(context)

# 查找实体间路径
paths = retriever.find_paths("MySQL", "Redis", max_depth=3)
```

#### 2.4 支持的实体和关系类型

**实体类型:**
| 类型 | 说明 | 示例 |
|------|------|------|
| `product` | 产品/组件 | MySQL, Redis |
| `api` | API/接口 | `/api/v1/users` |
| `config` | 配置项 | `max_connections` |
| `version` | 版本号 | `v1.0.0` |
| `dependency` | 依赖 | JDK, Python |
| `command` | 命令 | `mysql -u root` |

**关系类型:**
| 类型 | 说明 | 示例 |
|------|------|------|
| `depends_on` | 依赖 | MySQL → JDK |
| `requires` | 需要 | 安装 → 配置 |
| `belongs_to` | 属于 | API → 模块 |
| `configures` | 配置 | 参数 → 组件 |
| `related_to` | 关联 | MySQL ↔ Redis |

#### 2.5 使用 Neo4j (生产环境)

```python
from src.knowledge_graph import Neo4jStore, GraphBuildPipeline

# 连接 Neo4j
graph_store = Neo4jStore(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# 创建索引
graph_store.create_indexes()

# 使用 Neo4j 构建图谱
pipeline = GraphBuildPipeline(graph_store=graph_store)
pipeline.build(chunks)
```

---

### 三、🆕 数据驱动文档生成

从知识图谱和向量存储中严格提取数据生成技术文档，不编造内容。

#### 3.1 通过 API 生成

```bash
# 标准模式生成
curl -X POST "http://localhost:8000/api/generate/from-database" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_type": "api_reference",
    "title": "SmartHome Pro API 技术文档",
    "use_llm_formatting": true
  }'

# 简略模式 (更短的文档)
curl -X POST "http://localhost:8000/api/generate/from-database" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_type": "api_reference",
    "detail_level": "brief",
    "max_sections": 5
  }'

# 只包含特定实体类型
curl -X POST "http://localhost:8000/api/generate/from-database" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_type": "api_reference",
    "entity_types": ["api", "config", "error"],
    "include_relations": false
  }'
```

#### 3.2 文档长度和细节控制参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `detail_level` | string | 细节级别: `brief`, `standard`, `detailed` | `standard` |
| `max_entities_per_type` | int | 每种实体类型最多包含数量 | 按 detail_level |
| `max_sections` | int | 最大章节数 | 按 detail_level |
| `include_relations` | bool | 是否包含实体关系章节 | `true` |
| `entity_types` | list | 要包含的实体类型列表 | 全部 |
| `use_llm_formatting` | bool | 是否用 LLM 美化格式 | `true` |

**细节级别预设：**

| 级别 | 每类实体数 | 章节数 | 包含描述 | 包含别名 | 包含关系 |
|------|-----------|--------|----------|----------|----------|
| `brief` | 最多 5 | 最多 5 | ❌ | ❌ | ❌ |
| `standard` | 最多 20 | 最多 10 | ✅ | ❌ | ✅ |
| `detailed` | 无限制 | 无限制 | ✅ | ✅ | ✅ |

#### 3.3 通过 Python SDK

```python
from src.generation import DataDrivenWriter
from src.knowledge_graph import Neo4jStore
from src.storage import create_vector_store

# 初始化
graph_store = Neo4jStore(uri="bolt://localhost:7687", user="neo4j", password="graphrag123")
vector_store = create_vector_store()

# 创建写作器 (自定义控制参数)
writer = DataDrivenWriter(
    graph_store=graph_store,
    vector_store=vector_store,
    detail_level="standard",
    max_entities_per_type=10,
    max_sections=8,
    include_relations=True,
    entity_types=["api", "config", "command", "error"]
)

# 生成文档
document = writer.generate_with_llm_formatting(
    doc_type="api_reference",
    title="我的 API 文档"
)

# 保存为 Markdown
with open("output.md", "w") as f:
    f.write(document.to_markdown())

print(f"实体数: {document.entity_count}")
print(f"章节数: {len(document.sections)}")
```

#### 3.4 预览可用数据

```bash
# 预览数据库中的数据统计
curl "http://localhost:8000/api/generate/preview-data"

# 返回示例:
# {
#   "entity_count": 97,
#   "relation_count": 11,
#   "chunk_count": 585,
#   "entities_by_type": {"api": 11, "config": 33, "error": 5, ...}
# }
```

#### 3.5 🆕 文档导出 (Markdown/Word/PDF)

生成的文档可以导出为多种格式：

```bash
# 导出为 Word 文档 (DOCX)
curl -X POST "http://localhost:8000/api/generate/export/docx" \
  -F "content=# 文档标题

这是文档内容..." \
  -F "title=我的技术文档" \
  --output my_document.docx

# 导出为 PDF 文档
curl -X POST "http://localhost:8000/api/generate/export/pdf" \
  -F "content=# 文档标题

这是文档内容..." \
  -F "title=我的技术文档" \
  --output my_document.pdf
```

**支持的导出格式：**

| 格式 | 端点 | 说明 |
|------|------|------|
| Markdown | 直接下载 | 原始 Markdown 文本 |
| Word (DOCX) | `/api/generate/export/docx` | 使用 python-docx 生成 |
| PDF | `/api/generate/export/pdf` | 使用 WeasyPrint 生成，支持中文字体 |

**文件命名规则：**
- 格式：`{文档类型}_{标题}_{日期}.{扩展名}`
- 示例：`API参考文档_SmartHome_Pro_20260202.pdf`

---

### 四、RAG 问答 (支持 Graph RAG)

PDF 解析器会自动提取：

| 内容类型 | 处理方式 |
|----------|----------|
| 标题 | 识别层级 (H1-H6)，构建章节树 |
| 段落 | 保留格式，识别代码块 |
| 表格 | 转换为 Markdown 格式 |
| 图片 | 可选提取，生成多模态摘要 |
| 列表 | 保留层级结构 |

#### 1.4 扫描版 PDF 处理 (OCR)

系统自动检测扫描版 PDF，并启用 OCR 处理：

```python
from src.ingestion import PDFParser

# 创建支持 OCR 的解析器
parser = PDFParser(
    enable_ocr=True,                  # 启用 OCR
    ocr_provider="paddleocr",         # 使用 PaddleOCR (推荐)
    ocr_lang="ch",                    # 中英文混合
    ocr_confidence_threshold=0.6      # 置信度阈值
)

# 解析扫描版 PDF
result = parser.parse("scanned_doc.pdf")

# 或强制使用 OCR
result = parser.parse("doc.pdf", force_ocr=True)

# 检查 OCR 信息
print(f"是否扫描版: {result.metadata.get('is_scanned')}")
print(f"OCR 平均置信度: {result.metadata.get('ocr_confidence', 'N/A')}")
```

**OCR 提供者对比：**

| 提供者 | 准确率 | 表格识别 | 语言支持 | 成本 |
|--------|--------|----------|----------|------|
| **PaddleOCR** ⭐ | 高 | ✅ PPStructure | 80+ 语言 | 免费 |
| **Azure Doc Intelligence** | 极高 | ✅ 最强 | 50+ 语言 | 按量付费 |
| **Tesseract** | 中 | ❌ | 100+ 语言 | 免费 |

**切换 OCR 提供者：**

```python
# 使用 Azure Document Intelligence (高精度)
parser = PDFParser(
    enable_ocr=True,
    ocr_provider="azure"
)
# 需要设置环境变量:
# AZURE_FORM_RECOGNIZER_ENDPOINT=https://xxx.cognitiveservices.azure.com/
# AZURE_FORM_RECOGNIZER_KEY=xxx

# 使用 Tesseract (备选)
parser = PDFParser(
    enable_ocr=True,
    ocr_provider="tesseract"
)
```

**OCR 图像预处理：**

系统自动对扫描图像进行预处理以提高识别准确率：

- 自适应二值化 (Adaptive Threshold)
- 去噪 (Denoising)
- 倾斜校正 (Deskew)
- 对比度增强

#### 1.5 图片处理与多模态 RAG

系统支持对文档中的图片进行提取、理解、向量化和检索：

```
┌─────────────────────────────────────────────────────────────────┐
│                    图片处理流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ PDF 提取  │ →  │ AI 描述   │ →  │ 向量化   │ →  │ 检索     │  │
│  │ PyMuPDF  │    │ GPT-4V   │    │ CLIP    │    │ 文搜图    │  │
│  │          │    │ Claude   │    │ 文本向量 │    │ 图搜图    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**基础使用：**

```python
from src.ingestion import ImageProcessor, process_pdf_images

# 方式一：便捷函数
images = process_pdf_images(
    pdf_path="manual.pdf",
    vision_provider="openai",  # 使用 GPT-4V 生成描述
    generate_descriptions=True
)

for img in images:
    print(f"图片ID: {img.image_id}")
    print(f"类型: {img.image_type.value}")  # diagram, screenshot, chart...
    print(f"描述: {img.description[:200]}")
    print(f"保存位置: {img.image_path}")
    print("---")

# 方式二：精细控制
processor = ImageProcessor(
    output_dir="data/images",
    vision_provider="anthropic",  # 使用 Claude Vision
    min_image_size=100,           # 过滤小于100px的图片
    generate_descriptions=True
)

images = processor.process_document_images(
    pdf_path="architecture_doc.pdf",
    doc_id="arch_v1",
    generate_descriptions=True,
    generate_embeddings=True
)
```

**图片检索：**

```python
from src.ingestion import ImageRetriever
from src.storage import create_vector_store

# 初始化
vector_store = create_vector_store()
retriever = ImageRetriever(vector_store)

# 文本查询图片
results = retriever.search_by_text(
    query="系统架构图",
    top_k=5,
    filter_dict={"source_doc": "manual.pdf"}
)

for r in results:
    print(f"图片: {r['image_path']}")
    print(f"类型: {r['image_type']}")
    print(f"描述: {r['description'][:100]}")
    print(f"相似度: {r['score']:.3f}")

# 图片查询相似图片 (CLIP)
similar = retriever.search_by_image(
    image_path="query_image.png",
    top_k=5
)
```

**Vision 提供者对比：**

| 提供者 | 模型 | 特点 | 适用场景 |
|--------|------|------|----------|
| **OpenAI** | GPT-4o | 最强综合能力 | 复杂图表、代码截图 |
| **Anthropic** | Claude Sonnet | 细节描述精确 | 架构图、流程图 |
| **Google** | Gemini 1.5 Pro | 多语言、长上下文 | 多语言文档 |

**图片向量化策略：**

系统使用双向量策略，同时支持 "文搜图" 和 "图搜图"：

| 向量类型 | 模型 | 用途 | 维度 |
|----------|------|------|------|
| **文本向量** | BGE-M3 | 用描述文本搜图 | 1024 |
| **图片向量** | CLIP | 用图片搜相似图 | 512 |

```python
# 生成双向量
processor.generate_image_embedding(image, embed_type="both")

# 只生成文本向量 (更常用)
processor.generate_image_embedding(image, embed_type="text")
```

#### 1.6 表格处理与数据关系保持

表格是技术文档的核心内容，系统提供专门的表格处理模块确保数据关系的正确性：

```
┌─────────────────────────────────────────────────────────────────┐
│                    表格处理流程                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 结构提取  │ →  │ 语义理解  │ →  │ 多格式   │ →  │ 智能切分  │  │
│  │ pdfplumber│   │ 列类型    │    │ 输出     │    │ 保持完整 │  │
│  │ camelot  │    │ 表格分类  │    │ MD/JSON │    │ 行级检索 │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**基础使用：**

```python
from src.ingestion import TableProcessor, extract_tables

# 方式一：便捷函数
tables = extract_tables("manual.pdf")

for table in tables:
    print(f"表格ID: {table.table_id}")
    print(f"表格类型: {table.table_type.value}")  # data, comparison, specification...
    print(f"行数: {table.num_rows}, 列数: {table.num_cols}")
    print(f"表头: {table.headers}")
    print(f"Markdown:\n{table.to_markdown()}")
    print("---")

# 方式二：精细控制
processor = TableProcessor(
    detect_header=True,        # 自动检测表头
    infer_column_types=True,   # 推断列类型
    validate_structure=True    # 验证结构完整性
)

tables = processor.extract_tables_from_pdf(
    "config_guide.pdf",
    method="pdfplumber"  # 或 "camelot" (复杂表格)
)
```

**表格输出格式：**

```python
# 1. Markdown 格式 (用于 LLM 生成)
markdown = table.to_markdown()
# | 参数名 | 类型 | 默认值 | 说明 |
# |--------|------|--------|------|
# | timeout | int | 30 | 超时时间(秒) |

# 2. JSON 格式 (用于程序处理)
json_data = table.to_json()
# [{"参数名": "timeout", "类型": "int", "默认值": "30", "说明": "超时时间(秒)"}]

# 3. 自然语言描述 (用于检索匹配)
natural = table.to_natural_language()
# "这是一个关于「配置参数说明」的表格。表格共有 10 行 4 列。
#  列包括：参数名、类型、默认值、说明。
#  第1行：参数名是「timeout」，类型是「int」，默认值是「30」，说明是「超时时间(秒)」。"
```

**表格类型自动分类：**

| 类型 | 识别特征 | 示例 |
|------|----------|------|
| `specification` | 参数、配置、规格 | 硬件规格表、配置参数表 |
| `comparison` | 多列对比 | 功能对比表、版本差异表 |
| `mapping` | 代码、ID 映射 | 错误码表、状态码表 |
| `procedure` | 步骤、序号 | 操作步骤表、安装流程表 |
| `data` | 通用数据 | 测试数据表 |

**表格切分策略：**

```python
from src.ingestion import TableChunker, table_to_chunks

# 创建切分器
chunker = TableChunker(
    max_rows_per_chunk=20,      # 大表格分片阈值
    always_include_header=True,  # 分片时保留表头
    generate_row_chunks=True,    # 生成行级 chunks
    generate_summary=True        # 生成表格摘要
)

# 切分表格
chunks = chunker.chunk_table(table)

# 生成三类 chunks:
# 1. 表格整体/分片 (Markdown)
# 2. 行级 chunks (精确匹配)
# 3. 表格摘要 (自然语言)
```

**行级检索 (精确匹配表格数据)：**

```python
# 直接查询表格
results = table.query_by_column("参数名", "timeout")
# [{"参数名": "timeout", "类型": "int", "默认值": "30", "说明": "超时时间(秒)"}]

# 通过向量检索匹配表格中的具体行
# 用户问: "timeout 参数的默认值是多少？"
# 系统能匹配到具体的行级 chunk
```

**表格完整性验证：**

```python
# 验证表格结构
validation = processor.validate_table_structure(table)

print(f"是否有效: {validation['is_valid']}")
print(f"问题列表: {validation['issues']}")
# [{"type": "inconsistent_columns", "message": "列数不一致", "severity": "warning"}]

print(f"统计信息: {validation['stats']}")
# {"rows": 10, "cols": 4, "empty_ratio": 0.05, "has_header": True}
```

**列类型推断：**

系统自动识别列的数据类型，用于智能处理：

| 列类型 | 识别规则 | 示例值 |
|--------|----------|--------|
| `number` | 纯数值 | 100, 3.14, -5 |
| `percentage` | 含 % 符号 | 85%, 12.5% |
| `date` | 日期格式 | 2024-01-15 |
| `boolean` | 是/否/√/× | 是, Yes, ✓ |
| `code` | 代码样式 | `--verbose`, `$HOME` |
| `text` | 其他文本 | 描述性文字 |

---

### 二、检索问答 (RAG Chat)

#### 2.1 基础问答

```bash
# API 调用
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何配置 ProductX 的网络参数？",
    "top_k": 5,
    "use_rerank": true
  }'
```

#### 2.2 Python SDK

```python
from src.generation.rag_chat import RAGChat
from src.retrieval import HybridSearcher, Reranker

# 初始化
rag_chat = RAGChat(
    hybrid_searcher=hybrid_searcher,
    reranker=Reranker()
)

# 问答
response = await rag_chat.chat(
    query="如何配置网络参数？",
    filter_dict={"product": "ProductX"},  # 可选：过滤条件
    top_k=5
)

print(response["answer"])
print(f"引用 {len(response['citations'])} 个来源")

for cite in response["citations"]:
    print(f"  - {cite['source']} (相关度: {cite['score']:.2f})")
```

#### 2.3 🆕 智能查询增强与对话理解

系统支持多轮对话上下文理解和短查询自动增强：

```bash
# 带对话历史的问答
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "它的配置参数有哪些？",
    "top_k": 5,
    "conversation_history": [
      {"role": "user", "content": "OAuth 2.0 认证流程是什么？"},
      {"role": "assistant", "content": "SmartHome Pro 使用 OAuth 2.0 协议..."}
    ]
  }'
```

**查询增强触发条件：**

| 条件类型 | 示例 | 增强效果 |
|---------|------|----------|
| 短查询 | "OAuth" | → "OAuth 是什么，有什么功能及如何使用？" |
| 代词指代 | "它的配置参数？" | → "OAuth 2.0 的配置参数有哪些？" |
| 序数指代 | "回答第一个问题" | → 解析对话历史中的第一个问题 |
| 追问模式 | "继续说说" | → 结合上下文补充完整问题 |

**对话历史智能处理策略：**

```
┌─────────────────────────────────────────────────────────────┐
│                    对话历史智能处理                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  较早的对话 (> 3 轮前)          最近 3 轮对话               │
│  ┌────────────────────┐        ┌────────────────────┐      │
│  │ 提取主题摘要        │        │ 保持完整内容        │      │
│  │ - 用户问题前30字    │   +    │ - 用户消息: 200字   │      │
│  │ - 最多保留5个主题   │        │ - 助手回复: 150字   │      │
│  └────────────────────┘        └────────────────────┘      │
│            ↓                            ↓                   │
│  [历史讨论主题: OAuth、设备API...]  用户: xxx  助手: xxx    │
│                                                             │
│  总长度控制: 最多 1500 字符                                 │
└─────────────────────────────────────────────────────────────┘
```

**支持的上下文模式：**

| 模式类型 | 匹配词 |
|---------|--------|
| 代词指代 | "它"、"这个"、"那个"、"this"、"that" |
| 序数指代 | "第一"、"第二"、"上一个"、"之前" |
| 追问模式 | "还有"、"继续"、"详细说明"、"展开说说" |
| 对比模式 | "和"、"比较"、"区别"、"不同" |

#### 2.4 多路并行检索 (提高召回率)

```bash
# API: 多路并行检索
curl -X POST "http://localhost:8000/api/search/multi" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "API 认证方式",
    "top_k": 10
  }'
```

```python
from src.retrieval import MultiQueryRetriever, QueryTransformer

# 创建多路检索器
multi_retriever = MultiQueryRetriever(
    hybrid_searcher=hybrid_searcher,
    query_transformer=QueryTransformer(
        enable_expansion=True,   # 启用查询扩展
        enable_hyde=True,        # 启用 HyDE
        enable_terminology=True  # 启用术语词典
    ),
    max_parallel_queries=5
)

# 执行多路检索
result = multi_retriever.search("API 认证", top_k=10)

print(f"使用了 {len(result.queries_used)} 个查询版本:")
for q in result.queries_used:
    print(f"  - {q}")
print(f"召回 {len(result.results)} 个结果")
```

---

### 三、文档生成 (Doc Agent)

#### 3.1 生成技术文档

```bash
# API 调用
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "product": "ProductX",
    "doc_type": "installation_guide",
    "target_audience": "developer",
    "output_format": "markdown"
  }'
```

**支持的文档类型：**

| doc_type | 描述 | 典型章节 |
|----------|------|----------|
| `installation_guide` | 安装指南 | 环境要求、安装步骤、验证、常见问题 |
| `api_reference` | API 参考 | 认证、端点、请求/响应、错误码 |
| `user_manual` | 用户手册 | 概述、功能、操作、维护 |
| `release_notes` | 发布说明 | 新功能、改进、修复、已知问题 |
| `troubleshooting` | 故障排除 | 问题分类、诊断步骤、解决方案 |
| `configuration` | 配置指南 | 配置项、参数说明、最佳实践 |
| `quick_start` | 快速入门 | 5分钟上手、基础用法 |

#### 3.2 Python SDK

```python
from src.generation.doc_agent import DocAgent

# 创建文档生成代理
doc_agent = DocAgent(
    hybrid_searcher=hybrid_searcher,
    reranker=reranker
)

# 预览大纲
outline = await doc_agent.preview_outline(
    prompt="基于 ProductX 知识库生成安装指南",
    doc_type="installation_guide"
)

print("生成的大纲:")
for section in outline["sections"]:
    print(f"  {section['title']}")

# 生成完整文档
result = await doc_agent.generate_from_prompt(
    prompt="基于 ProductX 知识库生成安装指南",
    doc_type="installation_guide",
    output_format="markdown",  # 或 "docx", "html"
    output_path="output/installation_guide.md"
)

print(f"✅ 文档生成完成: {result['output_path']}")
print(f"   章节数: {len(result['sections'])}")
print(f"   一致性得分: {result['consistency_report']['overall_score']:.2f}")
```

#### 3.3 一致性检查

生成的文档会自动进行以下检查：

| 检查项 | 说明 |
|--------|------|
| 术语一致性 | 确保使用统一的术语表述 |
| 前后矛盾检测 | 检测章节间的逻辑矛盾 |
| 引用覆盖率 | 确保内容有据可查 |
| 格式规范 | 检查标题层级、列表格式等 |

---

### 四、术语词典

#### 4.1 查看术语扩展

```bash
# 扩展查询术语
curl "http://localhost:8000/api/terminology/expand?query=API认证"

# 响应示例:
# {
#   "original": "API认证",
#   "expanded": [
#     "API认证",
#     "Application Programming Interface认证",
#     "接口认证"
#   ],
#   "count": 3
# }
```

#### 4.2 添加自定义术语

```bash
curl -X POST "http://localhost:8000/api/terminology/add" \
  -H "Content-Type: application/json" \
  -d '{
    "term": "XYZ协议",
    "canonical": "XYZ Protocol",
    "aliases": ["xyz协议", "XYZ通信协议"],
    "abbreviations": ["XYZ"],
    "domain": "protocol"
  }'
```

#### 4.3 Python SDK

```python
from src.retrieval import get_terminology_dict

term_dict = get_terminology_dict()

# 规范化术语
canonical = term_dict.normalize("api")  # -> "Application Programming Interface"

# 扩展查询
expanded = term_dict.expand_query("配置 API")
# -> ["配置 API", "配置 Application Programming Interface", "设置 API"]

# 获取相关术语
related = term_dict.get_related_terms("API")
# -> ["REST API", "GraphQL", "SDK"]

# 添加自定义术语
term_dict.add_custom_term(
    term="MyProduct",
    canonical="MyProduct",
    aliases=["我的产品", "MP"],
    abbreviations=["MP"],
    domain="product"
)
```

---

### 五、召回率监控

#### 5.1 查看实时统计

```bash
curl http://localhost:8000/api/monitor/recall/stats

# 响应示例:
# {
#   "total_queries": 1234,
#   "zero_result_rate": 0.02,
#   "source_distribution": {
#     "hybrid": 0.65,
#     "vector_only": 0.25,
#     "bm25_only": 0.10
#   },
#   "avg_latency_ms": 156.3,
#   "avg_score": 0.72
# }
```

#### 5.2 生成召回率报告

```bash
# 获取过去 24 小时的报告
curl "http://localhost:8000/api/monitor/recall/report?hours=24"
```

#### 5.3 Python SDK

```python
from src.retrieval import RecallMonitor, MonitoredHybridSearcher

# 创建带监控的检索器
monitor = RecallMonitor(
    log_dir="data/recall_logs",
    alert_threshold=0.1,  # 零结果率超过 10% 告警
    enable_file_logging=True
)

monitored_searcher = MonitoredHybridSearcher(
    hybrid_searcher=hybrid_searcher,
    monitor=monitor
)

# 使用带监控的检索
results = monitored_searcher.search("查询内容")

# 获取报告
report = monitor.generate_report(hours=24)
print(f"总查询数: {report.total_queries}")
print(f"零结果率: {report.zero_result_rate:.2%}")
print(f"平均延迟: {report.avg_latency_ms:.1f}ms")

# 查看低召回查询
for query in report.low_recall_queries[:5]:
    print(f"  - {query['query']} (结果数: {query['num_results']})")
```

---

## 🔧 技术实现详解

### 一、PDF 解析 (Layout-Aware)

**核心类：** `src/ingestion/pdf_parser.py`

```python
class PDFParser:
    """
    布局感知的 PDF 解析器
    
    使用 PyMuPDF (fitz) + pdfplumber 组合：
    - PyMuPDF: 高效提取文本块、识别字体大小推断标题层级
    - pdfplumber: 精确提取表格
    """
```

**解析流程：**

```
PDF 文件
    ↓
1. 提取文本块 (TextBlock)
   - 坐标 (bbox)
   - 文本内容
   - 字体大小 → 推断层级
    ↓
2. 识别结构元素
   - 标题 (H1-H6): 字体大小 > 阈值
   - 段落: 正文文本
   - 代码块: 等宽字体
   - 列表: 特殊前缀 (•, -, 1.)
    ↓
3. 提取表格 (pdfplumber)
   - 表格边界检测
   - 单元格提取
   - 转换为 Markdown
    ↓
4. 构建章节树
   - 根据标题层级嵌套
   - 保留父子关系
    ↓
ParsedDocument
```

**关键配置：**

```python
# 标题层级识别阈值
HEADING_SIZE_THRESHOLDS = {
    1: 20,  # H1: 字体 >= 20pt
    2: 16,  # H2: 字体 >= 16pt
    3: 14,  # H3: 字体 >= 14pt
    4: 12,  # H4: 字体 >= 12pt
}
```

---

### 二、智能切分 (Parent-Child Indexing)

**核心类：** `src/ingestion/chunker.py`

**设计思想：**

```
问题: 传统切分面临矛盾
  - 小块 → 精准匹配，但上下文不足
  - 大块 → 上下文完整，但匹配噪声大

解决: 父子索引策略
  - 用小块 (child) 做检索
  - 命中后返回大块 (parent) 做上下文
```

**切分参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `child_chunk_size` | 400 tokens | 子块大小，用于精准检索 |
| `child_chunk_overlap` | 50 tokens | 子块重叠，避免边界截断 |
| `parent_chunk_size` | 1500 tokens | 父块大小，提供完整上下文 |
| `parent_chunk_overlap` | 200 tokens | 父块重叠 |

**代码示例：**

```python
from src.ingestion.chunker import SmartChunker

chunker = SmartChunker(
    child_chunk_size=400,
    parent_chunk_size=1500
)

# 切分文档
child_chunks, parent_chunks = chunker.chunk(parsed_doc)

# 每个 child_chunk 都有 parent_id 指向父块
for child in child_chunks:
    print(f"Child: {child.chunk_id}")
    print(f"Parent: {child.parent_id}")  # 用于检索后扩展上下文
```

---

### 三、混合检索 (Hybrid Search)

**核心类：** `src/retrieval/hybrid_search.py`

**为什么需要混合检索？**

| 场景 | 向量检索 | BM25 | 混合检索 |
|------|----------|------|----------|
| "如何部署应用" | ✅ 语义理解 | ❌ 词汇不匹配 | ✅ |
| "错误码 E1001" | ❌ 语义模糊 | ✅ 精确匹配 | ✅ |
| "配置 timeout 参数" | ⚠️ 部分理解 | ✅ 精确匹配 | ✅ |

**RRF 融合算法：**

```python
def rrf_score(rank: int, k: int = 60) -> float:
    """
    Reciprocal Rank Fusion
    
    RRF(d) = Σ 1/(k + rank_i(d))
    
    k=60 是经验值，平衡不同排序列表的贡献
    """
    return 1.0 / (k + rank)

# 融合示例:
# 文档 A: 向量排名 1, BM25 排名 3
# RRF(A) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

# 文档 B: 向量排名 5, BM25 排名 1  
# RRF(B) = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

# 文档 A 排名更高 (两边都不错 > 一边很好)
```

**加权配置：**

```python
hybrid_searcher = HybridSearcher(
    vector_weight=0.6,  # 向量权重
    bm25_weight=0.4,    # BM25 权重
    rrf_k=60            # RRF 参数
)
```

---

### 四、查询扩展 (Query Expansion)

**核心类：** `src/retrieval/query_transformer.py`

**三种扩展策略：**

#### 1. 术语词典扩展 (快速，无 LLM)

```python
# 输入: "配置 API"
# 输出: [
#   "配置 API",
#   "配置 Application Programming Interface",
#   "设置 API"
# ]
```

#### 2. LLM 驱动扩展

```python
# 输入: "如何配置网络"
# LLM 生成:
# - "网络配置方法"
# - "network configuration"
# - "设置网络参数步骤"
```

#### 3. HyDE (Hypothetical Document Embeddings)

```python
# 输入: "如何配置网络"
# LLM 生成假设文档:
"""
网络配置是系统部署的关键步骤。首先需要确定 IP 地址范围，
然后配置 DNS 服务器。对于防火墙设置，建议开放以下端口...
"""
# 用假设文档的 embedding 去检索，通常比问题本身更匹配文档
```

---

### 五、多路并行召回

**核心类：** `src/retrieval/multi_query_retriever.py`

**工作流程：**

```
原始查询
    ↓
QueryTransformer
    ↓
┌─────────────────────────────────┐
│ 生成多个查询版本:                │
│ 1. 原始查询                      │
│ 2. 术语词典扩展 (缩写→全称)      │
│ 3. LLM 改写版本                  │
│ 4. HyDE 假设文档                 │
└─────────────────────────────────┘
    ↓
并行执行 (ThreadPoolExecutor)
    ↓
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Query1 │ │Query2 │ │Query3 │ │Query4 │
│结果   │ │结果   │ │结果   │ │结果   │
└───────┘ └───────┘ └───────┘ └───────┘
    ↓         ↓         ↓         ↓
         多路 RRF 融合
              ↓
         最终结果
```

**优势：**

| 指标 | 单路检索 | 多路并行 | 提升 |
|------|----------|----------|------|
| Recall@10 | 75% | 89% | +14% |
| 延迟 (P50) | 150ms | 180ms | +20% |
| 零结果率 | 5% | 1.5% | -70% |

---

### 六、重排序 (Reranker)

**核心类：** `src/retrieval/reranker.py`

**为什么需要重排序？**

- 初次检索 (recall): 快速召回候选，可能有噪声
- 重排序 (precision): 精细排序，提升精度

**模型选择：**

| 模型 | 参数量 | 速度 | 精度 |
|------|--------|------|------|
| `BAAI/bge-reranker-v2-m3` | 568M | 中等 | 高 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 快 | 中 |

**使用示例：**

```python
from src.retrieval import Reranker

reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")

# 对检索结果重排序
reranked = reranker.rerank(
    query="如何配置网络",
    results=initial_results,
    top_k=5
)
```

---

### 七、文档生成流程

**核心类：** `src/generation/doc_agent.py`

```
需求输入
    ↓
┌─────────────────────────────────────────┐
│ OutlinePlanner (大纲规划)                │
│ - 根据 doc_type 选择模板                 │
│ - 结合知识库定制章节                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ SectionWriter (分节写作) × N             │
│ - 每节独立检索相关内容                   │
│ - LLM 生成，强制引用标注                 │
│ - Token 预算控制                         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ ConsistencyChecker (一致性检查)          │
│ - 术语一致性                             │
│ - 前后矛盾检测                           │
│ - 引用覆盖率                             │
│ - 格式规范                               │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 格式化输出                               │
│ - Markdown (.md)                         │
│ - Word (.docx)                           │
│ - HTML (.html)                           │
└─────────────────────────────────────────┘
```

---

## 📊 评估指标

### RAGAS 框架指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Faithfulness** | 生成内容是否忠实于检索结果 | > 0.9 |
| **Answer Relevance** | 是否真正回答了用户问题 | > 0.85 |
| **Context Precision** | 检索结果是否都有用 | > 0.8 |
| **Context Recall** | 是否检索到了关键信息 | > 0.85 |

### Hit Rate 评估

```python
from src.utils.evaluator import HitRateEvaluator

evaluator = HitRateEvaluator(retriever)

results = evaluator.evaluate(
    test_cases=[
        {"question": "如何安装?", "relevant_chunk_ids": ["chunk_001", "chunk_002"]},
        {"question": "配置参数", "relevant_chunk_ids": ["chunk_010"]},
    ],
    k_values=[1, 3, 5, 10]
)

# 输出:
# {
#   "hit_rate@1": 0.65,
#   "hit_rate@3": 0.82,
#   "hit_rate@5": 0.91,
#   "hit_rate@10": 0.97
# }
```

---

## ⚙️ 配置参考

### 完整配置项

```bash
# ==================== LLM ====================
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx

# 写作模型 (推荐 Claude 3.5 Sonnet)
LLM_WRITING_PROVIDER=anthropic
LLM_WRITING_MODEL=claude-3-5-sonnet-20241022

# 规划模型 (推荐 o3-mini)
LLM_PLANNING_PROVIDER=openai
LLM_PLANNING_MODEL=o3-mini

# ==================== Embedding ====================
EMBEDDING_PROVIDER=huggingface  # openai, huggingface, local
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DIMENSION=1024

# ==================== 向量库 ====================
VECTOR_PROVIDER=chroma  # chroma, milvus
VECTOR_CHROMA_PERSIST_DIR=./data/chroma
VECTOR_CHROMA_COLLECTION_NAME=tech_docs

# ==================== BM25 ====================
BM25_PROVIDER=memory  # memory, elasticsearch
BM25_PERSIST_PATH=./data/bm25_index.pkl

# ==================== Reranker ====================
RERANKER_ENABLED=true
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

# ==================== 切分 ====================
CHUNKING_CHILD_CHUNK_SIZE=400
CHUNKING_PARENT_CHUNK_SIZE=1500

# ==================== 检索 ====================
RETRIEVAL_VECTOR_WEIGHT=0.6
RETRIEVAL_BM25_WEIGHT=0.4
RETRIEVAL_INITIAL_TOP_K=30
RETRIEVAL_FINAL_TOP_K=10
```

### 模型推荐

| 任务 | 推荐模型 | 备选 | 说明 |
|------|----------|------|------|
| Embedding | `BAAI/bge-m3` | `text-embedding-3-large` | 中英双语，1024维 |
| Reranker | `BAAI/bge-reranker-v2-m3` | `cross-encoder` | 多语言支持 |
| 大纲规划 | `o3-mini` | `claude-3-5-sonnet` | 需要强推理 |
| 内容写作 | `claude-3-5-sonnet` | `gpt-4.1` | 写作质量高 |
| PDF 解析 | `gemini-1.5-pro` | - | 超长上下文，多模态 |

---

## 🔌 API 参考

### 核心端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/ingest` | 摄取文档 (路径) |
| `POST` | `/api/ingest/upload` | 上传并摄取 |
| `POST` | `/api/search` | 混合检索 |
| `POST` | `/api/search/multi` | 多路并行检索 |
| `POST` | `/api/chat` | RAG 问答 |
| `POST` | `/api/generate` | 生成文档 |
| `GET` | `/api/generate/outline` | 预览大纲 |

### 监控端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/monitor/recall/stats` | 召回统计 |
| `GET` | `/api/monitor/recall/report` | 召回报告 |
| `POST` | `/api/monitor/recall/reset` | 重置统计 |

### 术语端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/terminology/expand` | 术语扩展 |
| `GET` | `/api/terminology/domains` | 领域列表 |
| `POST` | `/api/terminology/add` | 添加术语 |

---

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_ingestion.py -v
pytest tests/test_api.py -v

# 带覆盖率
pytest --cov=src --cov-report=html
```

---

## 📈 性能基准

测试环境: 4 vCPU, 16GB RAM, NVIDIA T4

| 操作 | 延迟 (P50) | 延迟 (P99) | 吞吐量 |
|------|------------|------------|--------|
| PDF 摄取 (10页) | 2.3s | 5.1s | - |
| 混合检索 | 156ms | 380ms | 50 QPS |
| 多路检索 (5路) | 220ms | 520ms | 30 QPS |
| RAG 问答 | 1.8s | 3.5s | 10 QPS |
| 文档生成 (5章) | 45s | 90s | - |

---

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件
