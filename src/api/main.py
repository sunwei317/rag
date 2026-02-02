"""
FastAPI 服务
提供 RESTful API 接口
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pathlib import Path
import shutil
import uuid
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import settings


# ==================== Pydantic Models ====================

class IngestRequest(BaseModel):
    """文档摄取请求"""
    pdf_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "pdf_path": "/path/to/document.pdf",
                "metadata": {
                    "product": "ProductX",
                    "version": "1.0.0",
                    "doc_type": "user_manual"
                }
            }
        }


class ChatRequest(BaseModel):
    """问答请求"""
    question: str
    filter: Optional[Dict[str, Any]] = None
    top_k: int = 5
    use_rerank: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "如何配置网络参数？",
                "filter": {"product": "ProductX"},
                "top_k": 5
            }
        }


class ChatResponse(BaseModel):
    """问答响应"""
    answer: str
    references: List[Dict[str, Any]]
    query: str


class GenerateRequest(BaseModel):
    """文档生成请求"""
    product: str
    doc_type: str
    target_audience: str = "developer"
    version: Optional[str] = None
    additional_requirements: Optional[str] = None
    output_format: str = "markdown"
    
    class Config:
        json_schema_extra = {
            "example": {
                "product": "ProductX",
                "doc_type": "installation_guide",
                "target_audience": "developer",
                "output_format": "markdown"
            }
        }


class GenerateResponse(BaseModel):
    """文档生成响应"""
    title: str
    output_path: str
    sections_count: int
    consistency_passed: bool
    issues_count: int


class OutlinePreviewRequest(BaseModel):
    """大纲预览请求"""
    product: str
    doc_type: str
    target_audience: str = "developer"


class SearchRequest(BaseModel):
    """检索请求"""
    query: str
    top_k: int = 10
    filter: Optional[Dict[str, Any]] = None
    method: str = "hybrid"  # hybrid, vector, bm25


class SearchResponse(BaseModel):
    """检索响应"""
    results: List[Dict[str, Any]]
    total: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    components: Dict[str, bool]


# ==================== Application ====================

app = FastAPI(
    title="技术文档 RAG 系统",
    description="基于 RAG 的技术文档问答与生成系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 全局变量 (懒加载) ====================

_components = {
    "vector_store": None,
    "bm25_store": None,
    "embedder": None,
    "hybrid_searcher": None,
    "reranker": None,
    "rag_chat": None,
    "doc_agent": None,
    "ingestion_pipeline": None
}


def get_components():
    """懒加载组件"""
    if _components["vector_store"] is None:
        _init_components()
    return _components


def _init_components():
    """初始化组件"""
    logger.info("Initializing components...")
    
    from src.storage import create_vector_store, create_bm25_store
    from src.ingestion import Embedder, IngestionPipeline
    from src.retrieval import HybridSearcher, Reranker
    from src.generation import RAGChat, DocAgent
    
    # 存储
    _components["vector_store"] = create_vector_store(
        provider=settings.vector_store.provider,
        collection_name=settings.vector_store.chroma_collection_name,
        persist_directory=settings.vector_store.chroma_persist_dir
    )
    
    _components["bm25_store"] = create_bm25_store(
        provider=settings.bm25.provider,
        persist_path=settings.bm25.persist_path
    )
    
    # Embedder
    _components["embedder"] = Embedder(
        provider=settings.embedding.provider,
        model_name=settings.embedding.model_name
    )
    
    # 检索
    _components["hybrid_searcher"] = HybridSearcher(
        vector_store=_components["vector_store"],
        bm25_store=_components["bm25_store"],
        embedder=_components["embedder"],
        vector_weight=settings.retrieval.vector_weight,
        bm25_weight=settings.retrieval.bm25_weight
    )
    
    # Reranker
    if settings.reranker.enabled:
        try:
            _components["reranker"] = Reranker(
                model_name=settings.reranker.model_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
    
    # RAG Chat
    _components["rag_chat"] = RAGChat(
        hybrid_searcher=_components["hybrid_searcher"],
        reranker=_components["reranker"]
    )
    
    # Doc Agent
    _components["doc_agent"] = DocAgent(
        hybrid_searcher=_components["hybrid_searcher"],
        reranker=_components["reranker"]
    )
    
    # Ingestion Pipeline
    _components["ingestion_pipeline"] = IngestionPipeline(
        vector_store=_components["vector_store"],
        bm25_store=_components["bm25_store"],
        embedder=_components["embedder"]
    )
    
    logger.info("Components initialized successfully")


# ==================== 路由 ====================

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "技术文档 RAG 系统",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    components_status = {
        "vector_store": _components.get("vector_store") is not None,
        "bm25_store": _components.get("bm25_store") is not None,
        "embedder": _components.get("embedder") is not None,
        "reranker": _components.get("reranker") is not None
    }
    
    return HealthResponse(
        status="healthy" if all(components_status.values()) else "partial",
        version="1.0.0",
        components=components_status
    )


# ==================== 摄取接口 ====================

@app.post("/api/ingest", tags=["Ingestion"])
async def ingest_document(request: IngestRequest):
    """
    摄取文档
    
    将 PDF 文档解析、切分、向量化并存入知识库
    """
    components = get_components()
    pipeline = components["ingestion_pipeline"]
    
    if not request.pdf_path:
        raise HTTPException(status_code=400, detail="pdf_path is required")
    
    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")
    
    try:
        stats = pipeline.ingest_pdf(pdf_path, request.metadata)
        
        return {
            "success": True,
            "doc_id": stats.doc_id,
            "filename": stats.filename,
            "stats": {
                "total_pages": stats.total_pages,
                "parent_chunks": stats.parent_chunks,
                "child_chunks": stats.child_chunks,
                "total_tokens": stats.total_tokens
            }
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/upload", tags=["Ingestion"])
async def upload_and_ingest(
    file: UploadFile = File(...),
    product: Optional[str] = None,
    version: Optional[str] = None,
    doc_type: Optional[str] = None
):
    """
    上传并摄取文档
    """
    components = get_components()
    pipeline = components["ingestion_pipeline"]
    
    # 保存上传的文件
    upload_dir = settings.pdf_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 摄取
        metadata = {}
        if product:
            metadata["product"] = product
        if version:
            metadata["version"] = version
        if doc_type:
            metadata["doc_type"] = doc_type
        
        stats = pipeline.ingest_pdf(file_path, metadata)
        
        return {
            "success": True,
            "doc_id": stats.doc_id,
            "filename": file.filename,
            "stats": {
                "total_pages": stats.total_pages,
                "parent_chunks": stats.parent_chunks,
                "child_chunks": stats.child_chunks
            }
        }
    except Exception as e:
        logger.error(f"Upload and ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 检索接口 ====================

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    检索文档
    
    支持混合检索、向量检索、BM25 检索
    """
    components = get_components()
    searcher = components["hybrid_searcher"]
    
    try:
        if request.method == "hybrid":
            results = searcher.search(
                query=request.query,
                top_k=request.top_k,
                filter_dict=request.filter
            )
        elif request.method == "vector":
            results = searcher.vector_only_search(
                query=request.query,
                top_k=request.top_k,
                filter_dict=request.filter
            )
        elif request.method == "bm25":
            results = searcher.bm25_only_search(
                query=request.query,
                top_k=request.top_k,
                filter_dict=request.filter
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
        
        return SearchResponse(
            results=[r.to_dict() for r in results],
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 问答接口 ====================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    RAG 问答
    
    基于检索到的文档内容回答问题
    """
    components = get_components()
    rag_chat = components["rag_chat"]
    
    try:
        response = rag_chat.ask(
            question=request.question,
            filter_dict=request.filter,
            top_k=request.top_k,
            use_rerank=request.use_rerank
        )
        
        return ChatResponse(
            answer=response.answer,
            references=[r.to_dict() for r in response.references],
            query=response.query
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 文档生成接口 ====================

@app.post("/api/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_document(request: GenerateRequest):
    """
    生成技术文档
    
    根据产品和文档类型自动生成技术文档
    """
    components = get_components()
    doc_agent = components["doc_agent"]
    
    from src.generation import GenerationConfig
    
    try:
        config = GenerationConfig(
            product=request.product,
            doc_type=request.doc_type,
            target_audience=request.target_audience,
            version=request.version,
            additional_requirements=request.additional_requirements,
            output_format=request.output_format
        )
        
        result = doc_agent.generate(config)
        
        return GenerateResponse(
            title=result.document.title,
            output_path=result.output_path or "",
            sections_count=len(result.document.sections),
            consistency_passed=result.consistency_report.passed if result.consistency_report else True,
            issues_count=result.consistency_report.total_issues if result.consistency_report else 0
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/preview-outline", tags=["Generation"])
async def preview_outline(request: OutlinePreviewRequest):
    """
    预览文档大纲
    
    在生成文档前预览大纲结构
    """
    components = get_components()
    doc_agent = components["doc_agent"]
    
    try:
        outline_md = doc_agent.preview_outline(
            product=request.product,
            doc_type=request.doc_type,
            target_audience=request.target_audience
        )
        
        return {"outline": outline_md}
    except Exception as e:
        logger.error(f"Outline preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/generate/doc-types", tags=["Generation"])
async def list_doc_types():
    """
    列出支持的文档类型
    """
    return {
        "doc_types": [
            {"id": "installation_guide", "name": "安装指南"},
            {"id": "api_reference", "name": "API 参考文档"},
            {"id": "user_manual", "name": "用户手册"},
            {"id": "release_notes", "name": "发布说明"},
            {"id": "troubleshooting", "name": "故障排除指南"},
            {"id": "configuration", "name": "配置指南"},
            {"id": "quick_start", "name": "快速入门指南"}
        ]
    }


@app.get("/api/generate/download/{filename}", tags=["Generation"])
async def download_document(filename: str):
    """
    下载生成的文档
    """
    output_dir = Path("./output")
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# ==================== 知识库管理接口 ====================

@app.get("/api/kb/stats", tags=["Knowledge Base"])
async def get_kb_stats():
    """
    获取知识库统计信息
    """
    components = get_components()
    
    vector_count = 0
    bm25_count = 0
    
    try:
        if components["vector_store"]:
            vector_count = components["vector_store"].count
    except:
        pass
    
    try:
        if components["bm25_store"]:
            bm25_count = components["bm25_store"].count
    except:
        pass
    
    return {
        "vector_store": {
            "count": vector_count,
            "provider": settings.vector_store.provider
        },
        "bm25_store": {
            "count": bm25_count,
            "provider": settings.bm25.provider
        }
    }


# ==================== 召回率监控接口 ====================

@app.get("/api/monitor/recall/stats", tags=["Monitoring"])
async def get_recall_stats():
    """
    获取召回率统计摘要
    
    返回当前检索的实时统计数据
    """
    from src.retrieval import get_recall_monitor
    
    monitor = get_recall_monitor()
    return monitor.get_stats_summary()


@app.get("/api/monitor/recall/report", tags=["Monitoring"])
async def get_recall_report(hours: int = 24):
    """
    获取召回率报告
    
    生成指定时间范围内的召回率分析报告
    
    Args:
        hours: 报告时间范围 (小时)，默认 24 小时
    """
    from src.retrieval import get_recall_monitor
    
    monitor = get_recall_monitor()
    report = monitor.generate_report(hours=hours)
    
    return report.to_dict()


@app.post("/api/monitor/recall/reset", tags=["Monitoring"])
async def reset_recall_stats():
    """
    重置召回率统计
    
    清空当前的统计计数器 (不影响已持久化的日志)
    """
    from src.retrieval import get_recall_monitor
    
    monitor = get_recall_monitor()
    monitor.reset_stats()
    
    return {"message": "Recall stats reset successfully"}


# ==================== 多路召回接口 ====================

@app.post("/api/search/multi", tags=["Search"])
async def multi_query_search(request: SearchRequest):
    """
    多路并行检索
    
    使用查询扩展和多路召回提高检索质量
    """
    components = get_components()
    
    try:
        from src.retrieval import MultiQueryRetriever, QueryTransformer
        
        query_transformer = QueryTransformer(
            enable_expansion=True,
            enable_hyde=True,
            enable_terminology=True
        )
        
        multi_retriever = MultiQueryRetriever(
            hybrid_searcher=components["hybrid_searcher"],
            query_transformer=query_transformer,
            include_hyde=True
        )
        
        result = multi_retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_dict=request.filter
        )
        
        return {
            "results": [r.to_dict() for r in result.results],
            "total": len(result.results),
            "queries_used": result.queries_used,
            "stats": result.query_stats
        }
    except Exception as e:
        logger.error(f"Multi-query search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 术语词典接口 ====================

@app.get("/api/terminology/expand", tags=["Terminology"])
async def expand_query_terms(query: str):
    """
    扩展查询术语
    
    使用术语词典扩展查询中的术语
    """
    from src.retrieval import get_terminology_dict
    
    term_dict = get_terminology_dict()
    expanded = term_dict.expand_query(query)
    
    return {
        "original": query,
        "expanded": expanded,
        "count": len(expanded)
    }


@app.get("/api/terminology/domains", tags=["Terminology"])
async def list_terminology_domains():
    """
    列出术语词典的所有领域
    """
    from src.retrieval import get_terminology_dict
    
    term_dict = get_terminology_dict()
    domains = term_dict.list_domains()
    
    return {
        "domains": domains,
        "terms_per_domain": {
            d: len(term_dict.get_domain_terms(d)) for d in domains
        }
    }


@app.post("/api/terminology/add", tags=["Terminology"])
async def add_custom_term(
    term: str,
    canonical: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    abbreviations: Optional[List[str]] = None,
    domain: str = "custom"
):
    """
    添加自定义术语
    """
    from src.retrieval import get_terminology_dict
    
    term_dict = get_terminology_dict()
    term_dict.add_custom_term(
        term=term,
        canonical=canonical,
        aliases=aliases or [],
        abbreviations=abbreviations or [],
        domain=domain
    )
    
    return {"message": f"Term '{term}' added successfully"}


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup():
    """应用启动"""
    logger.info("Starting RAG API server...")
    settings.ensure_dirs()


@app.on_event("shutdown")
async def shutdown():
    """应用关闭"""
    logger.info("Shutting down RAG API server...")


# ==================== 主入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
