"""
Generation 模块
"""
from .rag_chat import RAGChat, ChatResponse, Reference
from .outline_planner import OutlinePlanner, DocumentOutline, SectionPlan
from .section_writer import SectionWriter, SectionContent, GeneratedDocument
from .consistency_checker import ConsistencyChecker, ConsistencyReport, ConsistencyIssue
from .doc_agent import DocAgent, GenerationConfig, GenerationResult, generate_document
from .data_driven_writer import DataDrivenWriter, DataDrivenDocument, DataSection

__all__ = [
    "RAGChat",
    "ChatResponse",
    "Reference",
    "OutlinePlanner",
    "DocumentOutline",
    "SectionPlan",
    "SectionWriter",
    "SectionContent",
    "GeneratedDocument",
    "ConsistencyChecker",
    "ConsistencyReport",
    "ConsistencyIssue",
    "DocAgent",
    "GenerationConfig",
    "GenerationResult",
    "generate_document",
    "DataDrivenWriter",
    "DataDrivenDocument",
    "DataSection"
]
