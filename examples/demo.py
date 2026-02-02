#!/usr/bin/env python3
"""
技术文档 RAG 系统 - 演示脚本

展示完整的工作流程:
1. 文档导入
2. RAG 对话
3. 文档生成
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.ingestion.pipeline import IngestionPipeline
from src.storage.vector_store import create_vector_store
from src.storage.bm25_store import create_bm25_store
from src.storage.metadata_store import MetadataStore
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.generation.rag_chat import RAGChat
from src.generation.doc_agent import DocAgent


async def demo_ingestion(pdf_path: str):
    """演示文档导入流程"""
    print("=" * 60)
    print("📥 文档导入演示")
    print("=" * 60)
    
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return None
    
    # 初始化存储
    vector_store = create_vector_store()
    bm25_store = create_bm25_store()
    metadata_store = MetadataStore()
    
    # 创建导入管道
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        bm25_store=bm25_store,
        metadata_store=metadata_store
    )
    
    # 导入文档
    print(f"正在处理文档: {pdf_path}")
    result = await pipeline.ingest_pdf(pdf_path)
    
    print(f"✅ 导入完成!")
    print(f"   文档ID: {result['document_id']}")
    print(f"   子块数量: {result['child_chunks_count']}")
    print(f"   父块数量: {result['parent_chunks_count']}")
    
    return {
        "vector_store": vector_store,
        "bm25_store": bm25_store,
        "metadata_store": metadata_store,
        "document_id": result["document_id"]
    }


async def demo_rag_chat(stores: dict):
    """演示 RAG 对话"""
    print("\n" + "=" * 60)
    print("💬 RAG 对话演示")
    print("=" * 60)
    
    # 初始化组件
    reranker = Reranker() if settings.reranker.enabled else None
    
    hybrid_searcher = HybridSearcher(
        vector_store=stores["vector_store"],
        bm25_store=stores["bm25_store"],
        metadata_store=stores["metadata_store"],
        reranker=reranker
    )
    
    rag_chat = RAGChat(
        hybrid_searcher=hybrid_searcher,
        metadata_store=stores["metadata_store"]
    )
    
    # 演示问答
    questions = [
        "这份文档的主要内容是什么?",
        "请解释文档中的关键概念",
    ]
    
    for question in questions:
        print(f"\n🤔 问题: {question}")
        print("-" * 40)
        
        response = await rag_chat.chat(question)
        
        print(f"📝 回答: {response['answer'][:500]}...")
        if response.get("citations"):
            print(f"📚 引用来源: {len(response['citations'])} 个")
            for i, cite in enumerate(response["citations"][:3], 1):
                print(f"   [{i}] {cite.get('source', 'Unknown')}")


async def demo_doc_generation(stores: dict):
    """演示文档生成"""
    print("\n" + "=" * 60)
    print("📄 文档生成演示")
    print("=" * 60)
    
    # 初始化组件
    reranker = Reranker() if settings.reranker.enabled else None
    
    hybrid_searcher = HybridSearcher(
        vector_store=stores["vector_store"],
        bm25_store=stores["bm25_store"],
        metadata_store=stores["metadata_store"],
        reranker=reranker
    )
    
    # 创建文档生成代理
    doc_agent = DocAgent(
        hybrid_searcher=hybrid_searcher,
        metadata_store=stores["metadata_store"]
    )
    
    # 生成文档
    prompt = "基于知识库内容,生成一份技术概述文档"
    print(f"📋 生成提示: {prompt}")
    print("-" * 40)
    
    # 先预览大纲
    print("\n📑 生成大纲...")
    outline = await doc_agent.preview_outline(
        prompt=prompt,
        doc_type="technical_spec"
    )
    
    print("生成的大纲:")
    for section in outline.get("sections", [])[:5]:
        print(f"   • {section.get('title', 'Untitled')}")
    
    # 生成完整文档
    print("\n📝 生成完整文档...")
    result = await doc_agent.generate_from_prompt(
        prompt=prompt,
        doc_type="technical_spec",
        output_format="markdown"
    )
    
    # 保存输出
    output_path = Path("output/generated_doc.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result["content"])
    
    print(f"✅ 文档生成完成!")
    print(f"   输出文件: {output_path}")
    print(f"   章节数量: {len(result.get('sections', []))}")
    
    # 显示一致性检查结果
    if result.get("consistency_report"):
        report = result["consistency_report"]
        print(f"\n📊 一致性检查:")
        print(f"   整体得分: {report.get('overall_score', 0):.2f}")
        if report.get("issues"):
            print(f"   发现问题: {len(report['issues'])} 个")


async def interactive_mode(stores: dict):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("🎯 进入交互式对话模式")
    print("   输入问题进行对话,输入 'quit' 退出")
    print("=" * 60)
    
    reranker = Reranker() if settings.reranker.enabled else None
    
    hybrid_searcher = HybridSearcher(
        vector_store=stores["vector_store"],
        bm25_store=stores["bm25_store"],
        metadata_store=stores["metadata_store"],
        reranker=reranker
    )
    
    rag_chat = RAGChat(
        hybrid_searcher=hybrid_searcher,
        metadata_store=stores["metadata_store"]
    )
    
    while True:
        try:
            question = input("\n🤔 你的问题: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                print("👋 再见!")
                break
            
            if not question:
                continue
            
            response = await rag_chat.chat(question)
            print(f"\n📝 回答:\n{response['answer']}")
            
            if response.get("citations"):
                print(f"\n📚 参考来源 ({len(response['citations'])} 个):")
                for i, cite in enumerate(response["citations"][:5], 1):
                    source = cite.get("source", "Unknown")
                    print(f"   [{i}] {source}")
        
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break


async def main():
    """主函数"""
    print("🚀 技术文档 RAG 系统 - 演示脚本")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # 导入文档
        stores = await demo_ingestion(pdf_path)
        
        if stores:
            # 演示 RAG 对话
            await demo_rag_chat(stores)
            
            # 演示文档生成
            await demo_doc_generation(stores)
            
            # 进入交互模式
            await interactive_mode(stores)
    else:
        print("使用方法: python demo.py <pdf_path>")
        print("\n示例:")
        print("  python demo.py ./docs/example.pdf")
        print("\n或者运行测试模式 (无需文档):")
        
        # 简单测试 - 只检查导入
        print("\n🔧 检查系统组件...")
        
        try:
            vector_store = create_vector_store()
            print("  ✅ 向量存储初始化成功")
        except Exception as e:
            print(f"  ❌ 向量存储初始化失败: {e}")
        
        try:
            bm25_store = create_bm25_store()
            print("  ✅ BM25 存储初始化成功")
        except Exception as e:
            print(f"  ❌ BM25 存储初始化失败: {e}")
        
        try:
            metadata_store = MetadataStore()
            print("  ✅ 元数据存储初始化成功")
        except Exception as e:
            print(f"  ❌ 元数据存储初始化失败: {e}")
        
        print("\n系统就绪! 请提供 PDF 文档开始使用。")


if __name__ == "__main__":
    asyncio.run(main())
