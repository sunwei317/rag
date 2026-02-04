#!/usr/bin/env python3
"""
Graph RAG 简化测试脚本
直接测试知识图谱核心功能
"""
import sys
import os
sys.path.insert(0, '/home/wei/rag')
os.chdir('/home/wei/rag')

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_graph_store():
    """测试图存储"""
    print_section("1. 图存储测试")
    
    from src.knowledge_graph.graph_store import InMemoryGraphStore
    from src.knowledge_graph.entity_extractor import Entity, EntityType
    from src.knowledge_graph.relation_builder import Relation, RelationType
    
    graph = InMemoryGraphStore()
    
    # 创建测试实体
    e1 = Entity(
        entity_id="e1",
        name="SmartHome Pro",
        entity_type=EntityType.PRODUCT,
        description="智能家居控制平台"
    )
    e2 = Entity(
        entity_id="e2", 
        name="OAuth 2.0",
        entity_type=EntityType.CONCEPT,
        description="认证协议"
    )
    e3 = Entity(
        entity_id="e3",
        name="Zigbee 3.0",
        entity_type=EntityType.CONCEPT,
        description="无线通信协议"
    )
    e4 = Entity(
        entity_id="e4",
        name="E1001",
        entity_type=EntityType.ERROR,
        description="设备离线错误"
    )
    
    # 添加实体
    for e in [e1, e2, e3, e4]:
        graph.add_entity(e)
    print(f"✅ 添加了 4 个实体")
    
    # 创建关系
    r1 = Relation(
        relation_id="r1",
        source_id="e1",
        target_id="e2",
        relation_type=RelationType.USES
    )
    r2 = Relation(
        relation_id="r2",
        source_id="e1",
        target_id="e3",
        relation_type=RelationType.USES
    )
    r3 = Relation(
        relation_id="r3",
        source_id="e4",
        target_id="e3",
        relation_type=RelationType.RELATED_TO
    )
    
    for r in [r1, r2, r3]:
        graph.add_relation(r)
    print(f"✅ 添加了 3 个关系")
    
    # 获取统计
    stats = graph.get_stats()
    print(f"📊 图谱统计: {stats}")
    
    # 测试查询
    entity = graph.get_entity("e1")
    print(f"✅ 查询实体 e1: {entity.name if entity else 'None'}")
    
    neighbors, relations = graph.get_neighbors("e1", max_depth=1)
    print(f"✅ e1 的邻居: {[n.name for n in neighbors]}")
    
    return graph

def test_entity_extraction_llm():
    """测试 LLM 实体抽取"""
    print_section("2. LLM 实体抽取测试")
    
    from src.knowledge_graph.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    
    text = """
    SmartHome Pro v2.5.0 是一个智能家居控制平台。
    它使用 OAuth 2.0 协议进行用户认证，支持授权码模式和客户端凭证模式。
    设备通过 Zigbee 3.0 协议通信，最多支持 200 个设备。
    如果设备离线，系统会返回错误码 E1001。
    配置文件为 config.yaml，日志级别可设置为 DEBUG 或 INFO。
    """
    
    print(f"📝 输入文本: {text[:100]}...")
    
    try:
        entities = extractor.extract_from_text(text, "test_chunk")
        print(f"✅ 抽取了 {len(entities)} 个实体:")
        for e in entities:
            print(f"   - {e.entity_type.value}: {e.name}")
            if e.description:
                print(f"     描述: {e.description[:50]}...")
        return entities
    except Exception as e:
        print(f"❌ 抽取失败: {e}")
        return []

def test_retrieval(graph):
    """测试图检索"""
    print_section("3. 图检索测试")
    
    from src.knowledge_graph.graph_retriever import GraphRetriever
    
    retriever = GraphRetriever(graph)
    
    queries = [
        "SmartHome Pro 使用什么认证协议？",
        "Zigbee 相关的错误有哪些？",
        "设备离线怎么办？",
    ]
    
    for query in queries:
        print(f"\n🔍 查询: {query}")
        try:
            result = retriever.retrieve(query, expand_depth=2)
            
            if result.is_empty():
                print("   没有找到相关实体")
            else:
                print(f"   找到 {len(result.entities)} 个实体, {len(result.relations)} 个关系")
                for e in result.entities[:3]:
                    print(f"   - {e.name} ({e.entity_type.value})")
                
                context = result.to_context_text(max_length=300)
                print(f"   上下文: {context[:150]}...")
        except Exception as e:
            print(f"   ❌ 查询失败: {e}")

def test_end_to_end_with_api():
    """测试通过 API 的端到端流程"""
    print_section("4. API 端到端测试")
    

    import requests
    from config.settings import settings
    base_url = f"http://{getattr(settings, 'api_host', 'localhost')}:{getattr(settings, 'api_port', 8000)}"
    
    # 检查服务状态
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code != 200:
            print("❌ API 服务未运行")
            return
        print("✅ API 服务正常")
    except:
        print("❌ 无法连接到 API 服务")
        return
    
    # 测试问答（使用已有知识库）
    questions = [
        "什么是 RAG？",
        "如何上传文档？",
    ]
    
    for q in questions:
        print(f"\n❓ 问题: {q}")
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"question": q},
                timeout=60
            )
            if resp.status_code == 200:
                answer = resp.json().get("answer", "")
                print(f"💬 回答: {answer[:200]}...")
            else:
                print(f"   状态码: {resp.status_code}")
        except Exception as e:
            print(f"   错误: {e}")

def main():
    print("\n" + "="*60)
    print("  Graph RAG 功能测试")
    print("="*60)
    
    # 1. 图存储测试
    graph = test_graph_store()
    
    # 2. LLM 实体抽取测试  
    test_entity_extraction_llm()
    
    # 3. 图检索测试
    test_retrieval(graph)
    
    # 4. API 端到端测试
    test_end_to_end_with_api()
    
    print_section("测试完成")
    print("✅ Graph RAG 核心功能测试完成！")

if __name__ == "__main__":
    main()
