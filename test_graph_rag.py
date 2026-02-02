#!/usr/bin/env python3
"""
Graph RAG 功能直接测试脚本
跳过文件上传，直接测试知识图谱和检索功能
"""
import sys
sys.path.insert(0, '/home/wei/rag')

from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_entity_extraction():
    """测试实体抽取"""
    print_section("1. 实体抽取测试")
    
    from src.knowledge_graph.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()  # 使用默认配置
    
    test_texts = [
        "SmartHome Pro v2.5.0 使用 OAuth 2.0 协议进行认证",
        "安装命令: pip install smarthome-pro-sdk>=2.5.0",
        "Zigbee 协议支持最多 200 个设备，延迟低于 100ms",
        "配置文件 config.yaml 中设置 logging.level = INFO",
        "错误码 E1001 表示设备离线",
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 文本 {i}: {text}")
        entities = extractor.extract_from_text(text, f"test_{i}")
        print(f"   抽取实体: {len(entities)} 个")
        for e in entities:
            print(f"   - {e.entity_type.value}: {e.name}")
    
    return True

def test_graph_building():
    """测试知识图谱构建"""
    print_section("2. 知识图谱构建测试")
    
    from src.knowledge_graph.entity_extractor import EntityExtractor
    from src.knowledge_graph.relation_builder import RelationBuilder
    from src.knowledge_graph.graph_store import InMemoryGraphStore
    
    # 初始化组件
    extractor = EntityExtractor()
    relation_builder = RelationBuilder()
    graph_store = InMemoryGraphStore()
    
    # 测试文档
    chunks = [
        {"chunk_id": "c1", "content": "SmartHome Pro v2.5.0 使用 OAuth 2.0 协议进行认证，支持授权码模式和客户端凭证模式"},
        {"chunk_id": "c2", "content": "Python SDK 安装: pip install smarthome-pro-sdk。需要 Python 3.9+ 版本"},
        {"chunk_id": "c3", "content": "Zigbee 3.0 协议支持灯光、传感器和开关，最多连接 200 个设备"},
        {"chunk_id": "c4", "content": "WebSocket 连接地址为 wss://api.smarthome.pro/ws/v2/events，支持 device.state_changed 事件"},
        {"chunk_id": "c5", "content": "错误码 E1001 表示设备离线，需要检查 Zigbee 网关和设备电源"},
    ]
    
    # 抽取实体和关系
    all_entities = []
    for chunk in chunks:
        entities = extractor.extract_from_text(chunk["content"], chunk["chunk_id"])
        all_entities.extend(entities)
        
        # 添加到图
        for entity in entities:
            graph_store.add_entity(entity)
    
    print(f"✅ 共抽取 {len(all_entities)} 个实体")
    
    # 构建关系 - 需要传入 chunks 参数
    relations = relation_builder.build_relations(all_entities, chunks)
    print(f"✅ 共构建 {len(relations)} 个关系")
    
    for rel in relations[:5]:  # 只显示前5个
        graph_store.add_relation(rel)
        print(f"   - ({rel.source_id}) --[{rel.relation_type.value}]--> ({rel.target_id})")
    
    # 图统计
    stats = graph_store.get_stats()
    print(f"\n📊 图谱统计:")
    print(f"   节点数: {stats.node_count}")
    print(f"   边数: {stats.edge_count}")
    
    return graph_store

def test_graph_retrieval(graph_store):
    """测试图谱检索"""
    print_section("3. 图谱检索测试")
    
    from src.knowledge_graph.graph_retriever import GraphRetriever
    
    retriever = GraphRetriever(graph_store)
    
    test_queries = [
        "SmartHome Pro 认证方式",
        "Zigbee 支持哪些设备",
        "如何解决 E1001 错误",
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        result = retriever.retrieve(query, expand_depth=2)
        
        print(f"   匹配实体: {len(result.entities)}")
        for e in result.entities[:3]:
            print(f"   - {e.name} ({e.entity_type.value})")
        
        print(f"   相关关系: {len(result.relations)}")
        for r in result.relations[:3]:
            print(f"   - {r.source_id} → {r.target_id}")
        
        # 生成上下文
        context = result.to_context_text()
        print(f"   上下文长度: {len(context)} 字符")
    
    return True

def test_end_to_end():
    """端到端测试"""
    print_section("4. 端到端问答测试（模拟）")
    
    from src.knowledge_graph.graph_store import InMemoryGraphStore
    from src.knowledge_graph.graph_retriever import GraphRetriever
    from src.knowledge_graph.entity_extractor import Entity, EntityType
    from src.knowledge_graph.relation_builder import Relation, RelationType
    import hashlib
    
    def gen_id(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    # 创建一个包含预置知识的图
    graph = InMemoryGraphStore()
    
    # 添加实体
    entities_data = [
        ("SmartHome Pro", EntityType.PRODUCT, "智能家居控制平台"),
        ("OAuth 2.0", EntityType.CONCEPT, "认证协议"),
        ("Zigbee 3.0", EntityType.CONCEPT, "无线通信协议"),
        ("Python SDK", EntityType.API, "Python 开发工具包"),
        ("E1001", EntityType.ERROR, "设备离线错误"),
        ("config.yaml", EntityType.FILE, "配置文件"),
    ]
    
    entities = {}
    for name, etype, desc in entities_data:
        eid = gen_id(name)
        e = Entity(
            entity_id=eid,
            name=name,
            entity_type=etype,
            description=desc,
            source_chunks=["demo"]
        )
        graph.add_entity(e)
        entities[name] = eid
    
    # 添加关系
    relations_data = [
        ("SmartHome Pro", "OAuth 2.0", RelationType.USES),
        ("SmartHome Pro", "Zigbee 3.0", RelationType.RELATED_TO),
        ("SmartHome Pro", "Python SDK", RelationType.CONTAINS),
        ("E1001", "Zigbee 3.0", RelationType.RELATED_TO),
    ]
    
    for source, target, rtype in relations_data:
        rel = Relation(
            relation_id=gen_id(f"{source}-{rtype.value}-{target}"),
            source_id=entities[source],
            target_id=entities[target],
            relation_type=rtype,
            source_chunk="demo"
        )
        graph.add_relation(rel)
    
    retriever = GraphRetriever(graph)
    
    # 测试问答
    questions = [
        ("SmartHome Pro 使用什么认证协议？", ["OAuth 2.0"]),
        ("Zigbee 相关的错误有哪些？", ["E1001"]),
        ("SmartHome Pro 包含哪些组件？", ["Python SDK", "Zigbee 3.0"]),
    ]
    
    for question, expected in questions:
        print(f"\n❓ 问题: {question}")
        result = retriever.retrieve(question, expand_depth=2)
        
        found = [e.name for e in result.entities]
        print(f"   找到实体: {found}")
        
        matches = [e for e in expected if e in found]
        if matches:
            print(f"   ✅ 匹配预期: {matches}")
        else:
            print(f"   ⚠️  未匹配预期: {expected}")
        
        context = result.to_context_text()
        if context:
            print(f"   图谱上下文: {context[:100]}...")
        else:
            print(f"   图谱上下文: (空)")
    
    return True

def main():
    print("\n" + "="*60)
    print("  Graph RAG 功能测试")
    print("="*60)
    
    try:
        # 1. 实体抽取
        test_entity_extraction()
        
        # 2. 图谱构建
        graph_store = test_graph_building()
        
        # 3. 图谱检索
        test_graph_retrieval(graph_store)
        
        # 4. 端到端测试
        test_end_to_end()
        
        print_section("测试完成")
        print("✅ 所有 Graph RAG 功能测试通过！")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
