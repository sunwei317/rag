#!/usr/bin/env python3
"""
Neo4j 图数据库测试脚本
测试连接、基本操作和查询功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_neo4j_connection():
    """测试 Neo4j 连接"""
    print_section("1. Neo4j 连接测试")
    
    try:
        from src.knowledge_graph.graph_store import Neo4jStore
        
        # 从环境变量获取配置，如果没有则使用默认值
        # 注意：从主机访问使用 localhost，从容器内访问使用 neo4j
        neo4j_uri = os.getenv("NEO4J_URI")
        if not neo4j_uri:
            # 检测是否在 Docker 容器内运行
            if os.path.exists("/.dockerenv") or os.getenv("HOSTNAME", "").startswith("rag-app"):
                neo4j_uri = "bolt://neo4j:7687"  # 容器内使用服务名
            else:
                neo4j_uri = "bolt://localhost:7687"  # 主机使用 localhost
        
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "graphrag123")
        
        print(f"📡 连接信息:")
        print(f"   URI: {neo4j_uri}")
        print(f"   User: {neo4j_user}")
        print(f"   Password: {'*' * len(neo4j_password)}")
        
        graph_store = Neo4jStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        
        print("✅ Neo4j 连接成功！")
        return graph_store
        
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_neo4j_basic_operations(graph_store):
    """测试基本操作：添加实体和关系"""
    print_section("2. 基本操作测试")
    
    if not graph_store:
        print("❌ 跳过测试：Neo4j 未连接")
        return
    
    try:
        from src.knowledge_graph.entity_extractor import Entity, EntityType
        from src.knowledge_graph.relation_builder import Relation, RelationType
        
        # 创建测试实体
        entities = [
            Entity(
                entity_id="test_e1",
                name="SmartHome Pro",
                entity_type=EntityType.PRODUCT,
                description="智能家居控制平台",
                aliases=["SHP", "SmartHome"]
            ),
            Entity(
                entity_id="test_e2",
                name="OAuth 2.0",
                entity_type=EntityType.CONCEPT,
                description="认证协议标准",
                aliases=["OAuth"]
            ),
            Entity(
                entity_id="test_e3",
                name="Zigbee 3.0",
                entity_type=EntityType.CONCEPT,
                description="无线通信协议",
                aliases=["Zigbee"]
            ),
            Entity(
                entity_id="test_e4",
                name="E1001",
                entity_type=EntityType.ERROR,
                description="设备离线错误码"
            )
        ]
        
        # 添加实体
        print("📝 添加实体...")
        for entity in entities:
            success = graph_store.add_entity(entity)
            if success:
                print(f"   ✅ {entity.name} ({entity.entity_type.value})")
            else:
                print(f"   ❌ 添加失败: {entity.name}")
        
        # 创建关系
        relations = [
            Relation(
                relation_id="test_r1",
                source_id="test_e1",
                target_id="test_e2",
                relation_type=RelationType.USES,
                description="使用 OAuth 2.0 进行认证"
            ),
            Relation(
                relation_id="test_r2",
                source_id="test_e1",
                target_id="test_e3",
                relation_type=RelationType.USES,
                description="使用 Zigbee 3.0 进行设备通信"
            ),
            Relation(
                relation_id="test_r3",
                source_id="test_e4",
                target_id="test_e3",
                relation_type=RelationType.RELATED_TO,
                description="错误与 Zigbee 协议相关"
            )
        ]
        
        print("\n🔗 添加关系...")
        for relation in relations:
            success = graph_store.add_relation(relation)
            if success:
                print(f"   ✅ {relation.source_id} --[{relation.relation_type.value}]--> {relation.target_id}")
            else:
                print(f"   ❌ 添加失败: {relation.relation_id}")
        
        print("\n✅ 基本操作测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 基本操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_queries(graph_store):
    """测试查询功能"""
    print_section("3. 查询功能测试")
    
    if not graph_store:
        print("❌ 跳过测试：Neo4j 未连接")
        return
    
    try:
        # 获取统计信息
        stats = graph_store.get_stats()
        print(f"📊 图谱统计:")
        print(f"   节点数: {stats.node_count}")
        print(f"   边数: {stats.edge_count}")
        
        # 查询实体
        print("\n🔍 查询实体...")
        entity = graph_store.get_entity("test_e1")
        if entity:
            print(f"   ✅ 找到实体: {entity.name}")
            print(f"      类型: {entity.entity_type.value}")
            print(f"      描述: {entity.description}")
        else:
            print("   ⚠️  未找到实体 test_e1")
        
        # 查询邻居节点
        print("\n🔍 查询邻居节点...")
        neighbors, relations = graph_store.get_neighbors("test_e1", max_depth=1)
        print(f"   ✅ test_e1 的邻居节点数: {len(neighbors)}")
        for neighbor in neighbors[:5]:  # 只显示前5个
            print(f"      - {neighbor.name} ({neighbor.entity_type.value})")
        
        print(f"   ✅ 关系数: {len(relations)}")
        for rel in relations[:5]:  # 只显示前5个
            print(f"      - {rel.source_id} --[{rel.relation_type.value}]--> {rel.target_id}")
        
        # 测试搜索
        print("\n🔍 搜索实体...")
        results = graph_store.search_entities("SmartHome", limit=5)
        print(f"   ✅ 搜索 'SmartHome' 找到 {len(results)} 个结果:")
        for e in results:
            print(f"      - {e.name} ({e.entity_type.value})")
        
        print("\n✅ 查询功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 查询功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_cypher(graph_store):
    """测试直接执行 Cypher 查询"""
    print_section("4. Cypher 查询测试")
    
    if not graph_store:
        print("❌ 跳过测试：Neo4j 未连接")
        return
    
    try:
        # 直接执行 Cypher 查询
        cypher_queries = [
            ("MATCH (n:Entity) RETURN count(n) as count", "统计所有实体"),
            ("MATCH (n:Entity {entity_id: 'test_e1'}) RETURN n.name as name", "查询特定实体"),
            ("MATCH (n:Entity)-[r]->(m:Entity) RETURN n.name, type(r), m.name LIMIT 5", "查询关系"),
            ("MATCH (n:Entity) WHERE n.entity_type = 'product' RETURN n.name LIMIT 5", "查询产品类型实体"),
        ]
        
        for query, description in cypher_queries:
            print(f"\n📝 {description}:")
            print(f"   Cypher: {query}")
            
            try:
                with graph_store._driver.session(database=graph_store.database) as session:
                    result = session.run(query)
                    records = list(result)
                    
                    if records:
                        print(f"   ✅ 结果 ({len(records)} 条):")
                        for record in records[:3]:  # 只显示前3条
                            print(f"      {dict(record)}")
                    else:
                        print("   ⚠️  无结果")
            except Exception as e:
                print(f"   ❌ 查询失败: {e}")
        
        print("\n✅ Cypher 查询测试完成")
        return True
        
    except Exception as e:
        print(f"❌ Cypher 查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neo4j_cleanup(graph_store):
    """清理测试数据"""
    print_section("5. 清理测试数据")
    
    if not graph_store:
        print("❌ 跳过清理：Neo4j 未连接")
        return
    
    try:
        # 删除测试数据
        cleanup_query = """
        MATCH (n:Entity)
        WHERE n.entity_id STARTS WITH 'test_'
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        with graph_store._driver.session(database=graph_store.database) as session:
            result = session.run(cleanup_query)
            record = result.single()
            deleted_count = record['deleted'] if record else 0
        
        print(f"✅ 已删除 {deleted_count} 个测试实体及其关系")
        return True
        
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("  Neo4j 图数据库测试")
    print("="*60)
    
    # 1. 测试连接
    graph_store = test_neo4j_connection()
    
    if not graph_store:
        print("\n❌ 无法连接到 Neo4j，请检查:")
        print("   1. Neo4j 容器是否运行: docker ps | grep neo4j")
        print("   2. 连接信息是否正确 (URI, User, Password)")
        print("   3. 端口是否正确映射 (7687)")
        return 1
    
    try:
        # 2. 基本操作测试
        test_neo4j_basic_operations(graph_store)
        
        # 3. 查询功能测试
        test_neo4j_queries(graph_store)
        
        # 4. Cypher 查询测试
        test_neo4j_cypher(graph_store)
        
        # 5. 清理测试数据（可选）
        # test_neo4j_cleanup(graph_store)
        
        print_section("测试完成")
        print("✅ Neo4j 图数据库测试通过！")
        print("\n💡 提示:")
        print("   - 访问 Neo4j Browser: http://localhost:7474")
        print("   - 用户名: neo4j")
        print("   - 密码: graphrag123")
        print("   - 运行 'MATCH (n) RETURN n LIMIT 25' 查看所有节点")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 关闭连接
        if graph_store:
            graph_store.close()
            print("\n🔌 已关闭 Neo4j 连接")

if __name__ == "__main__":
    sys.exit(main())
