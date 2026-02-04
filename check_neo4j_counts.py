#!/usr/bin/env python3
"""
检查 Neo4j 中的实体和关系数量
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def check_neo4j_counts():
    """检查 Neo4j 中的实体和关系数量"""
    try:
        from src.knowledge_graph.graph_store import Neo4jStore

        # 从环境变量获取配置
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "graphrag123")

        print(f"📡 连接信息:")
        print(f"   URI: {neo4j_uri}")
        print(f"   User: {neo4j_user}")
        print(f"   Password: {'*' * len(neo4j_password)}")

        # 连接到 Neo4j
        graph_store = Neo4jStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )

        print("✅ Neo4j 连接成功！")

        # 获取统计信息
        stats = graph_store.get_stats()
        print(f"\n📊 图谱总体统计:")
        print(f"   实体总数: {stats.node_count}")
        print(f"   关系总数: {stats.edge_count}")

        # 通过 Cypher 查询获取更详细的统计信息
        with graph_store._driver.session(database=graph_store.database) as session:
            # 统计不同类型的实体数量
            entity_types_query = """
            MATCH (n:Entity)
            RETURN n.entity_type as type, count(n) as count
            ORDER BY count DESC
            """
            
            print(f"\n📋 实体类型分布:")
            result = session.run(entity_types_query)
            for record in result:
                entity_type = record['type'] or 'UNDEFINED'
                count = record['count']
                print(f"   {entity_type}: {count}")

            # 统计不同类型的关系统计
            relation_types_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """
            
            print(f"\n🔗 关系类型分布:")
            result = session.run(relation_types_query)
            for record in result:
                rel_type = record['type']
                count = record['count']
                print(f"   {rel_type}: {count}")

            # 获取实体总数（另一种方法验证）
            total_entities_query = "MATCH (n:Entity) RETURN count(n) as count"
            result = session.run(total_entities_query)
            total_entities = result.single()['count']
            
            # 获取关系总数（另一种方法验证）
            total_relations_query = "MATCH ()-[r]->() RETURN count(r) as count"
            result = session.run(total_relations_query)
            total_relations = result.single()['count']

            print(f"\n🔍 详细统计 (通过 Cypher 查询验证):")
            print(f"   实体总数: {total_entities}")
            print(f"   关系总数: {total_relations}")

        # 关闭连接
        graph_store.close()
        print("\n🔌 已关闭 Neo4j 连接")

    except Exception as e:
        print(f"❌ 检查 Neo4j 统计信息失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_neo4j_counts()