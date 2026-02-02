"""
Graph RAG 使用示例

演示如何:
1. 从文档构建知识图谱
2. 使用 Graph RAG 进行增强检索和问答
"""
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph import (
    EntityExtractor,
    RelationBuilder,
    InMemoryGraphStore,
    GraphRetriever,
    GraphRAG,
    GraphBuildPipeline,
    create_graph_pipeline
)


def demo_entity_extraction():
    """演示实体抽取"""
    print("\n" + "="*60)
    print("1. 实体抽取示例")
    print("="*60)
    
    # 模拟的技术文档内容
    sample_text = """
    MySQL 8.0 是一个流行的关系型数据库管理系统。
    安装 MySQL 需要先安装 JDK 11 或更高版本。
    主要配置文件是 /etc/mysql/my.cnf，其中 max_connections 
    参数控制最大连接数，默认值为 151。
    
    MySQL 提供了 REST API 用于远程管理，主要端点包括:
    - GET /api/v1/status - 获取服务状态
    - POST /api/v1/query - 执行 SQL 查询
    """
    
    extractor = EntityExtractor()
    
    # 使用规则抽取 (不需要 LLM)
    entities = extractor.extract_with_rules(sample_text, chunk_id="demo_chunk_1")
    
    print("\n使用规则抽取的实体:")
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type.value})")
    
    # 如果配置了 OpenAI API，使用 LLM 抽取
    if os.getenv("OPENAI_API_KEY"):
        print("\n使用 LLM 抽取的实体:")
        entities = extractor.extract_from_text(sample_text, chunk_id="demo_chunk_1")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type.value}): {entity.description}")


def demo_graph_building():
    """演示图谱构建"""
    print("\n" + "="*60)
    print("2. 知识图谱构建示例")
    print("="*60)
    
    # 模拟的 chunks
    chunks = [
        {
            "chunk_id": "chunk_1",
            "content": """MySQL 8.0 需要 JDK 11 才能运行。
                         max_connections 配置项影响 MySQL 的并发性能。
                         推荐的安装平台是 Ubuntu 20.04 或 CentOS 7。""",
            "metadata": {"doc_id": "mysql_guide", "section_path": "安装/依赖"}
        },
        {
            "chunk_id": "chunk_2", 
            "content": """Redis 6.0 可以作为 MySQL 的缓存层使用。
                         两者配合可以显著提升读取性能。
                         Redis 的默认端口是 6379。""",
            "metadata": {"doc_id": "mysql_guide", "section_path": "性能优化/缓存"}
        },
        {
            "chunk_id": "chunk_3",
            "content": """API 网关使用 Nginx 作为反向代理。
                         Nginx 需要配置 upstream 指向 MySQL 服务。
                         配置文件位于 /etc/nginx/nginx.conf。""",
            "metadata": {"doc_id": "api_guide", "section_path": "部署/网关"}
        }
    ]
    
    # 创建内存图存储
    graph_store = InMemoryGraphStore()
    
    # 创建 Pipeline (使用规则抽取，不依赖 LLM)
    pipeline = GraphBuildPipeline(
        graph_store=graph_store,
        extract_rules=True
    )
    
    # 构建图谱
    result = pipeline.build(chunks)
    
    print(f"\n构建结果:")
    print(f"  - 实体数: {result.entity_count}")
    print(f"  - 关系数: {result.relation_count}")
    print(f"  - 耗时: {result.duration_seconds:.2f}s")
    
    # 查看图谱统计
    stats = graph_store.get_stats()
    print(f"\n图谱统计:")
    print(f"  - 节点总数: {stats.node_count}")
    print(f"  - 边总数: {stats.edge_count}")
    print(f"  - 实体类型分布: {stats.entity_types}")
    
    return graph_store


def demo_graph_retrieval(graph_store):
    """演示图谱检索"""
    print("\n" + "="*60)
    print("3. 图谱检索示例")
    print("="*60)
    
    retriever = GraphRetriever(graph_store)
    
    # 搜索实体
    print("\n搜索 'MySQL' 相关实体:")
    entities = graph_store.search_entities("MySQL", limit=5)
    for entity in entities:
        print(f"  - {entity.name} ({entity.entity_type.value})")
    
    # 检索子图
    print("\n检索查询相关子图: '如何安装 MySQL'")
    subgraph = retriever.retrieve("如何安装 MySQL", expand_depth=1)
    
    if not subgraph.is_empty():
        print(f"  - 找到 {len(subgraph.entities)} 个相关实体")
        print(f"  - 找到 {len(subgraph.relations)} 个关系")
        
        print("\n生成的图谱上下文:")
        context = subgraph.to_context_text(max_length=500)
        print(context)
    else:
        print("  - 未找到相关子图")


def demo_graph_rag():
    """演示 Graph RAG 完整流程"""
    print("\n" + "="*60)
    print("4. Graph RAG 完整示例")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[跳过] 需要设置 OPENAI_API_KEY 环境变量")
        return
    
    # 1. 创建并填充图存储
    graph_store = InMemoryGraphStore()
    
    # 模拟一些实体
    from src.knowledge_graph import Entity, EntityType, Relation, RelationType
    
    entities = [
        Entity(
            entity_id="e1",
            name="MySQL",
            entity_type=EntityType.PRODUCT,
            description="关系型数据库管理系统",
            aliases=["mysql", "MySQL Server"]
        ),
        Entity(
            entity_id="e2", 
            name="JDK",
            entity_type=EntityType.DEPENDENCY,
            description="Java 开发工具包",
            aliases=["Java JDK", "OpenJDK"]
        ),
        Entity(
            entity_id="e3",
            name="Redis",
            entity_type=EntityType.PRODUCT,
            description="内存键值存储",
            aliases=["redis-server"]
        ),
    ]
    
    relations = [
        Relation(
            relation_id="r1",
            source_id="e1",
            target_id="e2",
            relation_type=RelationType.REQUIRES,
            description="MySQL 运行需要 JDK"
        ),
        Relation(
            relation_id="r2",
            source_id="e1",
            target_id="e3",
            relation_type=RelationType.RELATED_TO,
            description="MySQL 常与 Redis 配合使用"
        ),
    ]
    
    for entity in entities:
        graph_store.add_entity(entity)
    for relation in relations:
        graph_store.add_relation(relation)
    
    # 2. 创建 Graph RAG
    graph_rag = GraphRAG(
        graph_store=graph_store,
        model="gpt-4.1-mini"
    )
    
    # 3. 执行查询
    question = "MySQL 需要什么依赖？"
    print(f"\n问题: {question}")
    
    # 仅图谱检索
    context = graph_rag.retrieve(question)
    
    print(f"\n检索结果:")
    print(f"  - 图谱实体: {len(context.subgraph.entities) if context.subgraph else 0}")
    print(f"\n图谱上下文:")
    print(context.graph_context)


def main():
    """主函数"""
    print("="*60)
    print("        Graph RAG 功能演示")
    print("="*60)
    
    # 1. 实体抽取
    demo_entity_extraction()
    
    # 2. 图谱构建
    graph_store = demo_graph_building()
    
    # 3. 图谱检索
    if graph_store:
        demo_graph_retrieval(graph_store)
    
    # 4. 完整 Graph RAG (需要 OpenAI API)
    demo_graph_rag()
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
