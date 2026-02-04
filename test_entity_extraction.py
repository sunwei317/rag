#!/usr/bin/env python3
"""
测试实体抽取功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

def test_entity_extraction():
    """测试实体抽取"""
    print("="*60)
    print("测试实体抽取功能")
    print("="*60)
    
    from src.knowledge_graph.entity_extractor import EntityExtractor
    
    extractor = EntityExtractor()
    
    # 测试文本
    test_text = """
    PostgreSQL是一个开源的关系型数据库管理系统。
    它支持SQL标准，提供了丰富的功能。
    版本15.0引入了新的性能优化特性。
    配置参数max_connections控制最大连接数。
    错误码23505表示唯一约束冲突。
    """
    
    print(f"\n测试文本:\n{test_text}")
    
    try:
        entities = extractor.extract_from_text(test_text, "test_chunk")
        print(f"\n✅ 抽取成功！找到 {len(entities)} 个实体:")
        for e in entities:
            print(f"  - {e.name} ({e.entity_type.value}): {e.description}")
        return len(entities) > 0
    except Exception as e:
        print(f"\n❌ 抽取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_entity_extraction()
    sys.exit(0 if success else 1)
