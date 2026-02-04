#!/usr/bin/env python3
"""
RAG 系统完整测试脚本
测试文档上传、知识图谱构建和问答功能
"""

import requests
import json
import time
from pathlib import Path

from config.settings import settings
BASE_URL = f"http://{getattr(settings, 'api_host', 'localhost')}:{getattr(settings, 'api_port', 8000)}"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(name, success, details=""):
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")

def test_health():
    """测试健康检查接口"""
    print_section("1. 健康检查")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        data = resp.json()
        print_result("API 服务状态", resp.status_code == 200, f"状态: {data.get('status')}")
        return True
    except Exception as e:
        print_result("API 服务状态", False, str(e))
        return False

def test_upload_document():
    """测试文档上传"""
    print_section("2. 文档上传")
    
    doc_path = Path("/home/wei/rag/test_docs/smart_home_api_guide.md")
    if not doc_path.exists():
        print_result("文档上传", False, "测试文档不存在")
        return None
    
    try:
        with open(doc_path, 'rb') as f:
            files = {'file': ('smart_home_api_guide.md', f, 'text/markdown')}
            resp = requests.post(
                f"{BASE_URL}/api/ingest/upload",
                files=files,
                timeout=120
            )
        
        if resp.status_code == 200:
            data = resp.json()
            doc_id = data.get('document_id') or data.get('doc_id') or data.get('id')
            chunks = data.get('chunks_count', data.get('chunks', 0))
            print_result("文档上传", True, f"文档ID: {doc_id}, 分块数: {chunks}")
            return doc_id
        else:
            print_result("文档上传", False, f"状态码: {resp.status_code}, 响应: {resp.text[:200]}")
            return None
    except Exception as e:
        print_result("文档上传", False, str(e))
        return None

def test_list_documents():
    """测试知识库统计"""
    print_section("3. 知识库状态")
    try:
        resp = requests.get(f"{BASE_URL}/api/kb/stats", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print_result("知识库状态", True)
            print(f"   向量数: {data.get('vector_count', 'N/A')}")
            print(f"   BM25文档数: {data.get('bm25_count', 'N/A')}")
            return True
        else:
            print_result("知识库状态", False, f"状态码: {resp.status_code}")
            return False
    except Exception as e:
        print_result("知识库状态", False, str(e))
        return False

def test_chat_queries():
    """测试问答功能"""
    print_section("4. 问答测试")
    
    test_questions = [
        "SmartHome Pro 支持哪些认证模式？",
        "如何获取访问令牌？请给出Python代码示例",
        "设备离线返回E1001错误该如何解决？",
        "WebSocket连接地址是什么？支持哪些事件类型？",
        "Zigbee协议最多支持多少个设备？延迟是多少？",
    ]
    
    success_count = 0
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        try:
            resp = requests.post(
                f"{BASE_URL}/api/chat",
                json={"question": question},
                timeout=120
            )
            
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get('answer', data.get('response', ''))
                sources = data.get('sources', data.get('contexts', []))
                
                # 截取答案前300字符显示
                answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
                print(f"💬 回答: {answer_preview}")
                if sources:
                    print(f"📚 来源数: {len(sources)}")
                
                success_count += 1
            else:
                print(f"❌ 请求失败: {resp.status_code}")
                print(f"   响应: {resp.text[:200]}")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
        
        time.sleep(1)  # 避免请求过快
    
    print(f"\n问答测试完成: {success_count}/{len(test_questions)} 成功")
    return success_count > 0

def test_graph_rag():
    """测试搜索功能"""
    print_section("5. 搜索测试")
    
    # 测试搜索查询
    search_queries = [
        "OAuth 认证",
        "设备控制 API",
    ]
    
    for query in search_queries:
        print(f"\n🔍 搜索: {query}")
        try:
            resp = requests.post(
                f"{BASE_URL}/api/search",
                json={"query": query, "top_k": 3},
                timeout=60
            )
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', data.get('documents', []))
                print(f"✅ 找到 {len(results)} 条结果")
                for i, r in enumerate(results[:2], 1):
                    content = r.get('content', r.get('text', ''))[:100]
                    print(f"   {i}. {content}...")
            else:
                print(f"❌ 状态码: {resp.status_code}")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")

def main():
    print("\n" + "="*60)
    print("  RAG 系统完整测试")
    print("  测试文档: SmartHome Pro API 技术文档")
    print("="*60)
    
    # 1. 健康检查
    if not test_health():
        print("\n⚠️  服务未启动，请先运行: docker-compose up -d")
        return
    
    # 2. 上传文档
    doc_id = test_upload_document()
    
    # 3. 列出文档
    test_list_documents()
    
    # 等待索引完成
    if doc_id:
        print("\n⏳ 等待文档索引完成 (5秒)...")
        time.sleep(5)
    
    # 4. 问答测试
    test_chat_queries()
    
    # 5. Graph RAG 测试
    test_graph_rag()
    
    print_section("测试完成")
    print("✅ RAG 系统测试流程已完成")
    print("📖 测试文档: test_docs/smart_home_api_guide.md")

if __name__ == "__main__":
    main()
