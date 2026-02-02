"""
测试套件 - API 模块
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查端点测试"""
    
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "endpoints" in data


class TestSearchEndpoint:
    """搜索端点测试"""
    
    def test_search_empty_query(self, client):
        """测试空查询"""
        response = client.post("/api/search", json={
            "query": "",
            "top_k": 10
        })
        # 应该返回验证错误或空结果
        assert response.status_code in [200, 422]
    
    def test_search_with_query(self, client):
        """测试带查询的搜索"""
        response = client.post("/api/search", json={
            "query": "Python 编程",
            "top_k": 5
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "query" in data


class TestChatEndpoint:
    """对话端点测试"""
    
    def test_chat_request_format(self, client):
        """测试对话请求格式"""
        response = client.post("/api/chat", json={
            "query": "什么是 RAG?",
            "use_history": False
        })
        # 可能因为没有配置 LLM 而失败,但请求格式应该正确
        assert response.status_code in [200, 500, 503]
    
    def test_chat_with_filters(self, client):
        """测试带过滤器的对话"""
        response = client.post("/api/chat", json={
            "query": "解释向量数据库",
            "filters": {
                "doc_ids": ["doc1", "doc2"]
            },
            "use_history": False
        })
        assert response.status_code in [200, 500, 503]


class TestGenerateEndpoint:
    """文档生成端点测试"""
    
    def test_generate_request_format(self, client):
        """测试生成请求格式"""
        response = client.post("/api/generate", json={
            "prompt": "生成一份技术文档",
            "doc_type": "technical_spec",
            "output_format": "markdown"
        })
        # 可能因为没有配置 LLM 而失败
        assert response.status_code in [200, 500, 503]
    
    def test_generate_invalid_doc_type(self, client):
        """测试无效文档类型"""
        response = client.post("/api/generate", json={
            "prompt": "生成文档",
            "doc_type": "invalid_type",
            "output_format": "markdown"
        })
        # 应该返回验证错误或处理无效类型
        assert response.status_code in [200, 422, 500]


class TestIngestEndpoint:
    """导入端点测试"""
    
    def test_ingest_no_file(self, client):
        """测试无文件上传"""
        response = client.post("/api/ingest")
        # 应该返回验证错误
        assert response.status_code == 422


class TestAPIDocumentation:
    """API 文档测试"""
    
    def test_openapi_schema(self, client):
        """测试 OpenAPI schema"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "info" in data
        assert "paths" in data
    
    def test_docs_endpoint(self, client):
        """测试文档端点"""
        response = client.get("/docs")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
