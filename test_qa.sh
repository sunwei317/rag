#!/bin/bash
# RAG 系统问答测试脚本
# 使用方法: ./test_qa.sh

BASE_URL="http://localhost:8000"

echo "============================================================"
echo "  RAG 系统问答测试"
echo "============================================================"

# 检查服务状态
echo -e "\n🔍 检查服务状态..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ 服务未运行，请先启动: docker-compose up -d"
    exit 1
fi
echo "✅ 服务正常运行"

# 测试函数
test_question() {
    local num=$1
    local question=$2
    local max_len=${3:-500}
    
    echo -e "\n============================================================"
    echo "📝 测试 $num: $question"
    echo "============================================================"
    
    response=$(curl -s -X POST "$BASE_URL/api/chat" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}")
    
    echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    answer = d.get('answer', '')
    print('💬 回答:')
    print(answer[:$max_len])
    if len(answer) > $max_len:
        print('...')
except Exception as e:
    print(f'❌ 解析错误: {e}')
"
}

# 执行测试
echo -e "\n开始问答测试...\n"

test_question 1 "SmartHome Pro 支持哪些认证模式？" 400

test_question 2 "如何获取访问令牌？给出Python代码示例" 800

test_question 3 "设备离线返回E1001错误怎么解决？" 700

test_question 4 "Zigbee协议最多支持多少个设备？延迟是多少？" 300

test_question 5 "WebSocket连接地址是什么？支持哪些事件类型？" 500

test_question 6 "API 速率限制是多少？" 400

test_question 7 "Python SDK 如何安装？" 400

echo -e "\n============================================================"
echo "✅ 测试完成"
echo "============================================================"
