#!/usr/bin/env python3
"""
快速入门脚本 - 启动 API 服务

用法:
    python run_server.py
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="技术文档 RAG 系统 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    args = parser.parse_args()
    
    # 检查必要目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "chroma").mkdir(exist_ok=True)
    (data_dir / "uploads").mkdir(exist_ok=True)
    
    # 检查环境变量文件
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("⚠️  未找到 .env 文件,请复制 .env.example 并配置 API 密钥")
            print("   cp .env.example .env")
            print("   然后编辑 .env 文件填入你的 API 密钥")
        else:
            print("⚠️  未找到配置文件,请参考 README.md 进行配置")
        sys.exit(1)
    
    # 启动服务
    import uvicorn
    
    print("🚀 启动技术文档 RAG 系统...")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   文档: http://{args.host}:{args.port}/docs")
    print()
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
