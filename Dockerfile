FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖 (包括 WeasyPrint 所需的 PDF 生成依赖和中文字体)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    # WeasyPrint 依赖
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    # 中文字体支持
    fonts-noto-cjk \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma /app/data/uploads /app/data/neo4j

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
