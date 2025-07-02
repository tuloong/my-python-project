# 多阶段构建的Docker文件
FROM python:3.9-slim as builder

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --user --no-cache-dir -r requirements.txt

# 生产环境阶段
FROM python:3.9-slim as production

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# 从builder阶段复制安装的包
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY src/ src/
COPY configs/ configs/
COPY *.py ./
COPY *.txt ./
COPY *.md ./

# 创建必要的目录
RUN mkdir -p data/raw data/processed data/external \
             outputs/models outputs/predictions \
             logs cache

# 创建非root用户
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
USER user

# 暴露端口（如果运行API服务）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src; print('OK')" || exit 1

# 默认命令
CMD ["python", "predict.py"]

# 开发环境阶段
FROM production as development

# 切换回root用户安装开发依赖
USER root

# 安装开发依赖
RUN pip install pytest pytest-cov black flake8 isort mypy jupyter

# 安装git（用于版本控制）
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 切换回user用户
USER user

# 暴露Jupyter端口
EXPOSE 8888

# 开发环境命令
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 