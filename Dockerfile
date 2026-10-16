# ── 阶段 1：构建前端 ──
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

# ── 阶段 2：生产运行时 ──
FROM python:3.11-slim

WORKDIR /app

# 系统依赖：中文字体 + PyMuPDF/OCR 运行时库
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk-extra \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .[pdf,ocr,retrieval,data,modeling,api,reporting]

# 复制前端构建产物
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 复制配置、提示词；私有数据与可选模型不打入镜像
COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY scripts/run_web_api.py ./scripts/

# 初始化运行时目录
RUN mkdir -p /app/output /app/tmp /app/artifacts/models

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "scripts/run_web_api.py", "--host", "0.0.0.0", "--port", "8000", "--no-browser"]
