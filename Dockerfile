# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim

# ============================================
# Layer 1: System dependencies (rarely changes)
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # OpenMP for torch
    libgomp1 \
    # RTSP streaming
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Layer 2: PyTorch + CUDA (large, rarely changes)
# CUDA build works on both GPU and CPU systems
# ============================================
ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================
# Layer 3: Other ML dependencies (moderate size, occasionally changes)
# ============================================
RUN pip install --no-cache-dir \
    ultralytics>=8.1.0 \
    timm>=0.9.0 \
    opencv-python>=4.8.0 \
    scipy>=1.11.0 \
    onnxruntime-gpu>=1.16.0

# ============================================
# Layer 4: App dependencies (small, may change more often)
# ============================================
RUN pip install --no-cache-dir \
    # Core
    httpx>=0.27.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    aiomqtt>=2.0.0 \
    Pillow>=10.0.0 \
    numpy>=1.26.0 \
    pyyaml>=6.0.0 \
    structlog>=24.1.0 \
    # Web/API
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    jinja2>=3.1.0 \
    aiosqlite>=0.19.0 \
    python-multipart>=0.0.6

# ============================================
# Layer 5: Application code (changes frequently)
# ============================================
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    DATA_DIR=/data \
    LOG_LEVEL=INFO

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --no-deps -e .

# ============================================
# Runtime configuration
# ============================================
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/api/health').raise_for_status()" || \
        python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "cat_watcher.web.app"]
