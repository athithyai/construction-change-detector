FROM python:3.11-slim

WORKDIR /app

# System deps — GDAL, OpenCV, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libgdal-dev gdal-bin \
    libspatialindex-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Torch with CUDA 12.1 (separate layer for better caching)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# App dependencies
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy application
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

ENV PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

CMD ["python", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
