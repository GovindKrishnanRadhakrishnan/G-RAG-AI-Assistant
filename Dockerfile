# ==============================================================================
# STAGE 1: Builder
# ==============================================================================
FROM python:3.11-slim AS builder

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# 1. Install CPU-only PyTorch first to prevent installing the 1.5GB+ CUDA version
# 2. Install the remaining requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Remove unnecessary files to reduce image size (test suites, pyc files, etc.)
RUN find /usr/local/lib/python3.11/site-packages/ -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages/ -name "__pycache__" -exec rm -rf {} +

# ==============================================================================
# STAGE 2: Runner
# ==============================================================================
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

WORKDIR /app

# Install curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed dependencies and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=appuser:appgroup . .

# Create directories for persistent volumes and set permissions
RUN mkdir -p /app/vector_db /home/appuser/.cache/huggingface && \
    chown -R appuser:appgroup /app/vector_db /home/appuser/.cache/huggingface

# Use the non-root user
USER appuser

# Expose FastAPI / HTML UI
EXPOSE 8001

# Default command runs the FastAPI + HTML app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
