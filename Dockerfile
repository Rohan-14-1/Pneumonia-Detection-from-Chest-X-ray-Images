# ============================================================
# Stage 1: Builder — install Python dependencies in isolation
# ============================================================
FROM python:3.11-slim AS builder

# Prevent .pyc files and force unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# ── Layer-cache optimization: copy requirements first ──
COPY requirements.txt .

# Create a virtual environment (makes it trivial to copy to runtime stage)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-only PyTorch FIRST (separate layer — ~250 MB, cached independently)
# Default PyTorch bundles CUDA libs (~2.3 GB) — Cloud Run has no GPUs
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skips torch/torchvision since already present)
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2: Runtime — minimal production image
# ============================================================
FROM python:3.11-slim AS runtime

# Prevent .pyc files and force unbuffered stdout/stderr for Cloud Logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cloud Run injects PORT; default to 8080
ENV PORT=8080

WORKDIR /app

# ── Copy pre-built venv from builder (no pip, no build tools carried over) ──
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Minimal OS libs required by OpenCV (cv2) at runtime ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── Security: run as non-root user ──
RUN adduser --disabled-password --gecos "" --no-create-home appuser

# ── Copy ONLY the files the API needs at runtime ──
# api.py        → Flask application entry point
# config/       → config.py (IMAGE_SIZE, DEVICE, MODEL_SAVE_PATH, etc.)
# models/       → cnn_model.py (CNNModel class definition)
COPY api.py .
COPY config/ config/
COPY models/ models/

# Switch to non-root user AFTER all file operations
USER appuser

# Document the port (informational; Gunicorn binds via $PORT)
EXPOSE 8080

# ── Gunicorn entrypoint ──
# exec:       replaces shell so Gunicorn gets SIGTERM directly for graceful shutdown
# workers 1:  Cloud Run scales horizontally (instances), not vertically (workers)
# threads 8:  handles concurrent requests within a single instance
# timeout 0:  Cloud Run manages request timeouts externally (default 300s)
# api:app:    import the `app` Flask object from api.py
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 8 \
    --timeout 0 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    api:app
