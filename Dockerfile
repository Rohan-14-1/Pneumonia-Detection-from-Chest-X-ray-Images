# ============================================================
# Stage 1: Builder - Install Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

# Prevent Python from creating .pyc files
# Force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Copy requirements first for Docker cache
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime libraries required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser

# Copy application files
COPY api.py .
COPY requirements.txt .

COPY config/ config/
COPY models/ models/
COPY checkpoints/ checkpoints/
COPY utils/ utils/

# Give ownership to application directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default HTTP port
EXPOSE 8080

# Start Gunicorn using Render's PORT environment variable
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 --access-logfile - --error-logfile - api:app"]