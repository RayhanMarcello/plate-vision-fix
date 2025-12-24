# PlateVision Backend Dockerfile for Railway
# Optimized for CPU-only deployment (< 4GB)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy requirements first
COPY requirements.txt .

# Install CPU-only PyTorch first, then other dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip && \
    rm -rf /root/.cache && \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy the rest of the application
COPY . .

# Create directories
RUN mkdir -p uploads detections

# Railway uses PORT environment variable
ENV HOST=0.0.0.0

CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port ${PORT:-8000}"]
