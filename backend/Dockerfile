# PlateVision Backend Dockerfile for Railway
# Optimized for smaller image size (< 4GB)

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with cache cleanup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy the rest of the application
COPY . .

# Create directories for uploads/detections
RUN mkdir -p uploads detections

# Railway uses PORT environment variable
ENV HOST=0.0.0.0

# Command to run the application
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port ${PORT:-8000}"]
