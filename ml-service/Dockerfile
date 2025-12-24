# PlateVision ML Service Dockerfile for Hugging Face Spaces

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p uploads

# Hugging Face Spaces uses port 7860
ENV PORT=7860
ENV HOST=0.0.0.0

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host $HOST --port $PORT"]
