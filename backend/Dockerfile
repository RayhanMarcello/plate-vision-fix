# PlateVision Backend Dockerfile for Railway
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for uploads/detections
RUN mkdir -p uploads detections

# Railway uses PORT environment variable
ENV HOST=0.0.0.0

# Command to run the application
# Using shell form to allow variable expansion
CMD ["sh", "-c", "uvicorn app.main:app --host $HOST --port ${PORT:-8000}"]
