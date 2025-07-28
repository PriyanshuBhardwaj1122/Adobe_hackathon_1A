# Use Python 3.9 slim image for AMD64 architecture
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyMuPDF
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main application
COPY main.py .

# Create input and output directories with proper permissions
RUN mkdir -p /app/input /app/output && \
    chmod 755 /app/input /app/output

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV OMP_NUM_THREADS=8

# Set user to avoid running as root (security best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["python", "main.py"]

