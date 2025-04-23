# Use Python 3.8 slim base
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better caching
COPY requirements.txt req.txt requirements3d.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r 

# Install torch with CUDA support
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install optional TTS for gradio demo
# RUN pip install TTS

# Create necessary directories
RUN mkdir -p checkpoints static src uploads results

# Copy application files
COPY api.py app_sadtalker.py inference.py launcher.py predict.py ./
COPY src/ ./src/
COPY static/ ./static/
# COPY checkpoints/ ./checkpoints/
# COPY examples/ ./examples/
COPY scripts/ ./scripts/

# Create required directories for API
RUN mkdir -p uploads results static

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]