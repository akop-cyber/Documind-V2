# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PDF processing and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . /app

# Create requirements.txt with all dependencies
RUN echo "fastapi==0.104.1\n\
uvicorn==0.24.0\n\
aiofiles==23.2.1\n\
pydantic==2.5.0\n\
haystack-ai==0.5.0\n\
sentence-transformers==2.2.2\n\
lancedb==0.3.11\n\
pyarrow==14.0.1\n\
huggingface-hub==0.19.4\n\
torch==2.1.1\n\
nltk==3.9.1\n\
scikit-learn==1.3.2" > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data for text processing
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Create directory for vector database
RUN mkdir -p /tmp/data

# Expose port (HuggingFace Spaces uses 7860 by default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/docs || exit 1

# Run the application
CMD ["python", "app.py"]
