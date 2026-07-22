# 1. Use Python 3.11 (This fixes the networkx error!)
FROM python:3.11-slim

# 2. Set environment variables to keep things running smoothly
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system tools needed to build certain Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 5. Create the directory where the database will store its files
RUN mkdir -p /tmp/data

# 6. Copy your requirements.txt first
COPY requirements.txt .

# 7. Install all your Python dependencies from your actual requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 8. Pre-download NLTK data so the app doesn't crash trying to download it on startup
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# 9. Copy the rest of your project files (app.py, loader.py, etc.)
COPY . .

# 10. Expose the port Hugging Face Spaces expects
EXPOSE 7860

# 11. Run your FastAPI app
CMD ["python", "app.py"]