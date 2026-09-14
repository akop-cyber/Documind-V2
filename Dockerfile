
FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860


WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir -p /tmp/data


COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

COPY . .

# 10. Expose the port Hugging Face Spaces expects
EXPOSE 7860

# 11. Run your FastAPI app
CMD ["python", "app.py"]
