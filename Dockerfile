FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY start.sh ./start.sh

# Make start.sh executable
RUN chmod +x start.sh

# Expose backend and frontend ports
EXPOSE 8000 8501

# Non-secret defaults — override at runtime via env_file or -e flags
ENV DOCUMENTS_DIR="/app/data/docs"
ENV VECTOR_STORE_DIR="/app/data/vector_store"
ENV COLLECTION_NAME="document_collection"
ENV MODEL_NAME="llama-3.3-70b-versatile"
ENV MODEL_TEMPERATURE=0.0
ENV CHAT_ENDPOINT_URL="http://localhost:8000/chat/answer"
ENV CHUNK_SIZE=512
ENV CHUNK_OVERLAP=100
ENV RETRIEVAL_TOP_K=15
ENV RERANK_TOP_K=5

# Run all services using start.sh
CMD ["/app/start.sh"]
