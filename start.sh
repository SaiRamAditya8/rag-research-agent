#!/bin/bash
set -e

# 1. Run document ingestion (one-time). Uncomment the line below if you need to ingest documents from DOCUMENTS_DIR and seed the vector store.
# python scripts/seed_vectorstore.py

# 2. Start backend API in background
uvicorn src.backend_src.main:app --host 0.0.0.0 --port 8000 &

# 3. Start frontend (Streamlit) in foreground
streamlit run src/frontend_src/app.py --server.port 8501 --server.address 0.0.0.0

