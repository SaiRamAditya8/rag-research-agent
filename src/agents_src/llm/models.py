from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

logger.info("Initializing shared models (Embedding & Reranker)...")

# Singleton instances
embed_model = HuggingFaceEmbedding()
rerank_model = CrossEncoder('BAAI/bge-reranker-large')
