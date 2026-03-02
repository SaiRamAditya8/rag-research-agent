import logging
from llama_index.embeddings.openai import OpenAIEmbedding
from sentence_transformers import CrossEncoder

from src.agents_src.config.agent_settings import AgentSettings

logger = logging.getLogger(__name__)

logger.info("Initialising shared models (Embedding & Reranker)...")

_settings = AgentSettings()

# text-embedding-3-small: fast, low cost, strong retrieval quality.
# Switch to "text-embedding-3-large" here for higher accuracy at ~5x the cost.
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=_settings.OPENAI_API_KEY,
)

# Cross-encoder reranker — local, no API cost, best-in-class for reranking small candidate sets.
rerank_model = CrossEncoder("BAAI/bge-reranker-large")