import logging
from typing import Dict, List

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq

from src.agents_src.config.agent_settings import AgentSettings
from src.agents_src.llm.models import embed_model, rerank_model

logger = logging.getLogger(__name__)


class RAGQueryTool:
    """
    Singleton RAG query tool.

    ChromaDB client, LlamaIndex index, and retriever are initialised ONCE at
    construction time and reused across all calls.  New documents added by
    paper_fetcher (which shares the same PersistentClient path) are visible
    automatically because the underlying ChromaDB collection is live.
    """

    def __init__(self):
        settings = AgentSettings()

        # Configure the global LlamaIndex LLM once
        Settings.llm = Groq(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            api_key=settings.GROQ_API_KEY,
        )
        Settings.embed_model = embed_model

        self._settings = settings
        self._rerank_model = rerank_model

        db = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
        chroma_collection = db.get_or_create_collection(settings.COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        self._retriever = self._index.as_retriever(
            similarity_top_k=settings.RETRIEVAL_TOP_K
        )
        self._synthesizer = get_response_synthesizer()

        logger.info(
            f"RAGQueryTool initialised | collection={settings.COLLECTION_NAME} "
            f"top_k={settings.RETRIEVAL_TOP_K} rerank_k={settings.RERANK_TOP_K}"
        )

    def query(self, query: str) -> Dict:
        """
        Retrieve relevant chunks, rerank, synthesize and return an answer.

        Returns:
            {"answer": str, "sources": List[str]}
        """
        logger.info(f"RAGQueryTool.query | query={query!r}")

        nodes = self._retriever.retrieve(query)
        if not nodes:
            logger.warning("RAGQueryTool: no nodes retrieved.")
            return {"answer": "No relevant context found.", "sources": []}

        # Cross-encoder reranking
        pairs = [(query, node.node.get_content()) for node in nodes]
        logger.info(f"Reranking {len(nodes)} chunks...")
        scores = self._rerank_model.predict(pairs)

        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in ranked[: self._settings.RERANK_TOP_K]]

        # Synthesize answer
        response = self._synthesizer.synthesize(query, nodes=top_nodes)

        # Deduplicated source list
        seen: set = set()
        sources: List[str] = []
        for node in top_nodes:
            meta = node.node.metadata or {}
            fname = meta.get("file_name") or meta.get("filename") or meta.get("source")
            if fname and fname not in seen:
                seen.add(fname)
                sources.append(fname)

        logger.info(f"RAGQueryTool.query complete | sources={sources}")
        return {"answer": str(response), "sources": sources}


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_instance: RAGQueryTool | None = None


def get_rag_tool() -> RAGQueryTool:
    """Return the module-level RAGQueryTool singleton, creating it on first call."""
    global _instance
    if _instance is None:
        _instance = RAGQueryTool()
    return _instance
