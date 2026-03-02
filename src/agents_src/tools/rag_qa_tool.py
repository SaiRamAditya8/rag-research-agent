import logging
from typing import Dict, List

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.agents_src.config.agent_settings import AgentSettings
from src.agents_src.llm.client import LLMClient
from src.agents_src.llm.models import embed_model, rerank_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthesis prompts
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = (
    "You are a precise research assistant. "
    "Answer questions strictly based on the provided research context. "
    "If the context does not contain enough information to answer the question, say so clearly. "
    "Do not hallucinate or add information beyond what is in the context."
)

_SYNTHESIS_USER = (
    "Answer the following question using ONLY the research context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}"
)


class RAGQueryTool:
    """
    Singleton RAG query tool.

    ChromaDB client, LlamaIndex index, and retriever are initialised ONCE at
    construction time and reused across all calls.  New documents ingested by
    paper_fetcher are visible immediately (shared PersistentClient path).

    Synthesis is performed by LLMClient (gpt-4o), keeping all LLM calls in
    one place and making the provider trivially swappable.
    """

    def __init__(self):
        settings = AgentSettings()
        self._settings = settings
        self._rerank_model = rerank_model
        self._llm = LLMClient()

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

        logger.info(
            f"RAGQueryTool initialised | collection={settings.COLLECTION_NAME} "
            f"top_k={settings.RETRIEVAL_TOP_K} rerank_k={settings.RERANK_TOP_K}"
        )

    def query(self, query: str) -> Dict:
        """
        Retrieve relevant chunks, rerank, synthesize via LLMClient, and return an answer.

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

        # Synthesize answer via LLMClient (gpt-4o)
        context = "\n\n---\n\n".join(node.node.get_content() for node in top_nodes)
        answer = self._llm.complete(
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": _SYNTHESIS_USER.format(context=context, query=query)},
            ],
            agent_name="QA Agent",
        )

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
        return {"answer": answer, "sources": sources}


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