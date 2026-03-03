import logging
from typing import Dict, List, Optional

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
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

    ChromaDB client, LlamaIndex index, and a global (unfiltered) retriever are
    initialised ONCE at construction time and reused across all calls.

    Filtered retrieval (by paper_title or project_id) creates a lightweight
    per-query retriever on top of the same shared index — no re-initialisation.

    Synthesis is performed by LLMClient (gpt-4o).
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
            f"top_k={settings.RETRIEVAL_TOP_K} rerank_k={settings.RERANK_TOP_K} "
            f"(global DB, paper-title-scoped at query time)"
        )

    def _build_retriever(
        self,
        paper_filter: Optional[List[str]],
        project_papers: Optional[List[str]],
    ):
        """
        Return a retriever scoped by paper title.
        Falls back to the global unfiltered retriever when no scope is given.

        Priority: paper_filter (explicit selection) > project_papers (all project papers) > global
        """
        titles = paper_filter or project_papers
        if not titles:
            return self._retriever

        f = MetadataFilter(
            key="paper_title",
            value=titles,
            operator=FilterOperator.IN,
        )
        return self._index.as_retriever(
            similarity_top_k=self._settings.RETRIEVAL_TOP_K,
            filters=MetadataFilters(filters=[f]),
        )

    def query(
        self,
        query: str,
        paper_filter: Optional[List[str]] = None,
        project_papers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Retrieve relevant chunks, rerank, synthesize via LLMClient, return answer.

        Args:
            query: the question to answer
            paper_filter: if set, restrict to these specific paper titles (user-specified)
            project_papers: if set (and paper_filter is empty), restrict to all papers
                            in the current project (by title)

        Returns:
            {"answer": str, "sources": List[str]}
        """
        logger.info(
            f"RAGQueryTool.query | query={query!r} "
            f"paper_filter={paper_filter} project_papers={project_papers}"
        )

        retriever = self._build_retriever(paper_filter, project_papers)
        nodes = retriever.retrieve(query)

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

        # Deduplicated source list — prefer paper_title, fall back to filename
        seen: set = set()
        sources: List[str] = []
        for node in top_nodes:
            meta = node.node.metadata or {}
            label = (
                meta.get("paper_title")
                or meta.get("file_name")
                or meta.get("filename")
                or meta.get("source")
            )
            if label and label not in seen:
                seen.add(label)
                sources.append(label)

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
