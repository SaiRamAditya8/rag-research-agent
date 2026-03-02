import logging

from src.agents_src.pipeline.intent import IntentPipeline
from src.agents_src.pipeline.answer import AnswerPipeline
from src.agents_src.utils.paper_fetcher import fetch_papers_and_ingest
from src.backend_src.memory.session_store import session_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level pipeline singletons — constructed once at import time so the
# RAGQueryTool (ChromaDB + LlamaIndex) and LLMClient (OpenAI) are shared
# across all requests.
# ---------------------------------------------------------------------------
_intent_pipeline = IntentPipeline()
_answer_pipeline = AnswerPipeline()


def get_answer(user_query: str, session_id: str = "default_session") -> dict:
    logger.info(f"Processing query for session {session_id}: {user_query!r}")

    # 1. Update memory with user message
    session_store.add_message(session_id, "user", user_query)

    # 2. Retrieve current memory state
    memory = session_store.get_memory(session_id)
    chat_summary = memory["chat_summary"]
    chat_buffer = memory["chat_buffer"]

    # 3. Classify intent
    intent = _intent_pipeline.run(
        user_query=user_query,
        chat_history=chat_buffer,
        chat_summary=chat_summary,
    )
    logger.info(f"Intent | fetch={intent.fetch} use_rag={intent.use_rag} request={intent.request!r}")

    # 4. Fetch papers if requested
    fetch_occurred = intent.fetch
    fetched_papers = []

    if fetch_occurred:
        result = fetch_papers_and_ingest(
            queries=intent.queries,
            categories=intent.categories,
        )
        fetched_papers = result if result else []

    papers_list = [p["title"] for p in fetched_papers if p and p.get("title")]

    # 5. Answer (RAG or chitchat — decided inside AnswerPipeline)
    answer_obj = _answer_pipeline.run(
        intent=intent,
        user_query=user_query,
        chat_history=chat_buffer,
        chat_summary=chat_summary,
        fetch=fetch_occurred,
        papers=papers_list,
    )
    rag_occurred = intent.use_rag

    # 6. Update memory with assistant response
    if answer_obj.answer:
        session_store.add_message(session_id, "assistant", answer_obj.answer)

    # 7. Update summary if triggered
    if session_store.summary_update_needed(session_id, fetch_occurred, rag_occurred):
        session_store.update_summary(session_id)

    return answer_obj.model_dump()