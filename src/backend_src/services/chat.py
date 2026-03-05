import logging

from src.agents_src.pipeline.intent import IntentPipeline
from src.agents_src.pipeline.answer import AnswerPipeline
from src.agents_src.utils.paper_fetcher import fetch_papers_and_ingest
from src.backend_src.projects.store import project_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level pipeline singletons — constructed once at import time so the
# RAGQueryTool (ChromaDB + LlamaIndex) and LLMClient (OpenAI) are shared
# across all requests.
# ---------------------------------------------------------------------------
_intent_pipeline = IntentPipeline()
_answer_pipeline = AnswerPipeline()


def get_answer(user_query: str, project_id: str = "default") -> dict:
    logger.info(f"Processing query for project {project_id}: {user_query!r}")

    # 1. Update memory + display history with user message
    project_store.add_message(project_id, "user", user_query)
    project_store.add_display_message(project_id, {"role": "user", "content": user_query})

    # 2. Retrieve current project state
    project = project_store.get_project(project_id)
    chat_summary = project.chat_summary
    chat_buffer = project.chat_buffer
    fetched_papers = project.fetched_papers

    # 3. Classify intent (with paper context so references can be resolved)
    intent = _intent_pipeline.run(
        user_query=user_query,
        chat_history=chat_buffer,
        chat_summary=chat_summary,
        fetched_papers=fetched_papers,
    )
    logger.info(
        f"Intent | fetch={intent.fetch} use_rag={intent.use_rag} "
        f"paper_filter={intent.paper_filter} request={intent.request!r}"
    )

    # 4. Fetch papers if requested
    fetch_occurred = intent.fetch
    newly_fetched = []

    if fetch_occurred:
        result = fetch_papers_and_ingest(
            queries=intent.queries,
            categories=intent.categories,
        )
        newly_fetched = result if result else []
        if newly_fetched:
            project_store.add_fetched_papers(project_id, newly_fetched)

    papers_list = [p["title"] for p in newly_fetched if p and p.get("title")]

    # 5. Answer (RAG or chitchat — decided inside AnswerPipeline)
    # Scope RAG to papers in this project; intent may further narrow to paper_filter.
    current_project_papers = [p["title"] for p in project_store.get_project(project_id).fetched_papers]
    answer_obj = _answer_pipeline.run(
        intent=intent,
        user_query=user_query,
        chat_history=chat_buffer,
        chat_summary=chat_summary,
        fetch=fetch_occurred,
        papers=papers_list,
        project_papers=current_project_papers,
    )
    rag_occurred = intent.use_rag

    # 6. Update memory + display history with assistant response
    current_papers = project_store.get_project(project_id).fetched_papers
    if answer_obj.answer:
        project_store.add_message(project_id, "assistant", answer_obj.answer)
        project_store.add_display_message(project_id, {
            "role": "assistant",
            "content": answer_obj.answer,
            "sources": answer_obj.sources,
            "tool_used": answer_obj.tool_used,
            "rationale": answer_obj.rationale,
            "fetched_papers": current_papers,
        })

    # 7. Update summary if triggered
    if project_store.summary_update_needed(project_id, fetch_occurred, rag_occurred):
        project_store.update_summary(project_id)

    # 8. Return answer + current project paper list for frontend sync
    result = answer_obj.model_dump()
    result["fetched_papers"] = project_store.get_project(project_id).fetched_papers
    return result
