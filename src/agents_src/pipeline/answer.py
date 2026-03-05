import json
import logging
from typing import List, Optional

from src.agents_src.llm.client import LLMClient
from src.agents_src.schemas import AnswerStructure, IntentOutput
from src.agents_src.tools.rag_qa_tool import get_rag_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ChitChat prompt (used when use_rag=False)
# ---------------------------------------------------------------------------

_CHITCHAT_SYSTEM = """\
You are a friendly and concise research assistant chatbot powered by OpenAI's GPT models \
(gpt-4o-mini for conversation, gpt-4o for research synthesis).
You help users find and understand research papers.
You handle conversational messages, greetings, and fetch acknowledgements.
Never mention internal flags, system state, or implementation details.
Keep responses natural and user-facing.
If asked what model or AI you are, mention you are powered by OpenAI GPT models.
"""

_CHITCHAT_USER = """\
User message: {user_query}
Conversation history: {chat_history}
Conversation summary: {chat_summary}
Was a paper fetch attempted this turn: {fetch}
Papers successfully fetched: {papers}

Instructions:
- If a fetch was attempted AND papers is non-empty: confirm success and list the paper titles clearly.
- If a fetch was attempted AND papers is empty: apologise and let the user know no papers were found.
- If no fetch: respond naturally to the user message. Reply briefly for greetings/small talk.
  For genuine questions, give a short best-effort answer without accessing any retrieval tools.

Respond in plain prose. Do not use any special formatting, headers, or bullet lists unless listing fetched papers.
"""


class AnswerPipeline:
    """
    Single pipeline that handles both RAG-grounded answers and conversational responses.

    Branches internally on intent.use_rag:
      - True  → retrieve from vector store, rerank, synthesize with gpt-4o (RAG path)
      - False → direct gpt-4o-mini call for chitchat / fetch-only acknowledgements
    """

    def __init__(self):
        self._rag = get_rag_tool()
        self._llm = LLMClient()

    def run(
        self,
        intent: IntentOutput,
        user_query: str,
        chat_history: List[dict],
        chat_summary: str,
        fetch: bool,
        papers: List[str],
        project_papers: Optional[List[str]] = None,
    ) -> AnswerStructure:
        if intent.use_rag:
            return self._rag_answer(
                intent.request, user_query, chat_history, chat_summary,
                fetch, papers, intent.paper_filter or [], project_papers,
            )
        else:
            return self._chitchat_answer(user_query, chat_history, chat_summary, fetch, papers)

    # ------------------------------------------------------------------
    # RAG path  (falls back to direct LLM if nothing is retrieved)
    # ------------------------------------------------------------------

    def _rag_answer(
        self,
        request: str,
        user_query: str,
        chat_history: List[dict],
        chat_summary: str,
        fetch: bool,
        papers: List[str],
        paper_filter: List[str],
        project_papers: Optional[List[str]],
    ) -> AnswerStructure:
        logger.info(
            f"AnswerPipeline._rag_answer | request={request!r} "
            f"paper_filter={paper_filter} project_papers={project_papers}"
        )

        rag_result = self._rag.query(
            request,
            paper_filter=paper_filter or None,
            project_papers=project_papers,
        )
        answer: str = rag_result.get("answer", "")
        sources: List[str] = rag_result.get("sources", [])

        # If nothing was retrieved, the vector store has no relevant content.
        # Fall back to a direct LLM answer rather than returning a dead-end message.
        # This also covers intent misclassification (e.g. "What model are you?" routed to RAG).
        if not sources:
            logger.info("AnswerPipeline: no sources retrieved — falling back to direct LLM answer.")
            return self._chitchat_answer(user_query, chat_history, chat_summary, fetch, papers)

        if not answer or answer.strip() == "Empty Response":
            answer = "The knowledge source does not contain the required information to answer this question."

        if fetch and papers:
            paper_list = "\n".join(f"  • {t}" for t in papers)
            answer = f"I fetched {len(papers)} paper(s) for you:\n{paper_list}\n\n" + answer
        elif fetch and not papers:
            answer = (
                "I tried to fetch papers for your request but could not find any. "
                "Please try a different search term.\n\n" + answer
            )

        return AnswerStructure(
            answer=answer,
            sources=sources,
            tool_used="RAG Retriever",
            rationale="Answer grounded in retrieved and reranked document chunks.",
        )

    # ------------------------------------------------------------------
    # Chitchat / fetch-acknowledgement path
    # ------------------------------------------------------------------

    def _chitchat_answer(
        self,
        user_query: str,
        chat_history: List[dict],
        chat_summary: str,
        fetch: bool,
        papers: List[str],
    ) -> AnswerStructure:
        logger.info(f"AnswerPipeline._chitchat_answer | fetch={fetch} papers={papers}")

        history_str = json.dumps(chat_history, ensure_ascii=False) if chat_history else "[]"
        summary_str = chat_summary.strip() if chat_summary else "No summary yet."

        user_prompt = _CHITCHAT_USER.format(
            user_query=user_query,
            chat_history=history_str,
            chat_summary=summary_str,
            fetch=fetch,
            papers=json.dumps(papers) if papers else "[]",
        )

        try:
            answer = self._llm.complete(
                messages=[
                    {"role": "system", "content": _CHITCHAT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                agent_name="ChitChat Agent",
            )
        except Exception as e:
            logger.error(f"AnswerPipeline chitchat LLM call failed: {e}")
            answer = "Sorry, I encountered an error. Please try again."

        rationale = "fetch-acknowledgement" if fetch else "chitchat response"
        return AnswerStructure(answer=answer, sources=[], tool_used=None, rationale=rationale)