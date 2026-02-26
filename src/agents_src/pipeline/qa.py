import logging
from typing import List

from src.agents_src.tools.rag_qa_tool import get_rag_tool
from src.agents_src.schemas import AnswerStructure

logger = logging.getLogger(__name__)


class QAPipeline:
    """
    Answers knowledge-seeking questions using RAG.

    Flow:
      1. Call RAGQueryTool.query(request) — retrieval + rerank + LlamaIndex synthesis.
      2. Optionally prepend a fetch acknowledgment (no extra LLM call).
      3. Return AnswerStructure.

    There is deliberately NO second LLM call here.  The LlamaIndex ResponseSynthesizer
    already produces a grounded answer; the CrewAI "agent framing" step was pure overhead.
    """

    def __init__(self):
        self._rag = get_rag_tool()

    def run(
        self,
        request: str,
        fetch: bool,
        papers: List[str],
    ) -> AnswerStructure:
        logger.info(f"QAPipeline.run | request={request!r} fetch={fetch}")

        rag_result = self._rag.query(request)
        answer: str = rag_result.get("answer", "")
        sources: List[str] = rag_result.get("sources", [])

        if not answer or answer.strip() == "Empty Response":
            answer = "The knowledge source does not contain the required information to answer this question."

        if fetch and papers:
            paper_list = "\n".join(f"  • {t}" for t in papers)
            fetch_note = (
                f"I fetched {len(papers)} paper(s) for you:\n{paper_list}\n\n"
            )
            answer = fetch_note + answer
        elif fetch and not papers:
            answer = (
                "I tried to fetch papers for your request but could not find any. "
                "Please try a different search term.\n\n" + answer
            )

        return AnswerStructure(
            answer=answer,
            sources=sources,
            tool_used="RAG Retriever",
            rationale="Answer grounded in retrieved and reranked document chunks via LlamaIndex.",
        )
