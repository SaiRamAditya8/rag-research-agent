import json
import logging
from typing import List

from src.agents_src.llm.client import LLMClient
from src.agents_src.schemas import AnswerStructure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a friendly and concise research assistant chatbot.
You handle conversational messages, greetings, and fetch acknowledgements.
Never mention internal flags, system state, or implementation details.
Keep responses natural and user-facing.
"""

_USER_PROMPT_TEMPLATE = """\
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


class ChitChatPipeline:
    """
    Handles greetings, small talk, and fetch-only confirmations via a direct LLM call.
    No retrieval tools are called here.
    """

    def __init__(self):
        self._llm = LLMClient()

    def run(
        self,
        user_query: str,
        chat_history: List[dict],
        chat_summary: str,
        fetch: bool,
        papers: List[str],
    ) -> AnswerStructure:
        logger.info(f"ChitChatPipeline.run | fetch={fetch} papers={papers}")

        history_str = json.dumps(chat_history, ensure_ascii=False) if chat_history else "[]"
        summary_str = chat_summary.strip() if chat_summary else "No summary yet."
        papers_str = json.dumps(papers) if papers else "[]"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            chat_history=history_str,
            chat_summary=summary_str,
            fetch=fetch,
            papers=papers_str,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            answer = self._llm.complete(messages, agent_name="ChitChat Agent")
        except Exception as e:
            logger.error(f"ChitChatPipeline LLM call failed: {e}")
            answer = "Sorry, I encountered an error. Please try again."

        rationale = "fetch-acknowledgement" if fetch else "chitchat response"

        return AnswerStructure(
            answer=answer,
            sources=[],
            tool_used=None,
            rationale=rationale,
        )
