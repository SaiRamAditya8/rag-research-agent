import json
import logging
from typing import List

from src.agents_src.llm.client import LLMClient
from src.agents_src.schemas import IntentOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise intent classification specialist for a research assistant chatbot.
Analyze user queries and chat history to determine research intent.
You MUST respond with valid JSON matching the schema described in the user message.
Do NOT include any explanation or markdown — only the raw JSON object.
"""

_USER_PROMPT_TEMPLATE = """\
Analyze the user's intent using the latest user message: "{user_query}"

Context:
- Recent history (last 5 turns): {chat_history}
- Conversation Summary: {chat_summary}

Determine whether research papers need to be fetched and/or whether a question needs to be answered using RAG.
Do NOT answer the user's question. Only analyze and decide intent.

Step 1: Determine Intent Flags (work from the RAW message — before any normalization)
- Set fetch = true if the raw message contains any of: fetch, download, find papers, search for papers, get the paper, pull papers.
  Even if the rest of the message has typos, fetch = true whenever one of these words is clearly present.
- Set use_rag = true ONLY if the raw message contains an explicit question or request for explanation IN ADDITION to (or instead of) a fetch.
  DO NOT set use_rag = true for a pure fetch request where the user only asked to get/find/download a paper with no follow-up question.
- use_rag = true examples: "fetch and explain SHAP", "what is attention?", "how does RLHF work", "fetch LoRA and summarize it"
- use_rag = false examples: "fetch the paper X", "get me papers on X", "download attention is all you need", "find papers on transformers"
- Chitchat (use_rag = false, fetch = false):
    - Greetings/small talk: "Hi", "Hello", "How are you?", "What's up?", "Thanks"
    - Questions about the assistant itself: "What model are you?", "Who made you?", "What can you do?", "Are you GPT?"

Step 2: Normalize the User Query
- Rewrite the user message into a fully self-contained, explicit question or topic.
- Resolve references: "it", "this", "that", "they" — use chat history to infer the subject.
- Resolve vague follow-ups: "Can you explain it?" → "Explain the transformer attention mechanism"
- REMOVE all fetch-related phrases (fetch, download, get the paper, find papers) from the normalized text.
- The result (request) must contain ONLY the coherent question or topic — no fetch verbs.
- CRITICAL: If the message is a pure fetch with no question, do NOT invent a question. Set request = the paper title or topic only.
Examples:
  "Can you explain it?" → "Explain the transformer attention mechanism"
  "Fetch papers and explain SHAP" → "Explain SHAP explanations"
  "fetch the paper attention si you ned" → "Attention Is All You Need"
  "get me papers on transformers" → "transformers"
- If purely chitchat, set request = original message.

Step 3: Create Query and Category Lists (only if fetch = true)
- If fetch = true, generate 1-5 short search queries (maximum 10 words each).
- CRITICAL: The FIRST query MUST be the specific paper title or exact topic the user named.
  Remove any fetch-related words from it.
- DELIMITER RULE: If the user encloses the title/topic in any delimiter, extract it EXACTLY as the first query:
  Curly braces: {curly_example}, Brackets: [content], Parentheses: (content),
  Double Quotes: "content", Single Quotes: 'content'
- TITLE LOOKUP vs TOPIC SEARCH — this distinction is critical:
  TITLE LOOKUP (user asks for a specific paper by name):
    Return ONLY that exact title as the SINGLE query. Do NOT add expansion queries.
  TOPIC SEARCH (user asks for papers on a subject):
    Generate 2-5 diverse queries using synonyms, paraphrases, and varying specificity.
- Generate a separate list of arXiv categories if reasonably confident, else use empty list.
- If fetch = false, return empty lists for both queries and categories.

Step 4: Return a JSON object (and nothing else) with this exact structure:
{{
  "fetch": <boolean>,
  "use_rag": <boolean>,
  "queries": ["query1", "query2"],
  "categories": ["cs.LG"],
  "request": "normalized self-contained question"
}}

Rules:
- If fetch is false, queries and categories must be empty lists.
- request must NOT contain fetch-related phrases and must be self-contained.
"""


class IntentPipeline:
    """
    Classifies user intent via a single direct Groq API call.

    Returns an IntentOutput Pydantic model with:
      fetch     — whether to fetch papers
      use_rag   — whether to run RAG
      queries   — arXiv search queries (populated only when fetch=True)
      categories — arXiv category hints (populated only when fetch=True)
      request   — normalized, self-contained question for downstream steps
    """

    def __init__(self):
        self._llm = LLMClient()

    def run(
        self,
        user_query: str,
        chat_history: List[dict],
        chat_summary: str,
    ) -> IntentOutput:
        history_str = json.dumps(chat_history, ensure_ascii=False) if chat_history else "[]"
        summary_str = chat_summary.strip() if chat_summary else "No summary yet."

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            chat_history=history_str,
            chat_summary=summary_str,
            curly_example="{{ content }}",
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = self._llm.complete(messages, agent_name="Intent Agent", json_mode=True)
            intent = IntentOutput.model_validate_json(raw)
            logger.info(
                f"IntentPipeline | fetch={intent.fetch} use_rag={intent.use_rag} "
                f"queries={intent.queries} request={intent.request!r}"
            )
            return intent
        except Exception as e:
            logger.error(f"IntentPipeline failed: {e}. Defaulting to chitchat.")
            return IntentOutput(
                fetch=False,
                use_rag=False,
                queries=[],
                categories=[],
                request=user_query,
            )
