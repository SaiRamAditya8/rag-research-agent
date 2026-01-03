from crewai import Task
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents_src.agents.chitchat_agent import chitchat_agent


class AnswerStructure(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    tool_used: str
    rationale: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
  user_query: str
  chat_history: List[ChatMessage]
  fetch: bool
  papers: List[str]
  request: str

chitchat_task = Task(
    agent=chitchat_agent,
    name="ChitChat Task",
    description="""
    A lightweight chitchat handler. This task is invoked when the Intent Agent set "use_rag" to false.

    You receive the following input fields (see ChatInput):
    - user_query: the original user message
    - chat_history: prior conversation (excluding the latest user message)
    - request: normalized query produced by the Intent Agent (may be empty)
    - fetch: whether a fetch attempt was made
    - papers: list of fetched paper titles (may be empty)

    Instructions:
    - Do NOT call any retrieval tools.
    - If the message is a greeting or casual chitchat (e.g., "hi", "thanks"), respond naturally and briefly.
    - If fetch = true and papers is non-empty, acknowledge the fetch attempt and mention the number or list of papers fetched.
    - If fetch = true but papers is empty, acknowledge that a fetch was attempted but no papers are available yet.
    - If the normalized request contains a real question but use_rag is false, provide a short, best-effort response based on general knowledge (do NOT hallucinate facts about fetched documents).
    - Keep responses concise, friendly, and user-facing.

    Output must match the AnswerStructure schema.

    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
      "answer": "Direct response to the query (1–3 paragraphs, clear and accurate).",
      "sources": None or empty list # No sources since RAG is not used,
      "tool_used": None,
      "rationale": "Brief explanation of why this answer was chosen (e.g., 'chitchat response' or 'no papers available')"
    }
    """,
    output_pydantic=AnswerStructure,
    input_pydantic=ChatInput,
)
