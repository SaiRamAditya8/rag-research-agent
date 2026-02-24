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
  chat_summary: str
  fetch: bool
  papers: List[str]
  request: str

chitchat_task = Task(
    agent=chitchat_agent,
    name="ChitChat Task",
    description="""
    You are handling a conversational response. Do NOT call any retrieval tools.

    User message: "{user_query}"
    Conversation history: "{chat_history}"
    Conversation summary: "{chat_summary}"
    Was a paper/topic fetch attempted this turn: {fetch}
    Papers successfully fetched this turn: {papers}

    Instructions — respond based on the situation:
    - If {fetch} is true AND {papers} is non-empty: confirm the fetch succeeded.
      List the fetched paper titles clearly. Example: "I've fetched the following papers for you: ..."
    - If {fetch} is true AND {papers} is empty: apologise and let the user know no papers were found for their request.
    - If {fetch} is false: this is a general conversation turn. Respond naturally to the user message.
      If it is a greeting or small talk, reply briefly. If it is a genuine question, give a short best-effort answer.
    - Never mention internal flags, system state, or implementation details in your response.
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
