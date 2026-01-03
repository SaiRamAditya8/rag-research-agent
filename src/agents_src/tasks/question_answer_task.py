from crewai import Task
from pydantic import BaseModel, Field
from typing import List, Optional

from src.agents_src.agents.question_answer_agent import qa_agent


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

qa_task = Task(
    agent=qa_agent,
    name="Question Answering Task",
    description="""
    Answer the user's question using Retrieval-Augmented Generation (RAG).

    You are given:
    - The original user message as: "{user_query}"
    - The normalized, self-contained question produced by the Intent Agent as `request`: "{request}"
    - The full chat history as: "{chat_history}"
    - A boolean `fetch` indicating whether a fetch attempt was made and `papers` containing fetched paper titles (may be empty)

    Instructions:
    - ALWAYS call the `rag_query_tool` with the normalized `request` to retrieve supporting context.
    - Use `user_query`, `chat_history`, `fetch`, and `papers` to add context and acknowledgements to the final answer.
    - Use `chat_history` only to resolve references and maintain minimal conversational continuity; the authoritative question to answer is `request`.
    - Synthesize a clear, accurate answer strictly grounded in retrieved evidence. Do NOT hallucinate.

    - If RAG retrieval returns no relevant results, clearly state: "The knowledge source does not contain the required information to answer this question." Do NOT fabricate answers.
    - In the final response, if `fetch` is true include an acknowledgement of the fetch attempt and the number of fetched papers (from `papers`).
    - Transparently include `tool_used` set to "RAG Retriever" and a short `rationale` about how the answer was generated.

    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
      "answer": "Direct, grounded answer to the normalized `request`. If `fetch` is true, prepend/append an acknowledgement about the fetch attempt and the number of fetched papers.",
      "sources": ["List of source files or documents used to generate the answer given by the RAG tool"],
      "tool_used": "RAG Retriever",
      "rationale": "Brief explanation of how the answer was derived from retrieved sources"
    }
    """,
    output_pydantic=AnswerStructure,
    input_pydantic=ChatInput,
)
