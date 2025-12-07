from crewai import Task
from pydantic import BaseModel
from typing import List, Optional

from src.agents_src.agents.question_answer_agent import qa_agent


class AnswerStructure(BaseModel):
    answer: str
    sources: list[str]
    tool_used: str
    rationale: str

class ChatMessage(BaseModel):
    role: str
    content: str

class IntentOutput(BaseModel):
    fetch: bool
    use_rag: bool
    papers: List[str]
    user_query: str
    chat_history: List[ChatMessage]

qa_task = Task(
    agent=qa_agent,
    name="Question Answering Task",
    description="""
    Answer the user query "{user_query}" using a Retrieval-Augmented Generation (RAG) pipeline.
    chat_history: "{chat_history}" only if "use_rag" is true. 
    Use "papers" as metadata if it is not None or empty to append to your answer that these documents have been fetched.
    
    Instructions:
    - If fetch is true and papers list is not empty, acknowledge that these papers have been fetched in your answer first.
    - Then retrieve relevant context from the document store using the RAG retriever tool only if use_rag is true and user_query contains a question.
    - Use the chat history if use_rag is true to provide context in your answer.
    - If use_rag is false, just respond by specifying the list of papers fetched in natural tone.
    - Prioritize evidence that directly addresses the query
    - Synthesize a clear, accurate answer grounded in the retrieved sources or chat history
    - If the query cannot be answered from the knowledge source or chat history, do not generate your own response.
      Instead, state clearly that the knowledge source does not contain the required information.
    - Provide transparency by including references, tool usage, and reasoning steps
    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
      "answer": "Direct response to the query (1–3 paragraphs, clear and accurate). 
                 If no answer is found and no papers were fetched, return: 'The knowledge source does not contain the required information.'",
      "sources": ["Pass through the papers list if not None or empty, else an empty list"],
      "tool_used": "Name of the retrieval/analysis tool invoked (e.g., RAG Retriever, VectorDB, ChatHistory, etc.)",
      "rationale": "Brief explanation of why this answer was chosen, or why no relevant information was found"
    }
    """,
    output_pydantic=AnswerStructure,
    input_pydantic=IntentOutput
)
