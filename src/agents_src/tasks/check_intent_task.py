from crewai import Task, Crew, Process
from pydantic import BaseModel
from typing import List, Optional

from src.agents_src.agents.check_intent_agent import intent_agent

class ChatMessage(BaseModel):
    role: str
    content: str

class IntentOutput(BaseModel):
    fetch: bool
    use_rag: bool
    papers : List[dict]
    user_query: str
    chat_history: List[ChatMessage]

intent_task = Task(
    agent=intent_agent,
    name="Check Intent Task",
    description="""
    Analyze the user's intent and decide whether research papers need to be fetched and/or a question needs to be answered.
    You must call the fetch_paper_tool EXACTLY ONCE if fetching is required.

    Step 1: Normalize User Query Using Chat History
    - Read the full chat history to understand the conversational context.
    - Rewrite the latest user query into a fully self-contained, explicit question or topic.
    - Resolve references such as:
    - "it", "this", "that", "they"
    - vague follow-ups like "Can you explain it?" or "What about its limitations?"
    - Use chat history to infer the subject if the user query is ambiguous.
    - Remove all fetch-related phrases such as:
    - "fetch", "download", "get the paper", "find papers"
    - The final user_query must contain ONLY the coherent question or topic.
    Examples:
    - "Can you explain it?" → "Explain the transformer attention mechanism"
    - "Fetch papers and explain SHAP" → "Explain SHAP explanations"

    Step 2: Determine Intent
    - Set fetch = true if user explicitly asks to fetch/download/find papers
    - Set use_rag = true if the NORMALIZED user_query contains a coherent question that needs answering
        (Even if fetch=true, if there's a question part, use_rag should be true)
    - Example: "Fetch papers on SHAP explanations and explain it"
        → fetch: true (fetch request present)
        → use_rag: true (question "explain SHAP" is present)
        → user_query: "Explain SHAP explanations"
    - user_query must contain ONLY the question part.

    Step 3: Create Diverse Query and Category Lists (only if fetch = true)
    - Generate 1–5 short and diverse search queries (maximum 10 words each).
    - Queries should:
    - Use synonyms and paraphrases
    - Vary specificity (broad to specific)
    - Avoid keyword repetition
    - Aim for semantic diversity
    - Generate a separate list of arXiv categories:
    - Include a category only if reasonably confident
    - Otherwise use null
    - The fetch tool will internally try all (query × category) combinations.

    Step 4: Call Tool ONCE
    - Call fetch_paper_tool exactly once.
    - Do NOT retry or call it again.
    - Take the tool response as-is and return it in the papers field.

    Step 5: Return Result
    - fetch: true or false
    - use_rag: true or false
    - papers: response returned directly by fetch_paper_tool
    - user_query: normalized, question-only query
    - chat_history: pass through unchanged

    IMPORTANT RULES:
    - Do NOT answer the user's question.
    - Do NOT rank or filter papers.
    - Do NOT modify the fetch tool response.
    - Always return valid JSON strictly matching the IntentOutput schema.
    """,
    expected_output="""
    Return a valid JSON object with the following structure:

    {
    "fetch": boolean,
    "use_rag": boolean,
    "papers": [
        {
        "title": "paper title",
        "url": "paper pdf or landing page url"
        }
    ],
    "user_query": "normalized, self-contained question or topic",
    "chat_history": [
        {
        "role": "user | assistant",
        "content": "message text"
        }
    ]
    }

    Notes:
    - If fetch is false, papers must be an empty list.
    - user_query must NOT contain fetch-related phrases.
    - user_query must be understandable without chat history.
    - chat_history must be passed through exactly as received.

    """,
    output_pydantic=IntentOutput,
    max_iterations=1,
)

if __name__ == "__main__":

    intent_crew = Crew(
        agents=[intent_agent],
        tasks=[intent_task],
        verbose=True,
    )

    input_data = {
        "user_query": "Can you fetch the paper Local Interpretable Model Agnostic Shap Explanations for machine learning models",
        "chat_history": []
    }

    result = intent_crew.kickoff(input_data)
    print(result)