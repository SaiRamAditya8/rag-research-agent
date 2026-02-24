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
    queries : List[str]
    categories : List[str]
    request : str

intent_task = Task(
    agent=intent_agent,
    name="Check Intent Task",
    description="""
    Analyze the user's intent using the latest user message "{user_query}".
    Context:
    - Recent history (last 5 turns): "{chat_history}"
    - Conversation Summary: "{chat_summary}"
    
    Determine whether research papers need to be fetched and/or whether a question needs to be answered using RAG.
    Do NOT call any tools. Only analyze and decide intent based on user input and history.

    Step 1: Normalize User Query Using Chat History
    - Read the full chat history to understand the conversational context.
    - Rewrite the latest user query into a fully self-contained, explicit question or topic.
    - Resolve references such as: "it", "this", "that", "they"
    - Resolve vague follow-ups like "Can you explain it?" or "What about its limitations?"
    - Use chat history to infer the subject if the user query is ambiguous.
    - Remove all fetch-related phrases like: "fetch", "download", "get the paper", "find papers"
    - The final request must contain ONLY the coherent question or topic.
    Examples:
    - "Can you explain it?" → "Explain the transformer attention mechanism"
    - "Fetch papers and explain SHAP" → "Explain SHAP explanations"

    Step 2: Determine Intent Flags (NO TOOL CALLS)
    - Set fetch = true if user explicitly asks to fetch/download/find/search for papers
    - Set use_rag = true if the NORMALIZED request is a KNOWLEDGE-SEEKING question (not chitchat/greetings)
    - Knowledge-seeking questions: "Explain X", "What is Y", "How does Z work", "Compare A and B"
    - Chitchat/greetings (use_rag = false): "How are you?", "Hi", "Hello", "What's up?", "How are you doing?"
    - Even if fetch=true, if there's a knowledge-seeking question part, use_rag should be true
    - Example: "Fetch papers on SHAP explanations and explain it"
        → fetch: true (fetch request present)
        → use_rag: true (knowledge-seeking question "explain SHAP" is present)
        → request: "Explain SHAP explanations"
    - request must contain ONLY the knowledge-seeking question part.
    - If request is only chitchat (greeting, how are you, small talk), set use_rag = false and request = original message

    Step 3: Create Query and Category Lists (only if fetch = true)
    - If fetch = true, generate 1–5 short search queries (maximum 10 words each).
    - CRITICAL: The FIRST query MUST be the specific paper title or exact topic the user named.
      - Remove any fetch-related words (e.g., "fetch", "download", "find", "get").
      - It should be exactly what the user is looking for.
    - DELIMITER RULE: If the user encloses the title/topic in any delimiter, extract it EXACTLY as the first query:
      - Curly braces: {{ content }}  |  Brackets: [content]  |  Parentheses: (content)
      - Double Quotes: "content"     |  Single Quotes: 'content'
      - Example: "fetch {{ Attention Is All You Need }}" → first query: "Attention Is All You Need"
    - TITLE LOOKUP vs TOPIC SEARCH — this distinction is critical for retrieval quality:
      - TITLE LOOKUP: user asks for a specific paper by name (e.g. "fetch Attention is all you need", "get the BERT paper"):
        - Return ONLY that exact title as the SINGLE query. Do NOT add topic expansion queries.
        - Adding diverse queries like "transformer self-attention" will surface wrong papers that compete with the correct one.
      - TOPIC SEARCH: user asks for papers on a subject (e.g. "fetch papers on attention mechanisms", "find recent RAG papers"):
        - Generate 2–5 diverse queries using synonyms, paraphrases, and varying specificity.
        - Aim for semantic diversity to surface a range of relevant papers.
    - Generate a separate list of arXiv categories:
        - Include a category only if reasonably confident
        - Otherwise use null
    - If fetch = false, return empty lists for both queries and categories.

    Step 4: Return Result
    - fetch: true or false
    - use_rag: true or false
    - queries: list of search queries (empty if fetch = false)
    - categories: list of arXiv categories (empty if fetch = false)
    - request: normalized, self-contained question

    IMPORTANT RULES:
    - Do NOT answer the user's question.
    - Always return valid JSON strictly matching the IntentOutput schema.
    """,
    expected_output="""
    Return a valid JSON object with the following structure:

    {
    "fetch": boolean,
    "use_rag": boolean,
    "queries": ["query1", "query2", ...],
    "categories": ["category1", "category2", ...],
    "request": "normalized, self-contained question or topic"
    }

    Notes:
    - If fetch is false, queries and categories must be empty lists.
    - request must NOT contain fetch-related phrases.
    - request must be understandable without chat history.
    - queries are diverse search queries generated to capture different aspects of the topic.
    - categories are arXiv categories that match the queries (can be null values if uncertain).

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

    user_query = "Hi there!"
    # user_query = "Can you fetch the paper Local Interpretable Model Agnostic Shap Explanations for machine learning models"

    input_data = {
        "user_query": user_query,
        "chat_history": []
    }

    result = intent_crew.kickoff(input_data)
    print(result)