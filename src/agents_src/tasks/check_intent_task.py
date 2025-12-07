from crewai import Task, Crew, Process
from pydantic import BaseModel
from typing import List

from src.agents_src.agents.check_intent_agent import intent_agent

class ChatMessage(BaseModel):
    role: str
    content: str

class IntentUse(BaseModel):
    title: str  
    category: str = "cs.AI"  # Default category

class IntentOutput(BaseModel):
    fetch: bool
    use_rag: bool
    papers: List[str]
    user_query: str
    chat_history: List[ChatMessage]


intent_task = Task(
    agent=intent_agent,
    name="Check Intent Task",
    description="""
    Understand user intent using the user query "{user_query}" and chat_history: "{chat_history}".
    Decide if the user wants to fetch any paper and fetch if needed.
    Pass through the chat history but modify the user query to separate out the question part if the user is asking a question.
    
    Instructions:
    - Do not call the tool multiple times.
    - Parse the full chat history and the latest user query.
    - Output MUST strictly follow the IntentOutput schema.
    - Decide if any paper needs to be fetched. Fetch only if the user explicitly asks for a paper or research article to be fetched.
    - If yes, use the fetch paper tool to fetch the most relevant paper.
    - Decide if RAG should be used for answering or if this is just a fetch request.
    - Do not answer any question.
    - Pass through the chat history.
    - Identify the question part in the user query.
    - Try and tweak the user query to only contain the question part if the user is asking a question.
    """,
    expected_output="""
    A structured JSON object with the following fields:
    {
      "fetch": "Boolean indicating if papers were fetched. This will be false if no paper fetch was needed or fetch failed",
      "use_rag": "Boolean indicating if RAG should be used for answering indicating if the user asked any question apart from fetching paper",
      "papers": "List of fetched papers with their titles and links returned by the fetch paper tool. If the tool failed to fetch or no fetch was needed, this will be None or an empty list.",
      "user_query": "Tweaked user query containing only the question part if the user is asking a question. If no question is asked, pass the user query as it is.",
      "chat_history": "The full chat history as a list of messages with roles and content. Pass it as it is."
    }
    """,
    output_pydantic=IntentOutput,
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