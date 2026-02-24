from crewai import Agent
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Check Intent Agent"
llm = get_llm_for_agent(name)


intent_agent = Agent(
    role="Check Intent Agent",
    llm=llm,
    goal="Analyze user intent to determine whether research papers need to be fetched and/or a question needs to be answered using RAG.",
    backstory="You are a precise intent classification specialist. Your job is to analyze user queries and chat history to determine if the user wants papers fetched (fetch flag) and/or if they have a question to answer (use_rag flag). You normalize queries and provide clear intent signals without performing any actions.",
    verbose=True,
)
