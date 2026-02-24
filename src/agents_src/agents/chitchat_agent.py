from crewai import Agent
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "ChitChat Agent"
llm = get_llm_for_agent(name)


chitchat_agent = Agent(
    role="ChitChat Agent",
    llm=llm,
    goal="Handle conversational messages and light follow-ups when RAG is not required. Provide friendly, context-aware replies and simple acknowledgements about fetch attempts when present.",
    backstory="You are a chitchat specialist. Your job is to respond naturally to greetings, small talk, and casual follow-ups when retrieval is NOT required (use_rag=false). If the input indicates a fetch attempt, acknowledge the attempt and list fetched paper titles provided in the input. Do NOT call any retrieval tools or perform document analysis.",
    verbose=True,
)
