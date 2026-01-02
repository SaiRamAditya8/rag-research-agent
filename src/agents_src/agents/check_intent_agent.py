from crewai import Agent

from src.agents_src.tools.fetch_paper_tool import fetch_paper_tool
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Check Intent Agent"
llm = get_llm_for_agent(name)


intent_agent = Agent(
    role="Check Intent Agent",
    llm=llm,
    tools=[fetch_paper_tool],
    goal="Analyze the latest user query and decide if papers need to be fetched. If yes, use the fetch_paper_tool to fetch and ingest them.",
    backstory="You are a precise intent classification specialist. Your job is to determine if the user wants new research papers fetched. "
                "If they do, analyze their query to extract search terms and categories, then use the fetch_paper_tool to fetch and ingest the papers. "
                "Do not answer questions yourself - only fetch papers if requested.",
    verbose=True,
)
