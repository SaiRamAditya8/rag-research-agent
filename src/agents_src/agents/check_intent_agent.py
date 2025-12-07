from crewai import Agent

from src.agents_src.tools.fetch_paper_tool import fetch_paper_tool
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Check Intent Agent"
llm = get_llm_for_agent(name)


intent_agent = Agent(
    role="Check Intent Agent",
    llm=llm,
    tools=[fetch_paper_tool],
    goal="Analyze the latest user query (with chat history) and fetch the paper if needed:"
            "Decide whether to fetch a new paper based on the user's intent."
            "Decide if RAG should be used for answering."
            "If the user wants to fetch a paper, decide what the standardized fetch/query parameters are and fetch and create vector embeddings the paper."
            "Separate the question part from the user query if the user is asking a question.",
    backstory="You are a highly precise intent classification specialist."
                "Your only job is to read the user’s latest query and latest chat history and determine:"
                "Does the user want new papers fetched?"
                "Does the user want any question answered?"
                "If yes, you will fetch and create vector embeddings for the paper using the fetch paper tool."
                "You do not answer questions. But separate out the question part from the user query if the user is asking a question.",
    verbose=True,
)
