from crewai import Agent

from src.agents_src.tools.rag_qa_tool import rag_query_tool
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Question Answer Agent"
llm = get_llm_for_agent(name)


qa_agent = Agent(
    role="Question Answer Agent",
    llm=llm,
    tools=[rag_query_tool],
    goal="Answer user questions by retrieving relevant context from documents using RAG. Always use the RAG tool to ground answers in evidence. Never hallucinate or speculate beyond what is found in the knowledge base.",
    backstory="You are a knowledge analyst specializing in retrieving and synthesizing information from document collections. Your role is to use the RAG tool to find relevant evidence and provide accurate, grounded answers. You never speculate beyond what you can verify in the documents, and you acknowledge when information is not available in the knowledge base.",
    verbose=True,
)
