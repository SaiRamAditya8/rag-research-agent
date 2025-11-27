# agentic-rag-chatbot

# Workflow:

## Tools
rag_query_tool -> When called by an agent, Returns the top 3 relevant Document chunks after comparing the user query across all the ingested documents using HuggingFaceEmbeddings

## Agents
qa_agent -> When used for a task, uses the llm specified in llm_configuration.py to give an answer for the user query. It has the rag_query_tool to search the knowledge base if needed.

## Tasks
qa_task -> Given the user_query and the chat_history, uses the qa_agent to return the response.

## backend_src

This folder contains the code to wrap all the Agent fucntionality in a FastAPI wrapper so that we can call the http://localhost:8000/chat/answer and provide the chat_history (which we get from frontend) to get the Chat Bot's response in total.

This parses the total chat history into user_query and chat_history for the agent 

## frontend_src

This contains the code for a simple streamlit interface that shows the chat history and user query. It uses the FastAPI endpoint mentioned in the .env to post the request with the whole chat_history and gets the output to show.
