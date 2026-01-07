import logging
from src.agents_src.crew import qa_crew, intent_crew, chitchat_crew
from src.agents_src.utils.paper_fetcher import fetch_papers_and_ingest
logger = logging.getLogger(__name__)


from src.backend_src.memory.session_store import session_store

def get_answer(user_query: str, session_id: str = "default_session") -> dict:
    logger.info(f"Processing query for session {session_id}: {user_query}")
    
    # 1. Update memory with user message
    session_store.add_message(session_id, "user", user_query)
    
    # 2. Retrieve current memory state
    memory = session_store.get_memory(session_id)
    chat_summary = memory["chat_summary"]
    chat_buffer = memory["chat_buffer"] # This replaces full chat_history
    
    # 3. Prepare input for Intent Agent
    # Note: We pass chat_buffer as 'chat_history' to match existing agent expectation of a list
    # But we also inject chat_summary into the context if possible. 
    # For now, we will pass chat_summary as part of the input dict, 
    # but the Task description needs to be updated to use it (next step).
    
    input_data = {
        "user_query": user_query,
        "chat_history": chat_buffer,
        "chat_summary": chat_summary
    }
    logger.debug(f"Input data for intent_crew: {input_data}")
    
    intent = intent_crew.kickoff(input_data)
    
    # Normalize intent result
    if not isinstance(intent, dict):
        try:
            intent = intent.to_dict()
        except Exception:
            intent = dict(intent)
    logger.info(f"Intent result: {intent}")

    fetched_papers = []
    fetch_occurred = False
    
    if intent.get("fetch"):
        fetch_occurred = True
        fetched_papers = fetch_papers_and_ingest(
                queries=intent.get("queries", []),
                categories=intent.get("categories", [])
        )

    use_rag = intent.get("use_rag", False)
    rag_occurred = False
    
    if use_rag:
        logger.info("Using QA Crew for response generation.")
        use_crew = qa_crew
        rag_occurred = True
    else:
        logger.info("Using ChitChat Crew for response generation.")
        use_crew = chitchat_crew
    
    papers_list = [p["title"] for p in fetched_papers if p and p.get("title")] if fetched_papers else []
    
    new_input_data = {
        "user_query": user_query,
        "chat_history": chat_buffer,
        "chat_summary": chat_summary,
        "fetch": intent.get("fetch", False),
        "papers": papers_list,
        "request": intent.get("request", ""),
    }
    
    result = use_crew.kickoff(new_input_data)
    result_dict = result.to_dict() if not isinstance(result, dict) else result
    
    # 4. Update memory with assistant response
    assistant_answer = result_dict.get("answer", "")
    if assistant_answer:
        session_store.add_message(session_id, "assistant", assistant_answer)
        
    # 5. Update summary if triggers met
    if session_store.summary_update_needed(session_id, fetch_occurred, rag_occurred):
        session_store.update_summary(session_id)
        
    return result_dict


# Example usage
# sample_chat_history = [
#     {"role": "user", "content": "Can you fetch and explain the paper on {Local Interpretable Model Agnostic Shap Explanations for machine learning models}"}]
# #     {"role": "assistant", "content": "Evolution is the scientific theory describing how all life forms on Earth change over successive generations through alterations in their genetic material, leading to the diversity of life seen today. This process involves changes in an organism's genetic makeup (genome), which result from processes like mutation and are influenced by natural selection, where individuals with advantageous traits for their environment leave more offspring."},
# #     {"role": "user", "content": "Explain in detail"}
# # ]
# response = get_answer(sample_chat_history)
# print(response)
