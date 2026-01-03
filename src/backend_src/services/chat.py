import logging
from src.agents_src.crew import qa_crew, intent_crew, chitchat_crew
from src.agents_src.tools.fetch_paper_tool import fetch_papers_and_ingest

logger = logging.getLogger(__name__)


def get_answer(chat_history: list) -> dict:
    logger.info(f"Received chat_history: {chat_history}")
    # get the last message in the chat_history as user_query
    last_user_message = chat_history[-1]
    user_query = last_user_message["content"]
    logger.info(f"Extracted user_query: {user_query}")
    # Remove the last user message from chat_history
    history_without_last = chat_history[:-1]
    input_data = {
        "user_query": user_query,
        "chat_history": history_without_last,
    }
    logger.debug(f"Input data for qa_crew: {input_data}")
    intent = intent_crew.kickoff(input_data)
    # kickoff may return a dict-like result; normalize to dict
    if not isinstance(intent, dict):
        try:
            intent = intent.to_dict()
        except Exception:
            intent = dict(intent)
    logger.info(f"Intent result: {intent}")

    fetched_papers = []

    if intent.get("fetch"):
        fetched_papers = fetch_papers_and_ingest(
                queries=intent.get("queries", []),
                categories=intent.get("categories", [])
        )
    if intent.get("use_rag"):
        logger.info("Using QA Crew for response generation.")
        use_crew = qa_crew
    else:
        logger.info("Using ChitChat Crew for response generation.")
        use_crew = chitchat_crew
    # fetched_papers is a list of dicts: [{"title": str, "url": str}, ...]
    # Filter out None and empty titles
    papers_list = [p["title"] for p in fetched_papers if p and p.get("title")] if fetched_papers else []
    logger.info(f"Papers list for response generation: {papers_list}")
    new_input_data = {
        "user_query": user_query,
        "chat_history": history_without_last,
        "fetch": intent.get("fetch", False),
        "papers": papers_list,
        "request": intent.get("request", ""),
    }
    result = use_crew.kickoff(new_input_data)
    result_dict = result.to_dict() if not isinstance(result, dict) else result
    return result_dict


# Example usage
# sample_chat_history = [
#     {"role": "user", "content": "Can you fetch and explain the paper Attention is all you need"}]
# #     {"role": "assistant", "content": "Evolution is the scientific theory describing how all life forms on Earth change over successive generations through alterations in their genetic material, leading to the diversity of life seen today. This process involves changes in an organism's genetic makeup (genome), which result from processes like mutation and are influenced by natural selection, where individuals with advantageous traits for their environment leave more offspring."},
# #     {"role": "user", "content": "Explain in detail"}
# # ]
# response = get_answer(sample_chat_history)
# print(response)
