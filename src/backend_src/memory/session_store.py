from src.agents_src.llm.get_llm import get_llm_for_agent
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SessionStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionStore, cls).__new__(cls)
            cls._instance.sessions = {}
            # Initialize summarizer LLM
            cls._instance.llm = get_llm_for_agent("Memory Assistant")
        return cls._instance

    def _get_or_create_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_buffer": [], # List[Dict] usually {"role": str, "content": str}
                "chat_summary": "",
                "turns_since_summary": 0
            }
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        session = self._get_or_create_session(session_id)
        session["chat_buffer"].append({"role": role, "content": content})
        session["turns_since_summary"] += 1
        
        # Trim buffer to max 5 items (rolling window)
        if len(session["chat_buffer"]) > 5:
            session["chat_buffer"] = session["chat_buffer"][-5:]

    def get_memory(self, session_id: str) -> Dict:
        session = self._get_or_create_session(session_id)
        return {
            "chat_summary": session["chat_summary"],
            "chat_buffer": session["chat_buffer"]
        }
    
    def summary_update_needed(self, session_id: str, fetch_occurred: bool, rag_occurred: bool) -> bool:
        """Check if summary update is needed based on triggers."""
        session = self._get_or_create_session(session_id)
        if fetch_occurred or rag_occurred:
            return True
        if session["turns_since_summary"] >= 5:
            return True
        return False

    def update_summary(self, session_id: str):
        """Use LLM to update the running summary of the conversation."""
        session = self._get_or_create_session(session_id)
        current_summary = session["chat_summary"]
        recent_history = session["chat_buffer"]
        
        # Don't summarize if there's no history to summarize
        if not recent_history:
            return

        prompt = f"""
        You are an expert summarizer for a research assistant AI. 
        Your goal is to maintain a concise but information-rich running summary of the conversation.
        
        Current Summary:
        {current_summary if current_summary else "No summary yet."}
        
        Recent Conversation:
        {recent_history}
        
        Instructions:
        1. update the Current Summary to include key information from the Recent Conversation.
        2. Focus on:
           - User's research, interests or specific questions.
           - Key papers fetched or discussed (titles, topics).
           - Important concepts explained.
           - Any specific constraints or preferences stated by the user.
        3. Drop transient chitchat (greetings, simple acks).
        4. Keep the summary coherent and chronological.
        5. Output ONLY the updated summary string.
        """
        
        try:
             # Call the LLM (LiteLLM wrapper usually has call or predict)
             # CrewAI's get_llm returns a LiteLLM object which supports .call()
             response = self.llm.call([{"role": "user", "content": prompt}])
             updated_summary = response
             
             # Create backup just in case response is structured
             if hasattr(response, 'content'):
                 updated_summary = response.content
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            updated_summary = current_summary # Fallback: keep old summary
        
        session["chat_summary"] = updated_summary
        session["turns_since_summary"] = 0
        logger.info(f"Updated summary for session {session_id}")

session_store = SessionStore()

