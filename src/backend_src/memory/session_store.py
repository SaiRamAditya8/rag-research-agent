import logging
from typing import Dict

from src.agents_src.llm.client import LLMClient

logger = logging.getLogger(__name__)


class SessionStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionStore, cls).__new__(cls)
            cls._instance.sessions: Dict[str, Dict] = {}
            cls._instance._llm = LLMClient()
        return cls._instance

    def _get_or_create_session(self, session_id: str) -> Dict:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_buffer": [],
                "chat_summary": "",
                "turns_since_summary": 0,
            }
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str):
        session = self._get_or_create_session(session_id)
        session["chat_buffer"].append({"role": role, "content": content})
        session["turns_since_summary"] += 1

        # Rolling window — keep last 5 messages
        if len(session["chat_buffer"]) > 5:
            session["chat_buffer"] = session["chat_buffer"][-5:]

    def get_memory(self, session_id: str) -> Dict:
        session = self._get_or_create_session(session_id)
        return {
            "chat_summary": session["chat_summary"],
            "chat_buffer": session["chat_buffer"],
        }

    def summary_update_needed(
        self, session_id: str, fetch_occurred: bool, rag_occurred: bool
    ) -> bool:
        session = self._get_or_create_session(session_id)
        if fetch_occurred or rag_occurred:
            return True
        if session["turns_since_summary"] >= 5:
            return True
        return False

    def update_summary(self, session_id: str):
        """Use the LLM to produce a running summary of the conversation."""
        session = self._get_or_create_session(session_id)
        current_summary = session["chat_summary"]
        recent_history = session["chat_buffer"]

        if not recent_history:
            return

        prompt = (
            "You are an expert summarizer for a research assistant AI.\n"
            "Your goal is to maintain a concise but information-rich running summary of the conversation.\n\n"
            f"Current Summary:\n{current_summary if current_summary else 'No summary yet.'}\n\n"
            f"Recent Conversation:\n{recent_history}\n\n"
            "Instructions:\n"
            "1. Update the Current Summary to include key information from the Recent Conversation.\n"
            "2. Focus on: user research interests, specific questions, key papers fetched or discussed, "
            "important concepts explained, and any constraints or preferences stated by the user.\n"
            "3. Drop transient chitchat (greetings, simple acks).\n"
            "4. Keep the summary coherent and chronological.\n"
            "5. Output ONLY the updated summary string — no preamble, no labels."
        )

        try:
            updated_summary = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                agent_name="Memory Assistant",
            )
            session["chat_summary"] = updated_summary
            session["turns_since_summary"] = 0
            logger.info(f"Updated summary for session {session_id}")
        except Exception as e:
            logger.error(f"Summarization failed for session {session_id}: {e}")
            # Keep the old summary on failure


session_store = SessionStore()
