import logging
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

from src.agents_src.config.agent_settings import AgentSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM configuration — edit model names and temperatures here.
# gpt-4o-mini : fast + cheap — intent classification, chitchat, memory.
# gpt-4o      : best reasoning — RAG synthesis where answer quality matters.
# ---------------------------------------------------------------------------
LLM_CONFIG = {
    "default":          {"model": "gpt-4o-mini", "temperature": 0.0},
    "Intent Agent":     {"model": "gpt-4o-mini", "temperature": 0.0},
    "QA Agent":         {"model": "gpt-4o",      "temperature": 0.0},
    "ChitChat Agent":   {"model": "gpt-4o-mini", "temperature": 0.7},
    "Memory Assistant": {"model": "gpt-4o-mini", "temperature": 0.1},
}


class LLMClient:
    """
    Singleton LLM client backed by LlamaIndex's OpenAI integration.

    To swap providers in future, change only:
      1. The import and _get_llm() instantiation below
      2. Model names in LLM_CONFIG above

    Usage:
        client = LLMClient()
        text  = client.complete(messages=[...], agent_name="Intent Agent")
        json_ = client.complete(messages=[...], agent_name="Intent Agent", json_mode=True)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = AgentSettings()
            cls._instance._llm_cache = {}
            logger.info("LLMClient initialised (LlamaIndex/OpenAI).")
        return cls._instance

    def _get_llm(self, agent_name: str) -> OpenAI:
        """Return a cached LlamaIndex OpenAI LLM for the given agent config."""
        if agent_name not in self._llm_cache:
            config = LLM_CONFIG.get(agent_name, LLM_CONFIG["default"])
            self._llm_cache[agent_name] = OpenAI(
                model=config["model"],
                temperature=config["temperature"],
                api_key=self._settings.OPENAI_API_KEY,
            )
            logger.debug(f"LLMClient: created LLM for agent='{agent_name}' model='{config['model']}'")
        return self._llm_cache[agent_name]

    def complete(
        self,
        messages: list[dict],
        agent_name: str = "default",
        json_mode: bool = False,
    ) -> str:
        """
        Call the LLM and return the assistant message content as a string.

        Args:
            messages:   OpenAI-style dicts: [{"role": "user"|"system"|"assistant", "content": "..."}]
            agent_name: Key into LLM_CONFIG to select model + temperature.
            json_mode:  If True, passes response_format={"type": "json_object"}.
                        The prompt MUST contain the word "JSON".
        """
        llm = self._get_llm(agent_name)
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        extra_kwargs = {}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        logger.debug(f"LLMClient.complete | agent={agent_name} | model={LLM_CONFIG.get(agent_name, LLM_CONFIG['default'])['model']} | json_mode={json_mode}")
        response = llm.chat(chat_messages, **extra_kwargs)
        return response.message.content
