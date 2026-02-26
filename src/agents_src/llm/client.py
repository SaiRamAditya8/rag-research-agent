import logging
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage

from src.agents_src.config.agent_settings import AgentSettings
from src.agents_src.llm.llm_configuration import LLM_CONFIG

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Singleton LLM client backed by LlamaIndex's Groq integration.

    To swap providers (e.g. to OpenAI or Ollama in future), change only:
      1. The import and instantiation in _build_llm()
      2. llm_configuration.py model names

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
            logger.info("LLMClient initialised (LlamaIndex/Groq).")
        return cls._instance

    def _get_llm(self, agent_name: str) -> Groq:
        """Return a cached LlamaIndex Groq LLM for the given agent config."""
        if agent_name not in self._llm_cache:
            config = LLM_CONFIG.get(agent_name, LLM_CONFIG["default"])
            self._llm_cache[agent_name] = Groq(
                model=config["model"],
                temperature=config["temperature"],
                api_key=self._settings.GROQ_API_KEY,
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
            json_mode:  If True, passes response_format={"type": "json_object"} to the API.
                        The prompt MUST contain the word "JSON" (Groq requirement).
        """
        llm = self._get_llm(agent_name)
        chat_messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]

        extra_kwargs = {}
        if json_mode:
            extra_kwargs["response_format"] = {"type": "json_object"}

        logger.debug(f"LLMClient.complete | agent={agent_name} | json_mode={json_mode}")
        response = llm.chat(chat_messages, **extra_kwargs)
        return response.message.content
