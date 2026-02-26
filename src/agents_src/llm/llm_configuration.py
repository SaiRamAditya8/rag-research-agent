# LLM configuration — model names are bare Groq model IDs (no "groq/" prefix).
LLM_CONFIG = {
    "default": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.0,
    },
    "Intent Agent": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.0,
    },
    "QA Agent": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.0,
    },
    "ChitChat Agent": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
    },
    "Memory Assistant": {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.1,
    },
}
