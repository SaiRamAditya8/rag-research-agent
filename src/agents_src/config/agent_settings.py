from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# load env variables from .env file
load_dotenv()


class AgentSettings(BaseSettings):
    OPENAI_API_KEY: str
    DOCUMENTS_DIR: str
    VECTOR_STORE_DIR: str
    COLLECTION_NAME: str

    # RAG chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    # RAG retrieval
    RETRIEVAL_TOP_K: int = 15   # initial vector search candidate count
    RERANK_TOP_K: int = 5       # chunks kept after cross-encoder reranking

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
        