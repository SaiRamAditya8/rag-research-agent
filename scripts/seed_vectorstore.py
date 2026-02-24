"""
Seed script: builds the ChromaDB vector store from documents in DOCUMENTS_DIR.
Run this once to pre-populate the vector store before starting the app,
or whenever you want to re-index a fresh set of local documents.

Usage:
    python scripts/seed_vectorstore.py
"""
import logging

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.agents_src.config.agent_settings import AgentSettings


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = AgentSettings()

logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()


def build_vector_store_from_documents():
    logger.info("Starting vector store ingestion process.")
    try:
        docs_dir_path = settings.DOCUMENTS_DIR
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME
        logger.info(f"Loading documents from directory: {docs_dir_path}")
        loader = SimpleDirectoryReader(input_dir=docs_dir_path)
        documents = loader.load_data()
        parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
        logger.info("Parsing documents into nodes.")
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Parsed {len(nodes)} nodes.")
        logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Building vector store index.")
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=embed_model
        )
        logger.info("Vector store built successfully.")
        return 0
    except Exception as e:
        logger.error(f"Error during vector store build: {e}")
        return 1


if __name__ == "__main__":
    build_vector_store_from_documents()
