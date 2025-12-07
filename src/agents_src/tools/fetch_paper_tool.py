import logging

from crewai.tools import tool
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import fitz  # PyMuPDF
import os
from typing import List, Optional
from llama_index.core import Document

import arxiv
from pathlib import Path

from src.rag_doc_ingestion.config.doc_ingestion_settings import DocIngestionSettings


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Load settings from environment variables
settings = DocIngestionSettings()

# def build_vector_store_from_documents():
#     logger.info("Starting vector store ingestion process.")
#     try:
#         docs_dir_path = settings.DOCUMENTS_DIR
#         vector_store_path = settings.VECTOR_STORE_DIR
#         collection_name = settings.COLLECTION_NAME
#         logger.info(f"Loading documents from directory: {docs_dir_path}")
#         loader = SimpleDirectoryReader(input_dir=docs_dir_path)
#         documents = loader.load_data()
#         # Create parser with chunking strategy
#         parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
#         logger.info("Parsing documents into nodes.")
#         nodes = parser.get_nodes_from_documents(documents)
#         logger.info(f"Parsed {len(nodes)} nodes.")
#         logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
#         db = chromadb.PersistentClient(path=vector_store_path)
#         # Create or retrieve the vector collection
#         chroma_collection = db.get_or_create_collection(name=collection_name)
#         logger.info(f"Creating Chroma vector store with collection name: {collection_name}")
#         vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
#         # Create storage context
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)
#         logger.info("Building vector store index.")
#         index = VectorStoreIndex(
#             nodes,
#             storage_context=storage_context,
#             vector_store=vector_store,
#             embed_model=embed_model
#         )
#         logger.info("Vector store build successfully.")
#         return 0
#     except Exception as e:
#         logger.error(f"Error during vector store build: {e}")
#         return 1

def _extract_text_from_pdf(path: str) -> str:
    """Extract text from a single PDF using PyMuPDF (fitz)."""
    text_chunks = []
    try:
        doc = fitz.open(path)
        for page in doc:
            text = page.get_text("text")
            if text:
                text_chunks.append(text)
        doc.close()
    except Exception as e:
        logger.exception(f"Failed to extract text from {path}: {e}")
    return "\n".join(text_chunks)


def build_vector_store_from_documents(pdf_paths: Optional[List[str]] = None) -> int:
    """
    Build a persistent Chroma vector store index.

    If `pdf_paths` is provided (list of absolute/relative PDF file paths),
    those PDFs are read and converted into Documents. Otherwise, the function
    falls back to reading documents from settings.DOCUMENTS_DIR using
    SimpleDirectoryReader (existing behaviour).
    """
    logger.info("Starting vector store ingestion process.")
    try:
        docs_dir_path = settings.DOCUMENTS_DIR
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        documents = []
        if pdf_paths:
            logger.info(f"Loading {len(pdf_paths)} PDF files from provided list.")
            for p in pdf_paths:
                p = os.path.expanduser(p)
                if not os.path.isfile(p):
                    logger.warning(f"PDF path not found or not a file: {p}")
                    continue
                text = _extract_text_from_pdf(p)
                if not text.strip():
                    logger.warning(f"No text extracted from PDF: {p}")
                    continue
                # create a llama_index Document (keep metadata)
                doc = Document(text=text, metadata={"source": p, "filename": os.path.basename(p)})
                documents.append(doc)
            if not documents:
                logger.error("No valid documents were created from provided PDF paths.")
                return 1
        # else:
        #     logger.info(f"Loading documents from directory: {docs_dir_path}")
        #     loader = SimpleDirectoryReader(input_dir=docs_dir_path)
        #     documents = loader.load_data()

        # Create parser with chunking strategy
        parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=50)
        logger.info("Parsing documents into nodes.")
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Parsed {len(nodes)} nodes.")

        # download & load embedding model
        logger.info("Loading HuggingFace embedding model...")
        embed_model = HuggingFaceEmbedding()

        logger.info(f"Initializing ChromaDB persistent client at: {vector_store_path}")
        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(name=collection_name)

        logger.info(f"Creating Chroma vector store with collection name: {collection_name}")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context and index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Building vector store index.")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=embed_model
        )

        logger.info("Vector store built successfully.")

        for pdf_path in pdf_paths:
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    logger.info(f"Deleted temporary PDF: {pdf_path}")
            except Exception as e:
                logger.warning(f"Could not delete PDF file {pdf_path}: {e}")

        return 0

    except Exception as e:
        logger.exception(f"Error during vector store build: {e}")
        return 1

class IntentUse(BaseModel):
    title: str  
    category: str = "cs.AI"  # Default category

@tool
def fetch_paper_tool(intent: IntentUse) -> dict:
    """
    Fetches a paper by querying the arxiv API.
    Creates Vector embeddings for the fetched paper and stores them in the vector store.
    Returns the list of fetched papers with their titles and links.

    Args:
        intent (IntentUse): The intent containing title and category of the paper to fetch. Eg: {"title": "Attention Is All You Need", "category": "cs.CL"}

    Returns:
        list: A list of fetched papers with their titles and links.

    Notes:
        - Requires proper title and category of the paper to query.
    """
    # Handle both dict and IntentUse object inputs
    if isinstance(intent, dict):
        intent = IntentUse(**intent)

    logger.info(f"Fetching papers with title: {intent.title} and category: {intent.category}")

    # Exact title + category search (title-only)
    query = f'ti:"{intent.title}" AND cat:{intent.category}'

    search = arxiv.Search(
        query=query,
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = list(search.results())
    if not results:
        return None  # no match
    docs_dir_path = settings.DOCUMENTS_DIR
    Path(docs_dir_path).mkdir(exist_ok=True)
    response=[]
    for paper in results:
        logger.info(f"Downloading: {paper.title}")
        paper.download_pdf(dirpath=docs_dir_path, filename=f"{paper.title}.pdf")
        response.append(paper.title)
    log_response = {
        "status": "success",
        "fetched_papers": response
    } 
    logger.info(f"Fetch response: {log_response}")
    build_vector_store_from_documents(pdf_paths=[os.path.join(docs_dir_path, f"{paper.title}.pdf") for paper in results])
    return response



if __name__ == "__main__":
    intent = IntentUse(title="Local Interpretable Model Agnostic Shap Explanations for machine learning models", category="cs.LG")
    result = fetch_paper_tool(intent)

    print(result)