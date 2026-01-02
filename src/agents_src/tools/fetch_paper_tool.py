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
        else:
            logger.info(f"Loading documents from directory: {docs_dir_path}")
            loader = SimpleDirectoryReader(input_dir=docs_dir_path)
            documents = loader.load_data()

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

@tool
def fetch_paper_tool(queries: List[str], categories: List[str] = None) -> dict:
    """
    Fetches academic papers using a multi-query search strategy.

    Args:
        queries: List of search query strings (1-5 items)
        categories: Optional list of arXiv categories corresponding to queries

    This tool performs multiple arXiv searches, aggregates and deduplicates 
    candidate papers, and ingests them into the vector store.

    Key characteristics:
    - Executes multiple arXiv queries (up to 5) per request
    - Treats arXiv categories as optional filters
    - Deduplicates papers across queries
    - Ingests papers deterministically
    - Deletes temporary PDF files after embedding

    The tool returns only successfully ingested paper titles and URLs.
    """

    all_results = []
    candidates = []
    seen_titles = set()
    
    # Default to single empty category if not provided
    if categories is None:
        categories = [""]
    
    # Search all combinations of queries and categories
    for query_text in queries:
        if not query_text or not query_text.strip():
            continue
        
        for category in categories:
            if category and category.strip():
                query = f'all:"{query_text}" AND cat:{category}'
                logger.info(f"Fetching papers with title: {query_text} and category: {category}")
            else:
                query = f'all:"{query_text}"'
                logger.info(f"Fetching papers with title: {query_text} without specific category")
        

        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for paper in search.results():
            norm_title = paper.title.lower().strip()
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                all_results.append(paper)
                candidates.append({
                    "title": paper.title,
                    "summary": paper.summary,
                    "pdf_url": paper.pdf_url,
                    "published": paper.published
                })

    if not all_results:
        return None  # no match
    
    docs_dir_path = settings.DOCUMENTS_DIR
    Path(docs_dir_path).mkdir(exist_ok=True)
    selected_papers = all_results
    pdf_paths = []
    response = []
    for paper in selected_papers:
        logger.info(f"Downloading: {paper.title}")
        safe_title = paper.title.replace("/", "_")[:100]
        paper.download_pdf(dirpath=docs_dir_path, filename=f"{safe_title}.pdf")
        pdf_paths.append(os.path.join(docs_dir_path, f"{safe_title}.pdf"))
        response.append({"title": paper.title, "url": paper.pdf_url})
    logger.info(f"Fetch response: {response}")
    build_vector_store_from_documents(pdf_paths=pdf_paths)
    return response

if __name__ == "__main__":
    queries = ["Attention is all you need", "Self-attention mechanisms"]
    categories = ["cs.LG", "cs.CL"]
    result = fetch_paper_tool(queries, categories)

    print(result)