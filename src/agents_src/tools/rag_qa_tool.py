import logging

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings, get_response_synthesizer
import chromadb
from sentence_transformers import CrossEncoder

from src.agents_src.config.agent_settings import AgentSettings

# Import shared model singletons
from src.agents_src.llm.models import embed_model, rerank_model

# Get a logger for this module
logger = logging.getLogger(__name__)


@tool
def rag_query_tool(query: str) -> dict:
    """
    Answers a query by retrieving relevant documents, reranking them for precision, and generating a response.
    Returns both the generated answer and the source file names from which the information was retrieved.

    Args:
        query (str): The input query string to be processed.

    Returns:
        dict: A dictionary with the following keys:
            - 'answer': The generated answer string.
            - 'sources': List of source file names used for retrieval.

    Notes:
        - Uses vector search to get top 10 chunks, then reranks with BAAI/bge-reranker-large.
        - The final answer is generated using the top 3 chunks after reranking.
        - Requires properly configured AgentSettings and access to the vector store.
    """

    settings = AgentSettings()
    vector_store_path = settings.VECTOR_STORE_DIR
    collection_name = settings.COLLECTION_NAME
    # Configure LLM
    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )
    # Load Chroma collection
    db = chromadb.PersistentClient(path=vector_store_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    # connect to the vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Load index from Chroma
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    # Create the retriever
    retriever = index.as_retriever(similarity_top_k=10) # Retrieve top 10 initially
    
    # Retrieve initial nodes
    initial_nodes = retriever.retrieve(query)
    
    if not initial_nodes:
        return {"answer": "No relevant context found.", "sources": []}

    # Prepare pairs for reranking: (query, chunk_text)
    pairs = [(query, node.node.get_content()) for node in initial_nodes]
    
    # Compute relevance scores using the Cross-Encoder
    logger.info(f"Reranking {len(initial_nodes)} chunks...")
    scores = rerank_model.predict(pairs)
    
    # Combine nodes with their scores and sort
    node_scores = sorted(zip(initial_nodes, scores), key=lambda x: x[1], reverse=True)
    
    # Select top K (K=3)
    top_k_nodes = [node for node, score in node_scores[:3]]
    
    # Create a response synthesizer
    response_synthesizer = get_response_synthesizer()
    
    # Generate the final answer using the top K chunks
    response = response_synthesizer.synthesize(query, nodes=top_k_nodes)
    
    # Extract source file names from top K chunks
    source_file_names = []
    if top_k_nodes:
        for node_with_score in top_k_nodes:
            metadata = node_with_score.node.metadata
            if metadata and isinstance(metadata, dict):
                file_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("source")
                if file_name:
                    source_file_names.append(file_name)
    
    # Filter out duplicates while preserving order
    seen = set()
    unique_sources = []
    for s in source_file_names:
        if s and s not in seen:
            seen.add(s)
            unique_sources.append(s)
    
    logger.info(f"Extracted sources after reranking: {unique_sources}")

    return {"answer": str(response),
            "sources": unique_sources}


# For direct testing, uncomment the code below and comment out @tool.
# When using CrewAI, uncomment @tool and comment out the test code.

# output = rag_query_tool(query="Explain SHAP")
# print(output)
# print(output["answer"])
# print(output["source_files"])
