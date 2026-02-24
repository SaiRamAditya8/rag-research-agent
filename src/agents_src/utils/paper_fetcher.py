import logging
import time
from difflib import SequenceMatcher
from typing import List, Optional

import chromadb
import fitz  # PyMuPDF
import os
import requests
from pathlib import Path

import arxiv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore

import numpy as np

from src.agents_src.config.agent_settings import AgentSettings
from src.agents_src.llm.models import embed_model, rerank_model


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = AgentSettings()

SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Minimum title similarity to treat a result as a confirmed match for the requested paper.
TITLE_MATCH_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MMR_LAMBDA = 0.7  # relevance weight in MMR; 1.0 = pure relevance, 0.0 = pure diversity


def _title_similarity(a: str, b: str) -> float:
    """Case-insensitive ratio of how similar two title strings are."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _mmr_select(
    candidates: List[dict],
    already_selected: List[dict],
    n: int,
) -> List[dict]:
    """
    Greedy MMR selection from `candidates`, given papers already chosen.
    Scores each step as: λ × rerank_score − (1−λ) × max_cosine_sim(candidate, selected)

    Requires each candidate and selected item to have an 'emb' key (unit-normed embedding).
    rerank_score must already be min-max normalised to [0, 1].
    """
    selected = list(already_selected)
    remaining = list(candidates)
    result = []

    while len(result) < n and remaining:
        best_score = -float("inf")
        best = None
        for cand in remaining:
            relevance = cand["rerank_score_norm"]
            if selected:
                redundancy = max(_cosine(cand["emb"], s["emb"]) for s in selected)
            else:
                redundancy = 0.0
            score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
            if score > best_score:
                best_score = score
                best = cand
        result.append(best)
        selected.append(best)
        remaining.remove(best)

    return result


def _extract_text_from_pdf(path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
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


def _normalize_title(title: str) -> str:
    return title.lower().strip()


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------

def _search_arxiv_by_title(title: str, max_results: int = 8) -> List[dict]:
    """
    Title-field ArXiv search — far more precise than all: for named papers.
    Tries exact phrase first, falls back to keyword-in-title.
    """
    candidates = []
    queries_to_try = [
        f'ti:"{title}"',   # exact phrase in title field
        f'ti:{title}',     # keyword match in title field (fallback)
    ]
    for q in queries_to_try:
        try:
            search = arxiv.Search(
                query=q,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(search.results())
            if results:
                for paper in results:
                    candidates.append({
                        "title": paper.title,
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "source": "arxiv",
                        "arxiv_obj": paper,
                    })
                logger.info(f"ArXiv title search '{q}' returned {len(results)} results.")
                break  # exact phrase matched — no need for keyword fallback
        except Exception as e:
            logger.warning(f"ArXiv title search failed for '{q}': {e}")
    return candidates


def _search_semantic_scholar(query: str, max_results: int = 5) -> List[dict]:
    """
    Semantic Scholar paper search. Good at finding specific papers by title.
    Returns only candidates where a downloadable PDF URL is available.
    """
    candidates = []
    try:
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,externalIds,openAccessPdf,year",
        }
        resp = requests.get(SEMANTIC_SCHOLAR_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        for paper in data:
            pdf_url = None
            # Prefer the direct openAccessPdf URL; fall back to constructing ArXiv URL
            if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url"):
                pdf_url = paper["openAccessPdf"]["url"]
            elif paper.get("externalIds", {}).get("ArXiv"):
                arxiv_id = paper["externalIds"]["ArXiv"]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            if pdf_url:
                candidates.append({
                    "title": paper.get("title", ""),
                    "summary": paper.get("abstract", "") or "",
                    "pdf_url": pdf_url,
                    "source": "semantic_scholar",
                    "arxiv_obj": None,
                })
        logger.info(f"Semantic Scholar search for '{query}' returned {len(candidates)} downloadable results.")
        time.sleep(1)  # respect S2 rate limit (1 req/sec without API key)
    except Exception as e:
        logger.warning(f"Semantic Scholar search failed for '{query}': {e}")
    return candidates


def _search_arxiv_broad(query_text: str, category: str = "", max_results: int = 5) -> List[dict]:
    """
    Topic-based broad ArXiv search (all: field). Used for supplementary results
    when looking for papers on a topic rather than a specific title.
    """
    candidates = []
    if category and category.strip():
        queries_to_try = [
            f'all:"{query_text}" AND cat:{category}',
            f'all:{query_text} AND cat:{category}',
        ]
    else:
        queries_to_try = [
            f'all:"{query_text}"',
            f'all:{query_text}',
        ]
    for q in queries_to_try:
        try:
            search = arxiv.Search(
                query=q,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            results = list(search.results())
            if results:
                for paper in results:
                    candidates.append({
                        "title": paper.title,
                        "summary": paper.summary,
                        "pdf_url": paper.pdf_url,
                        "source": "arxiv",
                        "arxiv_obj": paper,
                    })
                break
        except Exception as e:
            logger.warning(f"ArXiv broad search failed for '{q}': {e}")
    return candidates


# ---------------------------------------------------------------------------
# Vector store ingestion
# ---------------------------------------------------------------------------

def build_vector_store_from_documents(pdf_paths: Optional[List[str]] = None) -> int:
    """
    Build (or extend) the persistent Chroma vector store.

    If `pdf_paths` is provided those PDFs are read directly.
    Otherwise falls back to reading from settings.DOCUMENTS_DIR.
    Temporary PDFs are deleted after ingestion.
    """
    logger.info("Starting vector store ingestion process.")
    try:
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        documents = []
        if pdf_paths:
            logger.info(f"Loading {len(pdf_paths)} PDF files.")
            for p in pdf_paths:
                p = os.path.expanduser(p)
                if not os.path.isfile(p):
                    logger.warning(f"PDF path not found: {p}")
                    continue
                text = _extract_text_from_pdf(p)
                if not text.strip():
                    logger.warning(f"No text extracted from: {p}")
                    continue
                documents.append(Document(
                    text=text,
                    metadata={"source": p, "filename": os.path.basename(p)},
                ))
        else:
            from llama_index.core import SimpleDirectoryReader
            docs_dir_path = settings.DOCUMENTS_DIR
            logger.info(f"Loading documents from directory: {docs_dir_path}")
            loader = SimpleDirectoryReader(input_dir=docs_dir_path)
            documents = loader.load_data()

        if not documents:
            logger.error("No valid documents to ingest.")
            return 1

        parser = SimpleNodeParser.from_defaults(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Parsed {len(nodes)} nodes.")

        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            vector_store=vector_store,
            embed_model=embed_model,
        )
        logger.info("Vector store built successfully.")
        return 0

    except Exception as e:
        logger.exception(f"Error during vector store build: {e}")
        return 1

    finally:
        if pdf_paths:
            for pdf_path in pdf_paths:
                try:
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                        logger.info(f"Deleted temporary PDF: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not delete PDF {pdf_path}: {e}")


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

def _download_pdf(candidate: dict, docs_dir: str) -> Optional[str]:
    """
    Download the PDF for a candidate paper.
    Returns the local file path on success, None on failure.
    """
    safe_title = candidate["title"].replace("/", "_")[:100]
    pdf_path = os.path.join(docs_dir, f"{safe_title}.pdf")

    try:
        if candidate["source"] == "arxiv" and candidate.get("arxiv_obj"):
            candidate["arxiv_obj"].download_pdf(dirpath=docs_dir, filename=f"{safe_title}.pdf")
        else:
            # Semantic Scholar or any direct URL — stream download
            resp = requests.get(candidate["pdf_url"], stream=True, timeout=30)
            resp.raise_for_status()
            with open(pdf_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded PDF: {candidate['title']}")
        return pdf_path
    except Exception as e:
        logger.warning(f"Failed to download PDF for '{candidate['title']}': {e}")
        return None


# ---------------------------------------------------------------------------
# Main fetch entry point
# ---------------------------------------------------------------------------

def fetch_papers_and_ingest(
    queries: List[str],
    categories: List[str] = None,
    top_k: int = 3,
) -> Optional[List[dict]]:
    """
    Fetch and ingest academic papers using a two-phase strategy:

    Phase 1 — Title search (first query only):
        Searches ArXiv by title field and Semantic Scholar for the specific
        paper/topic the user named. Results with title similarity above
        TITLE_MATCH_THRESHOLD are treated as confirmed matches and pinned
        to the top of the final selection.

    Phase 2 — Broad topic search (all queries):
        Standard all-field ArXiv search across all queries and categories.
        Results fill remaining slots after confirmed matches.

    The two phases are merged, deduplicated, and ranked:
    - Confirmed title matches appear first (sorted by title similarity).
    - Remaining slots are filled by embedding-similarity ranked candidates.

    Returns a list of {"title": str, "url": str} for successfully ingested papers.
    """
    if not queries:
        return None

    if categories is None:
        categories = [""]

    first_query = queries[0].strip()
    seen_titles: set = set()
    confirmed: List[dict] = []   # high-confidence title matches
    broad: List[dict] = []       # topic-search candidates

    # ------------------------------------------------------------------
    # Phase 1: Title-specific search for the first (exact/primary) query
    # ------------------------------------------------------------------
    logger.info(f"Phase 1: Title search for '{first_query}'")

    title_candidates = _search_arxiv_by_title(first_query) + _search_semantic_scholar(first_query)

    for cand in title_candidates:
        norm = _normalize_title(cand["title"])
        if norm in seen_titles or not cand["title"]:
            continue
        sim = _title_similarity(first_query, cand["title"])
        cand["title_sim"] = sim
        if sim >= TITLE_MATCH_THRESHOLD:
            logger.info(f"Confirmed title match ({sim:.2f}): {cand['title']}")
            confirmed.append(cand)
            seen_titles.add(norm)
        else:
            # Below threshold — keep as broad candidate (title similarity stored for tiebreaking)
            broad.append(cand)
            seen_titles.add(norm)

    # ------------------------------------------------------------------
    # Phase 2: Broad topic search across all queries × categories
    # ------------------------------------------------------------------
    logger.info(f"Phase 2: Broad topic search across {len(queries)} queries")

    for query_text in queries:
        if not query_text or not query_text.strip():
            continue
        for category in categories:
            for cand in _search_arxiv_broad(query_text, category):
                norm = _normalize_title(cand["title"])
                if norm not in seen_titles and cand["title"]:
                    broad.append(cand)
                    seen_titles.add(norm)

    # ------------------------------------------------------------------
    # Phase 3: Score all candidates with cross-encoder + compute embeddings
    # for MMR redundancy penalty.
    # ------------------------------------------------------------------
    all_candidates = confirmed + broad
    if not all_candidates:
        return None

    # Cross-encoder relevance scores (single batch for efficiency)
    pairs = [[first_query, f"{c['title']} {c['summary']}"] for c in all_candidates]
    raw_scores = rerank_model.predict(pairs)
    for cand, score in zip(all_candidates, raw_scores):
        cand["rerank_score"] = float(score)

    # Min-max normalise scores to [0,1] so they're comparable with cosine similarity
    score_vals = [c["rerank_score"] for c in all_candidates]
    lo, hi = min(score_vals), max(score_vals)
    score_range = hi - lo if hi != lo else 1.0
    for cand in all_candidates:
        cand["rerank_score_norm"] = (cand["rerank_score"] - lo) / score_range

    # Compute unit-normed embeddings for MMR redundancy term
    for cand in all_candidates:
        emb = np.array(embed_model.get_text_embedding(f"{cand['title']} {cand['summary']}"))
        cand["emb"] = emb / (np.linalg.norm(emb) + 1e-10)

    # Sort confirmed by relevance (title lookup: pure relevance is correct)
    confirmed.sort(key=lambda c: c["rerank_score"], reverse=True)

    logger.info(
        f"Cross-encoder top confirmed: {confirmed[0]['title'] if confirmed else 'none'} "
        f"| top broad: {broad[0]['title'] if broad else 'none'}"
    )

    # ------------------------------------------------------------------
    # Phase 4: Select top_k
    # Confirmed matches fill slots first (user asked for a specific paper).
    # Remaining slots are filled via MMR over broad candidates — balancing
    # relevance and diversity so ingested papers cover different angles.
    # ------------------------------------------------------------------
    confirmed_selected = confirmed[:top_k]
    slots_remaining = top_k - len(confirmed_selected)

    if slots_remaining > 0 and broad:
        broad_selected = _mmr_select(broad, already_selected=confirmed_selected, n=slots_remaining)
    else:
        broad_selected = []

    selected = confirmed_selected + broad_selected

    if not selected:
        logger.warning("No papers found across all search strategies.")
        return None

    logger.info(f"Selected {len(selected)} papers for ingestion.")

    # ------------------------------------------------------------------
    # Phase 5: Download PDFs and ingest
    # ------------------------------------------------------------------
    docs_dir = settings.DOCUMENTS_DIR
    Path(docs_dir).mkdir(parents=True, exist_ok=True)

    pdf_paths = []
    response = []

    for cand in selected:
        pdf_path = _download_pdf(cand, docs_dir)
        if pdf_path:
            pdf_paths.append(pdf_path)
            response.append({"title": cand["title"], "url": cand["pdf_url"]})

    if not pdf_paths:
        logger.error("All PDF downloads failed.")
        return None

    build_vector_store_from_documents(pdf_paths=pdf_paths)
    logger.info(f"Ingested papers: {[r['title'] for r in response]}")
    return response


if __name__ == "__main__":
    queries = ["Attention is all you need"]
    categories = ["cs.CL"]
    result = fetch_papers_and_ingest(queries, categories)
    print(result)
