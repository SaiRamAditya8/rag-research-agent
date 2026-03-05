import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.backend_src.projects.store import project_store
from src.agents_src.utils.paper_fetcher import (
    ingest_arxiv_paper,
    ingest_doi_paper,
    ingest_uploaded_pdf,
    delete_paper_from_store,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/projects", tags=["projects"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""

class ArxivRequest(BaseModel):
    arxiv_input: str   # arXiv ID (e.g. "1706.03762") or full URL

class DoiRequest(BaseModel):
    doi: str           # e.g. "10.48550/arXiv.1706.03762" or full doi.org URL

class DeletePaperRequest(BaseModel):
    title: str         # exact title as stored in fetched_papers

class CopyPaperRequest(BaseModel):
    source_project_id: str
    title: str


# ---------------------------------------------------------------------------
# Project CRUD
# ---------------------------------------------------------------------------

@router.get("")
def list_projects():
    projects = project_store.list_projects()
    return [
        {
            "project_id": p.project_id,
            "name": p.name,
            "description": p.description,
            "paper_count": len(p.fetched_papers),
            "updated_at": p.updated_at,
        }
        for p in projects
    ]


@router.post("")
def create_project(request: CreateProjectRequest):
    project = project_store.create_project(
        name=request.name, description=request.description
    )
    return project.model_dump()


@router.get("/{project_id}")
def get_project(project_id: str):
    project = project_store.get_project(project_id)
    return project.model_dump()


@router.delete("/{project_id}")
def delete_project(project_id: str):
    if project_id not in project_store._projects:
        raise HTTPException(status_code=404, detail="Project not found")
    # Collect paper titles before removing the project record
    project = project_store.get_project(project_id)
    paper_titles = [p["title"] for p in project.fetched_papers]
    project_store.delete_project(project_id)
    # Delete chunks for papers no longer referenced by any remaining project
    remaining_projects = project_store.list_projects()
    for title in paper_titles:
        still_referenced = any(
            any(p["title"] == title for p in proj.fetched_papers)
            for proj in remaining_projects
        )
        if not still_referenced:
            delete_paper_from_store(title)
    return {"status": "deleted", "project_id": project_id}


# ---------------------------------------------------------------------------
# Paper management
# ---------------------------------------------------------------------------

@router.post("/{project_id}/papers/arxiv")
def add_arxiv_paper(project_id: str, request: ArxivRequest):
    paper = ingest_arxiv_paper(request.arxiv_input)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found or PDF could not be downloaded")
    project_store.add_fetched_papers(project_id, [paper])
    return paper


@router.post("/{project_id}/papers/doi")
def add_doi_paper(project_id: str, request: DoiRequest):
    paper = ingest_doi_paper(request.doi)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found or no open-access PDF available")
    project_store.add_fetched_papers(project_id, [paper])
    return paper


@router.post("/{project_id}/papers/upload")
async def upload_paper(project_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await file.read()
    paper = ingest_uploaded_pdf(content, file.filename)
    if not paper:
        raise HTTPException(status_code=500, detail="Failed to ingest PDF")
    project_store.add_fetched_papers(project_id, [paper])
    return paper


@router.delete("/{project_id}/papers")
def delete_paper(project_id: str, request: DeletePaperRequest):
    # Remove from this project's list first
    project = project_store.get_project(project_id)
    project.fetched_papers = [p for p in project.fetched_papers if p["title"] != request.title]
    project_store.save_project(project)
    # Only delete from the global vector store if no other project still references the title
    all_projects = project_store.list_projects()
    still_referenced = any(
        any(p["title"] == request.title for p in proj.fetched_papers)
        for proj in all_projects
    )
    if not still_referenced:
        delete_paper_from_store(request.title)
    return {"status": "deleted", "title": request.title}


@router.post("/{project_id}/papers/copy")
def copy_paper(project_id: str, request: CopyPaperRequest):
    # Chunks are global (tagged by paper_title only), so copy is just a metadata update.
    source = project_store.get_project(request.source_project_id)
    paper_meta = next((p for p in source.fetched_papers if p["title"] == request.title), None)
    if not paper_meta:
        raise HTTPException(status_code=404, detail="Paper not found in source project")
    project_store.add_fetched_papers(project_id, [paper_meta])
    return {"status": "copied", "title": request.title}
