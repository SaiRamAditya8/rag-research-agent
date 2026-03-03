import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents_src.config.agent_settings import AgentSettings
from src.agents_src.llm.client import LLMClient

logger = logging.getLogger(__name__)

_BUFFER_LIMIT = 5


class Project(BaseModel):
    project_id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: str
    fetched_papers: List[dict] = Field(default_factory=list)
    # Rolling 5-message buffer passed to LLM for context (content only)
    chat_buffer: List[dict] = Field(default_factory=list)
    chat_summary: str = ""
    turns_since_summary: int = 0
    # Full display history — every message with all UI metadata (sources, tool, etc.)
    chat_display_history: List[dict] = Field(default_factory=list)


class ProjectStore:
    """
    Singleton, file-backed store for Projects.

    Replaces the in-memory SessionStore.  Each Project owns its chat memory
    (buffer + summary) and the list of papers fetched into its vector store
    context.  All mutations are immediately persisted to disk.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            settings = AgentSettings()
            cls._instance._path = settings.PROJECTS_DATA_PATH
            cls._instance._llm = LLMClient()
            cls._instance._projects: Dict[str, Project] = {}
            cls._instance._load()
        return cls._instance

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        path = Path(self._path)
        if not path.exists():
            logger.info(f"ProjectStore: no file at {self._path}, starting empty.")
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for entry in raw:
                p = Project.model_validate(entry)
                self._projects[p.project_id] = p
            logger.info(f"ProjectStore: loaded {len(self._projects)} project(s) from {self._path}.")
        except Exception as e:
            logger.error(f"ProjectStore: failed to load {self._path}: {e}")

    def _save(self):
        path = Path(self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = [p.model_dump() for p in self._projects.values()]
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"ProjectStore: failed to save {self._path}: {e}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_projects(self) -> List[Project]:
        return sorted(self._projects.values(), key=lambda p: p.updated_at, reverse=True)

    def get_project(self, project_id: str) -> Project:
        if project_id not in self._projects:
            now = datetime.now(timezone.utc).isoformat()
            project = Project(
                project_id=project_id,
                name=project_id,
                created_at=now,
                updated_at=now,
            )
            self._projects[project_id] = project
            self._save()
        return self._projects[project_id]

    def create_project(self, name: str, description: str = "") -> Project:
        now = datetime.now(timezone.utc).isoformat()
        project = Project(
            project_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )
        self._projects[project.project_id] = project
        self._save()
        logger.info(f"ProjectStore: created project '{name}' ({project.project_id})")
        return project

    def save_project(self, project: Project):
        project.updated_at = datetime.now(timezone.utc).isoformat()
        self._projects[project.project_id] = project
        self._save()

    def delete_project(self, project_id: str):
        if project_id in self._projects:
            name = self._projects[project_id].name
            del self._projects[project_id]
            self._save()
            logger.info(f"ProjectStore: deleted project '{name}' ({project_id})")

    # ------------------------------------------------------------------
    # Chat memory
    # ------------------------------------------------------------------

    def add_message(self, project_id: str, role: str, content: str):
        project = self.get_project(project_id)
        project.chat_buffer.append({"role": role, "content": content})
        project.turns_since_summary += 1
        if len(project.chat_buffer) > _BUFFER_LIMIT:
            project.chat_buffer = project.chat_buffer[-_BUFFER_LIMIT:]
        self.save_project(project)

    def get_memory(self, project_id: str) -> dict:
        project = self.get_project(project_id)
        return {
            "chat_summary": project.chat_summary,
            "chat_buffer": project.chat_buffer,
        }

    def summary_update_needed(
        self, project_id: str, fetch_occurred: bool, rag_occurred: bool
    ) -> bool:
        project = self.get_project(project_id)
        if fetch_occurred or rag_occurred:
            return True
        return project.turns_since_summary >= _BUFFER_LIMIT

    def update_summary(self, project_id: str):
        project = self.get_project(project_id)
        if not project.chat_buffer:
            return
        prompt = (
            "You are an expert summarizer for a research assistant AI.\n"
            "Your goal is to maintain a concise but information-rich running summary of the conversation.\n\n"
            f"Current Summary:\n{project.chat_summary or 'No summary yet.'}\n\n"
            f"Recent Conversation:\n{project.chat_buffer}\n\n"
            "Instructions:\n"
            "1. Update the Current Summary to include key information from the Recent Conversation.\n"
            "2. Focus on: user research interests, specific questions, key papers fetched or discussed, "
            "important concepts explained, and any constraints or preferences stated by the user.\n"
            "3. Drop transient chitchat (greetings, simple acks).\n"
            "4. Keep the summary coherent and chronological.\n"
            "5. Output ONLY the updated summary string — no preamble, no labels."
        )
        try:
            updated = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                agent_name="Memory Assistant",
            )
            project.chat_summary = updated
            project.turns_since_summary = 0
            self.save_project(project)
            logger.info(f"ProjectStore: updated summary for project {project_id}")
        except Exception as e:
            logger.error(f"ProjectStore: summarization failed for {project_id}: {e}")

    def add_display_message(self, project_id: str, message: dict):
        """Append a full display message (with sources, tool_used, etc.) to the project history."""
        project = self.get_project(project_id)
        project.chat_display_history.append(message)
        self.save_project(project)

    # ------------------------------------------------------------------
    # Paper tracking
    # ------------------------------------------------------------------

    def add_fetched_papers(self, project_id: str, papers: List[dict]):
        if not papers:
            return
        project = self.get_project(project_id)
        existing_titles = {p["title"].lower() for p in project.fetched_papers}
        now = datetime.now(timezone.utc).isoformat()
        for paper in papers:
            if paper.get("title") and paper["title"].lower() not in existing_titles:
                project.fetched_papers.append({
                    "title": paper["title"],
                    "url": paper.get("url", ""),
                    "ingested_at": now,
                })
                existing_titles.add(paper["title"].lower())
        self.save_project(project)


project_store = ProjectStore()
