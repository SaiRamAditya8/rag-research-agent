import sys
import os
# Add project root to sys.path BEFORE any imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import requests
from src.frontend_src.config.frontend_settings import Settings

settings = Settings()
BACKEND = settings.CHAT_ENDPOINT_URL.rsplit("/chat/answer", 1)[0]

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None
if "projects" not in st.session_state:
    st.session_state.projects = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "creating_project" not in st.session_state:
    st.session_state.creating_project = False
if "project_papers" not in st.session_state:
    st.session_state.project_papers = []
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "add_paper_pid" not in st.session_state:
    st.session_state.add_paper_pid = None
if "create_project_error" not in st.session_state:
    st.session_state.create_project_error = ""


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _load_projects():
    try:
        resp = requests.get(f"{BACKEND}/projects", timeout=5)
        resp.raise_for_status()
        st.session_state.projects = resp.json()
    except Exception:
        st.session_state.projects = []


def _create_project(name: str, description: str = ""):
    try:
        resp = requests.post(
            f"{BACKEND}/projects",
            json={"name": name, "description": description},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()["project_id"]
    except Exception as e:
        st.error(f"Failed to create project: {e}")
        return None


def _delete_project(project_id: str):
    try:
        requests.delete(f"{BACKEND}/projects/{project_id}", timeout=5)
    except Exception:
        pass
    _load_projects()
    if st.session_state.active_project_id == project_id:
        st.session_state.active_project_id = None
        st.session_state.chat_history = []


def _switch_project(project_id: str):
    if st.session_state.active_project_id != project_id:
        st.session_state.active_project_id = project_id
        try:
            resp = requests.get(f"{BACKEND}/projects/{project_id}", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            st.session_state.chat_history = data.get("chat_display_history", [])
            st.session_state.project_papers = data.get("fetched_papers", [])
        except Exception:
            st.session_state.chat_history = []
            st.session_state.project_papers = []


def _refresh_project_papers(project_id: str):
    """Fetch the latest paper list from the backend and update session state."""
    try:
        resp = requests.get(f"{BACKEND}/projects/{project_id}", timeout=5)
        if resp.ok:
            st.session_state.project_papers = resp.json().get("fetched_papers", [])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Add Paper dialog
# ---------------------------------------------------------------------------

@st.dialog("Add Paper")
def add_paper_dialog(project_id: str):
    other_projects = [p for p in st.session_state.projects if p["project_id"] != project_id]
    method = st.radio(
        "Method", ["arXiv", "DOI", "Upload PDF", "Copy from Project"],
        horizontal=True, label_visibility="collapsed",
    )
    st.divider()

    if method == "arXiv":
        with st.form("arxiv_form"):
            arxiv_input = st.text_input("arXiv ID or URL", placeholder="1706.03762 or arxiv.org/abs/...")
            submitted = st.form_submit_button("Add", type="primary", use_container_width=True)
        if submitted:
            if not arxiv_input.strip():
                st.warning("Please enter an arXiv ID or URL.")
            else:
                with st.spinner("Fetching paper from arXiv…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/projects/{project_id}/papers/arxiv",
                            json={"arxiv_input": arxiv_input.strip()},
                            timeout=120,
                        )
                        if r.ok:
                            _refresh_project_papers(project_id)
                            _load_projects()
                            st.session_state.add_paper_pid = None
                            st.rerun()
                        else:
                            st.error(r.json().get("detail", "Failed to fetch paper."))
                    except Exception as e:
                        st.error(str(e))

    elif method == "DOI":
        with st.form("doi_form"):
            doi_input = st.text_input("DOI", placeholder="10.48550/arXiv.1706.03762")
            submitted = st.form_submit_button("Add", type="primary", use_container_width=True)
        if submitted:
            if not doi_input.strip():
                st.warning("Please enter a DOI.")
            else:
                with st.spinner("Resolving DOI…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/projects/{project_id}/papers/doi",
                            json={"doi": doi_input.strip()},
                            timeout=120,
                        )
                        if r.ok:
                            _refresh_project_papers(project_id)
                            _load_projects()
                            st.session_state.add_paper_pid = None
                            st.rerun()
                        else:
                            st.error(r.json().get("detail", "Failed to resolve DOI."))
                    except Exception as e:
                        st.error(str(e))

    elif method == "Upload PDF":
        uploaded = st.file_uploader(
            "PDF file", type=["pdf"],
            key=f"dialog_upload_{st.session_state.upload_key}",
        )
        if st.button("Add", type="primary", use_container_width=True):
            if not uploaded:
                st.warning("Please select a PDF file first.")
            else:
                with st.spinner("Ingesting PDF…"):
                    try:
                        r = requests.post(
                            f"{BACKEND}/projects/{project_id}/papers/upload",
                            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                            timeout=120,
                        )
                        if r.ok:
                            st.session_state.upload_key += 1
                            _refresh_project_papers(project_id)
                            _load_projects()
                            st.session_state.add_paper_pid = None
                            st.rerun()
                        else:
                            st.error(r.json().get("detail", "Failed to ingest PDF."))
                    except Exception as e:
                        st.error(str(e))

    elif method == "Copy from Project":
        if not other_projects:
            st.info("No other projects to copy from.")
        else:
            src_options = {p["name"]: p["project_id"] for p in other_projects}
            src_name = st.selectbox("Source project", list(src_options.keys()))
            src_pid = src_options[src_name]
            try:
                src_resp = requests.get(f"{BACKEND}/projects/{src_pid}", timeout=5)
                src_papers = src_resp.json().get("fetched_papers", []) if src_resp.ok else []
            except Exception:
                src_papers = []
            existing_titles = {p["title"] for p in st.session_state.project_papers}
            copyable = [p for p in src_papers if p["title"] not in existing_titles]
            if not copyable:
                st.info("No new papers to copy from this project.")
            else:
                selected_title = st.selectbox("Paper", [p["title"] for p in copyable])
                if st.button("Copy", type="primary", use_container_width=True):
                    with st.spinner("Copying paper…"):
                        try:
                            r = requests.post(
                                f"{BACKEND}/projects/{project_id}/papers/copy",
                                json={"source_project_id": src_pid, "title": selected_title},
                                timeout=30,
                            )
                            if r.ok:
                                _refresh_project_papers(project_id)
                                _load_projects()
                                st.session_state.add_paper_pid = None
                                st.rerun()
                            else:
                                st.error(r.json().get("detail", "Failed to copy paper."))
                        except Exception as e:
                            st.error(str(e))


# Always refresh projects list on every render to stay in sync with the backend
_load_projects()


# ---------------------------------------------------------------------------
# Sidebar — project management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Projects")

    if st.button("＋ New Project", use_container_width=True):
        st.session_state.creating_project = True

    if st.session_state.creating_project:
        with st.form("new_project_form", clear_on_submit=False):
            new_name = st.text_input("Project name", placeholder="e.g. Transformer Study")
            new_desc = st.text_area("Description (optional)", height=60)
            col1, col2 = st.columns(2)
            submitted = col1.form_submit_button("Create")
            cancelled = col2.form_submit_button("Cancel")
        if st.session_state.create_project_error:
            st.error(st.session_state.create_project_error)
        if submitted and new_name.strip():
            name = new_name.strip()
            existing_names = {p["name"].lower() for p in st.session_state.projects}
            if name.lower() in existing_names:
                st.session_state.create_project_error = f"A project named '{name}' already exists."
                st.rerun()
            else:
                st.session_state.create_project_error = ""
                pid = _create_project(name, new_desc.strip())
                if pid:
                    _switch_project(pid)
                st.session_state.creating_project = False
                st.rerun()
        if cancelled:
            st.session_state.create_project_error = ""
            st.session_state.creating_project = False
            st.rerun()

    st.divider()

    if not st.session_state.projects:
        st.caption("No projects yet. Create one above.")
    else:
        for project in st.session_state.projects:
            pid = project["project_id"]
            is_active = st.session_state.active_project_id == pid

            col_name, col_del = st.columns([5, 1])
            btn_label = f"**{project['name']}**" if is_active else project["name"]
            if col_name.button(btn_label, key=f"proj_{pid}", use_container_width=True):
                _switch_project(pid)
                st.rerun()
            if col_del.button("✕", key=f"del_{pid}", help="Delete project"):
                _delete_project(pid)
                st.rerun()

            # Paper list
            if is_active:
                current_papers = st.session_state.project_papers

                with st.expander(f"Papers ({len(current_papers)})", expanded=is_active):
                    for paper in current_papers:
                        title = paper.get("title", "")
                        col_t, col_x = st.columns([5, 1])
                        col_t.caption(f"• {title}")
                        if col_x.button("✕", key=f"delpaper_{pid}_{title[:20]}", help="Remove paper"):
                            try:
                                requests.delete(
                                    f"{BACKEND}/projects/{pid}/papers",
                                    json={"title": title},
                                    timeout=10,
                                )
                                _refresh_project_papers(pid)
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                            st.rerun()

                    st.divider()
                    if st.button("＋ Add Paper", key=f"open_add_{pid}", use_container_width=True):
                        st.session_state.add_paper_pid = pid
                        st.rerun()
            else:
                paper_count = project.get("paper_count", 0)
                if paper_count:
                    st.caption(f"Papers: {paper_count}")

            st.divider()


# ---------------------------------------------------------------------------
# Dialog trigger (must be outside the sidebar context)
# ---------------------------------------------------------------------------

if st.session_state.add_paper_pid:
    add_paper_dialog(st.session_state.add_paper_pid)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

if st.session_state.active_project_id is None:
    st.title("Research Assistant 🔬")
    st.info("Select a project from the sidebar or create a new one to start chatting.")
else:
    active_name = next(
        (p["name"] for p in st.session_state.projects
         if p["project_id"] == st.session_state.active_project_id),
        "Project",
    )
    st.title(f"🔬 {active_name}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = message.get("sources", [])
                tool_used = message.get("tool_used")
                rationale = message.get("rationale")
                if sources:
                    st.markdown(f"**Sources:** {', '.join(sources)}")
                if tool_used or rationale:
                    with st.expander("Show details (tool & rationale)"):
                        st.markdown(f"**Tool Used:** {tool_used or 'N/A'}")
                        st.markdown(f"**Rationale:** {rationale or 'N/A'}")

    user_prompt = st.chat_input("Ask about your research papers...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        payload = {
            "user_query": user_prompt,
            "project_id": st.session_state.active_project_id,
        }
        try:
            response = requests.post(settings.CHAT_ENDPOINT_URL, json=payload)
            response.raise_for_status()
            rj = response.json()
            assistant_response = rj.get("answer", "(No response)")
            tool_used = rj.get("tool_used")
            rationale = rj.get("rationale")
            sources = rj.get("sources", [])
            fetched_papers = rj.get("fetched_papers", [])
            if fetched_papers:
                st.session_state.project_papers = fetched_papers
        except Exception as e:
            assistant_response = f"Error: {e}"
            tool_used = None
            rationale = None
            sources = []
            fetched_papers = []

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": assistant_response,
            "tool_used": tool_used,
            "rationale": rationale,
            "sources": sources,
            "fetched_papers": fetched_papers,
        })

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            if tool_used or rationale:
                with st.expander("Show details (tool & rationale)"):
                    st.markdown(f"**Tool Used:** {tool_used or 'N/A'}")
                    st.markdown(f"**Rationale:** {rationale or 'N/A'}")

        _load_projects()
        st.rerun()
