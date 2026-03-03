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
            st.session_state.chat_history = resp.json().get("chat_display_history", [])
        except Exception:
            st.session_state.chat_history = []


# Load projects on first render
if not st.session_state.projects:
    _load_projects()


# ---------------------------------------------------------------------------
# Sidebar — project management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Projects")

    if st.button("＋ New Project", use_container_width=True):
        st.session_state.creating_project = True

    if st.session_state.creating_project:
        with st.form("new_project_form", clear_on_submit=True):
            new_name = st.text_input("Project name", placeholder="e.g. Transformer Study")
            new_desc = st.text_area("Description (optional)", height=60)
            col1, col2 = st.columns(2)
            submitted = col1.form_submit_button("Create")
            cancelled = col2.form_submit_button("Cancel")
        if submitted and new_name.strip():
            pid = _create_project(new_name.strip(), new_desc.strip())
            if pid:
                _switch_project(pid)
            st.session_state.creating_project = False
            st.rerun()
        if cancelled:
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
                # Get current paper list from last assistant message (most up to date)
                current_papers = []
                for msg in reversed(st.session_state.chat_history):
                    if msg.get("role") == "assistant" and "fetched_papers" in msg:
                        current_papers = msg["fetched_papers"]
                        break

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
                                # Update local chat history to reflect removal
                                proj_resp = requests.get(f"{BACKEND}/projects/{pid}", timeout=5)
                                if proj_resp.ok:
                                    updated = proj_resp.json()
                                    # Patch fetched_papers in all assistant messages in chat history
                                    for m in st.session_state.chat_history:
                                        if m.get("role") == "assistant":
                                            m["fetched_papers"] = updated.get("fetched_papers", [])
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                            st.rerun()

                    # Add paper section
                    st.markdown("**Add Paper**")
                    add_tab = st.radio("Method", ["arXiv", "DOI", "Upload PDF", "Copy from Project"],
                                       key=f"add_method_{pid}", horizontal=True, label_visibility="collapsed")

                    if add_tab == "arXiv":
                        arxiv_input = st.text_input("arXiv ID or URL", key=f"arxiv_{pid}",
                                                    placeholder="1706.03762 or arxiv.org/abs/...")
                        if st.button("Add", key=f"arxiv_add_{pid}") and arxiv_input.strip():
                            with st.spinner("Fetching from arXiv..."):
                                try:
                                    r = requests.post(
                                        f"{BACKEND}/projects/{pid}/papers/arxiv",
                                        json={"arxiv_input": arxiv_input.strip()},
                                        timeout=120,
                                    )
                                    if r.ok:
                                        st.success(f"Added: {r.json().get('title', '')}")
                                        _load_projects()
                                    else:
                                        st.error(r.json().get("detail", "Failed"))
                                except Exception as e:
                                    st.error(str(e))
                            st.rerun()

                    elif add_tab == "DOI":
                        doi_input = st.text_input("DOI", key=f"doi_{pid}",
                                                  placeholder="10.48550/arXiv.1706.03762")
                        if st.button("Add", key=f"doi_add_{pid}") and doi_input.strip():
                            with st.spinner("Resolving DOI..."):
                                try:
                                    r = requests.post(
                                        f"{BACKEND}/projects/{pid}/papers/doi",
                                        json={"doi": doi_input.strip()},
                                        timeout=120,
                                    )
                                    if r.ok:
                                        st.success(f"Added: {r.json().get('title', '')}")
                                        _load_projects()
                                    else:
                                        st.error(r.json().get("detail", "Failed"))
                                except Exception as e:
                                    st.error(str(e))
                            st.rerun()

                    elif add_tab == "Upload PDF":
                        uploaded = st.file_uploader("PDF file", type=["pdf"], key=f"upload_{pid}")
                        if uploaded and st.button("Add", key=f"upload_add_{pid}"):
                            with st.spinner("Ingesting PDF..."):
                                try:
                                    r = requests.post(
                                        f"{BACKEND}/projects/{pid}/papers/upload",
                                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                                        timeout=120,
                                    )
                                    if r.ok:
                                        st.success(f"Added: {r.json().get('title', '')}")
                                        _load_projects()
                                    else:
                                        st.error(r.json().get("detail", "Failed"))
                                except Exception as e:
                                    st.error(str(e))
                            st.rerun()

                    elif add_tab == "Copy from Project":
                        other_projects = [p for p in st.session_state.projects if p["project_id"] != pid]
                        if not other_projects:
                            st.caption("No other projects to copy from.")
                        else:
                            src_options = {p["name"]: p["project_id"] for p in other_projects}
                            src_name = st.selectbox("Source project", list(src_options.keys()),
                                                    key=f"copy_src_{pid}")
                            src_pid = src_options[src_name]
                            # Load papers available in source project
                            try:
                                src_resp = requests.get(f"{BACKEND}/projects/{src_pid}", timeout=5)
                                src_papers = src_resp.json().get("fetched_papers", []) if src_resp.ok else []
                            except Exception:
                                src_papers = []
                            # Filter out papers already in this project
                            existing_titles = {p["title"] for p in current_papers}
                            copyable = [p for p in src_papers if p["title"] not in existing_titles]
                            if not copyable:
                                st.caption("No new papers to copy from that project.")
                            else:
                                paper_titles = [p["title"] for p in copyable]
                                selected_title = st.selectbox("Paper to copy", paper_titles,
                                                              key=f"copy_paper_{pid}")
                                if st.button("Copy", key=f"copy_add_{pid}"):
                                    with st.spinner("Copying paper..."):
                                        try:
                                            r = requests.post(
                                                f"{BACKEND}/projects/{pid}/papers/copy",
                                                json={"source_project_id": src_pid, "title": selected_title},
                                                timeout=30,
                                            )
                                            if r.ok:
                                                st.success(f"Copied: {selected_title}")
                                                _load_projects()
                                            else:
                                                st.error(r.json().get("detail", "Failed"))
                                        except Exception as e:
                                            st.error(str(e))
                                    st.rerun()
            else:
                paper_count = project.get("paper_count", 0)
                if paper_count:
                    st.caption(f"Papers: {paper_count}")

            st.divider()


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
