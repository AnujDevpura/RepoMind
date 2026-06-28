import os
import sys
import io
import streamlit as st
import time
import logging

# Suppress the "LLM explicitly disabled" LlamaIndex info log — Settings.llm = None
# is intentional. Our chat LLM is LLMEngine (Groq/Ollama), not LlamaIndex's Settings.llm.
logging.getLogger("llama_index.core").setLevel(logging.WARNING)

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Page Config MUST be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="RepoMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Add project root to sys.path ──────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieval import Retriever
from src.llm import LLMEngine
from src.database import initialize_database, get_all_repositories, delete_repository
from src.ingestion import ingest_repo
from src.config import USE_RERANKER, RETRIEVE_LLM_PROVIDER, RETRIEVE_LLM_MODEL

# ══════════════════════════════════════════════════════════════════════════════
# CSS  —  Safe layout tweaks. Colors are handled by .streamlit/config.toml
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Font ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Hide Streamlit footer and default top-right menu ───── */
#MainMenu, footer { visibility: hidden !important; }

/* ── Main content centering ──────────────────────────────── */
.block-container {
    max-width: 850px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 2rem !important;
    padding-bottom: 6rem !important;
}

/* ── Chat input ──────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {
    border-radius: 14px !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
}

/* ── Code blocks ─────────────────────────────────────────── */
pre  { border-radius: 8px !important; }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# System init (cached — loads once, not on every chat message)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_system():
    # load_embed_model=True → loads bge-m3 on CPU for retrieval queries.
    # (The ingestion path uses load_embed_model=False and loads on CUDA instead.)
    initialize_database(load_embed_model=True)
    retriever  = Retriever(use_reranker=USE_RERANKER)
    llm_engine = LLMEngine(provider=RETRIEVE_LLM_PROVIDER, model_name=RETRIEVE_LLM_MODEL)
    return retriever, llm_engine

try:
    with st.spinner("Loading AI core…"):
        retriever, llm_engine = load_system()
except Exception as e:
    st.error(f"**System initialization failed:** {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Session state defaults
# ══════════════════════════════════════════════════════════════════════════════
if "messages"      not in st.session_state: st.session_state.messages      = []
if "selected_repo" not in st.session_state: st.session_state.selected_repo = "All Repositories"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown("""
        <div style="padding-bottom: 0.5rem;">
            <span style="font-size:1.2rem; font-weight:700; color:#ececec;">🧠 RepoMind</span><br>
            <span style="font-size:0.72rem; color:#6b7280;">AI-Powered Code Intelligence</span>
        </div>
    """, unsafe_allow_html=True)

    if st.button("✏️  New Chat", use_container_width=True, key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Repository navigation ────────────────────────────────
    st.markdown("<p style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Repositories</p>", unsafe_allow_html=True)

    all_repos = get_all_repositories()

    # All Repositories
    is_all = st.session_state.selected_repo == "All Repositories"
    label_all = ("● " if is_all else "  ") + "All Repositories"
    if st.button(label_all, use_container_width=True, key="repo_all"):
        if st.session_state.selected_repo != "All Repositories":
            st.session_state.selected_repo = "All Repositories"
            st.session_state.messages = []
        st.rerun()

    if not all_repos:
        st.caption("No repos ingested yet.")
    else:
        for repo in all_repos:
            is_active = st.session_state.selected_repo == repo
            label = ("● " if is_active else "  ") + "📁 " + repo
            if st.button(label, use_container_width=True, key=f"repo_{repo}"):
                if st.session_state.selected_repo != repo:
                    st.session_state.selected_repo = repo
                    st.session_state.messages = []
                st.rerun()

    st.divider()

    # ── Ingest new repo ──────────────────────────────────────
    st.markdown("<p style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Add Repository</p>", unsafe_allow_html=True)

    repo_url   = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo", label_visibility="collapsed", key="ingest_url")
    force_clone = st.checkbox("Force re-clone", key="force_clone")

    if st.button("⬆  Ingest", type="primary", use_container_width=True, key="ingest_btn"):
        if not repo_url.strip():
            st.error("Enter a GitHub URL first.")
        else:
            with st.spinner("Cloning and processing…"):
                try:
                    ingest_repo(repo_url.strip(), force_clone)
                    load_system.clear()
                    st.success("✅ Done!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    # ── Delete repos ─────────────────────────────────────────
    with st.expander("🗑  Manage Repos"):
        if all_repos:
            del_target = st.selectbox("Delete", ["Select…"] + all_repos, label_visibility="collapsed", key="del_select")
            if st.button("Delete", use_container_width=True, key="del_btn"):
                if del_target != "Select…":
                    delete_repository(del_target)
                    load_system.clear()
                    if st.session_state.selected_repo == del_target:
                        st.session_state.selected_repo = "All Repositories"
                    st.success(f"Deleted {del_target}")
                    time.sleep(1)
                    st.rerun()
        else:
            st.caption("No repositories to manage.")

    # ── Engine info ──────────────────────────────────────────
    st.divider()
    st.caption(f"**LLM:** {llm_engine.provider.upper()} · `{llm_engine.model_name}`")
    st.caption(f"**Reranker:** {'✓ On' if USE_RERANKER else '✗ Off'}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
selected_repo = st.session_state.selected_repo
prompt = st.chat_input(f"Ask about {selected_repo}…")

# ── Welcome / empty state ─────────────────────────────────────────────────────
if not st.session_state.messages and not prompt:
    st.markdown(f"""
    <div style="text-align:center; padding: 3.5rem 1rem 2rem;">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🧠</div>
        <h2 style="color:#ececec; font-weight:700; font-size:1.5rem; margin:0;">What can I help you understand?</h2>
        <p style="color:#6b7280; margin-top:0.4rem; font-size:0.9rem;">
            Searching across: <strong style="color:#9ca3af;">{selected_repo}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    starters = [
        ("🏗️ Architecture overview",  "Explain the overall architecture and main components of this repository."),
        ("📡 Service communication",   "How do the microservices communicate with each other?"),
        ("🔒 Auth & security",         "How does authentication and authorization work in this codebase?"),
        ("⚡ Data flow entry points",  "What are the main entry points and how does data flow through the system?"),
    ]
    c1, c2 = st.columns(2, gap="small")
    for i, (label, full_prompt) in enumerate(starters):
        col = c1 if i % 2 == 0 else c2
        with col:
            if st.button(label, use_container_width=True, key=f"starter_{i}"):
                st.session_state.messages.append({"role": "user", "content": full_prompt})
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Process new input ─────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        nodes      = []
        sources_md = ""

        # Retrieval phase
        with st.status("Searching codebase…", expanded=False) as status:
            try:
                nodes = retriever.search(prompt, repo_name=selected_repo)
                if not nodes:
                    status.update(label="No relevant code found", state="error")
                else:
                    n = len(nodes)
                    status.update(label=f"Found {n} relevant snippet{'s' if n != 1 else ''}", state="complete")
                    sources_md = "\n\n---\n**📚 Sources**\n"
                    for node in nodes:
                        score = getattr(node, "score", None)
                        meta  = getattr(node, "metadata", {})
                        fp    = meta.get("file_path", "Unknown")
                        score_str = f"{score:.2f}" if score is not None else "—"
                        sources_md += f"- `{fp}` · relevance {score_str}\n"
            except Exception as e:
                status.update(label=f"Search failed: {e}", state="error")

        # Generation phase
        if nodes:
            msgs = st.session_state.messages
            history = [
                (msgs[i]["content"], msgs[i + 1]["content"])
                for i in range(0, len(msgs) - 1, 2)
                if i + 1 < len(msgs)
            ]

            def generate():
                for chunk in llm_engine.stream_chat(prompt, nodes, history):
                    if chunk:
                        yield chunk

            try:
                response = st.write_stream(generate())
                if sources_md:
                    st.markdown(sources_md)
                    response += sources_md
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Generation error: {e}")