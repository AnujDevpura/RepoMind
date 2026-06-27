import os
import sys
import io
import streamlit as st
import time

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# --- Page Config & CSS ---
st.set_page_config(page_title="RepoMind", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for the "Anti-Streamlit" premium look
custom_css = """
<style>
/* Hide the top header and hamburger menu */
header {visibility: hidden;}

/* Hide the footer */
footer {visibility: hidden;}

/* Remove padding from the main block and adjust for full width */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 0rem !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
    max-width: 100% !important;
}

/* Premium dark theme for sidebar */
[data-testid="stSidebar"] {
    background-color: #121212 !important;
    border-right: 1px solid #2d2d30 !important;
}

/* Chat input styling */
[data-testid="stChatInputContainer"] {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 16px !important;
    padding-bottom: 10px !important;
}

/* Chat message bubbles */
[data-testid="chatAvatarIcon-user"] {
    background-color: #3b82f6 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #8b5cf6 !important;
}

/* Make headers look cleaner */
h1, h2, h3 {
    font-weight: 600 !important;
    color: #f3f4f6 !important;
}

/* Make buttons pop slightly */
.stButton>button {
    border-radius: 8px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Add project root to sys.path so 'src' module can be found when running via Streamlit
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports must happen after page config in Streamlit
from src.retrieval import Retriever
from src.llm import LLMEngine
from src.database import initialize_database, get_all_repositories, delete_repository
from src.ingestion import ingest_repo
from src.config import USE_RERANKER

# --- Cache Models ---
# We use @st.cache_resource so the Vector DB and LLM don't reload on every chat message!
@st.cache_resource(show_spinner=False)
def load_system():
    initialize_database()
    return Retriever(use_reranker=USE_RERANKER), LLMEngine()

try:
    with st.spinner("Initializing AI Core..."):
        retriever, llm_engine = load_system()
except Exception as e:
    st.error(f"Failed to initialize system: {e}")
    st.stop()

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("🧠 RepoMind")
    st.caption("AI-Powered Code Intelligence")
    st.divider()
    
    st.subheader("📂 Workspace Context")
    all_repos = ["All Repositories"] + get_all_repositories()
    selected_repo = st.selectbox(
        "Target Repository",
        options=all_repos,
        index=0,
        help="Scope AI search to a specific codebase"
    )
    
    st.divider()
    with st.expander("➕ Ingest New Repo"):
        repo_url = st.text_input("GitHub URL")
        force_clone = st.checkbox("Force Re-clone")
        if st.button("Start Ingestion", type="primary", use_container_width=True):
            if not repo_url:
                st.error("Please enter a URL.")
            else:
                with st.spinner("Cloning and processing... this may take a while."):
                    try:
                        ingest_repo(repo_url, force_clone)
                        # Clear cache so the retriever picks up the new index
                        load_system.clear()
                        st.success("✅ Ingestion complete!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
    
    with st.expander("🗑️ Manage Repos"):
        del_target = st.selectbox("Delete Repo", options=["Select..."] + get_all_repositories())
        if st.button("Delete", use_container_width=True):
            if del_target != "Select...":
                delete_repository(del_target)
                load_system.clear()
                st.success(f"Deleted {del_target}")
                time.sleep(1)
                st.rerun()
                
    st.divider()
    st.caption(f"**Engine:** {llm_engine.provider.upper()} ({llm_engine.model_name})")
    st.caption(f"**Reranker:** {'Enabled' if USE_RERANKER else 'Disabled'}")

# --- Main Chat Interface ---
# 1. Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Chat Input Trigger
if prompt := st.chat_input("Ask RepoMind about your codebase..."):
    
    # Append user message to state and display it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        sources_md = ""
        nodes = []
        
        # Retrieval Phase
        with st.status("Searching codebase...", expanded=True) as status:
            try:
                nodes = retriever.search(prompt, repo_name=selected_repo)
                if not nodes:
                    st.warning("No relevant code found in this repository.")
                    status.update(label="No results found", state="error")
                else:
                    st.write(f"Found {len(nodes)} relevant snippets.")
                    sources_md = "\n\n---\n**📚 Context Sources:**\n"
                    for i, node in enumerate(nodes, 1):
                        score = getattr(node, 'score', None)
                        metadata = getattr(node, 'metadata', {})
                        file_path = metadata.get('file_path', 'Unknown')
                        node_repo = metadata.get('repo_name', 'Unknown Repo')
                        score_str = f"{score:.2f}" if score is not None else "N/A"
                        sources_md += f"- `[{node_repo}] {file_path}` (Relevance: {score_str})\n"
                    status.update(label="Context retrieved successfully", state="complete")
            except Exception as e:
                status.update(label=f"Search failed: {e}", state="error")
        
        # Generation Phase
        if nodes:
            # Build history format expected by llm_engine
            history = []
            for i in range(0, len(st.session_state.messages)-1, 2):
                if i+1 < len(st.session_state.messages):
                    history.append((
                        st.session_state.messages[i]["content"], 
                        st.session_state.messages[i+1]["content"]
                    ))
            
            # Stream the response natively in Streamlit
            def generate():
                for chunk in llm_engine.stream_chat(prompt, nodes, history):
                    if chunk:
                        yield chunk
            
            try:
                # write_stream types out the chunks as they arrive
                response = st.write_stream(generate())
                
                # Append sources visually at the end of the stream
                if sources_md:
                    st.markdown(sources_md)
                    response += sources_md
                    
                # Save the final text back to state
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Generation error: {e}")