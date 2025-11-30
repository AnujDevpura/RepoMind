import os
import base64
import gradio as gr
import io
import sys
from src.retrieval import Retriever
from src.llm import LLMEngine
from src.database import initialize_database
from src.ingestion import ingest_repo

# --- Global State ---
print("⚙️ System Startup...")
try:
    initialize_database()  # Load Embeddings
    retriever = Retriever(use_reranker=True)  # Load Vector DB + Reranker
    llm_engine = LLMEngine()  # Load LLM Client
    print("✅ System Ready! Launching UI...")
except Exception as e:
    print(f"❌ Failed to initialize system: {e}")
    print("   Make sure:")
    print("   1. You've ingested at least one repository")
    print("   2. GROQ_API_KEY is set in .env file")
    raise

def handle_ingestion(repo_url: str, force_clone: bool):
    """
    Wrapper function to run ingestion and capture output for the UI.
    """
    if not repo_url or not repo_url.strip():
        return "❌ Error: Repository URL cannot be empty."

    # Redirect stdout to capture logs
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        ingest_repo(repo_url, force_clone=force_clone)
        
        # --- CRITICAL: Reload the retriever to see the new data ---
        global retriever
        print("\n🔄 Reloading retrieval engine with new data...")
        retriever = Retriever(use_reranker=True)
        print("✅ Retrieval engine reloaded.")
        
    except Exception as e:
        print(f"\n\n❌ INGESTION FAILED: {e}")
    finally:
        # Restore stdout and get the captured output
        sys.stdout = old_stdout
        log_output = captured_output.getvalue()

    # Use Markdown for better formatting in Gradio
    return f"```\n{log_output}\n```"


def retrieve_and_chat(message, history):
    """
    Core RAG Generator.
    Compatible with ChatInterface format.
    """
    if not message or not message.strip():
        return ""
    
    try:
        # 1. Retrieval
        print(f"🔍 User asked: {message}")
        nodes = retriever.search(message)
        
        if not nodes:
            response_text = "I couldn't find any relevant code in the ingested repository. Try rephrasing your question or ingest a different repo."
            return response_text
        
        # 2. Generation
        response_text = llm_engine.chat(message, nodes)
        
        # 3. Format Sources
        sources_html = "\n\n<details><summary>📚 <strong>Context Sources</strong></summary>\n\n"
        for i, node in enumerate(nodes, 1):
            score = getattr(node, 'score', None)
            metadata = getattr(node, 'metadata', {})
            file_path = metadata.get('file_path', 'Unknown')
            score_str = f"{score:.2f}" if score is not None else "N/A"
            sources_html += f"{i}. `{file_path}` (Relevance: {score_str})<br>"
        sources_html += "</details>"
        
        # 4. Return Complete Response
        return response_text + sources_html
                
    except Exception as e:
        import traceback
        print(f"❌ Error: {traceback.format_exc()}")
        return f"❌ Error: {str(e)}"

# --- Custom CSS ---
custom_css = """
/* Clean Header */
.header-container {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #2a387a 0%, #3c2657 100%);
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
}
.header-container h1 { 
    font-size: 2.5rem; 
    margin-bottom: 0.5rem; 
    color: white;
    font-weight: 700;
}
.header-container p {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.9);
}

/* Chat Styling */
.chat-window { 
    height: 600px !important; 
    border-radius: 8px;
}

/* Footer Styling */
.footer-text {
    text-align: center;
    margin-top: 20px;
    opacity: 0.6;
    font-size: 0.9rem;
}
"""

# --- Theme ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=("Inter", "system-ui", "sans-serif"),
)

# --- Gradio UI Layout ---
with gr.Blocks(title="RepoMind") as demo:

    # --- Asset Paths ---
    script_dir = os.path.dirname(__file__)
    avatar_path = os.path.abspath(os.path.join(script_dir, "..", "assets", "avatar.png"))

    # --- Encode Avatar for HTML ---
    with open(avatar_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    avatar_data_uri = f"data:image/png;base64,{encoded_string}"


    # Header
    gr.HTML(f"""
        <div class="header-container">
            <h1 style="display: flex; align-items: center; justify-content: center;">
                <img src="{avatar_data_uri}" style="height: 40px; margin-right: 10px;">RepoMind
            </h1>
            <p>Your AI-Powered Code Intelligence Assistant</p>
        </div>
    """)

    with gr.Tabs() as tabs:
        with gr.TabItem("💬 Chat with Repo", id=0):
            # Info Box
            gr.Markdown(
                """
                > **💡 Tip:** Ask about specific files, function logic, or architectural patterns. 
                > The system uses **RAG (Retrieval-Augmented Generation)** to cite its sources.
                """
            )

            # Main Chat Interface
            gr.ChatInterface(
                fn=retrieve_and_chat,
                chatbot=gr.Chatbot(
                    height=600,
                    avatar_images=(None, avatar_path),
                    render_markdown=True,
                ),
                textbox=gr.Textbox(
                    placeholder="How does the authentication middleware work?",
                    container=False,
                    scale=7
                ),
                examples=[
                    "What is the main purpose of this repository?",
                    "Explain the database schema.",
                    "What dependencies are used?",
                    "Summarize the main entry point of the app.",
                ],
            )

        with gr.TabItem("➕ Ingest New Repo", id=1):
            gr.Markdown(
                "## Ingest a New GitHub Repository\n"
                "Enter the URL of a public GitHub repository to add it to the vector database. "
                "This process may take several minutes depending on the size of the repository."
            )
            with gr.Row():
                repo_url_input = gr.Textbox(
                    label="GitHub Repository URL", 
                    placeholder="https://github.com/some-user/some-repo",
                    scale=4
                )
                force_clone_checkbox = gr.Checkbox(label="Force Re-clone if Exists", value=False)
            
            ingest_button = gr.Button("🚀 Ingest Repository", variant="primary")
            
            ingestion_status = gr.Markdown(label="Ingestion Status", value="*Awaiting ingestion...*")
            
            ingest_button.click(
                fn=handle_ingestion,
                inputs=[repo_url_input, force_clone_checkbox],
                outputs=ingestion_status
            )

    # Footer
    gr.HTML("""
        <div class="footer-text">
            RepoMind • Powered by LlamaIndex & ChromaDB
        </div>
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=theme, 
        css=custom_css
    )