import gradio as gr
from src.retrieval import Retriever
from src.llm import LLMEngine
from src.database import initialize_database

# --- Global State ---
# Initialize heavy models only once at startup
print("⚙️ System Startup...")
try:
    initialize_database() # Load Embeddings
    retriever = Retriever(use_reranker=True) # Load Vector DB + Reranker
    llm_engine = LLMEngine() # Load LLM Client
    print("✅ System Ready! Launching UI...")
except Exception as e:
    print(f"❌ Failed to initialize system: {e}")
    print("   Make sure:")
    print("   1. You've ingested at least one repository")
    print("   2. GROQ_API_KEY is set in .env file")
    raise

def generate_response(message, history):
    """
    The Core RAG Loop for Gradio:
    1. Retrieve relevant code chunks.
    2. Generate Answer with LLM.
    3. Format Sources for display.
    
    Args:
        message: User's question (history is ignored for stateless RAG)
    """
    if not message or not message.strip():
        return "Please enter a question about the codebase."
    
    try:
        # 1. Retrieval
        print(f"🔍 User asked: {message}")
        nodes = retriever.search(message)
        
        if not nodes:
            return "I couldn't find any relevant code in the ingested repository. Try rephrasing your question or ingest a different repo."

        # 2. Generation
        response_text = llm_engine.chat(message, nodes)
        
        # 3. Format Sources (Appended to the bottom)
        # We create a collapsible-style section for sources using Markdown
        source_text = "\n\n---\n### 📚 Context Sources\n"
        for i, node in enumerate(nodes, 1):
            metadata = getattr(node, 'metadata', {})
            file_path = metadata.get('file_path', 'Unknown')
            score = getattr(node, 'score', None)
            
            # Format score safely
            if score is not None and isinstance(score, (int, float)):
                score_str = f"{score:.2f}"
            else:
                score_str = "N/A"
            
            # Create a clean summary line
            source_text += f"**{i}. {file_path}** (Relevance: `{score_str}`)\n"
            
        return response_text + source_text

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error in generate_response: {error_details}")
        return f"❌ Error: {str(e)}\n\nPlease check the console for details."

# --- Gradio UI Layout ---
with gr.Blocks(title="RepoMind") as demo:
    gr.Markdown(
        """
        # 🧠 RepoMind
        ### Chat with your Codebase (RAG)
        """
    )
    
    # The Chat Interface
    chat_interface = gr.ChatInterface(
        fn=generate_response,
        chatbot=gr.Chatbot(height=600, render_markdown=True),
        textbox=gr.Textbox(
            placeholder="Ask about authentication, database schema, or specific functions...", 
            container=False, 
            scale=7
        ),
        title=None,
        description="Type a question below to analyze the ingested repository.",
        examples=[
            "How is authentication handled?",
            "Explain the project structure.",
            "What dependencies are used?",
            "How does the webhook system work?",
        ],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
        server_port=7860,
        share=False,
        show_error=True
    )