import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# --- Directory Paths ---
# We define the project root as the parent of the 'src' folder
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLONE_DIR = os.path.join(DATA_DIR, "cloned_repos")
CHROMA_PATH = os.path.join(DATA_DIR, "chromadb")
GRAPH_PATH = os.path.join(DATA_DIR, "graphdb")

# Ensure directories exist
os.makedirs(CLONE_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)

# --- Model Configs ---
# Option A (Better): "BAAI/bge-m3"
# Option B (Lite): "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# --- Retrieval / RAG Config ---
TOP_K = int(os.getenv("TOP_K", 3))

# --- Feature Flags ---
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

# --- LLM Configs (Ingestion) ---
INGEST_LLM_PROVIDER = os.getenv("INGEST_LLM_PROVIDER", "ollama").lower()
INGEST_LLM_MODEL = os.getenv("INGEST_LLM_MODEL", "qwen2.5-coder:7b")

# --- LLM Configs (Retrieval) ---
RETRIEVE_LLM_PROVIDER = os.getenv("RETRIEVE_LLM_PROVIDER", "groq").lower()
RETRIEVE_LLM_MODEL = os.getenv("RETRIEVE_LLM_MODEL", "llama-3.3-70b-versatile")
