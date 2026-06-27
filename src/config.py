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

# Ensure directories exist
os.makedirs(CLONE_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

# --- Model Configs ---
# Option A (Better): "BAAI/bge-m3"
# Option B (Lite): "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# --- Retrieval / RAG Config ---
CHUNK_SIZE = get_int_env("CHUNK_SIZE", 1500)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 200)
TOP_K = get_int_env("TOP_K", 40)
RERANK_TOP_K = get_int_env("RERANK_TOP_K", 12)
# Reranker model options:
# "BAAI/bge-reranker-v2-m3" (Most accurate, but slow on CPU)
# "cross-encoder/ms-marco-MiniLM-L-12-v2" (good accuracy)
# "cross-encoder/ms-marco-MiniLM-L-6-v2" (fastest, good enough accuracy)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# --- LLM Configs ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
