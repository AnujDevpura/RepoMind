import os
from pathlib import Path

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
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Retrieval Configs ---
TOP_K = 15
RERANK_TOP_K = 5
# Reranker model options (ranked by accuracy):
# "BAAI/bge-reranker-v2-m3"
# "cross-encoder/ms-marco-MiniLM-L-12-v2" (good accuracy)
# "cross-encoder/ms-marco-MiniLM-L-6-v2" (fastest, good enough accuracy)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- LLM Configs ---
# LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

# --- Ingestion Configs ---
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200