import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core.graph_stores import SimplePropertyGraphStore
from src.config import CHROMA_PATH, GRAPH_PATH, EMBEDDING_MODEL_NAME

load_dotenv()

def get_vector_store():
    """
    Initializes the ChromaDB client and sets up the storage context.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection("repomind_codebase")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def get_graph_store():
    """
    Initializes the SimplePropertyGraphStore.
    """
    # LlamaIndex SimplePropertyGraphStore persists to JSON
    graph_path = os.path.join(GRAPH_PATH, "graph_store.json")
    if os.path.exists(graph_path):
        try:
            return SimplePropertyGraphStore.from_persist_path(graph_path)
        except Exception as e:
            print(f"⚠️ Failed to load graph store: {e}, creating a new one.")
    return SimplePropertyGraphStore()

def initialize_database(load_embed_model: bool = True):
    """
    Sets up the global LlamaIndex settings.
    This must be called at the start of the application.

    Args:
        load_embed_model: If True (default, retrieval/app), load bge-m3 on CUDA.
            If False (ingestion path), defer — SemanticEnrichmentComponent will
            load bge-m3 on CUDA after evicting the Ollama LLM.
    """
    if load_embed_model:
        print(f"🔄 Loading Embedding Model: {EMBEDDING_MODEL_NAME} on CUDA...")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device="cuda",        # Always GPU — no CPU embedding ever
            embed_batch_size=16,
        )
        Settings.embed_model = embed_model
        print("✅ Embedding Model Loaded on CUDA.")
    else:
        # Ingestion path: bge-m3 will be loaded on CUDA by SemanticEnrichmentComponent
        # after Ollama is evicted, so we don't compete for VRAM here.
        Settings.embed_model = None
        print("ℹ️  Embedding model deferred — will load on CUDA during ingestion.")

    # Prevent LlamaIndex from automatically using OpenAI
    Settings.llm = None

    return get_vector_store()

def get_all_repositories():
    """
    Returns a list of unique repository names currently ingested by looking at the clone directory.
    """
    from src.config import CLONE_DIR
    if not os.path.exists(CLONE_DIR):
        return []
    
    repos = []
    for item in os.listdir(CLONE_DIR):
        if os.path.isdir(os.path.join(CLONE_DIR, item)):
            repos.append(item)
    return sorted(repos)

def delete_repository(repo_name: str):
    """
    Deletes all chunks belonging to a specific repository.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = db.get_collection("repomind_codebase")
    except ValueError:
        return False
        
    collection.delete(where={"repo_name": repo_name})
    return True