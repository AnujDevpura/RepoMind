import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
from src.config import CHROMA_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME

load_dotenv()

def get_vector_store():
    """
    Initializes the ChromaDB client and sets up the storage context.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection("repomind_codebase")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def initialize_database():
    """
    Sets up the global LlamaIndex settings.
    This must be called at the start of the application.
    """
    print(f"🔄 Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        trust_remote_code=True
    )
    
    # 1. Apply Global Embedding Model
    Settings.embed_model = embed_model
    print("✅ Embedding Model Loaded.")

    # 2. Prevent LlamaIndex from automatically using OpenAI if we don't want it to
    Settings.llm = None 
        
    return get_vector_store()

def get_all_repositories():
    """
    Returns a list of unique repository names currently ingested.
    """
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = db.get_collection("repomind_codebase")
    except ValueError:
        return [] # Collection doesn't exist yet
    
    data = collection.get(include=["metadatas"])
    repos = set()
    if data and data.get("metadatas"):
        for meta in data["metadatas"]:
            if meta and "repo_name" in meta:
                repos.add(meta["repo_name"])
    return sorted(list(repos))

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