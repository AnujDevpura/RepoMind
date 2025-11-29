import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from src.config import CHROMA_PATH, EMBEDDING_MODEL_NAME

def get_vector_store():
    """
    Initializes the ChromaDB client and sets up the storage context.
    """
    # 1. Initialize the ChromaDB client (Persistent)
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # 2. Create (or get) the specific collection for our code
    chroma_collection = db.get_or_create_collection("repomind_codebase")

    # 3. Create the LlamaIndex wrapper for Chroma
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return vector_store

def initialize_database():
    """
    Sets up the global LlamaIndex settings (Embedding model).
    This must be called at the start of the application.
    """
    print(f"🔄 Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    
    # Initialize the local HuggingFace embedding model
    # trust_remote_code=True is often needed for newer architectures like BGE
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        trust_remote_code=True
    )
    
    # Apply to Global Settings
    Settings.embed_model = embed_model
    print("✅ Embedding Model Loaded.")

    return get_vector_store()