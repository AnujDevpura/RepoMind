from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from src.config import CHROMA_PATH, TOP_K, RERANK_TOP_K, RERANK_MODEL
from src.database import get_vector_store

class Retriever:
    def __init__(self, use_reranker: bool = True):
        self.use_reranker = use_reranker
        self._index = self._load_index()
        
        # Initialize Reranker (Cross-Encoder)
        # Uses RERANK_MODEL from config (BAAI/bge-reranker-base by default)
        if self.use_reranker:
            print(f"🚀 Initializing Reranker: {RERANK_MODEL}...")
            self.reranker = SentenceTransformerRerank(
                model=RERANK_MODEL,
                top_n=RERANK_TOP_K  # Use config constant
            )

    def _load_index(self):
        """
        Loads the existing ChromaDB index from disk.
        If no index metadata exists, creates a new index from the existing vector store.
        This allows retrieval even if the index structure wasn't persisted.
        """
        print(f"📂 Loading Index from {CHROMA_PATH}...")
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        try:
            # Try to load existing index structure
            index = load_index_from_storage(
                storage_context=storage_context,
                store_nodes_override=True  # Needed for some vector stores to ensure node data is loaded
            )
            print("✅ Index loaded successfully")
            return index
        except (ValueError, KeyError):
            # No index metadata exists, but vector store might have data
            # Create a new index from the existing vector store
            print("⚠️ No index metadata found, creating index from vector store...")
            try:
                # Check if vector store has any data
                # Create index from existing vector store (nodes will be loaded from vector store)
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
                print("✅ Index created from existing vector store")
                return index
            except Exception as create_error:
                # Vector store is likely empty
                raise ValueError(
                    f"No index found and vector store appears to be empty ({create_error}). "
                    "Please ingest at least one repository first using: "
                    "python -m src.ingestion <repo_url>"
                ) from create_error
        except Exception as e:
            # Other unexpected errors
            print(f"⚠️ Error loading index: {e}")
            raise ValueError(
                f"Failed to load index: {e}. "
                "Make sure you've ingested at least one repository first."
            ) from e

    def search(self, query_text: str, top_k: int = None, rerank: bool = None):
        """
        Performs the 2-stage retrieval:
        1. Vector Search (Get top K candidates)
        2. Reranking (Filter to top RERANK_TOP_K)
        
        Args:
            query_text: Search query
            top_k: Number of initial candidates to retrieve (default: TOP_K from config)
            rerank: Whether to rerank results (default: self.use_reranker)
        
        Returns:
            List of nodes (reranked if rerank=True)
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if top_k is None:
            top_k = TOP_K
        if rerank is None:
            rerank = self.use_reranker
        
        print(f"🔍 Searching for: '{query_text}'")
        
        # 1. Vector Search (Retrieval)
        # Fetch more candidates if reranking (to give reranker options)
        # Otherwise use the requested top_k
        initial_k = top_k * 2 if rerank else top_k
        
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=initial_k,
        )
        
        # 2. Retrieve nodes
        nodes = retriever.retrieve(query_text)
        print(f"   📊 Found {len(nodes)} vector matches...")
        
        # 3. Rerank (Refinement)
        if rerank and len(nodes) > 0:
            print(f"   ✨ Reranking to top {RERANK_TOP_K} results...")
            nodes = self.reranker.postprocess_nodes(
                nodes, 
                query_str=query_text
            )
            # Reranker updates node.score with rerank scores
            # These are relevance scores (higher = better)
            print(f"   ✅ Reranked to {len(nodes)} results")
        elif not rerank and len(nodes) > top_k:
            # Limit to top_k if not reranking
            nodes = nodes[:top_k]
            
        return nodes

if __name__ == "__main__":
    # Test the Retrieval Engine
    import sys
    from src.database import initialize_database
    
    # Initialize database and embedding model
    print("🔧 Initializing database...")
    initialize_database()
    
    try:
        engine = Retriever(use_reranker=True)
        
        # Get query from command line or use default
        if len(sys.argv) > 1:
            test_query = " ".join(sys.argv[1:])
        else:
            test_query = "How does GitHub webhook authentication work?"
        
        print(f"\n{'='*60}")
        results = engine.search(test_query)
        print(f"{'='*60}\n")
        
        if not results:
            print("❌ No results found. Make sure you've ingested at least one repository.")
        else:
            print(f"🏆 Top {len(results)} Results:\n")
            for i, node in enumerate(results, 1):
                # Get reranker score (updated by SentenceTransformerRerank)
                # Score is relevance score: higher = more relevant
                score = getattr(node, 'score', None)
                if score is not None:
                    # Format score nicely (reranker scores are typically 0-1 or higher)
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                else:
                    score_str = "N/A"
                
                metadata = getattr(node, 'metadata', {})
                file_path = metadata.get('file_path', 'Unknown')
                content = node.get_content()[:300] if hasattr(node, 'get_content') else str(node)[:300]
                
                print(f"{'─'*60}")
                print(f"Result {i} (Relevance Score: {score_str})")
                print(f"File: {file_path}")
                print(f"Content:\n{content}...")
                print()
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)