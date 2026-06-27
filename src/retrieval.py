import sys
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from src.config import CHROMA_PATH, TOP_K, RERANK_TOP_K, RERANK_MODEL
from src.database import get_vector_store

class Retriever:
    def __init__(self, use_reranker: bool = True):
        self.use_reranker = use_reranker
        self._index = self._load_index()
        
        # 1. Initialize Reranker
        if self.use_reranker:
            print(f"🚀 Initializing Reranker: {RERANK_MODEL}...")
            self.reranker = SentenceTransformerRerank(
                model=RERANK_MODEL,
                top_n=RERANK_TOP_K
            )

    def _load_index(self):
        """
        Loads the existing ChromaDB index from disk.
        If no index metadata exists, creates a new index from the existing vector store.
        """
        print(f"📂 Loading Index from {CHROMA_PATH}...")
        vector_store = get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        try:
            index = load_index_from_storage(
                storage_context=storage_context,
                store_nodes_override=True 
            )
            print("✅ Index loaded successfully")
            return index
        except (ValueError, KeyError):
            print("⚠️ No index metadata found, creating index from vector store...")
            try:
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
                print("✅ Index created from existing vector store")
                return index
            except Exception as create_error:
                raise ValueError(
                    f"No index found and vector store appears to be empty ({create_error}). "
                    "Please ingest at least one repository first."
                ) from create_error
        except Exception as e:
            raise ValueError(f"Failed to load index: {e}") from e

    def search(self, query_text: str, repo_name: str = None, top_k: int = None, rerank: bool = None):
        """
        Performs retrieval (Vector Search -> Cross-Encoder Reranking)
        with optional repository filtering.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        target_rerank = rerank if rerank is not None else self.use_reranker
        k = top_k or (TOP_K * 2 if target_rerank else TOP_K)
        
        print(f"🔍 Executing Search for: '{query_text}'" + (f" in '{repo_name}'" if repo_name else ""))
        
        # Apply repo filter if provided
        filters = None
        if repo_name and repo_name != "All Repositories":
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="repo_name", value=repo_name)]
            )
            
        vector_retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=k,
            filters=filters
        )
        
        # 1. Retrieval Stage (Vector Search)
        nodes = vector_retriever.retrieve(query_text)
        print(f"   📊 Found {len(nodes)} candidates...")
        
        # 2. Reranking Stage
        if target_rerank and len(nodes) > 0:
            print(f"   ✨ Reranking to top {RERANK_TOP_K} results...")
            nodes = self.reranker.postprocess_nodes(nodes, query_str=query_text)
            print(f"   ✅ Reranked to {len(nodes)} results")
            
        return nodes

if __name__ == "__main__":
    # Test the Retrieval Engine
    from src.database import initialize_database
    
    print("🔧 Initializing database...")
    initialize_database()
    
    try:
        engine = Retriever(use_reranker=True)
        
        if len(sys.argv) > 1:
            test_query = " ".join(sys.argv[1:])
        else:
            test_query = "Where is the main training loop defined?"
        
        print(f"\n{'='*60}")
        results = engine.search(test_query)
        print(f"{'='*60}\n")
        
        if not results:
            print("❌ No results found. Make sure you've ingested at least one repository.")
        else:
            print(f"🏆 Top {len(results)} Results:\n")
            for i, node in enumerate(results, 1):
                score = getattr(node, 'score', None)
                score_str = f"{score:.4f}" if score is not None else "N/A"
                
                metadata = getattr(node, 'metadata', {})
                file_path = metadata.get('file_path', 'Unknown')
                content = node.get_content()[:300] if hasattr(node, 'get_content') else str(node)[:300]
                
                print(f"{'─'*60}")
                print(f"Result {i} (Relevance Score: {score_str})")
                print(f"File: {file_path}")
                print(f"Content:\n{content}...\n")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
