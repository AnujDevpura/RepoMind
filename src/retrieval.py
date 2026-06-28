import os
import argparse
from typing import List, Optional

from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.retrievers import VectorContextRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

from src.config import EMBEDDING_MODEL_NAME, TOP_K, RETRIEVE_LLM_PROVIDER, RETRIEVE_LLM_MODEL
from src.database import get_vector_store, get_graph_store
from src.llm import LLMEngine


class Retriever:
    """
    Manages the hybrid GraphRAG retrieval pipeline.

    Semantic Jump: locate EntityNodes via vector similarity search.
    Structural Blast Radius: traverse graph edges (path_depth=1) outward from those nodes
    to pull in their callers and callees, providing rich structural context.
    """

    def __init__(self, use_reranker: bool = False):
        vector_store = get_vector_store()
        graph_store = get_graph_store()

        # During retrieval, generation is handled by Groq (cloud) so VRAM is
        # completely free for bge-m3. Always use CUDA — never CPU.
        embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
            device="cuda",
            embed_batch_size=32,   # GPU is fast, higher batch is fine
        )
        Settings.embed_model = embed_model
        Settings.llm = None  # prevent accidental OpenAI fallback

        self._graph_store = graph_store
        self._vector_store = vector_store
        self._embed_model = embed_model

        self._index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store,
            vector_store=vector_store,
            embed_model=embed_model,
        )

        self._retriever = VectorContextRetriever(
            graph_store=graph_store,
            vector_store=vector_store,
            embed_model=embed_model,
            similarity_top_k=TOP_K,
            path_depth=1,      # traverse 1 edge out for blast radius
            include_text=True, # fetch original source chunks
        )

    def search(self, query: str, repo_name: Optional[str] = None) -> List[NodeWithScore]:
        """
        Run hybrid retrieval and optionally filter results by repository name.
        Returns an empty list if query is blank.
        """
        if not query or not query.strip():
            return []
        nodes = self._retriever.retrieve(query)
        if repo_name and repo_name != "All Repositories":
            nodes = [n for n in nodes if n.node.metadata.get("repo_name") == repo_name]
        return nodes


def main():
    parser = argparse.ArgumentParser(description="Query RepoMind GraphRAG (CLI)")
    parser.add_argument("query", type=str, help="The question to ask the codebase")
    args = parser.parse_args()

    print("🔧 Initializing retrieval pipeline...")
    retriever = Retriever()

    llm_engine = LLMEngine(provider=RETRIEVE_LLM_PROVIDER, model_name=RETRIEVE_LLM_MODEL)
    Settings.llm = llm_engine.llm

    print(f"\n{'='*60}")
    print("🚀 Running Hybrid Retrieval: Semantic Jump + Structural Blast Radius")
    print(f"{'='*60}\n")
    print(f"🗣️  Query: {args.query}\n")
    print("🤖 Thinking...")

    query_engine = retriever._index.as_query_engine(
        sub_retrievers=[retriever._retriever],
        llm=llm_engine.llm,
        similarity_top_k=TOP_K,
    )
    response = query_engine.query(args.query)

    print("\n🎯 Response:")
    print("-" * 60)
    print(response.response)
    print("-" * 60)

    print("\n🔍 Context Retrieved (The Blast Radius):")
    for i, node in enumerate(response.source_nodes):
        print(f"\n[Node {i+1}]")
        content = node.node.get_content()
        if len(content) > 400:
            content = content[:400] + "... [TRUNCATED]"
        print(f"Content:\n{content}")


if __name__ == "__main__":
    main()
