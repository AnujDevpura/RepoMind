# RepoMind

> AI-powered code intelligence using GraphRAG — query any codebase in natural language.

## What it Does

RepoMind ingests Git repositories and builds a **Property Knowledge Graph** of the codebase using a two-tier pipeline:

1. **Deterministic AST Extraction** — Tree-sitter parses 15+ languages and maps every function definition and call relationship into a graph. Zero LLM cost, zero hallucination.
2. **Semantic Enrichment** — A local Ollama LLM reads each function's source code and writes a one-sentence summary. This summary (not the raw code) gets embedded into ChromaDB.

When you ask a question, the **Hybrid Retriever** performs a **Semantic Jump** (vector similarity on summaries) then explodes outward through graph edges (**Structural Blast Radius**) to pull in related callers and callees — giving the answer LLM rich, structurally coherent context.

---

## Architecture

```
GitHub URL
    │
    ▼
[ Git Clone ]
    │
    ▼
[ FlatReader ] — load 215+ code files
    │
    ▼
┌──────────────────────────────────────────┐
│         LlamaIndex Transform Pipeline     │
│                                          │
│  1. ASTPropertyGraphExtractor            │
│     Tree-sitter → EntityNodes + Relations│
│     (FUNCTION defs, CALLS edges)         │
│                                          │
│  2. SemanticEnrichmentComponent          │
│     Local Ollama LLM → "summary" prop    │
│     Async Semaphore(4) + cache + retries │
└──────────────────────────────────────────┘
    │                    │
    ▼                    ▼
[ ChromaDB ]        [ graph_store.json ]
  (vectors of          (full property
   summaries)           graph on disk)
    │                    │
    └────────┬───────────┘
             ▼
    [ Hybrid Retriever ]
      VectorContextRetriever
      path_depth=1 (blast radius)
             │
             ▼
    [ Groq / GPT-OSS-120B ]
      stream_chat with citations
             │
             ▼
    [ Streamlit UI ]
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally with a model pulled:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```
- A [Groq API key](https://console.groq.com/) for Q&A synthesis

### 2. Install

```bash
git clone https://github.com/AnujDevpura/RepoMind.git
cd RepoMind
uv sync
```

### 3. Configure

Create a `.env` file in the project root:

```env
# Required: for Q&A synthesis (retrieval LLM)
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (defaults shown)
EMBEDDING_MODEL_NAME=BAAI/bge-m3
INGEST_LLM_PROVIDER=ollama
INGEST_LLM_MODEL=qwen2.5-coder:7b
RETRIEVE_LLM_PROVIDER=groq
RETRIEVE_LLM_MODEL=llama-3.3-70b-versatile
TOP_K=5
USE_RERANKER=false
```

### 4. Ingest a Repository

```bash
uv run python -m src.ingestion https://github.com/GoogleCloudPlatform/microservices-demo.git
```

This will clone the repo, extract the graph, generate LLM summaries, and embed them into ChromaDB. Ingestion is idempotent — restart safely if it crashes.

### 5. Launch the UI

```bash
uv run streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

### 6. CLI Query (Optional)

```bash
uv run python -m src.retrieval "How does the checkout service handle payment?"
```

### 7. Evaluation Pipeline

To evaluate the system using the Code-Aware LLM-as-a-Judge mechanism:

1. Generate a synthetic dataset (or use an existing `eval.json`):
```bash
uv run python -m src.evaluation.synthetic_data --repo microservices-demo --output eval_microservices.json
```
2. Run the 4-Pillar RAG Evaluator:
```bash
uv run python -m src.evaluation.run_eval eval_microservices.json --repo microservices-demo --output scorecard.json
```

---

## Project Structure

```
RepoMind/
├── README.md
├── requirements.txt
├── .env                          # API keys (create this)
├── .gitignore
├── data/
│   ├── cloned_repos/             # Git clones
│   ├── chromadb/                 # Vector embeddings (ChromaDB)
│   ├── graphdb/                  # Property graph (graph_store.json)
│   └── semantic_cache.json       # LLM summary cache (crash-safe)
├── scripts/
│   ├── reset_index.py            # Utility: wipe ChromaDB collection
│   └── repair_graph.py           # Utility: remove dangling triplets
└── src/
    ├── __init__.py
    ├── app.py                    # Streamlit UI
    ├── config.py                 # All configuration and env vars
    ├── database.py               # ChromaDB + graph store setup
    ├── ingestion.py              # Full ingestion pipeline entry point
    ├── ast_extractor.py          # Tree-sitter graph extractor (15+ langs)
    ├── semantic_enricher.py      # LLM enrichment with async + caching
    ├── retrieval.py              # Retriever class + CLI entry point
    ├── llm.py                    # LLMEngine (Groq / OpenAI / Ollama)
    └── evaluation/               # Comprehensive RAG Evaluator
        ├── synthetic_data.py     # Synthetic Code Q&A Generator
        ├── run_eval.py           # 4-Pillar single-shot judge evaluator
        ├── metrics.py            # Custom LlamaIndex Evaluation Class
        └── groq_client.py        # Robust client bypassing rate-limits
```

---

## Configuration Reference

All settings live in `src/config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | HuggingFace embedding model |
| `INGEST_LLM_PROVIDER` | `ollama` | LLM provider for summarization |
| `INGEST_LLM_MODEL` | `qwen2.5-coder:7b` | Model for summarization |
| `RETRIEVE_LLM_PROVIDER` | `groq` | LLM provider for Q&A |
| `RETRIEVE_LLM_MODEL` | `llama-3.3-70b-versatile` | Model for Q&A synthesis |
| `TOP_K` | `5` | Retrieved nodes per query |
| `USE_RERANKER` | `false` | Enable cross-encoder reranking |

---

## Hardware Notes

- **GPU (RTX 3050 4GB)**: Ollama LLM runs on GPU (~2.4GB VRAM). The embedding model (`bge-m3`) runs on CPU since there is no VRAM budget left.
- **Async throttling**: The Semantic Enrichment component uses `asyncio.Semaphore(4)` to avoid saturating the local GPU.
- **Crash safety**: LLM summaries are written to `data/semantic_cache.json` after every successful call. Restarting ingestion on a partially-complete run will skip already-summarized functions.

---

## Supported Languages

Tree-sitter extracts function graphs for: Python, JavaScript, TypeScript, Go, Java, C, C++, Rust, C#, Ruby, Swift, Kotlin, Scala, HTML, CSS.

---

**Made by Anuj Devpura**