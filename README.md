# RepoMind

> A powerful repository code search and retrieval system that lets you query and understand code using natural language.

## Overview

RepoMind transforms how you explore and understand codebases. By combining advanced embedding models with large language models, it enables you to search through repositories using natural language queries and get contextually relevant answers about code structure, functionality, and implementation details.

## Features

- **🔍 Intelligent Code Search** - Find specific code snippets, functions, or patterns across entire repositories
- **🧠 Contextual Understanding** - Leverage LLMs to understand code semantics, not just syntax
- **📦 Easy Repository Ingestion** - Index any git repository with a single command
- **🎯 Relevance Reranking** - Advanced reranking ensures the most relevant results surface first
- **💻 User-Friendly Interface** - Clean Gradio-based UI for seamless interaction
- **⚡ Fast Vector Search** - ChromaDB-powered vector store for lightning-fast retrieval

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AnujDevpura/RepoMind.git
cd RepoMind
```

### 2. Install Dependencies with uv

```bash
# uv automatically creates a virtual environment and installs dependencies
uv pip install -r requirements.txt
```


### 3. Configure API Keys

Create a `.env` file in the root directory:

```bash
# For Groq (recommended for fast inference)
GROQ_API_KEY=your_groq_api_key_here

# For OpenAI (alternative)
OPENAI_API_KEY=your_openai_api_key_here

# For HuggingFace Models
HF_TOKEN=your_hf_token_here

# Ollama runs locally, no API key needed
```

## Quick Start

### Launch the Application

```bash

# Activate the environment first
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate    # On Windows

# Run the UI
python -m src.app
```

The Gradio interface will launch and provide a local URL (typically `http://127.0.0.1:7860`).

### Ingest a Repository

1. Open the Gradio interface in your browser
2. Enter a git repository URL (e.g., `https://github.com/username/repo`)
3. Click "Ingest Repository" and wait for indexing to complete
4. Start querying your code!

### Query Examples

Try questions like:
- "How is authentication implemented?"
- "Show me all the API endpoints"
- "Where is error handling done?"
- "Explain the database schema"
- "Find functions that handle file uploads"

## Configuration

Configuration options are available in `src/config.py`:

| Option | Description | Default |
|--------|-------------|---------|
| `PROJECT_ROOT` | Root directory of the project | Auto-detected |
| `DATA_DIR` | Storage for repositories and databases | `./data` |
| `CLONE_DIR` | Cloned repositories location | `./data/cloned_repos` |
| `CHROMA_PATH` | ChromaDB database path | `./data/chroma_db` |

### Model Configuration

You can customize the embedding and LLM models in `src/config.py`:

```python
# Embedding model (HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM provider: "groq", "ollama", or "openai"
LLM_PROVIDER = "groq"
```

## Project Structure

```
RepoMind/
├── README.md
├── requirements.txt
├── .env                      # API keys (create this)
├── .gitignore
├── evaluation.ipynb         # Performance evaluation
├── assets/
│   └── avatar.png
├── data/
│   ├── cloned_repos/        # Cloned repositories
│   ├── chroma_db/           # Vector database
│   ├── tests.jsonl          # Test queries     
└── src/
    ├── __init__.py
    ├── app.py               # Main application entry
    ├── config.py            # Configuration settings
    ├── database.py          # ChromaDB interface
    ├── ingestion.py         # Repository processing
    ├── llm.py              # LLM integrations
    └── retrieval.py         # Search and reranking
```

## Dependencies

RepoMind builds on these excellent open-source projects:

- **LlamaIndex** - Data framework for LLM applications
- **ChromaDB** - Vector database for embeddings
- **Sentence Transformers** - State-of-the-art embeddings
- **Gradio** - Fast UI for ML applications
- **Tree-sitter** - Code parsing and analysis
- **Groq/Ollama** - Fast LLM inference

For the complete list, see `requirements.txt`.

**Made by Anuj Devpura**

*Have questions or feedback? Open an issue or start a discussion!*