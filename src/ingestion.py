import os
import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import re
import git
import traceback
from typing import List, Optional
from urllib.parse import urlparse
from pathlib import Path

from llama_index.core import Document
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.readers.file import FlatReader

# Internal imports
from src.config import CLONE_DIR, GRAPH_PATH, INGEST_LLM_PROVIDER, INGEST_LLM_MODEL
from src.database import get_vector_store, get_graph_store, initialize_database
from src.ast_extractor import ASTPropertyGraphExtractor
from src.semantic_enricher import SemanticEnrichmentComponent
from src.llm import LLMEngine


def extract_repo_name(repo_url: str) -> str:
    """
    Extract repository name from various URL formats.
    """
    repo_url = repo_url.split('?')[0].split('#')[0]
    
    if repo_url.startswith('git@'):
        repo_name = repo_url.split(':')[-1]
    else:
        parsed = urlparse(repo_url)
        repo_name = parsed.path.strip('/')
    
    repo_name = repo_name.replace('.git', '')
    repo_name = repo_name.split('/')[-1]
    repo_name = re.sub(r'[<>:"|?*]', '_', repo_name)
    
    if not repo_name:
        raise ValueError(f"Could not extract repository name from URL: {repo_url}")
    
    return repo_name


def clone_repo(repo_url: str, force_clone: bool = False) -> str:
    """
    Clones a Git repository to a local directory.
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")
    
    repo_url = repo_url.strip()
    repo_name = extract_repo_name(repo_url)
    repo_path = os.path.join(CLONE_DIR, repo_name)
    repo_path = os.path.normpath(repo_path)
    
    # Check if repo already exists
    if os.path.exists(repo_path):
        if force_clone:
            import shutil
            print(f"🗑️ Removing existing clone at {repo_path}...")
            shutil.rmtree(repo_path)
        else:
            print(f"📂 Repo already exists at {repo_path}, skipping clone...")
            try:
                git.Repo(repo_path)
                return repo_path
            except git.exc.InvalidGitRepositoryError:
                print("⚠️ Existing directory is not a valid git repo, re-cloning...")
                import shutil
                shutil.rmtree(repo_path)
    
    print(f"⬇️ Cloning {repo_url}...")
    try:
        git.Repo.clone_from(repo_url, repo_path, depth=1)
        print(f"✅ Successfully cloned to {repo_path}")
    except git.exc.GitCommandError as e:
        raise git.exc.GitCommandError(f"Failed to clone repository: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during clone: {e}") from e
    
    return repo_path


def get_language_from_extension(ext: str) -> str:
    """
    Map file extension to language for tagging metadata.
    """
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".php": "php",
        ".rb": "ruby",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".R": "r",
        ".html": "html",
        ".css": "css",
    }
    return language_map.get(ext.lower(), "python")


def parse_code_files(repo_path: str, max_file_size_mb: float = 5.0) -> List[Document]:
    """
    Walks through the repo, reads supported code files, and creates documents.
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    repo_path_obj = Path(repo_path).resolve()
    documents = []
    
    supported_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", 
        ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", 
        ".kt", ".scala", ".r", ".R", ".md", ".txt", ".json", ".yaml", 
        ".yml", ".xml", ".html", ".css", ".scss", ".sass", ".sh", ".bash"
    }
    
    skip_files = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
        "poetry.lock", "Pipfile.lock", "go.sum", "Cargo.lock",
        "requirements.txt", "requirements-dev.txt",
        ".gitignore", ".gitattributes", ".editorconfig",
        "tsconfig.json", "jsconfig.json",
    }
    
    skip_dirs = {
        ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", 
        "venv", "env", ".env", "dist", "build", ".pytest_cache", 
        ".mypy_cache", ".idea", ".vscode", ".vs", "target", "bin", 
        "obj", ".gradle", ".next", ".nuxt", "vendor", "bower_components"
    }
    
    print(f"🔍 Scanning files in {repo_path}...")
    
    reader = FlatReader()
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    files_processed = 0
    files_skipped = 0
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            
            if file.startswith('.'): 
                continue
            if file in skip_files: 
                files_skipped += 1
                continue
            if file.endswith('.lock'): 
                files_skipped += 1
                continue
            if file_path.suffix not in supported_extensions: 
                continue
            
            try:
                file_size = file_path.stat().st_size
                if file_size > max_file_size_bytes:
                    print(f"⚠️ Skipping large file: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
                    files_skipped += 1
                    continue
            except OSError:
                continue
            
            try:
                docs = reader.load_data(file_path)
                for doc in docs:
                    try:
                        rel_path = str(file_path.relative_to(repo_path_obj))
                    except ValueError:
                        rel_path = str(file_path)
                    
                    language = get_language_from_extension(file_path.suffix)
                    
                    doc.metadata = {
                        "file_path": rel_path,
                        "file_name": file,
                        "file_extension": file_path.suffix,
                        "language": language,
                        "repo_path": str(repo_path_obj),
                        "repo_name": repo_path_obj.name,
                    }
                
                documents.extend(docs)
                files_processed += 1
                
            except UnicodeDecodeError:
                print(f"⚠️ Skipping binary/non-UTF8 file: {file_path.name}")
                files_skipped += 1
            except Exception as e:
                print(f"⚠️ Failed to read {file_path.name}: {e}")
                files_skipped += 1
    
    print(f"✅ Loaded {len(documents)} documents from {files_processed} code files.")
    if files_skipped > 0:
        print(f"⚠️ Skipped {files_skipped} files.")
    
    return documents


def ingest_repo(repo_url: str, force_clone: bool = False) -> Optional[PropertyGraphIndex]:
    """
    Complete ingestion pipeline: Clone -> Parse -> Embed -> Store.
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")
    
    try:
        # 1. Clone repository
        print(f"\n{'='*60}")
        print(f"🚀 Starting ingestion for: {repo_url}")
        print(f"{'='*60}\n")
        
        local_path = clone_repo(repo_url, force_clone=force_clone)
        
        # 2. Load and parse code files
        raw_documents = parse_code_files(local_path)
        
        if not raw_documents:
            raise ValueError("No code files found in repository.")
        
        print(f"\n📄 Processing {len(raw_documents)} documents...")
        
        # 3. Embed, Extract Graph & Index
        print("🌐 Deterministically extracting Knowledge Graph (Zero API Cost for Structure)...")
        vector_store = get_vector_store()
        graph_store = get_graph_store()
        
        print(f"🤖 Semantically enriching graph nodes (1-sentence summaries) via {INGEST_LLM_PROVIDER} ({INGEST_LLM_MODEL})...")
        llm_engine = LLMEngine(provider=INGEST_LLM_PROVIDER, model_name=INGEST_LLM_MODEL)
        
        kg_extractors = [
            ASTPropertyGraphExtractor(),
            SemanticEnrichmentComponent(
                llm=llm_engine.llm,
                # Only pass model name for local Ollama — triggers VRAM eviction +
                # CUDA upgrade for the embedding phase that runs immediately after.
                ollama_model=INGEST_LLM_MODEL if INGEST_LLM_PROVIDER == "ollama" else None,
            )
        ]
        
        from llama_index.core.ingestion import IngestionPipeline

        # Run extraction pipeline manually so it doesn't lock the mock embedder
        pipeline = IngestionPipeline(transformations=kg_extractors)
        nodes = pipeline.run(documents=raw_documents, show_progress=True)
        
        from llama_index.core import Settings
        
        index = PropertyGraphIndex(
            nodes=[], 
            property_graph_store=graph_store,
            vector_store=vector_store,
            embed_model=Settings.embed_model,
        )
        
        # Batch nodes manually to prevent ChromaDB from exceeding its max batch size of 5461
        # (Each node contains many kg_nodes extracted by AST/LLM).
        batch_size = 50
        print(f"\n📦 Inserting {len(nodes)} document nodes into index in batches of {batch_size}...")
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            index.insert_nodes(batch)
            print(f"   Inserted {min(i + batch_size, len(nodes))}/{len(nodes)} documents...")

        
        # Persist the GraphStore
        graph_store.persist(os.path.join(GRAPH_PATH, "graph_store.json"))
        
        print(f"\n{'='*60}")
        print("✅ GraphRAG Ingestion Complete! Graph stored to disk and vectors in ChromaDB.")
        print(f"   Repository: {Path(local_path).name}")
        print(f"   Documents: {len(raw_documents)}")
        print(f"{'='*60}\n")
        
        return index
        
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}\n")
        raise
    except git.exc.GitCommandError as e:
        print(f"\n❌ Git Error: {e}\n")
        raise RuntimeError(f"Failed to clone repository: {e}") from e
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}\n")
        traceback.print_exc()
        raise RuntimeError(f"Ingestion failed: {e}") from e

if __name__ == "__main__":
    # Test Run
    print("🔧 Initializing database...")
    initialize_database(load_embed_model=False)

    
    # Get repo URL from command line or use default test repo
    if len(sys.argv) > 1:
        test_repo = sys.argv[1]
    else:
        test_repo = "https://github.com/psf/requests"
    
    try:
        ingest_repo(test_repo, force_clone=False)
    except Exception as e:
        print(f"\n💥 Ingestion failed: {e}")
        sys.exit(1)