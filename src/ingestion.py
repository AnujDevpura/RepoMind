import os
import re
import git
from typing import List, Optional
from urllib.parse import urlparse
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter
from llama_index.readers.file import FlatReader
from pathlib import Path
from src.config import CLONE_DIR
from src.database import get_vector_store
from tree_sitter_language_pack import get_parser


def extract_repo_name(repo_url: str) -> str:
    """
    Extract repository name from various URL formats.
    Handles: https://github.com/user/repo.git, https://github.com/user/repo, git@github.com:user/repo.git, etc.
    """
    # Remove query parameters and fragments
    repo_url = repo_url.split('?')[0].split('#')[0]
    
    # Handle SSH URLs (git@github.com:user/repo.git)
    if repo_url.startswith('git@'):
        repo_name = repo_url.split(':')[-1]
    else:
        # Handle HTTPS/HTTP URLs
        parsed = urlparse(repo_url)
        repo_name = parsed.path.strip('/')
    
    # Remove .git extension if present
    repo_name = repo_name.replace('.git', '')
    
    # Extract just the repo name (last part after /)
    repo_name = repo_name.split('/')[-1]
    
    # Sanitize the name (remove invalid characters for file paths)
    repo_name = re.sub(r'[<>:"|?*]', '_', repo_name)
    
    if not repo_name:
        raise ValueError(f"Could not extract repository name from URL: {repo_url}")
    
    return repo_name


def clone_repo(repo_url: str, force_clone: bool = False) -> str:
    """
    Clones a Git repository to a local directory.
    
    Args:
        repo_url: URL of the Git repository (HTTPS, HTTP, or SSH)
        force_clone: If True, remove existing clone and re-clone
    
    Returns:
        Path to the cloned repository
    
    Raises:
        ValueError: If URL is invalid or repository name cannot be extracted
        git.exc.GitCommandError: If cloning fails
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("Repository URL cannot be empty")
    
    repo_url = repo_url.strip()
    repo_name = extract_repo_name(repo_url)
    repo_path = os.path.join(CLONE_DIR, repo_name)
    repo_path = os.path.normpath(repo_path)  # Normalize path
    
    # Check if repo already exists
    if os.path.exists(repo_path):
        if force_clone:
            import shutil
            print(f"🗑️ Removing existing clone at {repo_path}...")
            shutil.rmtree(repo_path)
        else:
            print(f"📂 Repo already exists at {repo_path}, skipping clone...")
            # Verify it's a valid git repo
            try:
                git.Repo(repo_path)
                return repo_path
            except git.exc.InvalidGitRepositoryError:
                print("⚠️ Existing directory is not a valid git repo, re-cloning...")
                import shutil
                shutil.rmtree(repo_path)
    
    print(f"⬇️ Cloning {repo_url}...")
    try:
        git.Repo.clone_from(repo_url, repo_path, depth=1)  # Shallow clone for speed
        print(f"✅ Successfully cloned to {repo_path}")
    except git.exc.GitCommandError as e:
        raise git.exc.GitCommandError(f"Failed to clone repository: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during clone: {e}") from e
    
    return repo_path

def get_language_from_extension(ext: str) -> str:
    """
    Map file extension to language for CodeSplitter.
    Returns language code or 'python' as fallback.
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
    }
    return language_map.get(ext.lower(), "python")


def parse_code_files(repo_path: str, max_file_size_mb: float = 5.0) -> List[Document]:
    """
    Walks through the repo, reads supported code files, and creates documents.
    We verify file extensions to avoid ingesting images or binaries.
    
    Args:
        repo_path: Path to the repository root
        max_file_size_mb: Maximum file size in MB to process (default: 5MB)
    
    Returns:
        List of Document objects with metadata
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    repo_path_obj = Path(repo_path).resolve()
    documents = []
    
    # Extended list of supported extensions
    supported_extensions = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", 
        ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", 
        ".kt", ".scala", ".r", ".R", ".md", ".txt", ".json", ".yaml", 
        ".yml", ".xml", ".html", ".css", ".scss", ".sass", ".sh", ".bash"
    }
    
    # Files to skip (generated/lock files that aren't useful for code search)
    skip_files = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
        "poetry.lock", "Pipfile.lock", "go.sum", "Cargo.lock",
        "requirements.txt", "requirements-dev.txt",  # Usually just dependency lists
        ".gitignore", ".gitattributes", ".editorconfig",
        "tsconfig.json", "jsconfig.json",  # Config files, not code
    }
    
    # Directories to skip
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
        # Skip hidden and ignored directories
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip hidden files
            if file.startswith('.'):
                continue
            
            # Skip known generated/lock files
            if file in skip_files:
                files_skipped += 1
                continue
            
            # Skip lock files (any file ending in .lock)
            if file.endswith('.lock'):
                files_skipped += 1
                continue
            
            # Check extension
            if file_path.suffix not in supported_extensions:
                continue
            
            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_file_size_bytes:
                    print(f"⚠️ Skipping large file: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
                    files_skipped += 1
                    continue
            except OSError:
                continue
            
            try:
                # Load the file content
                docs = reader.load_data(file_path)
                
                # Attach metadata to each document
                for doc in docs:
                    # Get relative path from repo root
                    try:
                        rel_path = str(file_path.relative_to(repo_path_obj))
                    except ValueError:
                        # Fallback to absolute path if relative fails
                        rel_path = str(file_path)
                    
                    # Determine language from extension
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
        print(f"⚠️ Skipped {files_skipped} files (too large, binary, or errors).")
    
    return documents

def chunk_documents_by_language(documents: List[Document]) -> List:
    """
    Chunk documents using language-specific CodeSplitter when possible.
    Groups documents by language and chunks them appropriately.
    
    CRITICAL: Manually injects parsers using get_parser() due to llama-index bug
    where CodeSplitter fails to auto-load tree-sitter parsers.
    
    Args:
        documents: List of Document objects with language metadata
    
    Returns:
        List of chunked nodes
    """
    from llama_index.core.node_parser import SimpleNodeParser
    
    # Language mapping for tree-sitter (must match tree_sitter_language_pack names)
    # Some languages need special mapping
    parser_lang_map = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "cpp": "cpp",
        "c": "c",
        "csharp": "c_sharp",  # tree-sitter uses c_sharp
    }
    
    # Languages that support AST-aware CodeSplitter
    ast_supported_langs = set(parser_lang_map.keys())
    
    # Group documents by language
    docs_by_lang = {}
    for doc in documents:
        lang = doc.metadata.get("language", "python")
        if lang not in docs_by_lang:
            docs_by_lang[lang] = []
        docs_by_lang[lang].append(doc)
    
    all_nodes = []
    
    # Process each language group
    for lang, lang_docs in docs_by_lang.items():
        try:
            # Try language-specific CodeSplitter with manual parser injection
            if lang in ast_supported_langs:
                # Get the tree-sitter parser name
                parser_lang = parser_lang_map.get(lang, lang)
                
                # CRITICAL: Manually inject parser to work around llama-index bug
                parser = get_parser(parser_lang)
                
                splitter = CodeSplitter(
                    language=lang,
                    parser=parser,  # Manual injection!
                    chunk_lines=40,
                    chunk_lines_overlap=10,
                    max_chars=3000,
                )
                nodes = splitter.get_nodes_from_documents(lang_docs)
                
                # Verify we got AST-aware chunks (not just text splits)
                # AST chunks should have more structure-aware boundaries
                print(f"  ✓ AST-aware chunked {len(lang_docs)} {lang} files into {len(nodes)} nodes")
            else:
                # Fallback to SimpleNodeParser for unsupported languages
                splitter = SimpleNodeParser.from_defaults(
                    chunk_size=1024,
                    chunk_overlap=200,
                )
                nodes = splitter.get_nodes_from_documents(lang_docs)
                print(f"  ✓ Text-chunked {len(lang_docs)} {lang} files into {len(nodes)} nodes (no AST support)")
            
            all_nodes.extend(nodes)
            
        except Exception as e:
            # Fallback to SimpleNodeParser if CodeSplitter fails
            print(f"  ⚠️ CodeSplitter failed for {lang}, using text fallback: {e}")
            splitter = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=200,
            )
            nodes = splitter.get_nodes_from_documents(lang_docs)
            all_nodes.extend(nodes)
            print(f"  ✓ Fallback: Text-chunked {len(lang_docs)} {lang} files into {len(nodes)} nodes")
    
    return all_nodes


def ingest_repo(repo_url: str, force_clone: bool = False) -> Optional:
    """
    Complete ingestion pipeline: Clone -> Parse -> Chunk -> Embed -> Store.
    
    Args:
        repo_url: URL of the Git repository
        force_clone: If True, remove existing clone and re-clone
    
    Returns:
        VectorStoreIndex object or None if ingestion fails
    
    Raises:
        ValueError: If repository URL is invalid or no code files found
        RuntimeError: If ingestion fails at any stage
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
            raise ValueError(
                "No code files found in repository. "
                "Make sure the repository contains supported code files."
            )
        
        print(f"\n📄 Processing {len(raw_documents)} documents...")
        
        # 3. Chunk documents (language-aware)
        print("✂️ Chunking code files (AST-aware when possible)...")
        nodes = chunk_documents_by_language(raw_documents)
        
        if not nodes:
            raise ValueError("No chunks created from documents. Check file content.")
        
        print(f"🧩 Created {len(nodes)} semantic chunks.\n")
        
        # 4. Embed & Index (ChromaDB)
        print("💾 Saving to Vector Database (this may take a while)...")
        vector_store = get_vector_store()
        
        from llama_index.core import VectorStoreIndex, StorageContext
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index - this triggers embedding generation
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            # Settings.embed_model should be set in database.py via initialize_database()
        )
        
        print(f"\n{'='*60}")
        print("✅ Ingestion Complete! Embeddings stored in ChromaDB.")
        print(f"   Repository: {Path(local_path).name}")
        print(f"   Documents: {len(raw_documents)}")
        print(f"   Chunks: {len(nodes)}")
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
        raise RuntimeError(f"Ingestion failed: {e}") from e

if __name__ == "__main__":
    # Test Run
    import sys
    
    from src.database import initialize_database
    
    # Initialize database and embedding model
    print("🔧 Initializing database...")
    initialize_database()
    
    # Get repo URL from command line or use default test repo
    if len(sys.argv) > 1:
        test_repo = sys.argv[1]
    else:
        # Default test repo (small repository for testing)
        test_repo = "https://github.com/CoderAgent/SecureAgent"
    
    try:
        ingest_repo(test_repo, force_clone=False)
    except Exception as e:
        print(f"\n💥 Ingestion failed: {e}")
        sys.exit(1)