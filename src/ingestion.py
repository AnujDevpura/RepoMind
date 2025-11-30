import os
import re
import git
import sys
import traceback
from typing import List, Optional
from urllib.parse import urlparse
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser
from llama_index.readers.file import FlatReader

from tree_sitter_languages import get_parser
# Internal imports (assuming these exist in your project structure)
from src.config import CLONE_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.database import get_vector_store, initialize_database

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
        # Fixed: Removed the copy-paste error here that referenced SimpleNodeParser
        raise RuntimeError(f"Unexpected error during clone: {e}") from e
    
    return repo_path


def get_language_from_extension(ext: str) -> str:
    """
    Map file extension to language for CodeSplitter.
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


def chunk_documents_by_language(documents: List[Document]) -> List:
    """
    Chunk documents using language-specific CodeSplitter when possible.
    """
    
    # Language mapping 
    parser_lang_map = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "go": "go",
        "rust": "rust",
        "cpp": "cpp",
        "c": "c",
        "csharp": "c_sharp",
    }
    
    ast_supported_langs = set(parser_lang_map.keys())
    
    docs_by_lang = {}
    for doc in documents:
        lang = doc.metadata.get("language", "python")
        if lang not in docs_by_lang:
            docs_by_lang[lang] = []
        docs_by_lang[lang].append(doc)
    
    all_nodes = []
    
    for lang, lang_docs in docs_by_lang.items():
        lang_nodes = []
        
        # Filter out empty documents first
        valid_docs = []
        for doc in lang_docs:
            content = doc.get_content() if hasattr(doc, 'get_content') else (doc.text if hasattr(doc, 'text') else str(doc))
            if content and content.strip():
                valid_docs.append(doc)
            else:
                print(f"  ⚠️ Skipping empty document: {doc.metadata.get('file_name', 'unknown')}")
        
        if not valid_docs:
            print(f"  ⚠️ No valid documents for language: {lang}")
            continue
        
        try:
            # Try language-specific CodeSplitter with manual parser injection
            if lang in ast_supported_langs:
                # 1. Map 'python' -> 'python', 'csharp' -> 'c_sharp' etc.
                parser_lang = parser_lang_map.get(lang, lang)
                
                # 2. Get the parser using the library we just installed
                parser = get_parser(parser_lang)
                
                # 3. Manually pass it to CodeSplitter to bypass the ImportError
                splitter = CodeSplitter(
                    language=lang,
                    parser=parser,  # <--- CRITICAL INJECTION
                    chunk_lines=40,
                    chunk_lines_overlap=10,
                    max_chars=CHUNK_SIZE,
                )
                
                # Process documents one by one to catch individual failures
                for doc in valid_docs:
                    try:
                        doc_nodes = splitter.get_nodes_from_documents([doc])
                        lang_nodes.extend(doc_nodes)
                    except Exception as doc_error:
                        # If AST parsing fails for this specific doc, fall back to text splitting
                        print(f"  ⚠️ AST parsing failed for {doc.metadata.get('file_name', 'unknown')}, using text fallback for this file")
                        text_splitter = SimpleNodeParser.from_defaults(
                            chunk_size=1024,
                            chunk_overlap=200,
                        )
                        doc_nodes = text_splitter.get_nodes_from_documents([doc])
                        lang_nodes.extend(doc_nodes)
                
                print(f"  ✓ AST-aware chunked {len(valid_docs)} {lang} files into {len(lang_nodes)} nodes")
            else:
                raise ValueError(f"Language {lang} not supported for AST splitting")

        except Exception as e:
            print(f"  ⚠️ CodeSplitter failed for {lang}, using text fallback for all files.")
            print(f"     Error: {e}") 
            
            splitter = SimpleNodeParser.from_defaults(
                chunk_size=1024,
                chunk_overlap=200,
            )
            lang_nodes = splitter.get_nodes_from_documents(valid_docs)
            print(f"  ✓ Fallback: Text-chunked {len(valid_docs)} {lang} files into {len(lang_nodes)} nodes")
        
        all_nodes.extend(lang_nodes)
    
    return all_nodes

def ingest_repo(repo_url: str, force_clone: bool = False) -> Optional:
    """
    Complete ingestion pipeline: Clone -> Parse -> Chunk -> Embed -> Store.
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
        
        # 3. Chunk documents (language-aware)
        print("✂️ Chunking code files (AST-aware when possible)...")
        nodes = chunk_documents_by_language(raw_documents)
        
        if not nodes:
            raise ValueError("No chunks created from documents.")
        
        print(f"🧩 Created {len(nodes)} semantic chunks.\n")
        
        # 4. Embed & Index (ChromaDB)
        print("💾 Saving to Vector Database (this may take a while)...")
        vector_store = get_vector_store()
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create the index - this triggers embedding generation
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
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
        traceback.print_exc()
        raise RuntimeError(f"Ingestion failed: {e}") from e

if __name__ == "__main__":
    # Test Run
    print("🔧 Initializing database...")
    initialize_database()
    
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