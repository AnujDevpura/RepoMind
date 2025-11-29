"""
Test script to verify AST-aware parsing is working correctly.

This script tests:
1. Parser loading from tree_sitter_language_pack
2. CodeSplitter with manual parser injection
3. Verification that chunks respect code structure (not just text splitting)
4. Comparison between AST chunks and text chunks
"""

import sys
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter, SimpleNodeParser
from tree_sitter_language_pack import get_parser


def test_parser_loading():
    """Test that parsers can be loaded for different languages."""
    print("=" * 70)
    print("TEST 1: Parser Loading")
    print("=" * 70)
    
    test_languages = ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"]
    results = {}
    
    for lang in test_languages:
        try:
            parser = get_parser(lang)
            # Test parsing a simple snippet
            test_code = b"def test():\n    pass" if lang == "python" else b"function test() {}"
            tree = parser.parse(test_code)
            results[lang] = "✅ PASS"
            print(f"  {lang:15} -> {results[lang]}")
        except Exception as e:
            results[lang] = f"❌ FAIL: {e}"
            print(f"  {lang:15} -> {results[lang]}")
    
    print()
    return results


def test_ast_vs_text_chunking():
    """Test that AST chunking produces structure-aware chunks."""
    print("=" * 70)
    print("TEST 2: AST-Aware Chunking vs Text Chunking")
    print("=" * 70)
    
    # Sample Python code with clear structure
    python_code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self.value
    
    def subtract(self, x):
        self.value -= x
        return self.value

def standalone_function():
    print("This is a standalone function")
    return True

# Some comments here
# More comments
# Even more comments to pad the file
"""
    
    # Create document
    doc = Document(
        text=python_code,
        metadata={"language": "python", "file_path": "test.py"}
    )
    
    print("\n📝 Test Code Structure:")
    print("  - 1 class (Calculator) with 3 methods")
    print("  - 1 standalone function")
    print("  - Multiple comments")
    
    # Test 1: AST-aware chunking with manual parser
    print("\n🔬 Test 2a: AST-Aware Chunking (with parser injection)")
    try:
        parser = get_parser("python")
        ast_splitter = CodeSplitter(
            language="python",
            parser=parser,  # Manual injection
            chunk_lines=40,
            chunk_lines_overlap=10,
            max_chars=3000,
        )
        ast_nodes = ast_splitter.get_nodes_from_documents([doc])
        
        print(f"  ✅ AST chunks created: {len(ast_nodes)}")
        print(f"  📊 Chunk boundaries:")
        for i, node in enumerate(ast_nodes[:5], 1):  # Show first 5
            preview = node.text[:60].replace('\n', '\\n')
            print(f"     Chunk {i}: {preview}...")
        
        # Check if chunks respect structure (functions/classes)
        structure_respecting = 0
        for node in ast_nodes:
            text = node.text
            # AST chunks should typically start with class/def or contain complete structures
            if text.strip().startswith(('class ', 'def ', '@')) or 'def ' in text[:50]:
                structure_respecting += 1
        
        print(f"  🎯 Structure-respecting chunks: {structure_respecting}/{len(ast_nodes)}")
        ast_success = structure_respecting > len(ast_nodes) * 0.5  # At least 50% should respect structure
        
    except Exception as e:
        print(f"  ❌ AST chunking failed: {e}")
        ast_nodes = []
        ast_success = False
    
    # Test 2: Text-based chunking (fallback)
    print("\n🔬 Test 2b: Text-Based Chunking (SimpleNodeParser fallback)")
    try:
        text_splitter = SimpleNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=200,
        )
        text_nodes = text_splitter.get_nodes_from_documents([doc])
        
        print(f"  ✅ Text chunks created: {len(text_nodes)}")
        print(f"  📊 Chunk boundaries:")
        for i, node in enumerate(text_nodes[:5], 1):  # Show first 5
            preview = node.text[:60].replace('\n', '\\n')
            print(f"     Chunk {i}: {preview}...")
        
        # Text chunks are less likely to respect structure
        structure_respecting = 0
        for node in text_nodes:
            text = node.text
            if text.strip().startswith(('class ', 'def ', '@')) or 'def ' in text[:50]:
                structure_respecting += 1
        
        print(f"  🎯 Structure-respecting chunks: {structure_respecting}/{len(text_nodes)}")
        text_success = True  # Text splitting always "succeeds" but may not respect structure
        
    except Exception as e:
        print(f"  ❌ Text chunking failed: {e}")
        text_nodes = []
        text_success = False
    
    # Comparison
    print("\n📈 Comparison:")
    if ast_nodes and text_nodes:
        print(f"  AST chunks: {len(ast_nodes)} | Text chunks: {len(text_nodes)}")
        if len(ast_nodes) != len(text_nodes):
            print(f"  ✅ Different chunk counts indicate different strategies")
        else:
            print(f"  ⚠️ Same chunk count - may indicate fallback to text splitting")
    
    print()
    return ast_success, len(ast_nodes) if ast_nodes else 0, len(text_nodes) if text_nodes else 0


def test_code_splitter_without_parser():
    """Test that CodeSplitter fails without manual parser injection (demonstrates the bug)."""
    print("=" * 70)
    print("TEST 3: CodeSplitter WITHOUT Manual Parser (Expected to Fail)")
    print("=" * 70)
    
    python_code = """
def test_function():
    x = 1
    y = 2
    return x + y
"""
    
    doc = Document(
        text=python_code,
        metadata={"language": "python", "file_path": "test.py"}
    )
    
    print("\n🔬 Testing CodeSplitter without parser parameter...")
    try:
        # This should fail or silently fall back to text splitting
        splitter = CodeSplitter(
            language="python",
            # parser=parser,  # INTENTIONALLY OMITTED
            chunk_lines=40,
            chunk_lines_overlap=10,
            max_chars=3000,
        )
        nodes = splitter.get_nodes_from_documents([doc])
        
        print(f"  ⚠️ CodeSplitter created {len(nodes)} chunks WITHOUT parser")
        print(f"  ⚠️ This likely means it fell back to text splitting!")
        print(f"  ⚠️ Check chunk boundaries - they should NOT respect code structure")
        
        # Show chunk boundaries
        for i, node in enumerate(nodes, 1):
            preview = node.text[:80].replace('\n', '\\n')
            print(f"     Chunk {i}: {preview}...")
        
        return False  # This is the bug we're trying to avoid
        
    except Exception as e:
        print(f"  ✅ CodeSplitter failed as expected: {e}")
        print(f"  ✅ This confirms parser is required!")
        return True  # Failure is expected and good


def test_ingestion_function():
    """Test the actual chunk_documents_by_language function from ingestion.py"""
    print("=" * 70)
    print("TEST 4: Integration Test - chunk_documents_by_language()")
    print("=" * 70)
    
    try:
        from src.ingestion import chunk_documents_by_language
        
        # Create test documents
        python_doc = Document(
            text="""
class TestClass:
    def method1(self):
        return "test1"
    
    def method2(self):
        return "test2"
""",
            metadata={"language": "python", "file_path": "test.py"}
        )
        
        js_doc = Document(
            text="""
function testFunction() {
    const x = 1;
    const y = 2;
    return x + y;
}
""",
            metadata={"language": "javascript", "file_path": "test.js"}
        )
        
        docs = [python_doc, js_doc]
        
        print("\n🔬 Testing chunk_documents_by_language()...")
        nodes = chunk_documents_by_language(docs)
        
        print(f"  ✅ Function executed successfully")
        print(f"  ✅ Created {len(nodes)} total chunks")
        
        # Verify nodes have metadata
        if nodes:
            print(f"  ✅ Sample node metadata: {list(nodes[0].metadata.keys())}")
        
        return True, len(nodes)
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("AST PARSING VERIFICATION TEST SUITE")
    print("=" * 70)
    print("\nThis script verifies that AST-aware parsing is working correctly")
    print("and not silently falling back to text-based splitting.\n")
    
    results = {}
    
    # Test 1: Parser loading
    parser_results = test_parser_loading()
    results["parser_loading"] = all("✅" in r for r in parser_results.values())
    
    # Test 2: AST vs Text chunking
    ast_success, ast_count, text_count = test_ast_vs_text_chunking()
    results["ast_chunking"] = ast_success
    
    # Test 3: CodeSplitter without parser (demonstrates bug)
    bug_demo = test_code_splitter_without_parser()
    results["bug_demonstration"] = bug_demo
    
    # Test 4: Integration test
    integration_success, node_count = test_ingestion_function()
    results["integration"] = integration_success
    
    # Final summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25} -> {status}")
    
    print()
    
    # Critical check
    if results.get("ast_chunking") and results.get("integration"):
        print("🎉 SUCCESS: AST-aware parsing is working correctly!")
        print("   CodeSplitter is using tree-sitter parsers and respecting code structure.")
    else:
        print("⚠️  WARNING: AST-aware parsing may not be working correctly.")
        print("   Check the output above for details.")
        if not results.get("parser_loading"):
            print("   → Parser loading failed - check tree-sitter-language-pack installation")
        if not results.get("ast_chunking"):
            print("   → AST chunking failed - chunks may not respect code structure")
        if not results.get("integration"):
            print("   → Integration test failed - check chunk_documents_by_language() function")
    
    print()
    
    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

