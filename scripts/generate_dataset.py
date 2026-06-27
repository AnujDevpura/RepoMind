import time
import json
import os
import sys
import chromadb

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_index.core.schema import TextNode
from llama_index.core.llms import ChatMessage
from src.database import initialize_database
from src.llm import LLMEngine
from src.config import CHROMA_PATH


def get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def generate_evaluation_dataset():
    print("🔧 Initializing components...")
    initialize_database()
    llm_engine = LLMEngine()
    
    print("📂 Loading chunks directly from ChromaDB...")
    # Connect directly to the local ChromaDB storage
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = db.get_collection("repomind_codebase")
        data = collection.get(include=['documents', 'metadatas'])
    except Exception as e:
        print(f"❌ Could not access ChromaDB collection: {e}")
        print("Are you sure you ingested a repository first?")
        return

    documents = data.get('documents', [])
    metadatas = data.get('metadatas', [])
    
    nodes = []
    # Reconstruct LlamaIndex TextNodes
    for i, doc in enumerate(documents):
        if doc and len(doc.strip()) > 50:  # Skip empty or tiny fragments
            meta = metadatas[i] if metadatas else {}
            nodes.append(TextNode(text=doc, metadata=meta))

    if not nodes:
        print("❌ 0 chunks found. Please ingest a repository via the Gradio UI first.")
        return

    print(f"📊 Found {len(nodes)} valid chunks in the database.")
    
    chunk_count = min(get_env_int("DATASET_CHUNK_COUNT", 25), len(nodes))
    subset_nodes = nodes[:chunk_count] 
    print(f"🚀 Starting token-safe generation for {len(subset_nodes)} chunks...")
    
    output_file = os.getenv("EVAL_DATASET_PATH", "data/tests.jsonl")
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")
    
    # Custom system prompt for generation
    system_prompt = (
        "You are an expert software engineer generating a test dataset. "
        "Given the code context, generate exactly 2 diverse questions that can be answered using ONLY this context. "
        "Format your response EXACTLY as a JSON list of dictionaries, like this:\n"
        '[\n  {"question": "How does X work?", "reference_answer": "X works by..."},\n  {"question": "What is Y?", "reference_answer": "Y is a..."}\n]'
    )

    for i, node in enumerate(subset_nodes, 1):
        print(f"\nProcessing chunk {i} of {len(subset_nodes)}...")
        
        try:
            context_str = node.get_content()
            user_prompt = f"Code Context:\n{context_str}\n\nGenerate the JSON list of 2 question/answer pairs."
            
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
            
            # Use Groq to generate
            response = llm_engine.llm.chat(messages)
            response_text = response.message.content if hasattr(response, 'message') else str(response)
            
            # Clean up the response to parse JSON safely
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            qa_pairs = json.loads(response_text)
            
            # Append to file
            with open(output_file, 'a', encoding='utf-8') as f:
                for pair in qa_pairs:
                    pair["category"] = "synthetic_generated"
                    f.write(json.dumps(pair) + '\n')
            
            print("✅ Saved. Sleeping 10 seconds to respect the Groq rate limits...")
            time.sleep(10)
            
        except json.JSONDecodeError:
            print("⚠️ LLM did not return valid JSON. Skipping chunk.")
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Hit an error: {e}")
            print("Sleeping for 60 seconds to reset rate limit windows...")
            time.sleep(60)

if __name__ == "__main__":
    generate_evaluation_dataset()
