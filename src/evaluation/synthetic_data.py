import os
import json
import random
import asyncio
from typing import List, Dict

from src.database import get_graph_store, get_all_repositories
from src.llm import LLMEngine
from src.config import RETRIEVE_LLM_PROVIDER, RETRIEVE_LLM_MODEL

PROMPT = """
You are an expert developer looking at some source code from the repository '{repo_name}'.
Your task is to generate {num_questions} highly technical questions and their corresponding reference answers based ONLY on the provided code snippet.

The questions should be realistic questions a senior engineer would ask, such as:
- How does function X handle error Y?
- What is the relationship between struct A and interface B?
- Where does the data for component Z come from?

Code Snippet:
-------------------------
{code}
-------------------------

Output your questions and answers in JSON format exactly like this:
[
    {{"question": "How does X work?", "reference_answer": "X works by calling Y..."}},
    {{"question": "What does Z do?", "reference_answer": "Z processes..."}}
]

Output ONLY valid JSON.
"""

async def generate_synthetic_data(repo_name: str, num_snippets: int = 5, qs_per_snippet: int = 2) -> List[Dict]:
    """
    Connects to the GraphStore, samples `num_snippets` file nodes from a given repo,
    and uses the LLM to generate Q&A pairs for evaluation.
    """
    print(f"🧠 Generating synthetic dataset for '{repo_name}'...")
    
    # We grab nodes from the graph store
    graph_store = get_graph_store()
    
    # The property graph has nodes in `graph_store.get(properties={"repo_name": repo_name})`
    # Let's just fetch all nodes and filter by repo_name if present.
    # LlamaIndex PropertyGraphStore doesn't expose a simple get_all_nodes API easily without Cypher.
    # We'll pull from the local graph file if using SimplePropertyGraphStore.
    
    # A robust way is to read the raw json if it's the Simple store.
    from src.config import GRAPH_PATH
    graph_file = os.path.join(GRAPH_PATH, "graph_store.json")
    
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph store not found at {graph_file}")
        
    with open(graph_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # SimplePropertyGraphStore structure: data["nodes"] is a dict of id -> node dict
    nodes = data.get("nodes", {})
    
    # Filter for nodes that are actually documents (files) from this repo
    file_nodes = []
    for node_id, node_data in nodes.items():
        props = node_data.get("properties", {})
        if props.get("repo_name") == repo_name and props.get("file_path"):
            # We want nodes with actual code content
            # Wait, content might not be in properties. Let's see where text is.
            file_nodes.append(node_data)
            
    if not file_nodes:
        print(f"⚠️ No file nodes found for {repo_name} in GraphStore. Taking any nodes with code properties.")
        for node_id, node_data in nodes.items():
            if "code" in node_data.get("properties", {}):
                file_nodes.append(node_data)
                
    if not file_nodes:
        raise ValueError("Could not find any code nodes in the graph store.")
        
    # Sample nodes
    sampled = random.sample(file_nodes, min(num_snippets, len(file_nodes)))
    
    llm_engine = LLMEngine(provider=RETRIEVE_LLM_PROVIDER, model_name=RETRIEVE_LLM_MODEL)
    
    dataset = []
    
    for i, node in enumerate(sampled):
        props = node.get("properties", {})
        # Extract code: could be in properties['code'] or we might have to use the text if available
        code = props.get("code") or node.get("text") or str(props)
        
        # Trim if too large for prompt
        if len(code) > 8000:
            code = code[:8000] + "...[TRUNCATED]"
            
        print(f"   Generating questions for snippet {i+1}/{len(sampled)}...")
        
        prompt = PROMPT.format(repo_name=repo_name, num_questions=qs_per_snippet, code=code)
        
        try:
            res = await llm_engine.llm.acomplete(prompt)
            clean_text = res.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3]
                
            qas = json.loads(clean_text)
            for qa in qas:
                qa["category"] = "synthetic_generated"
                qa["source_node"] = node.get("id")
                dataset.append(qa)
        except Exception as e:
            print(f"   ⚠️ Failed to generate for snippet {i+1}: {e}")
            
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_name", help="Name of the repository to generate data for")
    parser.add_argument("--output", default="eval_dataset.json", help="Output JSON file")
    args = parser.parse_args()
    
    dataset = asyncio.run(generate_synthetic_data(args.repo_name))
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"\n✅ Successfully generated {len(dataset)} synthetic Q&A pairs in {args.output}")
