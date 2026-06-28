import os
import sys
import json
import asyncio
from typing import List, Dict

# Ensure we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database import initialize_database
from src.retrieval import Retriever
from src.llm import LLMEngine
from src.config import RETRIEVE_LLM_PROVIDER, RETRIEVE_LLM_MODEL, USE_RERANKER
from src.evaluation.metrics import ComprehensiveCodeRAGEvaluator

async def run_evaluation(dataset_path: str, repo_name: str, output_path: str):
    """
    Runs the Code RAG evaluation pipeline on a dataset of questions.
    """
    print(f"🚀 Starting Custom Code-Aware Evaluation on '{dataset_path}'...")
    
    # 1. Load Dataset
    dataset = []
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
    print(f"Loaded {len(dataset)} evaluation questions.")
    
    # 2. Initialize System
    initialize_database(load_embed_model=True)
    retriever = Retriever(use_reranker=USE_RERANKER)
    
    # For generation, we use the standard engine
    generation_engine = LLMEngine(provider=RETRIEVE_LLM_PROVIDER, model_name=RETRIEVE_LLM_MODEL)
    
    # For evaluation, we ideally use the same strong model to act as a judge
    eval_engine = LLMEngine(provider=RETRIEVE_LLM_PROVIDER, model_name=RETRIEVE_LLM_MODEL)
    if RETRIEVE_LLM_PROVIDER == "groq":
        from src.evaluation.groq_client import RobustGroqClient
        groq_client = RobustGroqClient(model_name=RETRIEVE_LLM_MODEL)
        evaluator = ComprehensiveCodeRAGEvaluator(groq_client=groq_client)
    else:
        evaluator = ComprehensiveCodeRAGEvaluator(llm=eval_engine.llm)
    
    results = []
    total_metrics = {"faithfulness": 0.0, "relevance": 0.0, "completeness": 0.0, "synthesization": 0.0}
    
    for i, item in enumerate(dataset):
        query = item.get("question")
        if not query:
            continue
            
        print(f"\n[{i+1}/{len(dataset)}] Q: {query}")
        
        # 3. Retrieve Context
        try:
            nodes = retriever.search(query, repo_name=repo_name)
            contexts = [n.node.get_content() for n in nodes]
            print(f"   ↳ Retrieved {len(contexts)} nodes.")
        except Exception as e:
            print(f"   ⚠️ Retrieval failed: {e}")
            contexts = []
            
        # 4. Generate Answer
        response_text = ""
        if contexts:
            try:
                # We use stream_chat but collect it into a single string
                chunks = list(generation_engine.stream_chat(query, nodes, history=[]))
                response_text = "".join(chunks)
            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
                
        # 5. Evaluate Metrics (Single-Shot)
        eval_res = await evaluator.aevaluate(
            query=query, response=response_text, contexts=contexts
        )
        
        try:
            feedback_data = json.loads(eval_res.feedback)
        except:
            feedback_data = {"error": eval_res.feedback}
            
        if "error" in feedback_data:
            print(f"   ⚠️ Evaluation Error: {feedback_data['error']}")
            results.append({
                "question": query,
                "error": feedback_data['error']
            })
            continue
            
        f_score = feedback_data["faithfulness"]["score"]
        r_score = feedback_data["relevance"]["score"]
        c_score = feedback_data["completeness"]["score"]
        s_score = feedback_data["synthesization"]["score"]
        
        print(f"   ↳ Faithfulness  : {f_score:.2f} | {feedback_data['faithfulness']['reason']}")
        print(f"   ↳ Relevance     : {r_score:.2f} | {feedback_data['relevance']['reason']}")
        print(f"   ↳ Completeness  : {c_score:.2f} | {feedback_data['completeness']['reason']}")
        print(f"   ↳ Synthesization: {s_score:.2f} | {feedback_data['synthesization']['reason']}")
        print(f"   ↳ OVERALL SCORE : {eval_res.score:.2f}")
        
        total_metrics["faithfulness"] += f_score
        total_metrics["relevance"] += r_score
        total_metrics["completeness"] += c_score
        total_metrics["synthesization"] += s_score
        
        results.append({
            "question": query,
            "reference_answer": item.get("reference_answer", ""),
            "generated_answer": response_text,
            "metrics": feedback_data,
            "overall_score": eval_res.score
        })
        
    # 6. Output Scorecard
    total = max(1, len(results))
    avg_f = total_metrics["faithfulness"] / total
    avg_r = total_metrics["relevance"] / total
    avg_c = total_metrics["completeness"] / total
    avg_s = total_metrics["synthesization"] / total
    
    print("\n" + "="*50)
    print("📊 ADVANCED EVALUATION SCORECARD")
    print("="*50)
    print(f"Total Questions Evaluated : {len(results)}")
    print(f"Average Code Faithfulness : {avg_f:.2f}")
    print(f"Average Struct. Relevance : {avg_r:.2f}")
    print(f"Average Answer Complete.  : {avg_c:.2f}")
    print(f"Average Code Synthesiz.   : {avg_s:.2f}")
    
    # Save detailed JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_faithfulness": avg_f,
                "avg_relevance": avg_r,
                "avg_completeness": avg_c,
                "avg_synthesization": avg_s,
                "total": len(results)
            },
            "results": results
        }, f, indent=2)
        
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to the eval_dataset.json or tests.jsonl")
    parser.add_argument("--repo", default="All Repositories", help="Repository to filter retrieval on")
    parser.add_argument("--output", default="eval_results.json", help="Output scorecard JSON")
    args = parser.parse_args()
    
    # Suppress verbose logging from llama_index
    import logging
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    asyncio.run(run_evaluation(args.dataset, args.repo, args.output))
