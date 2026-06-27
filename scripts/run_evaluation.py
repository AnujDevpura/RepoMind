import json
import time
import os
import sys
import re

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import Retriever
from src.llm import LLMEngine
from src.database import initialize_database 
from llama_index.core.llms import ChatMessage


def get_env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def extract_query_terms(question):
    terms = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", question))
    stopwords = {
        "what", "where", "when", "which", "within", "given", "code",
        "context", "class", "function", "purpose", "default", "value",
        "difference", "between", "implementation", "implementations",
    }
    return [term for term in terms if term.lower() not in stopwords]


def excerpt_around_terms(text, terms, max_chars=1600):
    if not text:
        return ""

    lower_text = text.lower()
    match_positions = [
        lower_text.find(term.lower())
        for term in terms
        if term and lower_text.find(term.lower()) != -1
    ]

    if match_positions:
        center = min(match_positions)
        start = max(0, center - max_chars // 3)
        end = min(len(text), start + max_chars)
        return text[start:end]

    return text[:max_chars]


def build_judge_context(nodes, question, max_chars_per_node=1600):
    terms = extract_query_terms(question)
    context_parts = []

    for idx, node in enumerate(nodes, 1):
        file_path = node.metadata.get("file_path", "Unknown")
        content = node.get_content() if hasattr(node, "get_content") else str(node)
        excerpt = excerpt_around_terms(content, terms, max_chars=max_chars_per_node)
        context_parts.append(f"\n--- Source {idx}: {file_path} ---\n{excerpt}\n")

    return "".join(context_parts) if context_parts else "No context found."

def extract_json_from_text(text):
    """Safely extracts JSON from LLM output even if wrapped in markdown or conversational text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

def run_evaluation():
    print("🔧 Initializing High-Fidelity Evaluation Pipeline...")
    try:
        initialize_database()  # <-- ADDED THIS (Loads BAAI/bge-m3 instead of OpenAI)
        retriever = Retriever(use_reranker=True)
        llm_engine = LLMEngine()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # Load the synthetic dataset
    dataset_path = os.getenv("EVAL_DATASET_PATH", "data/tests.jsonl")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            test_cases = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("❌ Dataset not found. Please run scripts/generate_dataset.py first.")
        return

    print(f"📊 Loaded {len(test_cases)} total test cases.")
    
    sample_size = min(get_env_int("EVAL_SAMPLE_SIZE", 10), len(test_cases))
    cooldown_seconds = get_env_int("EVAL_COOLDOWN_SECONDS", 20)
    test_cases = test_cases[:sample_size]
    print(f"🚀 Running benchmark on {sample_size} cases...\n")
    
    metrics = {
        "faithfulness": 0,
        "relevancy": 0,
        "context_precision": 0,
        "citation_accuracy": 0,
        "no_hallucination": 0
    }
    successful_evals = 0
    
    for i, test in enumerate(test_cases, 1):
        question = test['question']
        reference_answer = test.get('reference_answer', 'N/A')
        
        print(f"[{i}/{sample_size}] Q: {question[:80]}...")
        
        # 1. Run Retrieval
        nodes = retriever.search(question)
        source_paths = [n.metadata.get("file_path", "Unknown") for n in nodes]
        print(f"   Sources: {source_paths}")
        context_str = build_judge_context(nodes, question)
        
        # 2. Run Generation
        generated_answer = ""
        try:
            for token in llm_engine.stream_chat(question, nodes, history=[]):
                if token: generated_answer += token
        except Exception as e:
            print(f"   => ⚠️ Generation Error: {e}")
            time.sleep(20)
            continue
            
        # 3. LLM-as-a-Judge Evaluation
        eval_prompt = (
            "You are an expert grading system for a Code Retrieval-Augmented Generation (RAG) tool.\n"
            "Evaluate the Generated Answer against the Question, the Reference Answer, and the Retrieved Context.\n\n"
            f"Question: {question}\n"
            f"Reference Answer: {reference_answer}\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Generated Answer:\n{generated_answer}\n\n"
            "Score the following 5 metrics strictly based on the definitions below:\n"
            "1. 'context_precision' (1-5): Does the Retrieved Context contain the exact code/logic needed to answer the question? (5 = perfect context).\n"
            "2. 'faithfulness' (1-5): Is the Generated Answer supported ENTIRELY by the Retrieved Context? (5 = no external facts used).\n"
            "3. 'relevancy' (1-5): Does the Generated Answer completely and directly answer the Question? (5 = perfect answer).\n"
            "4. 'citation_accuracy' (1-5): Did the Generated Answer explicitly name the correct file paths (e.g., 'train.py') provided in the Context? (5 = perfect citations).\n"
            "5. 'no_hallucination' (0 or 1): Did the model invent any fake variables, functions, or classes? (1 = NO hallucinations, 0 = Hallucinated code).\n\n"
            "Output ONLY a raw JSON dictionary exactly like this:\n"
            '{"context_precision": 5, "faithfulness": 5, "relevancy": 4, "citation_accuracy": 5, "no_hallucination": 1}'
        )
        
        try:
            msg = [ChatMessage(role="user", content=eval_prompt)]
            eval_response = llm_engine.llm.chat(msg)
            resp_text = eval_response.message.content.strip()
            
            scores = extract_json_from_text(resp_text)
            
            if scores:
                cp = scores.get('context_precision', 0)
                f = scores.get('faithfulness', 0)
                r = scores.get('relevancy', 0)
                ca = scores.get('citation_accuracy', 0)
                nh = scores.get('no_hallucination', 0)
                
                print(f"   => Context: {cp}/5 | Faithfulness: {f}/5 | Relevancy: {r}/5 | Citations: {ca}/5 | Valid Code: {nh}/1")
                
                metrics["context_precision"] += cp
                metrics["faithfulness"] += f
                metrics["relevancy"] += r
                metrics["citation_accuracy"] += ca
                metrics["no_hallucination"] += nh
                successful_evals += 1
            else:
                print(f"   => ⚠️ Eval error: LLM did not return valid JSON. Response: {resp_text[:50]}...")
                
        except Exception as e:
            print(f"   => ⚠️ Eval API error: {e}")
        
        print(f"   ⏳ Cooling down API for {cooldown_seconds}s...")
        time.sleep(cooldown_seconds)
        
    # Calculate Final Percentages
    if successful_evals > 0:
        print("\n==================================================")
        print("🎯 FINAL REPOMIND ENGINE BENCHMARK (Out of 100%)")
        print("==================================================")
        print(f"Context Precision:  {(metrics['context_precision'] / successful_evals) * 20:5.1f}%  (Retrieval Quality)")
        print(f"Faithfulness:       {(metrics['faithfulness'] / successful_evals) * 20:5.1f}%  (Trust in Context)")
        print(f"Answer Relevancy:   {(metrics['relevancy'] / successful_evals) * 20:5.1f}%  (User Satisfaction)")
        print(f"Citation Accuracy:  {(metrics['citation_accuracy'] / successful_evals) * 20:5.1f}%  (File Tracing)")
        print(f"Code Authenticity:  {(metrics['no_hallucination'] / successful_evals) * 100:5.1f}%  (Non-Hallucinated Code)")
        print("==================================================")
    else:
        print("\n❌ Evaluation failed to complete successfully.")

if __name__ == "__main__":
    run_evaluation()
