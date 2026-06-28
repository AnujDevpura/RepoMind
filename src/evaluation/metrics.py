import json
from typing import Any, Optional, Sequence
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from llama_index.core.llms import LLM
from src.evaluation.groq_client import RobustGroqClient

COMPREHENSIVE_EVALUATION_PROMPT = """
You are an elite Principal Software Engineer evaluating the performance of a Code-Aware RAG system.
Your task is to evaluate a generated answer based on the user's query and the retrieved source code context.

You must evaluate the response across 4 distinct pillars using a strict 0-5 scale.

### PILLAR 1: Code Faithfulness (0-5)
Does the generated answer hallucinate code, logic, or dependencies not present in the context?
- 5: Perfectly faithful. Every class, function, or variable mentioned exists in the context.
- 4: Mostly faithful. Minor conceptual generalizations, but no hallucinated code structures.
- 3: Partially faithful. Accurately references some context but hallucinates a minor function or file.
- 2: Mostly hallucinated. Invents major dependencies or files not provided in the context.
- 1: Barely faithful. References context but is overwhelmed by hallucinations.
- 0: Completely unfaithful or explicitly admits it cannot answer due to lack of context.

### PILLAR 2: Structural Relevance (0-5)
Did the retrieved context provide the necessary code to answer the query?
- 5: Perfect context. Includes exact files, caller/callee graphs, and necessary imports.
- 4: Highly relevant. Contains the core logic needed, but might be missing a minor dependency.
- 3: Partially relevant. Contains related code but misses the specific logic asked for.
- 2: Barely relevant. Pulls from the right repository/module but completely misses the target logic.
- 1: Extremely irrelevant context, missing almost all relevant modules.
- 0: Completely irrelevant context.

### PILLAR 3: Answer Completeness (0-5)
How completely did the generated answer address the user's prompt?
- 5: Extremely comprehensive. Answers every part of the prompt accurately.
- 4: Strong answer, but lacks a tiny bit of depth or omits a secondary aspect of the prompt.
- 3: Moderate. Answers the primary question but ignores significant architectural nuances.
- 2: Poor. Barely addresses the core prompt or gives a superficial summary.
- 1: Extremely poor. Fails to address the core prompt entirely.
- 0: Completely fails to answer the prompt or gives an empty/irrelevant response.

### PILLAR 4: Code Synthesization & Readability (0-5)
How well is the technical explanation formatted and synthesized?
- 5: Excellent formatting. Uses Markdown cleanly, cites file paths, and breaks down complex logic.
- 4: Good formatting. Readable, but could use better structural grouping or clearer file citations.
- 3: Average. A bit of a wall of text, or fails to properly format code blocks.
- 2: Poor. Difficult to read, messy formatting, confusing transitions.
- 1: Extremely poor. Hard to parse visually.
- 0: Unreadable or incoherent structure.

---
User Query: {query}
---
Retrieved Context:
{context}
---
Generated Answer:
{answer}
---

Output your evaluation as a JSON object with the exact following structure. Output ONLY valid JSON:
{{
  "faithfulness": {{"score": <0-5>, "reasoning": "<explanation>"}},
  "relevance": {{"score": <0-5>, "reasoning": "<explanation>"}},
  "completeness": {{"score": <0-5>, "reasoning": "<explanation>"}},
  "synthesization": {{"score": <0-5>, "reasoning": "<explanation>"}}
}}
"""

class ComprehensiveCodeRAGEvaluator(BaseEvaluator):
    """
    Evaluates Code RAG responses across 4 pillars in a single-shot LLM pass for massive speedup.
    Maps discrete 1-5 scales to a 0.2-1.0 continuous float scale.
    """
    def __init__(self, llm: LLM = None, groq_client: RobustGroqClient = None):
        super().__init__()
        self._llm = llm
        self._groq_client = groq_client

    def _get_prompts(self) -> dict:
        return {}

    def _update_prompts(self, prompts: dict) -> None:
        pass

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        
        if not query or not response or not contexts:
            return EvaluationResult(score=0.0, feedback="Missing query, response, or context.")
            
        context_str = "\n\n".join(contexts)
        prompt = COMPREHENSIVE_EVALUATION_PROMPT.format(
            query=query, context=context_str, answer=response
        )
        
        if self._groq_client:
            res = await self._groq_client.acomplete(prompt)
        else:
            res = await self._llm.acomplete(prompt)
            
        try:
            # Clean up JSON formatting if LLM adds markdown blocks
            clean_text = res.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3]
                
            parsed = json.loads(clean_text)
            
            # Helper to safely map 0-5 to 0.0-1.0
            def map_score(key: str) -> tuple[float, str]:
                item = parsed.get(key, {})
                raw_score = float(item.get("score", 0.0))
                # Clamp between 0 and 5
                raw_score = max(0.0, min(5.0, raw_score))
                # Map 0-5 to 0.0-1.0
                mapped_score = raw_score / 5.0
                reason = item.get("reasoning", "")
                return mapped_score, reason
            
            f_score, f_reason = map_score("faithfulness")
            r_score, r_reason = map_score("relevance")
            c_score, c_reason = map_score("completeness")
            s_score, s_reason = map_score("synthesization")
            
            # We pack all metrics into a JSON string inside the feedback field 
            # to remain compliant with LlamaIndex's BaseEvaluator signature.
            structured_feedback = json.dumps({
                "faithfulness": {"score": f_score, "reason": f_reason},
                "relevance": {"score": r_score, "reason": r_reason},
                "completeness": {"score": c_score, "reason": c_reason},
                "synthesization": {"score": s_score, "reason": s_reason}
            })
            
            # The primary score can just be the average for standard LlamaIndex usage
            overall_score = (f_score + r_score + c_score + s_score) / 4.0
            
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                score=overall_score,
                feedback=structured_feedback
            )
            
        except Exception as e:
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                score=0.0,
                feedback=f'{{"error": "Failed to parse LLM evaluation JSON: {e}", "raw": {json.dumps(res.text)}}}'
            )
