import json
from typing import Any, Optional, Sequence
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from llama_index.core.llms import LLM
from src.evaluation.groq_client import RobustGroqClient

COMPREHENSIVE_EVALUATION_PROMPT = """
Evaluate the Generated Answer based on the User Query and Context.

Score these 4 metrics on a 0-5 scale (0=Fail, 5=Perfect).
1. Faithfulness: Does it hallucinate code/logic? (5=Perfectly faithful)
2. Relevance: Did the context provide the necessary code? (5=Perfect context)
3. Completeness: Did the answer address the prompt? (5=Very comprehensive)
4. Synthesization: How well is it formatted? (5=Excellent formatting)

---
Query: {query}
---
Context:
{context}
---
Answer:
{answer}
---

Output valid JSON exactly like this:
{{
  "faithfulness": {{"score": <0-5>, "reasoning": "<1 sentence>"}},
  "relevance": {{"score": <0-5>, "reasoning": "<1 sentence>"}},
  "completeness": {{"score": <0-5>, "reasoning": "<1 sentence>"}},
  "synthesization": {{"score": <0-5>, "reasoning": "<1 sentence>"}}
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
        
        # Hard-cap context length to prevent blowing past Groq 12k TPM limits
        max_chars = 30000
        if len(context_str) > max_chars:
            context_str = context_str[:max_chars] + "\n\n...[CONTEXT TRUNCATED DUE TO TOKEN LIMITS]..."
            
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
