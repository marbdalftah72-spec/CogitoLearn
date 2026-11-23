from typing import Dict, Any, List
from ..llm_adapters.gemini_orchestrator import GeminiOrchestrator

VERIFY_PROMPT = """You are a strict fact-checker.
Check whether EVERY claim in the Answer is supported by the Context.

Context:
{context}

Question: {question}

Answer: {answer}

Return ONLY JSON:
{{"grounded": true/false, "score": <0-10>, "unsupported_claims": ["..."]}}"""

class NLIVerifier:
    """
    Fact checks LLM responses against retrieved evidence context.
    """
    def __init__(self, llm: GeminiOrchestrator, threshold: int = 7):
        self.llm = llm
        self.threshold = threshold

    def verify(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        prompt = VERIFY_PROMPT.format(context=context, question=question, answer=answer)
        resp = self.llm.call_llm(prompt, temperature=0.0, max_tokens=250)
        out = self.llm.parse_json(resp)
        if not out:
            return {"grounded": False, "score": 0, "unsupported_claims": ["verifier parse error"]}
        
        is_grounded = bool(out.get("grounded")) and out.get("score", 0) >= self.threshold
        out["grounded"] = is_grounded
        return out
