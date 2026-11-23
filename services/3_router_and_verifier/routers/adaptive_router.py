import logging
from typing import Dict, Any
from ..llm_adapters.gemini_orchestrator import GeminiOrchestrator

logger = logging.getLogger(__name__)

ROUTER_PROMPT = """You are a query router for an academic lecture assistant.
Classify the student's message into exactly one route:
- "qa": a specific question about the lecture content.
- "summarize_topic": asks to summarize/explain briefly a specific topic or section.
- "summarize_all": asks to summarize the whole lecture/document.
- "out_of_scope": greetings, chit-chat, or unrelated to academic lecture content.

The message may be in Arabic, English, or mixed.
Return ONLY JSON: {{"route": "...", "topic": "<topic or empty>"}}

Message: {question}
"""

class AdaptiveRouter:
    """
    Routes user query to appropriate pipeline:
    qa, summarize_topic, summarize_all, out_of_scope
    """
    def __init__(self, llm: GeminiOrchestrator):
        self.llm = llm

    def route_query(self, question: str) -> Dict[str, Any]:
        prompt = ROUTER_PROMPT.format(question=question)
        resp = self.llm.call_llm(prompt, temperature=0.0, max_tokens=100)
        out = self.llm.parse_json(resp)
        if out and out.get("route") in {"qa", "summarize_topic", "summarize_all", "out_of_scope"}:
            return out
            
        # Fallback heuristic
        q = question.lower()
        if any(w in q for w in ["summarize the lecture", "summary of the lecture", "لخص المحاضرة", "لخص كل"]):
            return {"route": "summarize_all", "topic": ""}
        if any(w in q for w in ["summar", "لخص", "تلخيص"]):
            return {"route": "summarize_topic", "topic": question}
        if any(w in q for w in ["hi", "hello", "ازيك", "صباح الخير", "مساء الخير"]):
            return {"route": "out_of_scope", "topic": ""}
        return {"route": "qa", "topic": ""}
