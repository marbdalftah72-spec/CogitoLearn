import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from common.contracts import VerifiedResponse, AtomicClaim
from .llm_adapters.gemini_orchestrator import GeminiOrchestrator
from .routers.adaptive_router import AdaptiveRouter
from .routers.query_decomposer import QueryDecomposer
from .verifiers.nli_verifier import NLIVerifier

logger = logging.getLogger(__name__)

QA_PROMPT = """You are an AI Academic Tutor.
Answer the student's question using ONLY the provided context.
If the answer is not in the context, reply exactly: "The information is not available in the lecture."
Reply in the same language as the question. Be accurate and concise.
{extra}
Context:
{context}

Student Question:
{question}

Answer:"""

SUMMARY_PROMPT = """You are an AI Academic Tutor.
Summarize the following lecture content into clear, well-organized key points.
Use ONLY the provided content. Do not add outside information.
Reply in the same language as the student's request.

Student request: {question}

Content:
{context}

Summary:"""

PART_PROMPT = """Summarize the following lecture excerpt into concise bullet points.
Use ONLY the excerpt. Keep key definitions, advantages and disadvantages.

Excerpt:
{content}

Bullet points:"""

FINAL_PROMPT = """You are an AI Academic Tutor.
Combine the partial summaries below into ONE clear, well-organized summary of the lecture.
Use ONLY the partial summaries. Reply in the same language as the student's request.

Student request: {question}

Partial summaries:
{partials}

Final summary:"""

NOT_AVAILABLE = "المعلومة دي مش موجودة في المحاضرة. / The information is not available in the lecture."
OUT_OF_SCOPE = "أنا مساعد أكاديمي، اسألني عن محتوى المحاضرة أو اطلب تلخيص. / I can only help with lecture content."

class RAGEngine:
    """
    Main Service 3 Engine coordinating:
    - Adaptive Routing
    - Qdrant Vector Retrieval (Dense BGE-M3)
    - Response Generation (Groq/Gemini LLM)
    - Fact-checking Verification
    - Map-Reduce Full Lecture Summarization
    """
    def __init__(self, memory_service=None, llm: Optional[GeminiOrchestrator] = None, min_score: float = 0.40):
        self.memory_service = memory_service
        self.llm = llm or GeminiOrchestrator()
        self.router = AdaptiveRouter(self.llm)
        self.decomposer = QueryDecomposer()
        self.verifier = NLIVerifier(self.llm)
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 3) -> List[Any]:
        if not self.memory_service:
            return []
        q_emb = self.memory_service.embedder.encode_query(query)
        return self.memory_service.qdrant.search(
            collection_name=self.memory_service.collection_name,
            query_vector=q_emb,
            limit=top_k,
            score_threshold=self.min_score
        )

    def get_all_chunks(self) -> List[Any]:
        if not self.memory_service:
            return []
        return self.memory_service.qdrant.scroll_all(self.memory_service.collection_name)

    def build_context(self, points: List[Any]) -> str:
        return "\n\n".join(f"[Chunk {p.payload.get('chunk_id')}]\n{p.payload.get('text')}" for p in points)

    def summarize_all_lecture(self, question: str, batch_chars: int = 4500) -> str:
        points = self.get_all_chunks()
        texts = []
        for i, p in enumerate(points):
            words = p.payload.get("text", "").split()
            texts.append(" ".join(words if i == 0 else words[20:]))

        batches, cur = [], ""
        for t in texts:
            if cur and len(cur) + len(t) > batch_chars:
                batches.append(cur)
                cur = ""
            cur += " " + t
        if cur:
            batches.append(cur)

        partials = [self.llm.call_llm(PART_PROMPT.format(content=b), max_tokens=350) for b in batches]
        return self.llm.call_llm(FINAL_PROMPT.format(question=question, partials="\n\n".join(partials)), max_tokens=900)

    def answer_question(self, question: str, strict: bool = False) -> Tuple[Optional[str], List[Any]]:
        points = self.retrieve(question, top_k=3)
        if not points:
            return None, []
        context = self.build_context(points)
        extra = "IMPORTANT: Every claim must be directly supported by the context. Do not add anything else." if strict else ""
        prompt = QA_PROMPT.format(extra=extra, context=context, question=question)
        answer = self.llm.call_llm(prompt)
        return answer, points

    def summarize(self, question: str, route: str, topic: str = "") -> Tuple[Optional[str], List[Any]]:
        if route == "summarize_all":
            return self.summarize_all_lecture(question), []
        points = self.retrieve(topic or question, top_k=6)
        if not points:
            return None, []
        context = self.build_context(points)
        answer = self.llm.call_llm(SUMMARY_PROMPT.format(question=question, context=context), max_tokens=900)
        return answer, points

    def ask(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        route_info = self.router.route_query(question)
        route = route_info["route"]

        if route == "out_of_scope":
            return {"route": route, "answer": OUT_OF_SCOPE, "verified": None, "sources": []}

        if route == "summarize_all":
            answer, _ = self.summarize(question, route)
            return {"route": route, "answer": answer, "verified": "skipped (full lecture)", "sources": ["all"]}

        for attempt in range(2):
            if route == "qa":
                answer, points = self.answer_question(question, strict=(attempt == 1))
            else:
                answer, points = self.summarize(question, route, route_info.get("topic", ""))

            if answer is None or "not available in the lecture" in answer.lower():
                return {"route": route, "answer": NOT_AVAILABLE, "verified": None, "sources": []}

            context = self.build_context(points)
            check = self.verifier.verify(question, answer, context)
            if check["grounded"]:
                sources = [p.payload.get("chunk_id") for p in points]
                return {
                    "route": route,
                    "answer": answer,
                    "verified": True,
                    "score": check.get("score"),
                    "sources": sources
                }

        return {"route": route, "answer": NOT_AVAILABLE, "verified": False, "sources": []}
