import os
import json
import re
import time
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class GeminiOrchestrator:
    """
    LLM orchestrator supporting Groq / OpenAI compatible APIs or Google Gemini.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b", base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def call_llm(self, prompt: str, system: str = "You are an AI Academic Tutor.", temperature: float = 0.0, max_tokens: int = 800, retries: int = 5) -> str:
        client = self._get_client()
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code in (413, 429) and attempt < retries - 1:
                    wait = 10 * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2)
        return ""

    @staticmethod
    def parse_json(text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
