"""
Groq LLM Provider
=================
Wraps the Groq API into a simple callable that the agent loop uses.
Matches SimuHome's pattern: provider.generate(messages, response_format=schema)
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# SimuHome ChatMessage is just a dataclass with role + content
# We accept either that or plain dicts
def _to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    return {"role": msg.role, "content": msg.content}


class GroqProvider:
    """
    Drop-in LLM provider for both BaseReActAgent and TKGAgent.

    Usage:
        llm = GroqProvider()
        text = llm.generate(messages, response_format=schema)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ):
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(
        self,
        messages: list[Any],
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Call Groq and return the assistant text.
        Retries on rate-limit (429) and server errors (502/503).
        """
        raw_messages = [_to_dict(m) for m in messages]

        # Groq supports JSON mode but not full json_schema — use json_object
        groq_format: Optional[dict] = None
        if response_format and response_format.get("type") == "json_schema":
            groq_format = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=raw_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if groq_format:
                    kwargs["response_format"] = groq_format

                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                is_server = "502" in err or "503" in err

                if attempt < self.max_retries - 1 and (is_rate or is_server):
                    wait = self.retry_delay * (2 ** attempt) if is_rate else self.retry_delay
                    print(f"  [Groq] {'Rate limit' if is_rate else 'Server error'} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue

                raise RuntimeError(f"Groq call failed after {attempt+1} attempts: {e}") from e

        raise RuntimeError("Groq: max retries exhausted")