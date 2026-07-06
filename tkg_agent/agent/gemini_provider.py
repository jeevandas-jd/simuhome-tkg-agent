"""
Gemini LLM Provider
===================
Wraps the Google Gemini API into a simple callable that the agent loop uses.
Matches SimuHome's pattern: provider.generate(messages, response_format=schema)
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def _to_gemini_content(msg: Any) -> types.Content:
    """Converts a standard chat message dictionary/dataclass to a GenAI Content object."""
    role = msg.role if hasattr(msg, "role") else msg.get("role")
    content = msg.content if hasattr(msg, "content") else msg.get("content")
    
    # Map typical roles to Gemini accepted roles ('user', 'model')
    if role in ["assistant", "model"]:
        g_role = "model"
    else:
        g_role = "user"
        
    return types.Content(
        role=g_role,
        parts=[types.Part.from_text(text=content)]
    )


class GeminiProvider:
    """
    Drop-in LLM provider for both BaseReActAgent and TKGAgent.

    Usage:
        llm = GeminiProvider()
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
        # Defaulting to a strong, fast reasoning/general model
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initializes using GEMINI_API_KEY environment variable automatically
        self._client = genai.Client()

    def generate(
        self,
        messages: list[Any],
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Call Gemini and return the assistant text.
        Retries on rate-limit (429) and server errors (500/503).
        """
        # Convert incoming list into the correct SDK content array structure
        contents = [_to_gemini_content(m) for m in messages]

        # Configure the request config mapping
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }

        # Gemini supports full structured JSON schema outputs natively
        if response_format and response_format.get("type") == "json_schema":
            config_kwargs["response_mime_type"] = "application/json"
            # If your schema specifies a nested 'schema' key, pass it directly
            if "schema" in response_format:
                config_kwargs["response_schema"] = response_format["schema"]

        config = types.GenerateContentConfig(**config_kwargs)

        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return resp.text.strip()

            except Exception as e:
                err = str(e)
                # Catching common Gemini API resource exhausted / server errors
                is_rate = "429" in err or "quota" in err.lower() or "exhausted" in err.lower()
                is_server = "500" in err or "503" in err or "bad gateway" in err.lower()

                if attempt < self.max_retries - 1 and (is_rate or is_server):
                    wait = self.retry_delay * (2 ** attempt) if is_rate else self.retry_delay
                    print(f"  [Gemini] {'Rate limit' if is_rate else 'Server error'} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue

                raise RuntimeError(f"Gemini call failed after {attempt+1} attempts: {e}") from e

        raise RuntimeError("Gemini: max retries exhausted")