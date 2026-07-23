"""
Kaggle LLM Provider
====================
Drop-in replacement for GroqProvider.
Points at the FastAPI server running in your Kaggle notebook.

Usage — in .env set:
    KAGGLE_LLM_URL=https://abc123.ngrok-free.app/generate

Then swap in your code:
    # was:  from tkg_agent.agent.groq_provider import GroqProvider
    # now:  from tkg_agent.agent.kaggle_provider import KaggleProvider as GroqProvider
"""
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def _to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    return {"role": msg.role, "content": msg.content}


class KaggleProvider:
    \"""
    Calls the FastAPI /generate endpoint running in your Kaggle notebook.
    Interface is identical to GroqProvider — swap with one import change.
    \"""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        timeout: int = 120,          # Kaggle can be slow on first call
    ):
        self.endpoint  = endpoint or os.getenv("KAGGLE_LLM_URL")
        if not self.endpoint:
            raise ValueError(
                "Set KAGGLE_LLM_URL in your .env or pass endpoint= to KaggleProvider.\n"
                "Example: KAGGLE_LLM_URL=https://abc123.ngrok-free.app/generate"
            )
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.max_retries  = max_retries
        self.retry_delay  = retry_delay
        self.timeout      = timeout

    def generate(
        self,
        messages: list[Any],
        response_format: Optional[dict] = None,   # accepted but ignored — model handles JSON natively
    ) -> str:
        raw_messages = [_to_dict(m) for m in messages]

        payload = {
            "messages":    raw_messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["response"].strip()

            except requests.exceptions.Timeout:
                print(f"  [Kaggle] Timeout on attempt {attempt+1} — retrying in {self.retry_delay}s")
                time.sleep(self.retry_delay)

            except requests.exceptions.ConnectionError:
                print(f"  [Kaggle] Connection error — is the ngrok tunnel still open?")
                time.sleep(self.retry_delay)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  [Kaggle] Error: {e} — retrying in {self.retry_delay}s")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Kaggle LLM call failed after {self.max_retries} attempts: {e}") from e

        raise RuntimeError("Kaggle provider: max retries exhausted")
"""

"""
Groq LLM Provider
=================
Wraps the Groq API into a simple callable that the agent loop uses.
Matches SimuHome's pattern: provider.generate(messages, response_format=schema)
"""
"""
Kaggle LLM Provider
====================
Wraps a self-hosted vLLM server (running on Kaggle, exposed via a Cloudflare
tunnel) into a simple callable that the agent loop uses.
Matches SimuHome's pattern: provider.generate(messages, response_format=schema)
"""

import os
import time
from typing import Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _to_openai_message(msg: Any) -> dict:
    """Converts a standard chat message dictionary/dataclass to an OpenAI-style dict."""
    role = msg.role if hasattr(msg, "role") else msg.get("role")
    content = msg.content if hasattr(msg, "content") else msg.get("content")

    # Normalize Gemini-style 'model' role back to 'assistant' if present
    if role == "model":
        role = "assistant"

    return {"role": role, "content": content}


class KaggleProvider:
    """
    Drop-in LLM provider for both BaseReActAgent and TKGAgent.

    Usage:
        llm = KaggleProvider()
        text = llm.generate(messages, response_format=schema)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ):
        # Defaulting to env vars set for the Kaggle-hosted vLLM endpoint
        self.base_url = base_url or os.getenv("KAGGLE_API_URL")
        if not self.base_url:
            raise ValueError(
                "No base_url provided. Set KAGGLE_API_URL env var or pass "
                "base_url= explicitly (e.g. 'https://<tunnel>.trycloudflare.com/v1')."
            )

        self.model = model or os.getenv("KAGGLE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # vLLM doesn't validate the API key by default, but the SDK requires
        # a non-empty string to be passed in.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("KAGGLE_API_KEY", "not-needed"),
        )

    def generate(
        self,
        messages: list[Any],
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Call the Kaggle-hosted vLLM server and return the assistant text.
        Retries on rate-limit (429), server errors (500/502/503), and
        connection errors (tunnel restarted, notebook sleeping, etc).
        """
        chat_messages = [_to_openai_message(m) for m in messages]

        config_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # vLLM supports OpenAI-style structured JSON output via response_format
        if response_format and response_format.get("type") == "json_schema":
            config_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": response_format.get("schema", {}),
            }
        elif response_format and response_format.get("type") == "json_object":
            config_kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**config_kwargs)
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                is_server = any(code in err for code in ("500", "502", "503", "504"))
                is_conn = "connection" in err.lower() or "timeout" in err.lower()

                if attempt < self.max_retries - 1 and (is_rate or is_server or is_conn):
                    wait = self.retry_delay * (2 ** attempt) if is_rate else self.retry_delay
                    reason = "Rate limit" if is_rate else ("Server error" if is_server else "Connection issue")
                    print(f"  [Kaggle] {reason} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue

                raise RuntimeError(f"Kaggle call failed after {attempt+1} attempts: {e}") from e

        raise RuntimeError("Kaggle: max retries exhausted")