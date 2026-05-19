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
    """
    Calls the FastAPI /generate endpoint running in your Kaggle notebook.
    Interface is identical to GroqProvider — swap with one import change.
    """

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
