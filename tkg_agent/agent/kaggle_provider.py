"""
Kaggle / Remote LLM Provider
============================
Wraps a remote FastAPI inference endpoint into the same interface
used by SimuHome agents.

Compatible with:
- BaseReActAgent
- TKGReActAgent

Matches:
    provider.generate(messages, response_format=schema)

Designed for:
- Kaggle-hosted inference
- Cloudflare/ngrok endpoints
- self-hosted lightweight models
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# SimuHome ChatMessage compatibility
def _to_dict(msg: Any) -> dict:

    if isinstance(msg, dict):
        return msg

    return {
        "role": msg.role,
        "content": msg.content,
    }


class KaggleProvider:
    """
    Drop-in remote inference provider.

    Usage:
        llm = KaggleProvider()
        text = llm.generate(messages, response_format=schema)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 300,
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ):

        self.endpoint = (
            endpoint
            or os.getenv("KAGGLE_LLM_ENDPOINT")
        )

        if not self.endpoint:
            raise ValueError(
                "KAGGLE_LLM_ENDPOINT not configured"
            )

        self.endpoint = self.endpoint.rstrip("/")

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(
        self,
        messages: list[Any],
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Call remote inference endpoint and return assistant text.
        Retries on:
        - connection failures
        - timeout errors
        - 502/503 server issues
        """

        raw_messages = [
            _to_dict(m)
            for m in messages
        ]

        payload = {
            "messages": raw_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Preserve compatibility with existing interface
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(self.max_retries):

            try:

                response = requests.post(
                    f"{self.endpoint}/generate",
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

                if "response" not in data:
                    raise RuntimeError(
                        f"Malformed response: {data}"
                    )

                return data["response"].strip()

            except Exception as e:

                err = str(e).lower()

                is_timeout = (
                    "timeout" in err
                )

                is_server = (
                    "502" in err
                    or "503" in err
                    or "504" in err
                )

                is_connection = (
                    "connection" in err
                    or "refused" in err
                    or "remote end closed" in err
                )

                retryable = (
                    is_timeout
                    or is_server
                    or is_connection
                )

                if (
                    attempt < self.max_retries - 1
                    and retryable
                ):

                    wait = self.retry_delay * (2 ** attempt)

                    print(
                        f"  [KaggleProvider] "
                        f"Connection/server issue — retrying in {wait:.0f}s"
                    )

                    time.sleep(wait)

                    continue

                raise RuntimeError(
                    f"KaggleProvider failed after "
                    f"{attempt+1} attempts: {e}"
                ) from e

        raise RuntimeError(
            "KaggleProvider: max retries exhausted"
        )