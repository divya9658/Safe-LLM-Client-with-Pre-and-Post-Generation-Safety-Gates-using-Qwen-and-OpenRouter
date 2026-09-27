"""
OpenRouter Client
=================

Thin wrapper around OpenRouter's OpenAI-compatible Chat Completions
endpoint, pinned to a Qwen chat model. No prompts reach this module
unless they have already passed the Pre-Filter.
"""

import os
from typing import Any, Dict

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pinned model. Documented Qwen chat variant served via OpenRouter.
MODEL_ID = "qwen/qwen-2.5-7b-instruct"

DEFAULT_TIMEOUT_SECONDS = 30


class OpenRouterError(Exception):
    """Raised for any network, auth, or parsing failure calling OpenRouter."""


def call_qwen_model(prompt: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """
    Calls the Qwen model via OpenRouter's chat completions endpoint.

    Params:
        prompt (str): The safe (pre-filtered) user input.
        timeout (int): Request timeout in seconds.

    Returns:
        str: The model's text response.

    Raises:
        OpenRouterError: If the request fails, times out, returns a
            non-2xx status, or the response body is not in the expected
            shape. Callers should catch this rather than let the pipeline
            crash.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set in the environment. "
            "Copy .env.example to .env and provide a valid key."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Recommended (not strictly required) by OpenRouter so requests
        # are attributable in their dashboard/analytics.
        "HTTP-Referer": "https://github.com/example/safe-llm-gateway",
        "X-Title": "Safe LLM Gateway",
    }

    payload: Dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OpenRouterError(f"Network error calling OpenRouter: {exc}") from exc

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(
            f"Malformed response from OpenRouter: {exc}"
        ) from exc
