from __future__ import annotations
import os
from typing import Optional
from langchain_anthropic import ChatAnthropic

try:
    # ChatOllama lives in langchain-community
    # from langchain_community.chat_models import ChatOllama  # type: ignore
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None  # type: ignore


def get_default_model():
    """Return the default chat model for DeepAgents.

    Behavior is controlled via environment variables:
    - DEEPAGENTS_MODEL_PROVIDER: one of {"anthropic", "ollama"}. Defaults to "anthropic".
    - DEEPAGENTS_MODEL_NAME: overrides the model name for the selected provider.
    - DEEPAGENTS_MAX_TOKENS: max tokens for Anthropic (default: 64000).
    - DEEPAGENTS_OLLAMA_BASE_URL or OLLAMA_BASE_URL: base URL for Ollama (default: http://localhost:11434).
    """
    provider = os.getenv("DEEPAGENTS_MODEL_PROVIDER", "anthropic").lower()

    if provider in {"ollama", "local"}:
        if ChatOllama is None:
            raise ImportError(
                "Ollama selected but langchain-community is not installed.\n"
                "Please install it: pip install langchain-community"
            )

        model_name = os.getenv("DEEPAGENTS_MODEL_NAME", "llama3")
        base_url = (
            os.getenv("DEEPAGENTS_OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        )
        return ChatOllama(model=model_name, base_url=base_url)

    # Default to Anthropic
    model_name = os.getenv("DEEPAGENTS_MODEL_NAME", "claude-sonnet-4-20250514")
    max_tokens = int(os.getenv("DEEPAGENTS_MAX_TOKENS", "64000"))
    return ChatAnthropic(model_name=model_name, max_tokens=max_tokens)
