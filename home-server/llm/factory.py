"""Build the active LLMClient from settings.

This is the single seam Streamlit Settings hits when the user switches
providers. Calling code never instantiates clients directly.
"""
from __future__ import annotations

from llm.base import LLMClient
from llm.gemini_client import GeminiClient
from llm.grok_client import GrokClient


def build_client(provider: str, *, xai_api_key: str = "", google_api_key: str = "") -> LLMClient:
    provider = provider.lower().strip()
    if provider == "grok":
        if not xai_api_key:
            raise ValueError("Grok provider selected but XAI_API_KEY is empty.")
        return GrokClient(api_key=xai_api_key)
    if provider == "gemini":
        if not google_api_key:
            raise ValueError("Gemini provider selected but GOOGLE_API_KEY is empty.")
        return GeminiClient(api_key=google_api_key)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Expected 'grok' or 'gemini'.")
