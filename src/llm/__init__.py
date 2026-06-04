"""LLM provider abstraction (Ollama / OpenAI / OpenAI-compatible) with BYOK support."""

from .provider import (
    LLMConfig,
    resolve_config,
    get_langchain_chat,
    get_openai_client,
    chat,
    chat_stream,
    embed,
    list_models,
)

__all__ = [
    "LLMConfig",
    "resolve_config",
    "get_langchain_chat",
    "get_openai_client",
    "chat",
    "chat_stream",
    "embed",
    "list_models",
]
