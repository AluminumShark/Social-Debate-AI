"""
Unified LLM provider abstraction.

Goals:
- One place to configure the chat/embedding backend (Ollama by default, any
  OpenAI-compatible endpoint, or OpenAI itself).
- Bring-Your-Own-Key (BYOK): callers may pass per-request overrides (model /
  base_url / api_key) coming from the frontend. These are used only for that
  request and are never persisted or logged.

The default config is read from environment variables (.env). See env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "ollama"            # ollama | openai | openai-compatible
    model: str = "qwen3.6:latest"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    embedding_model: str = "embeddinggemma:latest"
    temperature: float = 0.7
    max_tokens: int = 512

    def redacted(self) -> Dict[str, Any]:
        """Safe representation for logging (never expose the key)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else "",
            "embedding_model": self.embedding_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def _default_config() -> LLMConfig:
    return LLMConfig(
        provider=_env("LLM_PROVIDER", "ollama") or "ollama",
        model=_env("LLM_MODEL", "qwen3.6:latest") or "qwen3.6:latest",
        base_url=_env("LLM_BASE_URL", "http://localhost:11434/v1")
        or "http://localhost:11434/v1",
        api_key=_env("LLM_API_KEY", "ollama") or "ollama",
        embedding_model=_env("LLM_EMBEDDING_MODEL", "embeddinggemma:latest")
        or "embeddinggemma:latest",
        temperature=float(_env("LLM_TEMPERATURE", "0.7") or 0.7),
        max_tokens=int(_env("LLM_MAX_TOKENS", "512") or 512),
    )


def _byok_allowed() -> bool:
    return _env("ALLOW_BYOK", "true").lower() in ("1", "true", "yes")


# Keys a frontend request is allowed to override.
_OVERRIDABLE = {"provider", "model", "base_url", "api_key", "embedding_model",
                "temperature", "max_tokens"}


def resolve_config(overrides: Optional[Dict[str, Any]] = None) -> LLMConfig:
    """
    Merge env defaults with per-request (BYOK) overrides.

    `overrides` typically comes from the frontend JSON body, e.g.
    {"model": "gpt-5.5", "base_url": "https://api.openai.com/v1",
     "api_key": "sk-...", "provider": "openai"}.
    Only known fields are honored, and only if ALLOW_BYOK is enabled.
    """
    cfg = _default_config()
    if not overrides or not _byok_allowed():
        return cfg

    clean: Dict[str, Any] = {}
    for k, v in overrides.items():
        if k not in _OVERRIDABLE or v in (None, ""):
            continue
        if k == "temperature":
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        elif k == "max_tokens":
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        clean[k] = v
    return replace(cfg, **clean) if clean else cfg


# --------------------------------------------------------------------------- #
# Clients
# --------------------------------------------------------------------------- #
def get_openai_client(config: LLMConfig):
    """Return a raw OpenAI SDK client pointed at the configured endpoint."""
    from openai import OpenAI

    # Ollama/openai-compatible endpoints require *some* api_key string.
    return OpenAI(api_key=config.api_key or "not-needed", base_url=config.base_url)


def get_langchain_chat(config: LLMConfig, streaming: bool = False):
    """Return a langchain ChatOpenAI bound to the configured endpoint."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        base_url=config.base_url,
        api_key=config.api_key or "not-needed",
        streaming=streaming,
    )


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def chat(messages: List[Dict[str, str]], config: Optional[LLMConfig] = None) -> str:
    """Blocking chat completion. `messages` is OpenAI-style role/content dicts."""
    config = config or _default_config()
    client = get_openai_client(config)
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def chat_stream(
    messages: List[Dict[str, str]], config: Optional[LLMConfig] = None
) -> Iterator[str]:
    """Yield content deltas token-by-token for streaming UIs."""
    config = config or _default_config()
    client = get_openai_client(config)
    stream = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


def embed(texts: List[str], config: Optional[LLMConfig] = None) -> List[List[float]]:
    """Return embedding vectors for a list of texts."""
    config = config or _default_config()
    client = get_openai_client(config)
    resp = client.embeddings.create(model=config.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def list_models(config: Optional[LLMConfig] = None) -> List[str]:
    """List model ids available at the endpoint (best-effort)."""
    config = config or _default_config()
    try:
        client = get_openai_client(config)
        return [m.id for m in client.models.list().data]
    except Exception:
        return []
