"""
GPT/LLM client interface.

Thin backward-compatible wrapper over `src/llm` (provider abstraction).
Defaults come from environment variables (.env) — by default a local/LAN
Ollama exposing an OpenAI-compatible API. See env.example.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

# Allow `from llm import ...` when imported from project root or src/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import LLMConfig, resolve_config, chat as _chat, chat_stream as _chat_stream  # noqa: E402

# Resolved default config (provider/model/base_url/key from env)
_DEFAULT = resolve_config()
DEFAULT_MODEL = _DEFAULT.model
DEFAULT_MAX_TOKENS = _DEFAULT.max_tokens
DEFAULT_TEMPERATURE = _DEFAULT.temperature

print(f"[LLM] Default backend: {_DEFAULT.provider}/{_DEFAULT.model} @ {_DEFAULT.base_url}")


def _config_for(
    model: Optional[str], max_tokens: Optional[int], temperature: Optional[float]
) -> LLMConfig:
    from dataclasses import replace

    overrides = {}
    if model:
        overrides["model"] = model
    if max_tokens:
        overrides["max_tokens"] = max_tokens
    if temperature is not None:
        overrides["temperature"] = temperature
    return replace(_DEFAULT, **overrides) if overrides else _DEFAULT


def chat(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Single-prompt chat completion against the configured LLM backend.

    Kept for backward compatibility with the legacy orchestrator. New code
    should use `src.llm.chat(messages, config)` directly.
    """
    cfg = _config_for(model, max_tokens, temperature)
    try:
        return _chat([{"role": "user", "content": prompt}], cfg)
    except Exception as e:  # noqa: BLE001
        print(f"LLM call failed: {e}")
        return "I understand your point. Let me think about this issue from a different perspective."


def chat_messages(messages: List[Dict[str, str]], config: Optional[LLMConfig] = None) -> str:
    """Multi-message chat completion (role/content dicts)."""
    try:
        return _chat(messages, config or _DEFAULT)
    except Exception as e:  # noqa: BLE001
        print(f"LLM call failed: {e}")
        return "I understand your point. Let me think about this issue from a different perspective."


def chat_stream(messages: List[Dict[str, str]], config: Optional[LLMConfig] = None):
    """Streaming chat completion (yields content deltas)."""
    return _chat_stream(messages, config or _DEFAULT)
