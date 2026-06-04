"""
LLM-based debate-strategy labeling (replaces the old keyword heuristic).

Labels a piece of argument text as one of four rhetorical strategies. Used to
supervise the GNN strategy head with real labels instead of keyword counts.
Falls back to a keyword guess if the LLM is unavailable.
"""

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

STRATEGIES = ["aggressive", "defensive", "analytical", "empathetic"]
STRATEGY_TO_ID = {s: i for i, s in enumerate(STRATEGIES)}

_PROMPT = """Classify the dominant rhetorical strategy of this debate argument as \
exactly one of: aggressive, defensive, analytical, empathetic.
- aggressive: attacks/refutes the opponent, points out flaws
- defensive: restates/protects its own position
- analytical: logic, data, evidence, structured reasoning
- empathetic: acknowledges feelings, finds common ground

Reply with ONLY the single word.

Argument:
\"\"\"%s\"\"\""""

_cache = {}


def _keyword_label(text: str) -> str:
    t = text.lower()
    scores = {
        "aggressive": sum(w in t for w in ("wrong", "flawed", "incorrect", "fallacy")),
        "defensive": sum(w in t for w in ("maintain", "still believe", "my point", "defend")),
        "analytical": sum(w in t for w in ("data", "study", "research", "evidence", "because")),
        "empathetic": sum(w in t for w in ("understand", "feel", "agree", "appreciate")),
    }
    return max(scores, key=scores.get) if any(scores.values()) else "analytical"


def label_strategy(text: str, use_llm: bool = True) -> str:
    text = (text or "").strip()
    if not text:
        return "analytical"
    key = text[:200]
    if key in _cache:
        return _cache[key]

    label = None
    if use_llm:
        try:
            from llm import chat, resolve_config
            raw = chat([{"role": "user", "content": _PROMPT % text[:1500]}], resolve_config())
            m = re.search(r"aggressive|defensive|analytical|empathetic", raw.lower())
            if m:
                label = m.group(0)
        except Exception:  # noqa: BLE001
            label = None
    if label is None:
        label = _keyword_label(text)

    _cache[key] = label
    return label


def label_id(text: str, use_llm: bool = True) -> int:
    return STRATEGY_TO_ID[label_strategy(text, use_llm)]
