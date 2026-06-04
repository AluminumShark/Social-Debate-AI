"""
RL policy network + inference interface.

Single canonical Actor-Critic (PPONetwork, 768-d state = context embedding from
the same encoder used everywhere). `select_strategy` runs the *trained* policy
on the real debate context and falls back to a keyword heuristic only if no
model is available or an error occurs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Strategy mappings
STRATEGIES = ['aggressive', 'defensive', 'analytical', 'empathetic']
STRATEGY_TO_ID = {s: i for i, s in enumerate(STRATEGIES)}

MODEL_PATH = Path("data/models/ppo_policy.pt")
STATE_DIM = 768


class PPONetwork(nn.Module):
    """Actor-Critic policy used for BOTH training and inference."""

    def __init__(self, state_dim=STATE_DIM, action_dim=4, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        feats = self.shared(state)
        return self.actor(feats), self.critic(feats)

    def select_action(self, state):
        from torch.distributions import Categorical
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()


# Global model instance
_policy_model = None


def _load_model():
    global _policy_model
    if _policy_model is not None:
        return _policy_model
    try:
        if MODEL_PATH.exists():
            ckpt = torch.load(str(MODEL_PATH), map_location='cpu')
            cfg = ckpt.get('config', {})
            _policy_model = PPONetwork(
                state_dim=cfg.get('state_dim', STATE_DIM),
                action_dim=cfg.get('action_dim', 4),
                hidden_dim=cfg.get('hidden_dim', 256),
            )
            _policy_model.load_state_dict(ckpt['network_state_dict'])
            _policy_model.eval()
            print(f"[RL] Loaded trained policy from {MODEL_PATH}")
        else:
            _policy_model = "untrained"
            print("[RL] No trained policy found; using heuristic strategy selection")
    except Exception as e:  # noqa: BLE001
        print(f"[RL] Failed to load policy ({e}); using heuristic")
        _policy_model = "untrained"
    return _policy_model


def _keyword_strategy(query: str) -> str:
    q = (query or "").lower()
    if any(w in q for w in ['wrong', 'stupid', 'ridiculous', 'nonsense', 'absurd']):
        return 'aggressive'
    if any(w in q for w in ['research', 'study', 'data', 'evidence', 'statistics']):
        return 'analytical'
    if any(w in q for w in ['understand', 'feel', 'experience', 'perspective', 'concern']):
        return 'empathetic'
    return 'defensive'


def select_strategy(query, context="", social_context=None):
    """Pick a debate strategy.

    Uses the trained PPO policy on the real context embedding when available;
    otherwise falls back to a keyword heuristic.
    """
    model = _load_model()
    if model == "untrained":
        return _keyword_strategy(query)

    try:
        from llm import embed, resolve_config
        text = (context or query or "").strip()
        if not text:
            return _keyword_strategy(query)
        vec = embed([text[:2000]], resolve_config())[0]
        state = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs, _ = model(state)
        return STRATEGIES[int(torch.argmax(probs, dim=-1).item())]
    except Exception as e:  # noqa: BLE001
        print(f"[RL] policy inference failed ({e}); using heuristic")
        return _keyword_strategy(query)


def choose_snippet(state_text, pool):
    if not pool:
        return "No evidence available"
    state_words = set(state_text.lower().split())
    best_snippet, best_score = "", 0
    for item in pool:
        content = item.get('content', '')
        content_words = set(content.lower().split())
        overlap = len(state_words & content_words)
        score = overlap / max(len(state_words), 1)
        if score > best_score:
            best_score, best_snippet = score, content
    return best_snippet if best_snippet else pool[0].get('content', 'No evidence available')


def predict_quality(text):
    """Lightweight heuristic quality estimate (0..1).

    Response quality at debate time is scored by the LLM judge; this remains a
    cheap prior used by the RL tool wrapper.
    """
    words = text.split()
    length_score = min(len(words) / 50, 1.0) if len(words) < 100 else 0.8
    complexity_score = min(len(text.split('.')) / 5, 1.0)
    evidence_words = ['research', 'study', 'data', 'according', 'evidence']
    evidence_score = sum(1 for w in evidence_words if w in text.lower()) / 10
    return min(length_score * 0.4 + complexity_score * 0.3 + evidence_score * 0.3, 1.0)


PPONetwork.predict_quality = staticmethod(predict_quality)


def get_policy_network(model_path=None):
    return _load_model()
