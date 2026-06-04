"""
A/B ablation evaluation: does each ML module actually improve debates?

Runs the same topics under increasing module sets and compares the LLM judge's
per-turn scores. This is the honest test for "are GNN/RL/RAG worth it?".

Configs:
  LLM-only   : RAG off, GNN off, RL off
  +RAG       : RAG on
  +RAG+GNN   : RAG + GNN
  full       : RAG + GNN + RL

Metric: mean per-turn judge persuasion / evidence / attack (already computed by
the judge during each debate — no extra LLM calls), averaged over topics.

Usage:
  python scripts/eval_ab.py                       # default topics, 2 rounds
  LLM_MODEL=gemma3:4b python scripts/eval_ab.py   # faster model
"""

import argparse
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TOPICS = [
    "Should governments provide a universal basic income?",
    "Should social media platforms be legally liable for user content?",
    "Should cities ban private cars from downtown areas?",
    "Should university education be free for all citizens?",
]

CONFIGS = {
    "LLM-only": {"USE_RAG": "false", "USE_GNN": "false", "USE_RL": "false"},
    "+RAG":     {"USE_RAG": "true",  "USE_GNN": "false", "USE_RL": "false"},
    "+RAG+GNN": {"USE_RAG": "true",  "USE_GNN": "true",  "USE_RL": "false"},
    "full":     {"USE_RAG": "true",  "USE_GNN": "true",  "USE_RL": "true"},
}

AGENTS = [
    {"id": "Agent_A", "initial_stance": 0.8, "initial_conviction": 0.7},
    {"id": "Agent_B", "initial_stance": -0.6, "initial_conviction": 0.7},
    {"id": "Agent_C", "initial_stance": 0.0, "initial_conviction": 0.7},
]


def run_one(orch, topic, rounds):
    persuasion, evidence, attack = [], [], []
    for ev in orch.stream_debate(topic, AGENTS, max_rounds=rounds):
        if ev["type"] == "turn_end":
            e = ev.get("effects", {})
            persuasion.append(e.get("persuasion_score", 0.0))
            evidence.append(e.get("evidence_score", 0.0))
            attack.append(e.get("attack_score", 0.0))
    return {
        "persuasion": mean(persuasion) if persuasion else 0.0,
        "evidence": mean(evidence) if evidence else 0.0,
        "attack": mean(attack) if attack else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--topics", type=int, default=len(TOPICS))
    ap.add_argument("--out", default="docs/eval_results.md")
    args = ap.parse_args()

    from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator

    topics = TOPICS[: args.topics]
    results = {}
    for cfg_name, flags in CONFIGS.items():
        os.environ.update(flags)
        orch = create_langgraph_orchestrator()  # picks up flags per-turn
        agg = {"persuasion": [], "evidence": [], "attack": []}
        for t in topics:
            print(f"[eval] {cfg_name} :: {t[:50]}...")
            r = run_one(orch, t, args.rounds)
            for k in agg:
                agg[k].append(r[k])
        results[cfg_name] = {k: mean(v) for k, v in agg.items()}

    # Render markdown table
    lines = [
        "# Ablation results (A/B evaluation)",
        "",
        f"Topics: {len(topics)} · rounds: {args.rounds} · "
        f"judge: LLM per-turn scores (higher = better).",
        "",
        "| Config | Persuasion | Evidence | Attack |",
        "|--------|-----------:|---------:|-------:|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['persuasion']:.3f} | {m['evidence']:.3f} | {m['attack']:.3f} |")
    base = results.get("LLM-only", {})
    full = results.get("full", {})
    if base and full:
        dp = (full["persuasion"] - base["persuasion"])
        de = (full["evidence"] - base["evidence"])
        lines += [
            "",
            f"**full vs LLM-only:** persuasion {dp:+.3f}, evidence {de:+.3f}.",
            "",
            "_Interpretation: if the deltas are ~0, the ML modules add little to "
            "debate quality and should be treated as experimental; if clearly "
            "positive, they earn their place._",
        ]
    out = "\n".join(lines)
    print("\n" + out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out, encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
