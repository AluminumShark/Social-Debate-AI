"""
Generate demo/sample_debate.json by running one real debate.

Run:  python scripts/make_demo.py [topic]
Uses the env LLM config (defaults to Ollama). Pick a fast model via
LLM_MODEL for quicker generation, e.g. LLM_MODEL=gemma3:12b.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator  # noqa: E402


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else (
        "Should governments provide a universal basic income?"
    )
    agent_configs = [
        {"id": "Agent_A", "initial_stance": 0.8, "initial_conviction": 0.7},
        {"id": "Agent_B", "initial_stance": -0.6, "initial_conviction": 0.7},
        {"id": "Agent_C", "initial_stance": 0.0, "initial_conviction": 0.7},
    ]

    orch = create_langgraph_orchestrator()

    rounds = {}
    summary = {}
    for ev in orch.stream_debate(topic, agent_configs, max_rounds=2):
        if ev["type"] == "turn_end":
            rd = rounds.setdefault(ev["round"], {"round": ev["round"], "responses": [], "agents": {}})
            rd["responses"].append({
                "agent_id": ev["agent"],
                "content": ev["content"],
                "effects": ev.get("effects", {}),
            })
            rd["agents"] = ev.get("agent_states", {})
        elif ev["type"] == "summary":
            summary = ev["summary"]

    demo = {
        "topic": topic,
        "rounds": [rounds[k] for k in sorted(rounds)],
        "summary": summary,
        "orchestrator": "langgraph",
        "demo": True,
    }

    out = ROOT / "demo" / "sample_debate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(demo, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out} ({len(demo['rounds'])} rounds)")


if __name__ == "__main__":
    main()
