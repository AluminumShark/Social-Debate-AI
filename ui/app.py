"""
Flask web app for Social Debate AI.

Single orchestrator (LangGraph). Streams debates as Server-Sent Events, supports
Bring-Your-Own-Key, persists debates to SQLite with shareable links, and serves a
no-key demo. All LLM/embedding traffic flows through src/llm (the provider seam).
"""

import os
import sys
import json
import time
import threading
from collections import deque
from pathlib import Path

from flask import (Flask, render_template, request, jsonify, Response,
                   stream_with_context)

# Make project root and src/ importable.
_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator  # noqa: E402
from utils.config_loader import ConfigLoader  # noqa: E402
from llm import resolve_config, list_models  # noqa: E402
from storage import save_debate, get_debate, list_debates  # noqa: E402

app = Flask(__name__)

orchestrator = None
config = None

# --- simple per-IP rate limiter (protects the shared default LLM backend) ---
_rl_lock = threading.Lock()
_rl_hits = {}


def _rate_limited(ip: str) -> bool:
    limit = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))
    if limit <= 0:
        return False
    now = time.time()
    with _rl_lock:
        dq = _rl_hits.setdefault(ip, deque())
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= limit:
            return True
        dq.append(now)
    return False


def initialize_system() -> bool:
    """Load config + build the (single) LangGraph orchestrator."""
    global orchestrator, config
    try:
        config = ConfigLoader.load("debate")
        cfg = resolve_config()
        print(f"[init] LLM: {cfg.provider}/{cfg.model} @ {cfg.base_url}")
        orchestrator = create_langgraph_orchestrator()
        print("[init] orchestrator ready")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[init] failed: {e}")
        return False


def _agent_configs():
    """Agent configs from debate.yaml, with sensible defaults."""
    defaults = [
        {"id": "Agent_A", "initial_stance": 0.8, "initial_conviction": 0.7},
        {"id": "Agent_B", "initial_stance": -0.6, "initial_conviction": 0.7},
        {"id": "Agent_C", "initial_stance": 0.0, "initial_conviction": 0.7},
    ]
    try:
        d = (config or {}).get("debate", {})
        names = d.get("agents") or [c["id"] for c in defaults]
        acfg = d.get("agent_configs", {})
        out = [{
            "id": n,
            "initial_stance": acfg.get(n, {}).get("initial_stance", 0.0),
            "initial_conviction": acfg.get(n, {}).get("initial_conviction", 0.7),
        } for n in names]
        return out or defaults
    except Exception:  # noqa: BLE001
        return defaults


def _byok_overrides(data: dict) -> dict:
    """Per-request LLM overrides from the frontend; used only for this request."""
    llm = (data or {}).get("llm") or {}
    if not isinstance(llm, dict):
        return {}
    allowed = {"provider", "model", "base_url", "api_key", "temperature", "max_tokens"}
    return {k: v for k, v in llm.items() if k in allowed and v not in (None, "")}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html", title="Social Debate AI",
                           description="AI-powered debate simulation")


@app.route("/d/<debate_id>")
def shared_debate(debate_id):
    """Render the app; the frontend loads and replays the stored debate."""
    return render_template("index.html", title="Social Debate AI",
                           description="Shared debate", shared_id=debate_id)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.route("/api/config")
def api_config():
    cfg = resolve_config()
    byok = os.environ.get("ALLOW_BYOK", "true").lower() in ("1", "true", "yes")
    try:
        available = list_models(cfg)
    except Exception:  # noqa: BLE001
        available = []
    return jsonify({"byok_allowed": byok, "provider": cfg.provider,
                    "default_model": cfg.model, "base_url": cfg.base_url,
                    "available_models": available})


@app.route("/api/debate/stream", methods=["POST"])
def debate_stream():
    """Stream a debate as Server-Sent Events (token-level), then persist it."""
    if _rate_limited(request.remote_addr or "unknown"):
        return jsonify({"error": "Rate limit exceeded, please slow down"}), 429
    if not orchestrator:
        return jsonify({"error": "System not initialized"}), 503

    data = request.get_json(silent=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400
    topic = topic[:300]

    max_rounds = (config or {}).get("debate", {}).get("max_rounds", 5)
    try:
        max_rounds = int(data.get("max_rounds", max_rounds))
    except (TypeError, ValueError):
        pass
    max_rounds = max(1, min(max_rounds, 10))

    llm_cfg = resolve_config(_byok_overrides(data))

    agents = _agent_configs()
    req_agents = data.get("agents")
    if isinstance(req_agents, list) and req_agents:
        clean = []
        for i, a in enumerate(req_agents[:6]):
            if isinstance(a, dict):
                clean.append({
                    "id": str(a.get("id") or f"Agent_{chr(65 + i)}"),
                    "initial_stance": float(a.get("initial_stance", 0.0)),
                    "initial_conviction": float(a.get("initial_conviction", 0.7)),
                })
        if clean:
            agents = clean

    def event_stream():
        rounds_map, summary = {}, {}
        try:
            for ev in orchestrator.stream_debate(topic, agents,
                                                 max_rounds=max_rounds, llm_config=llm_cfg):
                if ev["type"] == "turn_end":
                    rd = rounds_map.setdefault(ev["round"],
                                               {"round": ev["round"], "responses": [], "agents": {}})
                    rd["responses"].append({"agent_id": ev["agent"], "content": ev["content"],
                                            "effects": ev.get("effects", {})})
                    rd["agents"] = ev.get("agent_states", {})
                elif ev["type"] == "summary":
                    summary = ev["summary"]
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            try:
                did = save_debate(topic, [rounds_map[k] for k in sorted(rounds_map)], summary)
                yield f"data: {json.dumps({'type': 'saved', 'id': did})}\n\n"
            except Exception as e:  # noqa: BLE001
                print(f"[store] save failed: {e}")
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/demo")
def api_demo():
    demo_path = _ROOT / "demo" / "sample_debate.json"
    if demo_path.exists():
        with open(demo_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Demo data not available"}), 404


@app.route("/api/debate/<debate_id>")
def api_get_debate(debate_id):
    data = get_debate(debate_id)
    return jsonify(data) if data else (jsonify({"error": "Not found"}), 404)


@app.route("/api/debates")
def api_recent():
    return jsonify({"debates": list_debates(20)})


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "ready": orchestrator is not None,
                    "orchestrator": "langgraph", "timestamp": time.time()})


@app.route("/api/graph")
def api_graph():
    if orchestrator:
        return jsonify({"success": True, "graph": orchestrator.get_graph_visualization()})
    return jsonify({"success": False, "message": "Not initialized"}), 503


# Lightweight no-ops kept for frontend compatibility (client manages its own UI state).
@app.route("/api/init", methods=["POST"])
def api_init():
    ready = orchestrator is not None or initialize_system()
    return jsonify({"success": bool(ready), "orchestrator_type": "langgraph"})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    return jsonify({"success": True})


@app.route("/api/export")
def api_export():
    return jsonify({"success": True, "data": {"debates": list_debates(50)}})


if __name__ == "__main__":
    if initialize_system():
        debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
        host = os.environ.get("FLASK_HOST", "0.0.0.0")
        port = int(os.environ.get("FLASK_PORT", "5000"))
        print(f"Server ready at http://localhost:{port} (debug={debug})")
        app.run(debug=debug, host=host, port=port)
    else:
        print("Initialization failed")
        sys.exit(1)
