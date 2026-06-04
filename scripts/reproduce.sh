#!/usr/bin/env bash
# Reproduce the Social Debate AI experiment end to end.
#
#   ./scripts/reproduce.sh           # light: deps -> RAG index (seed/CMV) -> ablation
#   ./scripts/reproduce.sh full      # full: download CMV + train GNN/RL + ablation (GPU recommended)
#
# Requires Python 3.10+ and an LLM backend reachable via .env
# (LLM_BASE_URL / LLM_API_KEY; default is a local/LAN Ollama).
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-light}"
PY="${PY:-python}"

echo "==> [1/4] Setup"
uv sync 2>/dev/null || pip install -e ".[dev]"
[ -f .env ] || { cp env.example .env; echo "    created .env (edit LLM_BASE_URL/LLM_API_KEY for your backend)"; }

if [ "$MODE" = "full" ]; then
  echo "==> [2/4] Download + clean CMV, then train GNN/RL"
  "$PY" train_all.py --all
else
  echo "==> [2/4] Build FAISS index (CMV pairs if present, else seed evidence)"
  "$PY" scripts/build_rag_index.py
fi

echo "==> [3/4] Tests"
USE_LLM_JUDGE=false "$PY" -m pytest tests/ -q || echo "    (tests reported issues)"

echo "==> [4/4] Ablation experiment"
"$PY" scripts/eval_ab.py

echo "==> Done. Results: docs/eval_results.md"
echo "    Start the app with:  make run   (or: python ui/app.py)"
