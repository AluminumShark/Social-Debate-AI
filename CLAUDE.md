# CLAUDE.md

Working guide for this repo. Full design rationale lives in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DECISIONS.md](docs/DECISIONS.md).

## What this is

Social Debate AI: a multi-agent debate system where three AI agents (support /
oppose / neutral) debate a topic. A LangGraph turn loop coordinates RL (strategy),
GNN (persuasion signal), RAG (evidence) and an LLM judge; the LLM generates the
arguments. Frontend is Flask + Bootstrap with token streaming over SSE.

Positioning: a learning / portfolio / engineering-skeleton showcase. Documentation
is written without emoji.

## Commands

```bash
uv sync                              # install deps
cp env.example .env                  # configure (defaults to local/LAN Ollama; no cloud key)
uv run python ui/app.py              # dev server  (http://localhost:5000)
gunicorn -k gthread -w 2 --threads 8 -t 0 -b 0.0.0.0:5000 wsgi:app   # prod
uv run pytest tests/ -q              # tests
uv run ruff check .                  # lint
python train_all.py --all           # data (CMV) -> FAISS -> graph GNN -> PPO
python scripts/eval_ab.py            # ablation A/B (writes docs/eval_results.md)
```

## Architecture (single orchestrator)

```
Browser (SSE, BYOK) -> Flask ui/app.py -> LangGraphDebateOrchestrator.stream_debate
  per turn: analyze (RL + GNN + RAG, ablation-toggleable) -> fuse
            -> stream tokens -> LLM judge -> update stance/conviction
  all LLM/embedding calls go through src/llm/provider.py (Ollama / OpenAI / BYOK)
  shared encoder: embeddinggemma 768-d (RAG + GNN + RL + judge)
```

Source map:
- `src/llm/provider.py` — the single LLM/embedding seam; `resolve_config()` merges
  env defaults with per-request BYOK overrides.
- `src/orchestrator/` — `langgraph_orchestrator.py` (the only orchestrator;
  `stream_debate` for SSE, `run_debate` for batch), `debate_state.py`, `debate_tools.py`.
- `src/rag/` — `vector_retriever.py` (FAISS, primary), `simple_retriever.py` (fallback).
- `src/gnn/` — `social_encoder.py` (model + inference), `train_graph.py`, `strategy_label.py`.
- `src/rl/` — `policy_network.py` (PPONetwork + inference), `ppo_trainer.py`, `train_ppo.py`.
- `src/storage/` — `debate_store.py` (SQLite, shareable ids).
- `ui/app.py` — Flask routes; `wsgi.py` — gunicorn entry.

## Conventions

- black + ruff, line length 100, target py310.
- Config: `configs/*.yaml` for hyperparameters; secrets only in `.env` (gitignored).
- Tests in `tests/unit` and `tests/integration`. Set `USE_LLM_JUDGE=false` to skip
  judge LLM calls in tests.

## Honest status (see docs/eval_results.md)

- RAG: working; ablation shows it helps (+5% evidence).
- LLM judge: working (keyword fallback).
- GNN / RL: experimental. The ablation shows no demonstrated debate-quality gain in
  this setup; that result is confounded with weak training (GNN collapses to the
  majority class under class imbalance). Kept as integration demonstrations.

## Gotchas

- Models/index live under `data/` (gitignored). Without them the app degrades
  gracefully (GNN -> neutral prior, RL -> keyword strategy, RAG -> empty/seed).
- Default LLM is a local/LAN Ollama via the OpenAI-compatible API; large models
  (qwen3.6, 26GB) spill to CPU on a 16GB GPU and are slow — prefer a model that
  fits VRAM, or set a small model for bulk jobs (labeling/eval).
- Do not commit `.env`. BYOK keys are never stored server-side.
