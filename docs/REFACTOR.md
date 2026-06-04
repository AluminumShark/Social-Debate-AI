# Refactor Plan

Goal: make the codebase **read well** (this is a portfolio/showcase — clean
structure is itself a deliverable) without changing behavior. Each phase is an
independent PR, ordered so a safety net exists before risky moves.

Principle: **no behavior change per phase.** Tests + one manual debate run are
the green light between phases.

---

## Current smells (evidence)
- `sys.path.insert(...)` path surgery in almost every module (provider, gpt_client,
  langgraph_orchestrator, debate_tools, vector_retriever, trainers, app, wsgi).
- **Dual orchestrator**: `parallel_orchestrator.py` + `langgraph_orchestrator.py`;
  state-update logic duplicated again in `ui/app.py:_run_langgraph_debate`.
- `ui/app.py` (~650 lines) mixes routing, business logic, and state reconstruction.
- **3 retrievers** (`simple_retriever`, `retriever`[Chroma/OpenAI], `vector_retriever`[FAISS])
  and **2 index builders** (`build_index`[Chroma], `scripts/build_rag_index`[FAISS]).
- **2 GNN trainers** (`train_supervised`[old 2-node + synthetic fallback] vs `train_graph`).
- Config overlap: `ConfigLoader` + `configs/*.yaml` + `.env`; `system.yaml` still
  references `gpt-3.5-turbo` / OpenAI key fields (stale vs the provider seam).

---

## Phase 0 — Safety net (do first)
- Add characterization tests around current behavior: provider `resolve_config`
  (env + BYOK merge), `vector_retriever` retrieve, `storage` round-trip,
  `stream_debate` event sequence (mock the LLM), API smoke (`/api/config`,
  `/api/demo`, a short stream).
- Document the "one manual run" check (`python ui/app.py` → one debate).
- **Risk: none.** Outcome: a green baseline to refactor against.

## Phase 1 — Make it an installable package (kills path hacks)
- Move `src/` to a proper package `social_debate/` (or keep `src/` but add a
  `[tool.setuptools] package-dir` mapping) and `pip install -e .`.
- Delete every `sys.path.insert` and switch to absolute imports
  (`from social_debate.llm import ...`).
- Update `wsgi.py`, `train_all.py`, `scripts/*` imports.
- **Risk: medium (import churn).** Biggest readability win; do it early.

## Phase 2 — One orchestrator, one source of state logic
- Make `LangGraphDebateOrchestrator` the single orchestrator.
- Delete `parallel_orchestrator.py` and the duplicated state math in
  `app.py:_run_langgraph_debate` / `_run_legacy_debate`; the app calls the
  orchestrator and serializes its output only.
- Keep step-mode only if used; otherwise drop `/api/debate_round` legacy path.
- **Risk: medium.** Removes the worst duplication.

## Phase 3 — Thin the Flask app (blueprints + service layer)
- Split `ui/app.py` into:
  - `ui/routes/` (debate, debate_share, system) Flask blueprints — thin.
  - `social_debate/service/debate_service.py` — orchestration + persistence glue.
- Routes parse/validate/stream; the service does the work.
- **Risk: low–medium.** Outcome: each file has one job.

## Phase 4 — Prune superseded modules
- RAG: keep `vector_retriever` (FAISS) as primary, `simple_retriever` as the
  explicit fallback; **remove** the Chroma/OpenAI `retriever.py` + `build_index.py`
  (or move under `legacy/` with a note).
- GNN: keep `train_graph.py`; **remove** `train_supervised.py` (old 2-node) or
  fold its synthetic-data path into a clearly-labeled fixture for tests.
- Remove `run_flask.py` / `setup.py` if redundant with `wsgi.py` / `pyproject`.
- **Risk: low.** Less code to read = better showcase.

## Phase 5 — One config source of truth
- Make `.env` (via `src/llm` + a small settings module) the source for runtime;
  keep `configs/*.yaml` only for debate/model hyperparameters actually read.
- Delete stale `system.yaml` API/OpenAI/gpt-3.5 fields; document each remaining key.
- **Risk: low.**

## Phase 6 — Polish for review
- Type hints on public functions; `ruff`/`black` clean; optional `mypy`.
- Docstrings that state intent (not restate code).
- Expand tests to new modules; ensure CI runs them.
- README: GIF/screenshot of streaming, live-demo link, ablation table front-and-center.
- **Risk: none.**

---

## Target structure (after)
```
social_debate/
  llm/            provider.py (the one LLM seam)
  orchestrator/   debate_orchestrator.py (single), state.py, tools.py
  rag/            vector_retriever.py (+ simple_retriever fallback)
  gnn/            model.py, train_graph.py, strategy_label.py
  rl/             policy.py, ppo_trainer.py
  storage/        debate_store.py
  service/        debate_service.py
  settings.py     (env/config single source)
ui/
  app.py          (app factory, ~40 lines)
  routes/         debate.py, share.py, system.py
  templates/ static/
scripts/          prepare_cmv.py, build_rag_index.py, eval_ab.py, make_demo.py
docker/  docs/  tests/
```

## Sequencing & effort (rough)
0 → 1 → 2 → 3 → 4 → 5 → 6. Phases 0–1 unblock everything; 2–3 remove the most
duplication; 4–5 shrink surface area; 6 is presentation. Each phase is small
enough to land and verify independently.

## Explicit non-goals
No new ML, no production ops (k8s/scaling), no framework swaps. This refactor is
about clarity and honesty of the existing system.
