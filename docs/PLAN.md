# Roadmap

Where this project came from, what is shipped (v1), and what is intentionally
left for later (v2). For the honest evaluation of the ML modules see
[eval_results.md](eval_results.md); for trade-offs see [DECISIONS.md](DECISIONS.md).

## Starting point (problems that were fixed)

The initial version oversold its ML. Concretely:

- GNN inference was fed random noise instead of real text features.
- RAG was keyword overlap, not the advertised vector search.
- Response scoring was keyword counting.
- The trained RL policy was never connected to inference.
- Training silently fell back to synthetic random data when the corpus was absent.
- The default model was hardcoded and outdated; the debate ran fully blocking
  (no streaming); two orchestrators duplicated state logic.

## v1 (shipped)

- Token-streaming debates over Server-Sent Events.
- Local-first LLM (OpenAI-compatible) with Bring-Your-Own-Key in the browser;
  keys are never stored server-side. A no-key demo mode replays a recorded debate.
- Real FAISS vector RAG over the ChangeMyView corpus.
- LLM-as-a-judge scoring (keyword fallback).
- One LLM/embedding seam; one shared embedding space across RAG/GNN/RL/judge.
- Single LangGraph orchestrator; SQLite persistence with shareable links.
- Ops: production WSGI entry, app + training Docker images, per-IP rate limiting,
  debug off, CI, tests.
- Graceful degradation when models/keys are absent.
- Honest docs: no research-grade claims; module value reported via an ablation.

## v2 (backlog)

1. Replace heuristic-rooted strategy labels with outcome-derived labels (fixes the
   supervision source shared by GNN and RL).
2. Give the GNN a genuinely graph-shaped inference task, or rename it to a
   persuasion classifier.
3. Train RL from real debate outcomes (judge / delta deltas) rather than imitating
   the GNN.
4. Use LangGraph's checkpointer on the streaming path, or simplify it away.
5. Cache per-turn embeddings to cut latency.

## Guiding principle

Accessibility and honesty over feature count. Each change keeps the app runnable
and is verified by tests plus a manual debate run. The ablation result decides
whether the ML modules graduate from experimental or stay labeled as such.
