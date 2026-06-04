# Design Decisions & Trade-offs

This document records the engineering decisions behind Social Debate AI, why they
were made, and — honestly — where the limits are. It is written for readers
evaluating the project as an engineering showcase.

## 1. One LLM seam instead of scattered SDK calls
**Decision:** all chat/embedding traffic goes through `src/llm/provider.py`
(`resolve_config → chat / chat_stream / embed`).
**Why:** swapping Ollama ↔ OpenAI ↔ a user-supplied (BYOK) endpoint becomes a
config change, and per-request overrides are trivial.
**Trade-off:** a thin indirection layer; the payoff is that nothing else in the
codebase imports an LLM SDK directly.

## 2. One shared embedding space (embeddinggemma, 768-d)
**Decision:** RAG, GNN, RL and the judge all embed text with the same model.
**Why:** the original code trained the GNN on distilbert features but ran
inference on different vectors — a silent representation mismatch. Unifying the
encoder removes that class of bug.
**Trade-off:** the whole system is coupled to one small embedding model; if it is
weak, every module is weak. Acceptable for a showcase; for production you would
benchmark several encoders.

## 3. Local-first + Bring-Your-Own-Key
**Decision:** default to a local/LAN Ollama; let users paste their own key in the
browser (kept client-side, never stored/logged server-side); offer a no-key demo.
**Why:** zero-friction trial is the single biggest driver of "people actually try
it." An API-key wall kills that.
**Trade-off:** BYOK shifts trust to the client; we mitigate by never persisting keys.

## 4. Token streaming via a hand-rolled loop
**Decision:** `stream_debate()` drives the turn nodes directly and streams tokens
over SSE, rather than streaming through the compiled LangGraph.
**Why:** simplest path to a token-level UX.
**Trade-off (honest):** this means LangGraph's headline feature — checkpointing /
human-in-the-loop — is **not** exploited. LangGraph is currently closer to a
state container than a load-bearing engine. A principled v2 would either use the
compiled graph with a checkpointer or drop the dependency.

## 5. Ablation switches + an A/B harness
**Decision:** `USE_RL/USE_GNN/USE_RAG` toggles, plus `scripts/eval_ab.py`
comparing `LLM-only → +RAG → +GNN → full`.
**Why:** the central honest question is "do the ML modules actually improve
debates?" The architecture is built so that question can be *measured*, not
assumed. Results are reported even when a module shows little gain.

## 6. Honest take on the ML modules
Applied with one consistent statistical ruler: at the ablation's scale (3 topics ×
2 rounds, one small judge) *no* module reaches significance, so the take below is
about evidence *strength*, not proof.
- **RAG** is the best-supported module: it has the clearest mechanism (real semantic
  retrieval of CMV evidence) and the only directional signal on the metric it should
  affect (evidence, +0.039). That is "consistent with helping," not "proven" — and we
  resist calling it a confirmed win while calling GNN/RL inconclusive on the same data.
- **GNN** is a node-classification model over real reply trees. At inference it
  scores a single context; calling it a "graph" neural net is accurate for training
  but thin at serving time. In practice its training collapsed to the majority class
  (class imbalance), so it never produced a trustworthy persuasion signal — its
  ablation non-result is uninformative rather than a clean negative.
- **RL/PPO** selects a strategy label. Its reward is derived from the GNN's
  suitability signal — an improvement over the original (a synthetic environment with
  constant rewards, never even connected to inference), but the deeper problem is
  structural: because the GNN signal is broken, the policy faithfully optimizes a
  mis-measured target. Stacking RL on a broken estimator can't win by construction;
  this is the project's clearest worked example of the "no clean reward signal" trap.
- The LLM does the overwhelming majority of debate quality. The ML modules are
  diagnostic demonstrations whose marginal value — and, for GNN/RL, whose instructive
  failure — is reported empirically.

## 7. Offline/online split with graceful degradation
**Decision:** training runs in a CUDA Docker image on a GPU box; the web app
loads artifacts and falls back to neutral/keyword behavior when models are absent.
**Why:** the app must run for anyone (no GPU, no trained models) while still
benefiting from training when available.

## What would change for a "real" v3
- Replace strategy supervision rooted in heuristics with outcome-derived labels.
- Give the GNN a genuinely graph-shaped inference task (or rename it honestly).
- Train RL from real debate outcomes (judge/delta deltas), not GNN imitation.
- Use LangGraph's checkpointer or remove it; consolidate the legacy orchestrator.
- Add persistence (SQLite) + shareable debate links; cache per-turn embeddings.

## Security notes (known, by design for a showcase)

- **BYOK is server-side request forgery (SSRF)-adjacent.** A request may override
  `base_url`, and the server then makes outbound calls to it. That is intended
  (users point at their own endpoint), but on an untrusted public deployment you
  should allowlist `base_url` (or disable the override). Keys are used per-request
  and never stored or logged.
- **BYOK covers generation only.** Embeddings (RAG/GNN context) and the LLM judge
  use the server's default backend, not the per-request key — so the host still
  bears that cost. Routing every call through the per-request config is a v2 item.
- **Rate limiting is per-process and in-memory.** With multiple gunicorn workers the
  effective limit is per-worker; use a shared store (e.g. Redis) for a real limit.

## What was deliberately *not* done
Production ops (k8s, autoscaling), squeezing model accuracy, and multi-tenant
concerns — out of scope for a portfolio/showcase whose value is clean seams,
honesty, and measurability.
