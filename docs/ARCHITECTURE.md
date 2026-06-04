# Architecture

Social Debate AI is a multi-agent debate system that combines a frontier LLM
(for argument generation) with trained ML modules (RAG / GNN / RL) and a
LangGraph-style turn loop. It is built as a **portfolio / engineering-skeleton
showcase**: the emphasis is clean seams, honest measurement, and reproducibility.
Its real subject is **measurement and diagnosis** — the ablation harness exists to
show *whether* each module helps and, where one doesn't, *why* (see
[Honest status](#honest-status) and [eval_results](eval_results.md)).

## System overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Frontend  (templates/index.html + static/js/modern-app.js)             │
│    topic input · live token streaming · BYOK key (localStorage) · demo     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ HTTP / Server-Sent Events
┌───────────────────────────────▼──────────────────────────────────────────┐
│ 2. Service   ui/app.py   (gunicorn ← wsgi.py)                              │
│    /api/debate/stream (SSE) · /api/config · /api/demo · /api/health        │
│    per-IP rate limit · debug off · CORS                                    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │ stream_debate(topic, agents, llm_cfg)
┌───────────────────────────────▼──────────────────────────────────────────┐
│ 3. Orchestration   orchestrator/langgraph_orchestrator.py                  │
│    per turn:                                                                │
│      ① analyze (parallel): RL strategy · GNN persuasion · RAG evidence     │
│         └─ ablation switches: USE_RL / USE_GNN / USE_RAG                    │
│      ② fuse  ③ generate (token streaming)  ④ judge (LLM)  ⑤ update state    │
└──────┬───────────────┬───────────────┬───────────────┬───────────────────┘
       │               │               │               │
┌──────▼─────┐  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼───────────────────┐
│ RL policy  │  │ GNN          │ │ RAG          │ │ Judge                    │
│ rl/policy_ │  │ gnn/social_  │ │ rag/vector_  │ │ orchestrator/debate_     │
│ network.py │  │ encoder.py   │ │ retriever.py │ │ tools.evaluate_response  │
│ ppo_policy │  │ gnn_persua-  │ │ FAISS        │ │ _effects (LLM judge)     │
│ .pt        │  │ sion.pt      │ │ (CMV index)  │ │                          │
└──────┬─────┘  └──────┬──────┘ └──────┬──────┘ └──────┬───────────────────┘
       └───────────────┴───────────────┴───────────────┘
                                 │  single exit
┌───────────────────────────────▼──────────────────────────────────────────┐
│ 4. LLM seam   src/llm/provider.py   ← the ONE place LLM/embeddings flow     │
│    resolve_config(env defaults + BYOK overrides) → chat/chat_stream/embed   │
└──────┬───────────────────┬────────────────────┬──────────────────────────┘
       ▼                   ▼                     ▼
   Ollama (default)    OpenAI-compatible      BYOK endpoint (user's key)
   qwen3 chat · embeddinggemma 768-d  ◄══ one shared vector space ══►
                           (RAG / GNN / RL / judge all use it)

┌──────────────────────────────────────────────────────────────────────────┐
│ 5. Offline training   (Docker GPU; CUDA 12.8 / cu128)   train_all.py --all │
│   prepare_cmv.py ─► data/raw/pairs.jsonl + threads.jsonl (reply trees)      │
│        ├─► scripts/build_rag_index.py ───────► FAISS (data/rag/faiss)       │
│        ├─► gnn/train_graph.py ───────────────► gnn_persuasion.pt            │
│        │      real conversation graphs + LLM strategy labels                │
│        └─► rl/train_ppo.py ──────────────────► ppo_policy.pt                │
│               state = real comment embeddings; reward = real delta + GNN    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key design decisions

1. **One LLM seam (`src/llm/provider.py`).** Every chat/embedding call goes
   through it. Swapping Ollama ↔ OpenAI ↔ a user's BYOK key is a config change,
   not a code change.
2. **One shared embedding space (`embeddinggemma`, 768-d).** RAG, GNN, RL and
   the judge all consume the same vectors, so training and inference use the
   same encoder — no representation mismatch.
3. **Local-first + BYOK.** Defaults to a local/LAN Ollama (no cloud key). Users
   can paste their own key in the browser; it is never stored or logged server-side.
4. **Ablation switches.** `USE_RL/USE_GNN/USE_RAG` let the A/B harness measure
   each module's marginal contribution (see `scripts/eval_ab.py`).
5. **Offline/online split.** Heavy training runs in a GPU Docker image; the web
   app loads the resulting artifacts and degrades gracefully if they are absent.

## Honest status

| Component | State |
|-----------|-------|
| LLM streaming, BYOK, demo, provider seam | ✅ production-usable |
| RAG (FAISS over CMV) | best-supported module: directional +0.039 on evidence + clean mechanism, but within judge noise at this scale |
| Judge (LLM scoring) | working, with keyword fallback |
| GNN (graph persuasion model) | experimental; no measurable gain, and training collapsed to the majority class |
| RL (PPO strategy policy) | experimental; no measurable gain; reward derived from the broken GNN, so it can't win by construction |

The GNN/RL modules are diagnostic demonstrations. Their contribution is reported
empirically in [the ablation results](eval_results.md), with the statistical caveat
applied **evenly**: at this sample size and judge, *no* module reaches significance —
RAG included. The harness yields a diagnosis, not certification. RAG is the only
module with both a directional signal and a clean mechanism (so it is the one worth
keeping, though unproven); the GNN collapsed to the majority class so its non-result
is uninformative; and RL's reward is derived from that broken GNN, so the GNN → RL
chain fails by construction rather than by chance. They are kept as labeled
experiments whose failure is itself the finding, not oversold.

Note: the persuasion the system actually delivers comes from the LLM and is scored by
the judge, which is **independent of the GNN**. The GNN's broken signal is a *would-be*
add-on, not the source of the debates' persuasiveness — see the note under the table in
[eval_results](eval_results.md).
See [DECISIONS.md](DECISIONS.md) for trade-offs and what would change for v2/production.
