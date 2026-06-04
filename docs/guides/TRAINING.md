# Training Guide

Trains the two models used by the debate system (the graph GNN and the PPO
policy) and builds the FAISS RAG index. All embeddings use the project's LLM
provider (`embeddinggemma` by default) so **no cloud key is required** — set
`LLM_BASE_URL` / `LLM_EMBEDDING_MODEL` in `.env`.

## One command

```bash
python train_all.py --all      # data -> FAISS index -> graph GNN -> PPO policy
```

Individual steps:

```bash
python train_all.py --data     # download + clean CMV  -> data/raw/{pairs,threads}.jsonl
python train_all.py --rag      # build FAISS index      -> data/rag/faiss/
python train_all.py --gnn      # train graph GNN        -> data/models/gnn_persuasion.pt
python train_all.py --rl       # train PPO policy       -> data/models/ppo_policy.pt
```

## Pipeline

1. **Data** — `scripts/prepare_cmv.py` downloads the Cornell ConvoKit
   ChangeMyView corpus and writes:
   - `data/raw/pairs.jsonl` — submission / delta-comment pairs (for RAG)
   - `data/raw/threads.jsonl` — full reply trees (for the graph GNN)
2. **RAG index** — `scripts/build_rag_index.py` embeds documents with
   `embeddinggemma` and writes a FAISS index. With no CMV present it falls back
   to `data/seed_evidence.jsonl`.
3. **Graph GNN** — `src/gnn/train_graph.py` trains node classification on the
   reply trees. Node features are `embeddinggemma` embeddings (same encoder used
   at inference); strategy labels come from an LLM (`src/gnn/strategy_label.py`).
4. **PPO policy** — `src/rl/train_ppo.py` trains a strategy policy whose state is
   a real comment embedding and whose reward blends the real delta outcome with
   the GNN's predicted strategy suitability.

## Hardware

- A GPU is optional and only used for training; CPU works (the models are small).
- For RTX 50-series / Blackwell GPUs use **CUDA 12.8 / PyTorch cu128** — the
  provided `docker/Dockerfile.train` + `docker/docker-compose.train.yml` are set up
  for this:

```bash
docker compose -f docker/docker-compose.train.yml run --rm train
```

## Tunables

- `python -m src.gnn.train_graph --max-nodes N --strategy-llm-cap N --epochs N`
  (strategy labeling calls the LLM; use a small/fast `LLM_MODEL` and a modest cap).
- `python -m src.rl.train_ppo --episodes N`
- Hyperparameters also live in `configs/gnn.yaml` and `configs/rl.yaml`.

## Honesty note

The GNN's delta-accuracy can look high purely because of class imbalance (most
conversation nodes are non-delta). What matters is whether the modules improve
debates, which is measured in [eval_results.md](../eval_results.md) — run it with:

```bash
python scripts/eval_ab.py
```
