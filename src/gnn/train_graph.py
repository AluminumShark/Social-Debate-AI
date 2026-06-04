"""
Graph GNN training on REAL CMV conversation graphs (v2).

Unlike the old 2-node submission->comment version, this trains node
classification over actual reply trees (data/raw/threads.jsonl):
  - node features: embeddinggemma 768-d text embeddings (same encoder as inference)
  - node labels:  delta (binary) + strategy (LLM-labeled) + quality (proxy)
Saves a checkpoint compatible with src/gnn/social_encoder.py.

Caps (env or args) keep training time tractable on Ollama embeddings:
  --max-nodes (embedding budget), --strategy-llm-cap (LLM strategy labels)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm import embed as _llm_embed, resolve_config  # noqa: E402
from gnn.social_encoder import PersuasionGNN  # noqa: E402
from gnn.strategy_label import label_id  # noqa: E402


def load_graphs(path, max_nodes, strategy_llm_cap):
    from torch_geometric.data import Data

    cfg = resolve_config()
    convos = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                convos.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Budget nodes for embedding
    selected, total = [], 0
    for c in convos:
        n = len(c["nodes"])
        if total + n > max_nodes:
            continue
        selected.append(c)
        total += n
    print(f"[graph] {len(selected)} conversations, {total} nodes (embedding...)")

    # Embed all node texts
    texts = [nd["text"] for c in selected for nd in c["nodes"]]
    vecs = []
    for i in tqdm(range(0, len(texts), 64), desc="Embedding"):
        vecs.extend(_llm_embed(texts[i:i + 64], cfg))
    vecs = [np.asarray(v, dtype=np.float32) for v in vecs]

    # Strategy labels (LLM up to cap, keyword fallback after)
    print(f"[graph] Labeling strategies (LLM cap={strategy_llm_cap})...")
    data_list, vi, labeled = [], 0, 0
    for c in selected:
        nodes = c["nodes"]
        x = torch.tensor(np.stack(vecs[vi:vi + len(nodes)]), dtype=torch.float)
        y_delta, y_strategy, y_quality = [], [], []
        for nd in nodes:
            y_delta.append(1.0 if nd["is_delta"] else 0.0)
            y_quality.append(0.75 if nd["is_delta"] else 0.4)
            use_llm = labeled < strategy_llm_cap
            y_strategy.append(label_id(nd["text"], use_llm=use_llm))
            labeled += 1
        vi += len(nodes)
        edge_index = torch.tensor(c["edges"], dtype=torch.long).t().contiguous()
        data_list.append(Data(
            x=x, edge_index=edge_index,
            y_delta=torch.tensor(y_delta, dtype=torch.float),
            y_strategy=torch.tensor(y_strategy, dtype=torch.long),
            y_quality=torch.tensor(y_quality, dtype=torch.float),
        ))
    return data_list


def train(args):
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_list = load_graphs(args.threads, args.max_nodes, args.strategy_llm_cap)
    if not data_list:
        raise SystemExit("No graphs loaded — run prepare_cmv.py first")

    split = max(1, int(0.85 * len(data_list)))
    train_loader = DataLoader(data_list[:split], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data_list[split:], batch_size=args.batch_size)

    model = PersuasionGNN(input_dim=768, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run(loader, train_mode):
        model.train() if train_mode else model.eval()
        tot, correct_d, n = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            if train_mode:
                opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            ld = F.binary_cross_entropy_with_logits(out["delta"].squeeze(-1), batch.y_delta)
            lq = F.mse_loss(out["quality"].squeeze(-1), batch.y_quality)
            ls = F.cross_entropy(out["strategy"], batch.y_strategy)
            loss = ld + lq + ls
            if train_mode:
                loss.backward()
                opt.step()
            tot += float(loss) * batch.num_graphs
            pred = (torch.sigmoid(out["delta"].squeeze(-1)) > 0.5).float()
            correct_d += int((pred == batch.y_delta).sum())
            n += batch.y_delta.numel()
        return tot / max(len(loader), 1), correct_d / max(n, 1)

    best, out_path = float("inf"), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run(train_loader, True)
        with torch.no_grad():
            va_loss, va_acc = run(val_loader, False)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}: train {tr_loss:.3f}/acc {tr_acc:.3f} | "
                  f"val {va_loss:.3f}/Delta Acc {va_acc:.3f}")
        if va_loss < best:
            best = va_loss
            torch.save({
                "config": {"input_dim": 768, "hidden_dim": args.hidden_dim, "num_strategies": 4},
                "model_state": model.state_dict(),
                "node_to_idx": {},
                "performance": {"delta_acc": float(va_acc)},
            }, out_path)
    print(f"Saved graph GNN -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="data/raw/threads.jsonl")
    ap.add_argument("--output", default="data/models/gnn_persuasion.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-nodes", type=int, default=8000)
    # LLM strategy labels are slow on large models; keep the default modest.
    # Use a small/fast LLM_MODEL (e.g. a 4B) for labeling.
    ap.add_argument("--strategy-llm-cap", type=int, default=800)
    train(ap.parse_args())
