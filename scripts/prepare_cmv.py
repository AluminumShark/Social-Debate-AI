"""
Download the Cornell ConvoKit ChangeMyView corpus and convert it to:
  1. data/raw/pairs.jsonl   — {submission, delta_comment} pairs (RAG / simple GNN)
  2. data/raw/threads.jsonl — real conversation graphs (v2 graph GNN):
       {conv_id, nodes:[{id,text,author,is_delta}], edges:[[src,dst],...]}
     edges follow reply_to (child -> parent and parent -> child).

Usage:
  python scripts/prepare_cmv.py --corpus winning-args-corpus            # both
  python scripts/prepare_cmv.py --max-convos 1500                       # cap graphs
  python scripts/prepare_cmv.py --pairs-only                            # legacy pairs

`convokit` must be installed (project dep). First run downloads the corpus.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "pairs.jsonl"
GRAPH_OUT = ROOT / "data" / "raw" / "threads.jsonl"


def _is_delta(utt) -> bool:
    meta = getattr(utt, "meta", {}) or {}
    for key in ("success", "delta", "is_delta", "awarded_delta"):
        v = meta.get(key)
        if v in (1, True, "1", "true", "True"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="winning-args-corpus")
    ap.add_argument("--max", type=int, default=None, help="max pairs")
    ap.add_argument("--max-convos", type=int, default=2000,
                    help="max conversations exported as graphs (training-time control)")
    ap.add_argument("--pairs-only", action="store_true")
    ap.add_argument("--graph-only", action="store_true")
    args = ap.parse_args()

    from convokit import Corpus, download

    print(f"Downloading/loading corpus: {args.corpus} ...")
    corpus = Corpus(filename=download(args.corpus))
    corpus.print_summary_stats()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_pairs, n_graphs, n_nodes = 0, 0, 0

    pf = None if args.graph_only else open(OUT, "w", encoding="utf-8")
    gf = None if args.pairs_only else open(GRAPH_OUT, "w", encoding="utf-8")

    try:
        for convo in corpus.iter_conversations():
            try:
                ids = convo.get_utterance_ids()
                root = convo.get_utterance(ids[0])
            except Exception:
                continue

            # ---- pairs export (submission + each delta comment) ----
            if pf is not None and (args.max is None or n_pairs < args.max):
                submission = {
                    "id": getattr(root, "id", ""),
                    "title": (root.meta or {}).get("title", ""),
                    "selftext": root.text or "",
                    "score": (root.meta or {}).get("score", 0),
                }
                for utt in convo.iter_utterances():
                    if utt.id == root.id or not _is_delta(utt) or not (utt.text or "").strip():
                        continue
                    pf.write(json.dumps({
                        "submission": submission,
                        "delta_comment": {"id": utt.id, "body": utt.text,
                                          "score": (utt.meta or {}).get("score", 0)},
                    }, ensure_ascii=False) + "\n")
                    n_pairs += 1
                    if args.max and n_pairs >= args.max:
                        break

            # ---- graph export (real reply tree) ----
            if gf is not None and n_graphs < args.max_convos:
                utts = list(convo.iter_utterances())
                if len(utts) < 2:
                    continue
                idx = {u.id: i for i, u in enumerate(utts)}
                nodes = [{
                    "id": u.id,
                    "text": (u.text or "")[:1500],
                    "author": getattr(u, "speaker", None).id if getattr(u, "speaker", None) else "",
                    "is_delta": bool(_is_delta(u)),
                } for u in utts]
                edges = []
                for u in utts:
                    parent = getattr(u, "reply_to", None)
                    if parent and parent in idx:
                        c, p = idx[u.id], idx[parent]
                        edges.append([c, p])
                        edges.append([p, c])  # undirected message passing
                if not edges:
                    continue
                gf.write(json.dumps({"conv_id": convo.id, "nodes": nodes, "edges": edges},
                                    ensure_ascii=False) + "\n")
                n_graphs += 1
                n_nodes += len(nodes)
    finally:
        if pf:
            pf.close()
        if gf:
            gf.close()

    if pf is not None:
        print(f"Wrote {n_pairs} pairs -> {OUT}")
    if gf is not None:
        print(f"Wrote {n_graphs} conversation graphs ({n_nodes} nodes) -> {GRAPH_OUT}")


if __name__ == "__main__":
    main()
