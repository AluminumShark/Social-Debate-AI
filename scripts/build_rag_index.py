"""
Build the FAISS vector index used by RAG, embedding with the project's LLM
provider (Ollama `embeddinggemma` by default — no external key needed).

Sources (first that exists wins, unless --source given):
  1. data/raw/pairs.jsonl   (CMV submission/delta_comment pairs)
  2. data/seed_evidence.jsonl ({"text":..., "metadata":{...}} per line)

Usage:
  python scripts/build_rag_index.py [--source PATH] [--max N] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rag.vector_retriever import build_index  # noqa: E402


def _docs_from_pairs(path: Path, max_n=None):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_n and len(docs) >= max_n:
                break
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue
            sub = pair.get("submission", {})
            body = sub.get("selftext") or sub.get("title", "")
            if body:
                docs.append({"text": body[:2000], "metadata": {
                    "type": "submission", "id": sub.get("id", ""), "score": sub.get("score", 0)}})
            dc = pair.get("delta_comment", {})
            if dc and dc.get("body"):
                docs.append({"text": dc["body"][:2000], "metadata": {
                    "type": "delta_comment", "score": dc.get("score", 0),
                    "persuasion_success": True}})
    return docs


def _docs_from_seed(path: Path, max_n=None):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_n and len(docs) >= max_n:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("text"):
                docs.append({"text": obj["text"], "metadata": obj.get("metadata", {})})
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default="data/rag/faiss")
    args = ap.parse_args()

    pairs = ROOT / "data" / "raw" / "pairs.jsonl"
    seed = ROOT / "data" / "seed_evidence.jsonl"

    if args.source:
        src = Path(args.source)
        docs = _docs_from_pairs(src, args.max) if "pairs" in src.name else _docs_from_seed(src, args.max)
    elif pairs.exists():
        print(f"Using CMV pairs: {pairs}")
        docs = _docs_from_pairs(pairs, args.max)
    elif seed.exists():
        print(f"Using seed evidence: {seed}")
        docs = _docs_from_seed(seed, args.max)
    else:
        raise SystemExit("No source found. Provide --source or create data/raw/pairs.jsonl")

    print(f"Prepared {len(docs)} docs; building FAISS index -> {args.out}")
    n = build_index(docs, index_dir=args.out)
    print(f"Done. Indexed {n} docs.")


if __name__ == "__main__":
    main()
