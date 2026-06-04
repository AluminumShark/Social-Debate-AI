"""
FAISS-backed semantic retriever.

Uses the project's LLM provider for embeddings (Ollama `embeddinggemma` by
default — no external API key required), so RAG is real vector search rather
than keyword overlap. Falls back gracefully when the index is missing.

Index layout (under `index_dir`, default data/rag/faiss):
  - index.faiss   : normalized inner-product (cosine) index
  - docs.json     : [{"text": ..., "metadata": {...}}, ...] aligned to vectors
  - meta.json     : {"embedding_model": ..., "dim": ...}
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from llm import LLMConfig, resolve_config, embed  # noqa: E402


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: Dict


def _embed_batched(texts: List[str], cfg: LLMConfig, batch_size: int = 64) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        vecs.extend(embed(texts[i:i + batch_size], cfg))
    arr = np.asarray(vecs, dtype="float32")
    # Normalize for cosine similarity via inner product.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class VectorRetriever:
    def __init__(self, index_dir: str = "data/rag/faiss", llm_config: Optional[LLMConfig] = None):
        self.index_dir = Path(index_dir)
        self.cfg = llm_config or resolve_config()
        self.index = None
        self.docs: List[Dict] = []
        self._load()

    def _load(self):
        import faiss  # imported lazily so the app starts without faiss

        idx = self.index_dir / "index.faiss"
        docs = self.index_dir / "docs.json"
        if not (idx.exists() and docs.exists()):
            raise FileNotFoundError(f"FAISS index not found in {self.index_dir}")
        self.index = faiss.read_index(str(idx))
        with open(docs, "r", encoding="utf-8") as f:
            self.docs = json.load(f)
        print(f"[VectorRetriever] Loaded {len(self.docs)} docs from {self.index_dir}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or not self.docs:
            return []
        q = _embed_batched([query], self.cfg)
        scores, idxs = self.index.search(q, min(top_k, len(self.docs)))
        out: List[Dict] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or int(i) >= len(self.docs):
                continue
            d = self.docs[int(i)]
            out.append({
                "content": d.get("text", ""),
                "similarity_score": float(score),
                "metadata": d.get("metadata", {}),
            })
        return out

    def get_stats(self) -> Dict:
        return {"total_documents": len(self.docs), "index_dir": str(self.index_dir)}


def build_index(
    docs: List[Dict],
    index_dir: str = "data/rag/faiss",
    llm_config: Optional[LLMConfig] = None,
    batch_size: int = 64,
) -> int:
    """Build and persist a FAISS index from `docs` (list of {text, metadata})."""
    import faiss

    cfg = llm_config or resolve_config()
    texts = [d["text"] for d in docs if d.get("text")]
    docs = [d for d in docs if d.get("text")]
    if not texts:
        raise ValueError("No non-empty documents to index")

    print(f"[build_index] Embedding {len(texts)} docs with {cfg.embedding_model}...")
    mat = _embed_batched(texts, cfg, batch_size=batch_size)
    dim = mat.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    with open(out / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": cfg.embedding_model, "dim": dim, "count": len(docs)}, f)
    print(f"[build_index] Saved {len(docs)} vectors (dim={dim}) to {out}")
    return len(docs)
