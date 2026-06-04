#!/usr/bin/env bash
# Download the pre-trained models + FAISS RAG index from the GitHub release
# and extract them into data/ (so the app runs the full trained stack without
# retraining). See docs/eval_results.md for what these modules contribute.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/AluminumShark/Social-Debate-AI/releases/download/models-v1/social-debate-models.tar.gz"
TMP="$(mktemp -t sdai-models-XXXX.tar.gz)"

echo "Downloading pre-trained models + FAISS index..."
curl -fL -o "$TMP" "$URL"
tar -xzf "$TMP"          # restores data/models/* and data/rag/faiss/*
rm -f "$TMP"
echo "Done:"
echo "  data/models/gnn_persuasion.pt, ppo_policy.pt"
echo "  data/rag/faiss/index.faiss, docs.json"
