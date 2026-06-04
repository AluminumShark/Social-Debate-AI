"""
Unified training pipeline for Social Debate AI.

Steps (in order):
  1. data : download CMV (ConvoKit) -> data/raw/pairs.jsonl   (scripts/prepare_cmv.py)
  2. rag  : build FAISS vector index from pairs                (scripts/build_rag_index.py)
  3. gnn  : train persuasion GNN                               (src.gnn.train_supervised)
  4. rl   : train PPO policy                                   (src.rl.train_ppo)

Examples:
  python train_all.py --all          # data + rag + gnn + rl
  python train_all.py --data         # only fetch/convert CMV
  python train_all.py --rag          # only (re)build FAISS index
  python train_all.py --gnn --rl     # train models, skip data/rag
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd, description):
    print(f"\n=== {description} ===")
    print("Running:", " ".join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"Done in {time.time() - start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Social Debate AI training pipeline")
    ap.add_argument("--all", action="store_true", help="run data + rag + gnn + rl")
    ap.add_argument("--data", action="store_true", help="download + convert CMV corpus")
    ap.add_argument("--rag", action="store_true", help="build FAISS vector index")
    ap.add_argument("--gnn", action="store_true", help="train GNN")
    ap.add_argument("--rl", action="store_true", help="train RL policy")
    ap.add_argument("--cmv-corpus", default="winning-args-corpus")
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()

    # Default to --all when nothing selected
    if not any([args.all, args.data, args.rag, args.gnn, args.rl]):
        args.all = True

    steps = []
    if args.all or args.data:
        steps.append((["python", "scripts/prepare_cmv.py", "--corpus", args.cmv_corpus],
                      "Download + convert CMV corpus"))
    if args.all or args.rag:
        steps.append((["python", "scripts/build_rag_index.py"],
                      "Build FAISS vector index"))
    if args.all or args.gnn:
        steps.append((["python", "-m", "src.gnn.train_graph"],
                      "Train graph GNN model (real conversation graphs)"))
    if args.all or args.rl:
        steps.append((["python", "-m", "src.rl.train_ppo", "--episodes", str(args.episodes)],
                      "Train RL policy"))

    print("Social Debate AI Training Pipeline")
    print("Steps:", ", ".join(d for _, d in steps))

    try:
        for cmd, desc in steps:
            run(cmd, desc)
        print("\nTraining pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nStep failed (exit {e.returncode}): {e.cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
