"""
Unified training management for Social Debate AI
"""

import subprocess
import time
import argparse
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True, cwd=Path.cwd())
    finally:
        elapsed = time.time() - start_time
        if elapsed < 300:  # Less than 5 minutes
            print(f"Completed in {elapsed:.1f}s")
        else:
            print(f"Failed after {elapsed:.1f}s")

def main():
    parser = argparse.ArgumentParser(
        description="Social Debate AI unified training management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_all.py                    # Train all models
  python train_all.py --skip-rag         # Skip RAG training
  python train_all.py --rag-chroma       # Use Chroma for RAG
        """)
    
    parser.add_argument('--skip-rag', action='store_true',
                       help='Skip RAG index building')
    parser.add_argument('--rag-chroma', action='store_true',
                       help='Use Chroma vector database for RAG')
    
    args = parser.parse_args()
    
    print("Social Debate AI Training Pipeline")
    print(f"Config: RAG={'skip' if args.skip_rag else 'train'}, "
          f"Vector DB={'chroma' if args.rag_chroma else 'simple'}")
    
    training_steps = []
    
    # RAG training
    if not args.skip_rag:
        if args.rag_chroma:
            training_steps.append((
                ["python", "-m", "src.rag.build_index", "--type", "chroma"],
                "Building Chroma vector index"
            ))
        else:
            training_steps.append((
                ["python", "-m", "src.rag.train"],
                "Building simple RAG index"
            ))
    
    # GNN training
    training_steps.append((
        ["python", "-m", "src.gnn.train_supervised"],
        "Training GNN model"
    ))
    
    # RL training
    training_steps.append((
        ["python", "-m", "src.rl.train_ppo", "--episodes", "1000"],
        "Training RL policy"
    ))
    
    # Execute training
    print("\nStarting training pipeline...")
    
    try:
        for cmd, desc in training_steps:
            run_command(cmd, desc)
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 