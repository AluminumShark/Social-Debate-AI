"""
PPO training script
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rl.ppo_trainer import PPOTrainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def plot_training(rewards, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.7)
    
    if len(rewards) >= 20:
        window = min(20, len(rewards) // 5)
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), avg, 'r-', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="data/models")
    args = parser.parse_args()
    
    print("Training PPO debate policy...")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = PPOTrainer()
    
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    
    losses = trainer.train(args.episodes)
    
    model_path = output_dir / "ppo_policy.pt"
    trainer.save(str(model_path))
    
    if losses:
        log = {
            'episodes': args.episodes,
            'final_loss': losses[-1] if losses else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }
        
        with open(output_dir / "training_log.json", 'w') as f:
            json.dump(log, f, indent=2)
        
        plot_training(losses, output_dir / "training_curve.png")
        
        print(f"Training complete")
        print(f"Final loss: {losses[-1]:.3f}")
        print(f"Model saved: {model_path}")

if __name__ == "__main__":
    main() 