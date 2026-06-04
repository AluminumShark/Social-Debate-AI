"""
PPO trainer for debate strategy selection.

Grounded (not synthetic):
- States are REAL context embeddings sampled from the CMV corpus
  (embeddinggemma, 768-d) — the same encoder used at inference.
- Rewards come from the trained GNN's per-strategy persuasion prediction for
  that state (data-grounded), so the policy learns "which strategy suits this
  context" rather than memorizing a constant. No live LLM calls per step, so
  training stays cheap. Falls back to a mild prior if no GNN/data is present.
"""

import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl.policy_network import PPONetwork, STRATEGIES  # noqa: E402


@dataclass
class DebateTransition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float


class DebateEnvironment:
    """Debate environment backed by real CMV embeddings + GNN reward."""

    def __init__(self, threads_path="data/raw/threads.jsonl",
                 pairs_path="data/raw/pairs.jsonl", pool_size=600, max_rounds=5):
        self.strategies = STRATEGIES
        self.max_rounds = max_rounds
        self.round = 0
        self._gnn = self._load_gnn()
        # pool entries: (embedding_tensor, delta_outcome in {0.0,1.0})
        self.pool = self._build_state_pool(Path(threads_path), Path(pairs_path), pool_size)
        self.state, self.outcome = self.pool[0]

    def _load_gnn(self):
        try:
            from gnn import social_encoder
            if getattr(social_encoder, "_PERSUASION_MODEL", None) is not None:
                print("[RL] Reward source: trained GNN persuasion model")
                return social_encoder
            print("[RL] GNN model not trained; using prior reward")
        except Exception as e:  # noqa: BLE001
            print(f"[RL] GNN unavailable ({e}); using prior reward")
        return None

    def _build_state_pool(self, threads_path, pairs_path, pool_size):
        from llm import embed, resolve_config

        # Prefer real conversation nodes (carry real delta outcome labels).
        items = []  # (text, delta_outcome)
        if threads_path.exists():
            with open(threads_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for nd in c.get("nodes", []):
                        if (nd.get("text") or "").strip():
                            items.append((nd["text"][:2000], 1.0 if nd.get("is_delta") else 0.0))
                    if len(items) >= pool_size:
                        break
        elif pairs_path.exists():
            with open(pairs_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    dc = p.get("delta_comment", {})
                    if dc.get("body"):
                        items.append((dc["body"][:2000], 1.0))  # pairs are all delta
                    if len(items) >= pool_size:
                        break

        if not items:
            print("[RL] No CMV data found; using random state pool")
            return [(torch.randn(768), 0.5) for _ in range(64)]

        items = items[:pool_size]
        print(f"[RL] Embedding {len(items)} CMV states (with delta outcomes)...")
        cfg = resolve_config()
        texts = [t for t, _ in items]
        vecs = []
        for i in range(0, len(texts), 64):
            vecs.extend(embed(texts[i:i + 64], cfg))
        return [(torch.tensor(v, dtype=torch.float32), out)
                for v, (_, out) in zip(vecs, items)]

    def reset(self):
        self.round = 0
        self.state, self.outcome = random.choice(self.pool)
        return self.state

    def step(self, action):
        reward = self._reward(self.state, self.outcome, action)
        self.round += 1
        self.state, self.outcome = random.choice(self.pool)
        done = self.round >= self.max_rounds
        return self.state, reward, done

    def _reward(self, state, outcome, action):
        """Outcome-grounded reward.

        Combines (a) the REAL delta outcome of this context with (b) the GNN's
        predicted suitability of the chosen strategy. High reward = pick the
        strategy the data associates with success, in contexts that actually
        succeeded.
        """
        strategy = self.strategies[action]
        suitability = 0.25
        if self._gnn is not None:
            try:
                pred = self._gnn.predict_persuasion(state.numpy())
                suitability = pred.get("strategy_scores", {}).get(strategy, 0.25)
            except Exception:  # noqa: BLE001
                pass
        # 0.5 weight on real outcome, 0.5 on strategy suitability.
        reward = 0.5 * outcome + 0.5 * suitability
        return float(np.clip(reward, 0.0, 1.0))


class PPOTrainer:
    def __init__(self, state_dim=768, action_dim=4, lr=3e-4):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.env = DebateEnvironment()
        self.memory = deque(maxlen=10000)
        self.epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.update_epochs = 4
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_trajectory(self, num_episodes=10):
        trajectories = []
        for _ in range(num_episodes):
            state = self.env.reset()
            ep_rewards, ep_trans = [], []
            done = False
            while not done:
                action, log_prob, value = self.network.select_action(state.unsqueeze(0))
                next_state, reward, done = self.env.step(action)
                ep_trans.append(DebateTransition(
                    state=state, action=action, reward=reward, next_state=next_state,
                    done=done, log_prob=log_prob.item(), value=value.item()))
                ep_rewards.append(reward)
                state = next_state
            self._calculate_advantages(ep_trans)
            trajectories.extend(ep_trans)
            self.episode_rewards.append(sum(ep_rewards))
            self.episode_lengths.append(len(ep_rewards))
        return trajectories

    def _calculate_advantages(self, transitions):
        returns, G = [], 0
        for t in reversed(transitions):
            G = t.reward + self.gamma * G
            returns.insert(0, G)
        values = [t.value for t in transitions] + [0]
        advantages, gae = [], 0
        for i in reversed(range(len(transitions))):
            delta = (transitions[i].reward
                     + self.gamma * values[i + 1] * (1 - transitions[i].done)
                     - values[i])
            gae = delta + self.gamma * self.gae_lambda * (1 - transitions[i].done) * gae
            advantages.insert(0, gae)
        adv = torch.tensor(advantages, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for i, t in enumerate(transitions):
            t.return_value = returns[i]
            t.advantage = adv[i].item()

    def update_policy(self, trajectories):
        states = torch.stack([t.state for t in trajectories])
        actions = torch.tensor([t.action for t in trajectories], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in trajectories])
        returns = torch.tensor([t.return_value for t in trajectories], dtype=torch.float32)
        advantages = torch.tensor([t.advantage for t in trajectories], dtype=torch.float32)

        total_losses = []
        for _ in range(self.update_epochs):
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            total_losses.append(total_loss.item())
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        return float(np.mean(total_losses))

    def train(self, num_iterations=1000, episodes_per_iteration=10):
        print("Starting PPO training...")
        losses = []
        for iteration in range(num_iterations):
            trajectories = self.collect_trajectory(episodes_per_iteration)
            losses.append(self.update_policy(trajectories))
            if iteration % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-episodes_per_iteration:])
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.3f}")
        print("Training completed!")
        return losses

    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': {'state_dim': 768, 'action_dim': 4, 'hidden_dim': 256},
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }, path)
        print(f"Model saved to {path}")

    def save(self, path):
        self.save_model(path)

    def load_model(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.network.load_state_dict(ckpt['network_state_dict'])
        self.episode_rewards = ckpt.get('episode_rewards', [])
        self.episode_lengths = ckpt.get('episode_lengths', [])


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train(num_iterations=200, episodes_per_iteration=5)
    trainer.save_model("data/models/ppo_policy.pt")
