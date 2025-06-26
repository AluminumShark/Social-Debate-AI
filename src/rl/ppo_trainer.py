"""
PPO trainer for debate strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Tuple
from dataclasses import dataclass
from torch.distributions import Categorical
import matplotlib.pyplot as plt

@dataclass
class DebateTransition:
    """Debate transition record"""
    state: torch.Tensor  # Current state
    action: int  # Selected strategy
    reward: float  # Received reward
    next_state: torch.Tensor  # Next state
    done: bool  # Whether finished
    log_prob: float  # Log probability of action
    value: float  # State value

class PPONetwork(nn.Module):
    """PPO network architecture (Actor-Critic)"""
    
    def __init__(self, state_dim=768, action_dim=4, hidden_dim=256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        """Forward propagation"""
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    
    def select_action(self, state):
        """Select action"""
        action_probs, state_value = self.forward(state)
        
        # Create action distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action), state_value.squeeze()

class DebateEnvironment:
    """Debate environment"""
    
    def __init__(self):
        self.strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
        self.reset()
    
    def reset(self):
        """Reset environment"""
        # Debate state
        self.current_stance = 0.0  # -1 to 1
        self.conviction = 0.7  # 0 to 1
        self.round = 0
        self.max_rounds = 5
        
        # Initialize state vector (simplified)
        self.state = torch.randn(768)  # Simulated BERT embedding
        
        return self.state
    
    def step(self, action):
        """Execute action"""
        strategy = self.strategies[action]
        
        # Simulate opponent response and debate outcome
        reward = self._calculate_reward(action)
        
        # Update state
        self.round += 1
        self.state = torch.randn(768)  # New state after opponent response
        
        done = self.round >= self.max_rounds
        
        return self.state, reward, done
    
    def _calculate_reward(self, action):
        """Calculate reward based on strategy effectiveness"""
        strategy = self.strategies[action]
        
        # Simplified reward calculation
        base_reward = 0.0
        
        if strategy == 'analytical':
            # Analytical strategy generally effective
            base_reward = 0.8
        elif strategy == 'empathetic':
            # Empathetic strategy good for building rapport
            base_reward = 0.7
        elif strategy == 'defensive':
            # Defensive strategy maintains position
            base_reward = 0.5
        elif strategy == 'aggressive':
            # Aggressive strategy risky but potentially high reward
            base_reward = random.choice([0.3, 0.9])
        
        # Add some noise
        noise = np.random.normal(0, 0.1)
        reward = base_reward + noise
        
        return np.clip(reward, 0, 1)

class PPOTrainer:
    """PPO trainer"""
    
    def __init__(self, state_dim=768, action_dim=4, lr=3e-4):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        self.env = DebateEnvironment()
        self.memory = deque(maxlen=10000)
        
        # PPO hyperparameters
        self.epsilon = 0.2  # Clipping parameter
        self.gamma = 0.99   # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.update_epochs = 4
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
    
    def collect_trajectory(self, num_episodes=10):
        """Collect trajectory data"""
        trajectories = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_rewards = []
            episode_transitions = []
            
            done = False
            while not done:
                # Select action
                action, log_prob, value = self.network.select_action(state.unsqueeze(0))
                
                # Execute action
                next_state, reward, done = self.env.step(action)
                
                # Store transition
                transition = DebateTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob.item(),
                    value=value.item()
                )
                
                episode_transitions.append(transition)
                episode_rewards.append(reward)
                
                state = next_state
            
            # Calculate returns and advantages
            self._calculate_advantages(episode_transitions)
            trajectories.extend(episode_transitions)
            
            self.episode_rewards.append(sum(episode_rewards))
            self.episode_lengths.append(len(episode_rewards))
        
        return trajectories
    
    def _calculate_advantages(self, transitions):
        """Calculate GAE advantages"""
        returns = []
        advantages = []
        
        # Calculate returns
        G = 0
        for transition in reversed(transitions):
            G = transition.reward + self.gamma * G
            returns.insert(0, G)
        
        # Calculate advantages using GAE
        values = [t.value for t in transitions] + [0]  # Add terminal value
        advantages = []
        
        gae = 0
        for i in reversed(range(len(transitions))):
            delta = (transitions[i].reward + 
                    self.gamma * values[i + 1] * (1 - transitions[i].done) - 
                    values[i])
            gae = delta + self.gamma * self.gae_lambda * (1 - transitions[i].done) * gae
            advantages.insert(0, gae)
        
        # Normalize advantages
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store in transitions
        for i, transition in enumerate(transitions):
            transition.return_value = returns[i]
            transition.advantage = advantages[i].item()
    
    def update_policy(self, trajectories):
        """Update policy using PPO"""
        # Convert to tensors
        states = torch.stack([t.state for t in trajectories])
        actions = torch.tensor([t.action for t in trajectories], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in trajectories])
        returns = torch.tensor([t.return_value for t in trajectories])
        advantages = torch.tensor([t.advantage for t in trajectories])
        
        total_losses = []
        # PPO update
        for _ in range(self.update_epochs):
            # Forward pass
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            
            # Calculate new log probabilities
            new_log_probs = dist.log_prob(actions)
            
            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus for exploration
            entropy = dist.entropy().mean()
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            total_losses.append(total_loss.item())
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        return np.mean(total_losses)
    
    def train(self, num_iterations=1000, episodes_per_iteration=10):
        """Training main loop"""
        print("Starting PPO training...")
        
        losses = []
        for iteration in range(num_iterations):
            # Collect data
            trajectories = self.collect_trajectory(episodes_per_iteration)
            
            # Update policy and get loss
            loss = self.update_policy(trajectories)
            losses.append(loss)
            
            # Print progress
            if iteration % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-episodes_per_iteration:])
                avg_length = np.mean(self.episode_lengths[-episodes_per_iteration:])
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.3f}, Avg Length = {avg_length:.1f}")
        
        print("Training completed!")
        return losses
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
        print(f"Model saved to {path}")
    
    def save(self, path):
        """Save model (backward compatibility alias)"""
        self.save_model(path)
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        print(f"Model loaded from {path}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Create trainer
    trainer = PPOTrainer()
    
    # Train
    trainer.train(num_iterations=500, episodes_per_iteration=5)
    
    # Save model
    trainer.save_model("models/ppo_debate_strategy.pt")
    
    # Plot results
    trainer.plot_training_progress() 