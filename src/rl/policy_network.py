"""
RL policy network interface
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
import random
from pathlib import Path

# Strategy mappings
STRATEGIES = ['aggressive', 'defensive', 'analytical', 'empathetic']
STRATEGY_TO_ID = {s: i for i, s in enumerate(STRATEGIES)}

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=901, hidden_size=256, num_strategies=4):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        return action_probs, state_value

# Global model instance
_policy_model = None

def _load_model():
    global _policy_model
    if _policy_model is not None:
        return _policy_model
        
    model_path = Path("data/models/policy/pytorch_model.bin")
    
    try:
        _policy_model = PolicyNetwork()
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            _policy_model.load_state_dict(state_dict)
            _policy_model.eval()
            print(f"Loaded policy model from {model_path}")
        else:
            print("No trained model found, using random initialization")
    except Exception as e:
        print(f"Failed to load model: {e}")
        _policy_model = PolicyNetwork()  # Fallback to random
    
    return _policy_model

def select_strategy(query, context="", social_context=None):
    # Simple heuristic-based strategy selection
    query_lower = query.lower()
    
    # Check for aggressive indicators
    aggressive_words = ['wrong', 'stupid', 'ridiculous', 'nonsense', 'absurd']
    if any(word in query_lower for word in aggressive_words):
        return 'aggressive'
    
    # Check for analytical indicators  
    analytical_words = ['research', 'study', 'data', 'evidence', 'statistics']
    if any(word in query_lower for word in analytical_words):
        return 'analytical'
    
    # Check for empathetic indicators
    empathetic_words = ['understand', 'feel', 'experience', 'perspective', 'concern']
    if any(word in query_lower for word in empathetic_words):
        return 'empathetic'
    
    # Default to defensive for neutral cases
    return 'defensive'

def choose_snippet(state_text, pool):
    if not pool:
        return "No evidence available"
    
    # Simple scoring based on text overlap
    state_words = set(state_text.lower().split())
    
    best_snippet = ""
    best_score = 0
    
    for item in pool:
        content = item.get('content', '')
        content_words = set(content.lower().split())
        
        # Calculate overlap score
        overlap = len(state_words & content_words)
        score = overlap / max(len(state_words), 1)
        
        if score > best_score:
            best_score = score
            best_snippet = content
    
    return best_snippet if best_snippet else pool[0].get('content', 'No evidence available')

class PolicyNetwork:
    def predict_quality(self, text):
        # Simple quality estimation based on text features
        words = text.split()
        
        # Length factor (prefer moderate length)
        length_score = min(len(words) / 50, 1.0) if len(words) < 100 else 0.8
        
        # Complexity factor (sentence variety)
        sentences = text.split('.')
        complexity_score = min(len(sentences) / 5, 1.0)
        
        # Evidence factor (keywords)
        evidence_words = ['research', 'study', 'data', 'according', 'evidence']
        evidence_score = sum(1 for word in evidence_words if word in text.lower()) / 10
        
        quality = (length_score * 0.4 + complexity_score * 0.3 + evidence_score * 0.3)
        return min(quality, 1.0)

def get_policy_network(model_path=None):
    return _load_model()
