"""
Social network encoder using GNN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch import load
from pathlib import Path
import numpy as np

class PersuasionGNN(nn.Module):
    
    def __init__(self, input_dim=768, hidden_dim=256, num_strategies=4):
        super().__init__()
        
        # GraphSAGE layers
        self.conv1 = tgnn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = tgnn.SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = tgnn.SAGEConv(hidden_dim, 128)
        
        # Attention layer
        self.attention = tgnn.GATConv(128, 128, heads=4, concat=False)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Multi-task heads - 調整結構以匹配檢查點 (0, 1, 2, 3)
        self.delta_head = nn.Sequential(
            nn.Linear(128, 64),     # 第0層
            nn.ReLU(),              # 第1層
            nn.Dropout(0.3),        # 第2層
            nn.Linear(64, 1)        # 第3層
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64),     # 第0層
            nn.ReLU(),              # 第1層
            nn.Dropout(0.3),        # 第2層
            nn.Linear(64, 1)        # 第3層
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(128, 64),     # 第0層
            nn.ReLU(),              # 第1層
            nn.Dropout(0.3),        # 第2層
            nn.Linear(64, num_strategies)  # 第3層
        )
    
    def forward(self, x, edge_index, batch=None):
        # Graph convolution
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # Attention layer
        x = self.attention(x, edge_index)
        x = self.dropout(x)
        
        # If batch information exists, perform global pooling
        if batch is not None:
            x = tgnn.global_mean_pool(x, batch)
        
        # Multi-task prediction
        delta_pred = self.delta_head(x)
        quality_pred = self.quality_head(x)
        strategy_pred = self.strategy_head(x)
        
        return {
            'delta': delta_pred,
            'quality': quality_pred,
            'strategy': strategy_pred,
            'embeddings': x
        }

# Load pre-trained model
_ckpt_path = Path('data/models/gnn_social.pt')
_persuasion_path = Path('data/models/gnn_persuasion.pt')

# Use supervised model first
if _persuasion_path.exists():
    print("Loading supervised GNN model")
    _persuasion_ckpt = load(str(_persuasion_path), map_location='cpu')
    _PERSUASION_MODEL = PersuasionGNN(
        input_dim=_persuasion_ckpt['config']['input_dim'],
        hidden_dim=_persuasion_ckpt['config']['hidden_dim']
    )
    _PERSUASION_MODEL.load_state_dict(_persuasion_ckpt['model_state'])
    _PERSUASION_MODEL.eval()
    _NODE_TO_IDX = _persuasion_ckpt['node_to_idx']
    print(f"Model performance: Delta accuracy={_persuasion_ckpt['performance']['delta_acc']:.3f}")
else:
    _PERSUASION_MODEL = None
    _NODE_TO_IDX = None

# Old model as backup (optional)
if _ckpt_path.exists():
    _ckpt = load(str(_ckpt_path), map_location='cpu')
    _EMB = _ckpt['emb']
    _IDX = _ckpt['node2idx']
else:
    # Old model doesn't exist, but it's okay if we have supervised model
    _EMB = None
    _IDX = None

# Strategy name mapping
STRATEGY_NAMES = {
    0: 'aggressive',
    1: 'defensive',
    2: 'analytical',
    3: 'empathetic'
}

def social_encoder(author, k=8):
    """Get author's social encoding (string format)"""
    if _IDX is None or _EMB is None:
        return "UNK"
    
    idx = _IDX.get(author)
    return "UNK" if idx is None else " ".join(f"{x:.4f}" for x in _EMB[idx][:k])

def social_vec(agent_id):
    """Get Agent's social vector"""
    # If supervised model exists, use its embedding
    if _PERSUASION_MODEL is not None and _NODE_TO_IDX is not None:
        idx = _NODE_TO_IDX.get(agent_id)
        if idx is not None:
            # Use model's embedding layer output
            with torch.no_grad():
                # Create dummy input (single node)
                x = torch.zeros(1, _PERSUASION_MODEL.conv1.in_channels)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                outputs = _PERSUASION_MODEL(x, edge_index)
                embedding = outputs['embeddings'][0].numpy()
                
                # Ensure it's 128-dimensional
                if len(embedding) >= 128:
                    return embedding[:128].tolist()
                else:
                    # Pad to 128 dimensions
                    padded = np.zeros(128)
                    padded[:len(embedding)] = embedding
                    return padded.tolist()
    
    # Use old model
    if _IDX is not None and _EMB is not None:
        idx = _IDX.get(agent_id)
        if idx is not None and idx < len(_EMB):
            embedding = _EMB[idx].numpy() if hasattr(_EMB[idx], 'numpy') else _EMB[idx]
            if len(embedding) >= 128:
                return embedding[:128].tolist()
            else:
                padded = np.zeros(128)
                padded[:len(embedding)] = embedding
                return padded.tolist()
    
    # If not found, use deterministic random vector based on agent_id
    np.random.seed(hash(agent_id) % 2**32)
    return np.random.rand(128).tolist()

def predict_persuasion(text_features, agent_id=None):
    """Predict persuasion-related metrics"""
    if _PERSUASION_MODEL is None:
        return {
            'delta_probability': 0.5,
            'quality_score': 0.5,
            'best_strategy': 'analytical',
            'strategy_scores': {'aggressive': 0.25, 'defensive': 0.25, 
                              'analytical': 0.25, 'empathetic': 0.25}
        }
    
    with torch.no_grad():
        # Prepare input - add placeholder for argument features
        # Model expects 770-dimensional input (768 text features + 2 argument features)
        if len(text_features) == 768:
            # Add default argument features
            argument_features = np.array([0.5, 0.5])  # Placeholder values
            full_features = np.concatenate([text_features, argument_features])
        else:
            full_features = text_features
            
        # Create single-node graph
        x = torch.tensor(full_features, dtype=torch.float32).unsqueeze(0)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Model prediction
        outputs = _PERSUASION_MODEL(x, edge_index)
        
        # Process results
        delta_prob = torch.sigmoid(outputs['delta']).item()
        quality_score = torch.sigmoid(outputs['quality']).item()
        strategy_logits = outputs['strategy'].squeeze()
        strategy_probs = torch.softmax(strategy_logits, dim=0).numpy()
        
        # Get best strategy
        best_strategy_idx = np.argmax(strategy_probs)
        best_strategy = STRATEGY_NAMES[best_strategy_idx]
        
        # Create strategy score dictionary
        strategy_scores = {
            STRATEGY_NAMES[i]: float(prob) 
            for i, prob in enumerate(strategy_probs)
        }
        
        return {
            'delta_probability': delta_prob,
            'quality_score': quality_score,
            'best_strategy': best_strategy,
            'strategy_scores': strategy_scores
        }

def get_social_influence_score(agent_id):
    """Get social influence score"""
    social_vector = social_vec(agent_id)
    
    # Calculate influence score based on social vector
    # Use the average absolute value of first 10 dimensions
    influence_dims = social_vector[:10]
    influence_score = np.mean(np.abs(influence_dims))
    
    # Normalize to 0-1 range
    influence_score = np.clip(influence_score, 0, 1)
    
    return float(influence_score)