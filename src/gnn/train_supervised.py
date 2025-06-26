"""
GNN training for persuasion prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import networkx as nx
from transformers import AutoTokenizer, AutoModel

class PersuasionGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_strategies=4):
        super().__init__()
        
        self.conv1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = gnn.SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = gnn.SAGEConv(hidden_dim, hidden_dim // 2)
        
        self.attention = gnn.GATConv(hidden_dim // 2, hidden_dim // 2, heads=4, concat=False)
        
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self.strategy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_strategies)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        x = self.attention(x, edge_index)
        x = self.dropout(x)
        
        if batch is not None:
            x = gnn.global_mean_pool(x, batch)
        
        return {
            'delta': self.delta_head(x),
            'quality': self.quality_head(x),
            'strategy': self.strategy_head(x),
            'embeddings': x
        }

class PersuasionDataset:
    def __init__(self, pairs_path='data/raw/pairs.jsonl'):
        self.pairs_path = Path(pairs_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        
        self.strategies = {
            'aggressive': 0,
            'defensive': 1,
            'analytical': 2,
            'empathetic': 3
        }
        
    def encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, truncation=True, max_length=512, 
                                  padding=True, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
    
    def encode_batch(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, truncation=True, max_length=512, 
                                      padding=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.encoder(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def classify_strategy(self, text: str, is_delta: bool) -> int:
        text_lower = text.lower()
        
        aggressive = sum(1 for w in ['wrong', 'incorrect', 'flawed'] if w in text_lower)
        defensive = sum(1 for w in ['defend', 'maintain', 'believe'] if w in text_lower)
        analytical = sum(1 for w in ['analyze', 'consider', 'examine'] if w in text_lower)
        empathetic = sum(1 for w in ['understand', 'appreciate', 'feel'] if w in text_lower)
        
        scores = {
            'aggressive': aggressive,
            'defensive': defensive,
            'analytical': analytical + (2 if is_delta else 0),
            'empathetic': empathetic + (1 if is_delta else 0)
        }
        
        best = max(scores.keys(), key=lambda k: scores[k])
        return self.strategies[best]
    
    def build_graph(self) -> Tuple[Data, Dict]:
        print("Building interaction graph...")
        
        interactions = []
        if not self.pairs_path.exists():
            print("Dataset not found, creating synthetic data...")
            return self._create_synthetic_data()
        
        with open(self.pairs_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                try:
                    pair = json.loads(line)
                    submission = pair['submission']
                    delta_comment = pair.get('delta_comment', {})
                    
                    if delta_comment.get('body'):
                        interactions.append({
                            'submission_text': submission.get('selftext', '') or submission.get('title', ''),
                            'comment_text': delta_comment.get('body', ''),
                            'is_delta': True,
                            'score': delta_comment.get('score', 0)
                        })
                        
                except Exception as e:
                    continue
        
        if not interactions:
            return self._create_synthetic_data()
        
        print(f"Loaded {len(interactions)} interactions")
        
        # Create embeddings
        all_texts = []
        for item in interactions:
            all_texts.append(item['submission_text'])
            all_texts.append(item['comment_text'])
        
        print("Encoding texts...")
        embeddings = self.encode_batch(all_texts, batch_size=16)
        
        # Build graph data
        node_features = []
        labels = {'delta': [], 'quality': [], 'strategy': []}
        
        for i, item in enumerate(interactions):
            # Submission node
            sub_emb = embeddings[i * 2]
            node_features.append(sub_emb)
            labels['delta'].append(0)
            labels['quality'].append(0.5)
            labels['strategy'].append(self.classify_strategy(item['submission_text'], False))
            
            # Comment node
            com_emb = embeddings[i * 2 + 1]
            node_features.append(com_emb)
            labels['delta'].append(1 if item['is_delta'] else 0)
            labels['quality'].append(min(item['score'] / 10.0, 1.0))
            labels['strategy'].append(self.classify_strategy(item['comment_text'], item['is_delta']))
        
        # Create edges (simple: submission -> comment)
        edge_index = []
        for i in range(len(interactions)):
            edge_index.append([i * 2, i * 2 + 1])  # submission -> comment
            edge_index.append([i * 2 + 1, i * 2])  # comment -> submission
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        
        data = Data(
            x=torch.tensor(np.array(node_features), dtype=torch.float),
            edge_index=edge_index,
            y_delta=torch.tensor(labels['delta'], dtype=torch.float),
            y_quality=torch.tensor(labels['quality'], dtype=torch.float),
            y_strategy=torch.tensor(labels['strategy'])
        )
        
        stats = {
            'num_nodes': len(node_features),
            'num_edges': edge_index.size(1),
            'delta_ratio': np.mean(labels['delta']),
            'avg_quality': np.mean(labels['quality'])
        }
        
        return data, stats
    
    def _create_synthetic_data(self):
        print("Creating synthetic training data...")
        
        # Create simple synthetic graph
        np.random.seed(42)
        num_nodes = 100
        
        node_features = np.random.randn(num_nodes, 768)
        
        # Create random edges
        edges = []
        for i in range(0, num_nodes - 1, 2):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        
        edge_index = torch.tensor(edges).t().contiguous()
        
        # Create synthetic labels
        delta_labels = np.random.choice([0, 1], num_nodes, p=[0.7, 0.3])
        quality_labels = np.random.uniform(0, 1, num_nodes)
        strategy_labels = np.random.choice([0, 1, 2, 3], num_nodes)
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y_delta=torch.tensor(delta_labels, dtype=torch.float),
            y_quality=torch.tensor(quality_labels, dtype=torch.float),
            y_strategy=torch.tensor(strategy_labels)
        )
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.size(1),
            'delta_ratio': np.mean(delta_labels),
            'avg_quality': np.mean(quality_labels),
            'synthetic': True
        }
        
        return data, stats

def train_gnn(epochs=50, hidden_dim=256, lr=0.001, output_path='data/models/gnn_persuasion.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    dataset = PersuasionDataset()
    data, stats = dataset.build_graph()
    data = data.to(device)
    
    print(f"Graph stats: {stats}")
    
    # Split data
    num_nodes = data.x.size(0)
    indices = torch.randperm(num_nodes)
    train_idx = indices[:int(0.8 * num_nodes)]
    val_idx = indices[int(0.8 * num_nodes):]
    
    # Create model
    model = PersuasionGNN(input_dim=768, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        
        # Multi-task loss
        delta_loss = F.binary_cross_entropy_with_logits(
            out['delta'][train_idx].squeeze(), 
            data.y_delta[train_idx]
        )
        quality_loss = F.mse_loss(
            out['quality'][train_idx].squeeze(), 
            data.y_quality[train_idx]
        )
        strategy_loss = F.cross_entropy(
            out['strategy'][train_idx], 
            data.y_strategy[train_idx]
        )
        
        total_loss = delta_loss + quality_loss + strategy_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_delta_loss = F.binary_cross_entropy_with_logits(
                    val_out['delta'][val_idx].squeeze(), 
                    data.y_delta[val_idx]
                )
                val_quality_loss = F.mse_loss(
                    val_out['quality'][val_idx].squeeze(), 
                    data.y_quality[val_idx]
                )
                val_strategy_loss = F.cross_entropy(
                    val_out['strategy'][val_idx], 
                    data.y_strategy[val_idx]
                )
                val_loss = val_delta_loss + val_quality_loss + val_strategy_loss
                
                print(f"Epoch {epoch}: Train Loss {total_loss:.4f}, Val Loss {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), output_path)
    
    print(f"Training complete. Model saved to {output_path}")
    return model, stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=str, default="data/models/gnn_persuasion.pt")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train_gnn(args.epochs, args.hidden_dim, args.lr, args.output) 