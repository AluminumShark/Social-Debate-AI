# Social Debate AI Training Guide

*English | [](#chinese-version)*

This guide provides detailed instructions on how to train the three core models of Social Debate AI: RAG, GNN, and RL.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Quick Training](#quick-training)
- [RAG System Training](#rag-system-training)
- [GNN Model Training](#gnn-model-training)
- [RL Model Training](#rl-model-training)
- [Training Monitoring](#training-monitoring)
- [FAQ](#faq)

## Environment Setup

### 1. Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, RTX 3060+ GPU, 8-core CPU
- **Storage**: At least 20GB (for data and models)

### 2. Environment Configuration
```bash
# Create virtual environment
conda create -n social_debate python=3.8
conda activate social_debate

# Install PyTorch (according to your CUDA version)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU version
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### 3. API Key Setup
```bash
# Create .env file
cp env.example .env

# Edit .env, add your OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here
```

## Quick Training

### Train All Models (Recommended)
```bash
python train_all.py --all
```

This will train in sequence:
1. GNN social network model
2. RL strategy selection model
3. RAG retrieval index

Estimated total time: 30-60 minutes (depending on hardware)

### Individual Model Training
```bash
# Train GNN only
python train_all.py --gnn

# Train RL only
python train_all.py --rl

# Build RAG index only
python train_all.py --rag        # Simple index (fast)
python train_all.py --rag-chroma  # Chroma vector index (complete)
python train_all.py --rag-both    # Build both indexes
```

## RAG System Training

RAG (Retrieval-Augmented Generation) system is responsible for retrieving relevant evidence to support debates.

### 1. Simple Index (Quick Test)
```bash
python train_all.py --rag
```

- **Processing Time**: ~5 minutes
- **Document Count**: 45,974 documents
- **Index Size**: ~50MB
- **Use Case**: Quick testing and development

### 2. Chroma Vector Index (Production)
```bash
python train_all.py --rag-chroma
```

- **Processing Time**: ~20-30 minutes
- **Document Chunks**: 94,525 chunks
- **Index Size**: ~500MB
- **Embedding Cost**: ~$0.02 (using OpenAI API)
- **Use Case**: Production environment, high-quality retrieval

### 3. Configuration
Edit `configs/rag.yaml`:

```yaml
chroma:
  embedding:
    batch_size: 500  # Increase batch size for faster processing
    model: "text-embedding-3-small"  # Optional: text-embedding-3-large

indexing:
  quality_filter:
    min_score: 10    # Minimum score filter
    min_length: 50   # Minimum text length
```

## GNN Model Training

GNN (Graph Neural Network) model uses supervised learning to predict persuasion success rate and optimal strategies.

### 1. Training Command
```bash
python train_all.py --gnn
```

### 2. Training Architecture
- **Model Type**: GraphSAGE + GAT attention mechanism
- **Task Type**: Multi-task learning
  - Delta prediction (binary classification)
  - Quality scoring (regression)
  - Strategy classification (multi-class)
- **Training Data**: Delta/non-delta comments from CMV dataset
- **Training Time**: ~10-15 minutes (GPU)

### 3. Monitor Training
Training process displays:
```
Epoch 10/50, Loss: 1.2345, Delta Acc: 0.5678, Quality MAE: 2.3456, Strategy Acc: 0.4567
Epoch 20/50, Loss: 0.8901, Delta Acc: 0.6234, Quality MAE: 1.8901, Strategy Acc: 0.5678
Epoch 30/50, Loss: 0.5678, Delta Acc: 0.6789, Quality MAE: 1.4567, Strategy Acc: 0.6234
```

### 4. Training Performance
- **Delta Accuracy**: ~67-70%
- **Strategy Accuracy**: ~64-67%
- **Quality Prediction MAE**: ~1.2-1.5

### 5. Configuration Adjustment
Edit `configs/gnn.yaml`:

```yaml
training:
  epochs: 50           # Can increase to 100 for better results
  batch_size: 32       # Adjust based on GPU memory
  learning_rate: 0.001 # Adjustable learning rate
  
model:
  hidden_dim: 768      # BERT embedding dimension
  num_layers: 3        # Number of GNN layers
  dropout: 0.1         # Dropout rate
```

## RL Model Training

RL (Reinforcement Learning) model uses PPO algorithm to learn optimal debate strategies.

### 1. Training Command
```bash
python train_all.py --rl
```

### 2. PPO Training Architecture
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Network Architecture**: Actor-Critic dual networks
- **State Space**: Text embeddings + stance + round + history
- **Action Space**: 4 strategies (aggressive, defensive, analytical, empathetic)
- **Training Time**: ~20-30 minutes (1000 episodes)

### 3. Environment Design
Debate environment simulates real interactions:
- **Initial State**: Random stance and belief
- **State Transition**: Based on strategy effectiveness
- **Reward Mechanism**:
  - Strategy effectiveness reward (-1 to +1)
  - Persuasion success reward (+5)
  - Strategy diversity reward (+0.1)
- **Termination**: Surrender or maximum rounds reached

### 4. Training Monitoring
```
Episode 100/1000, Avg Reward: -2.34, Policy Loss: 0.456, Value Loss: 1.234
Episode 200/1000, Avg Reward: -0.89, Policy Loss: 0.234, Value Loss: 0.789
Episode 500/1000, Avg Reward: 1.23, Policy Loss: 0.123, Value Loss: 0.456
Episode 1000/1000, Avg Reward: 2.56, Policy Loss: 0.089, Value Loss: 0.234
```

### 5. Configuration Optimization
Edit `configs/rl.yaml`:

```yaml
ppo:
  episodes: 1000       # Number of training episodes
  max_steps: 50        # Maximum steps per episode
  batch_size: 64       # Batch size
  learning_rate: 3e-4  # Learning rate
  gamma: 0.99          # Discount factor
  clip_epsilon: 0.2    # PPO clipping parameter
  
environment:
  reward_scale: 1.0    # Reward scaling
  persuasion_bonus: 5  # Persuasion success reward
  diversity_bonus: 0.1 # Strategy diversity reward
```

## Training Monitoring

### 1. Check Training Status
```bash
# View model files
ls -la data/models/

# Expected output:
# gnn_social.pt      (GNN model)
# policy/            (RL model directory)
# rag/               (RAG index)
```

### 2. Validate Models
```bash
# Run system integrity test
python test_system_integrity.py
```

### 3. GPU Usage Monitoring
```bash
# NVIDIA GPU
nvidia-smi -l 1

# Watch GPU memory usage
watch -n 1 nvidia-smi
```

## FAQ

### Q1: CUDA out of memory
**Solution**:
- Reduce batch size: adjust `batch_size` in config files
- Use CPU training: install CPU version of PyTorch
- Use mixed precision training (automatically enabled)

### Q2: OpenAI API Error
**Solution**:
- Check if API Key is correctly set
- Confirm API quota is sufficient
- Use simple index instead of Chroma index

### Q3: Training takes too long
**Solution**:
- Use GPU acceleration
- Reduce training epochs
- Use pre-trained models

### Q4: Poor model performance
**Solution**:
- Increase training epochs
- Adjust learning rate
- Ensure data quality

## Training Recommendations

1. **First Use**: Train with default parameters to familiarize with the process
2. **Optimize Results**: Gradually adjust parameters and observe changes
3. **Production Deployment**: Use complete dataset and more training epochs
4. **Regular Updates**: Retrain periodically as new data accumulates

## Advanced Training

### 1. Custom Dataset
```python
# Prepare your data in JSONL format
# Place in data/raw/ directory
# Modify data paths in config files
```

### 2. Model Fine-tuning
```python
# Modify model architecture
# Edit src/gnn/social_encoder.py
# Edit src/rl/policy_network.py
```

### 3. Distributed Training
```bash
# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python train_all.py --all
```

---

**Tip**: If you encounter issues during training, check log files or submit an Issue.

**Support**: For assistance, contact your-email@example.com

---

## Chinese Version

# Social Debate AI 

*[English](#social-debate-ai-training-guide) | *

 Social Debate AI RAGGNN  RL

## 

- [](#)
- [](#)
- [RAG ](#rag-)
- [GNN ](#gnn-)
- [RL ](#rl-)
- [](#)
- [](#)

## 

### 1. 
- ****8GB RAM, 4 CPU
- ****16GB RAM, RTX 3060+ GPU, 8 CPU
- **** 20GB

### 2. 
```bash
# 
conda create -n social_debate python=3.8
conda activate social_debate

#  PyTorch ( CUDA )
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 
pip install torch torchvision torchaudio

# 
pip install -r requirements.txt
```

### 3. API Key 
```bash
#  .env 
cp env.example .env

#  .env OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here
```

## 

### 
```bash
python train_all.py --all
```


1. GNN 
2. RL 
3. RAG 

30-60 

### 
```bash
#  GNN
python train_all.py --gnn

#  RL
python train_all.py --rl

#  RAG 
python train_all.py --rag        # 
python train_all.py --rag-chroma  # Chroma 
python train_all.py --rag-both    # 
```

## RAG 

RAG (Retrieval-Augmented Generation) 

### 1. 
```bash
python train_all.py --rag
```

- **** 5 
- ****45,974 
- **** 50MB
- ****

### 2. Chroma 
```bash
python train_all.py --rag-chroma
```

- **** 20-30 
- ****94,525 
- **** 500MB
- **** $0.02 OpenAI API
- ****

### 3. 
 `configs/rag.yaml`

```yaml
chroma:
  embedding:
    batch_size: 500  # 
    model: "text-embedding-3-small"  #  text-embedding-3-large

indexing:
  quality_filter:
    min_score: 10    # 
    min_length: 50   # 
```

## GNN 

GNN (Graph Neural Network) 

### 1. 
```bash
python train_all.py --gnn
```

### 2. 
- ****GraphSAGE + GAT 
- ****
  - Delta 
  - 
  - 
- ****CMV  delta/non-delta comments
- **** 10-15 GPU

### 3. 

```
Epoch 10/50, Loss: 1.2345, Delta Acc: 0.5678, Quality MAE: 2.3456, Strategy Acc: 0.4567
Epoch 20/50, Loss: 0.8901, Delta Acc: 0.6234, Quality MAE: 1.8901, Strategy Acc: 0.5678
Epoch 30/50, Loss: 0.5678, Delta Acc: 0.6789, Quality MAE: 1.4567, Strategy Acc: 0.6234
```

### 4. 
- **Delta ** 67-70%
- **** 64-67%
- ** MAE** 1.2-1.5

### 5. 
 `configs/gnn.yaml`

```yaml
training:
  epochs: 50           #  100 
  batch_size: 32       #  GPU 
  learning_rate: 0.001 # 
  
model:
  hidden_dim: 768      # BERT 
  num_layers: 3        # GNN 
  dropout: 0.1         # Dropout 
```

## RL 

RL (Reinforcement Learning)  PPO 

### 1. 
```bash
python train_all.py --rl
```

### 2. PPO 
- ****Proximal Policy Optimization (PPO)
- ****Actor-Critic 
- **** +  +  + 
- ****4aggressivedefensiveanalyticalempathetic
- **** 20-30 1000 episodes

### 3. 

- ****
- ****
- ****
  - -1  +1
  - +5
  - +0.1
- ****

### 4. 
```
Episode 100/1000, Avg Reward: -2.34, Policy Loss: 0.456, Value Loss: 1.234
Episode 200/1000, Avg Reward: -0.89, Policy Loss: 0.234, Value Loss: 0.789
Episode 500/1000, Avg Reward: 1.23, Policy Loss: 0.123, Value Loss: 0.456
Episode 1000/1000, Avg Reward: 2.56, Policy Loss: 0.089, Value Loss: 0.234
```

### 5. 
 `configs/rl.yaml`

```yaml
ppo:
  episodes: 1000       # 
  max_steps: 50        # 
  batch_size: 64       # 
  learning_rate: 3e-4  # 
  gamma: 0.99          # 
  clip_epsilon: 0.2    # PPO 
  
environment:
  reward_scale: 1.0    # 
  persuasion_bonus: 5  # 
  diversity_bonus: 0.1 # 
```

## 

### 1. 
```bash
# 
ls -la data/models/

# 
# gnn_social.pt      (GNN )
# policy/            (RL )
# rag/               (RAG )
```

### 2. 
```bash
# 
python test_system_integrity.py
```

### 3. GPU 
```bash
# NVIDIA GPU
nvidia-smi -l 1

#  GPU 
watch -n 1 nvidia-smi
```

## 

### Q1: CUDA out of memory
****
-  `batch_size`
-  CPU  CPU  PyTorch
- 

### Q2: OpenAI API 
****
-  API Key 
-  API 
-  Chroma 

### Q3: 
****
-  GPU 
- 
- 

### Q4: 
****
- 
- 
- 

## 

1. ****
2. ****
3. ****
4. ****

## 

### 1. 
```python
#  JSONL 
#  data/raw/ 
# 
```

### 2. 
```python
# 
#  src/gnn/social_encoder.py
#  src/rl/policy_network.py
```

### 3. 
```bash
#  GPU
CUDA_VISIBLE_DEVICES=0,1 python train_all.py --all
```

---

**** Issue

**** your-email@example.com 