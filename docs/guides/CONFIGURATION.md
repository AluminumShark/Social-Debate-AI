# Configuration Guide

*English | [](#chinese-version)*

This guide provides detailed explanations of all configuration files in the Social Debate AI system.

## Configuration Files Overview

The system uses YAML format configuration files, located in the `configs/` directory:

- **system.yaml** - System overall configuration
- **debate.yaml** - Debate system core configuration
- **gnn.yaml** - Graph Neural Network configuration
- **rl.yaml** - Reinforcement Learning configuration
- **rag.yaml** - Retrieval Augmented Generation configuration

## system.yaml - System Overall Configuration

### Basic Settings
```yaml
version: "1.0.0"          # System version
mode: "production"        # Run mode: development/production/debug
```

### Module Management
```yaml
modules:
  rl:
    enabled: true         # Whether to enable RL module
    type: "ppo"          # RL type: ppo/dqn/a2c
  gnn:
    enabled: true
    type: "supervised"    # GNN type: supervised/unsupervised
  rag:
    enabled: true
    type: "hybrid"       # RAG type: simple/chroma/hybrid
```

### Resource Configuration
- GPU memory allocation
- CPU thread limit
- memory usage limit

## debate.yaml - Debate System Configuration

### Agent Settings
```yaml
agent_configs:
  Agent_A:
    initial_stance: 0.8      # Initial stance (-1 to 1)
    initial_conviction: 0.7  # Initial belief strength (0 to 1)
    personality: "analytical"
    strategy_preference:     # Strategy preference weights
      analytical: 0.4
      empathetic: 0.3
      defensive: 0.2
      aggressive: 0.1
```

### Victory Conditions
```yaml
victory_conditions:
  surrender_threshold: 0.4        # Surrender belief threshold
  stance_neutral_threshold: 0.2   # Neutral stance threshold
  consecutive_persuasion: 3       # Consecutive persuaded rounds
```

### Strategy Fusion
```yaml
strategy_fusion:
  method: "weighted_average"      # Fusion method
  weights:
    rl_policy: 0.4               # RL strategy weight
    gnn_suggestion: 0.3          # GNN suggestion weight
    personality: 0.3             # Personality preference weight
```

## gnn.yaml - GNN Configuration

### Model Architecture
```yaml
model:
  architecture: "PersuasionGNN"
  input_dim: 770                 # BERT(768) + argument features(2)
  hidden_dim: 256
  num_layers: 3
  conv_type: "GraphSAGE"
```

### Multi-task Learning
```yaml
task_weights:
  delta_prediction: 0.5          # Delta prediction weight
  quality_regression: 0.3        # Quality scoring weight
  strategy_classification: 0.2    # Strategy classification weight
```

### Training Parameters
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001
```

## rl.yaml - PPO Reinforcement Learning Configuration

### PPO Algorithm Parameters
```yaml
ppo:
  episodes: 1000                 # Training episodes
  max_steps: 50                  # Max steps per episode
  learning_rate: 3e-4
  gamma: 0.99                    # Discount factor
  clip_epsilon: 0.2              # PPO clipping parameter
```

### Debate Environment
```yaml
environment:
  reward_scale: 1.0
  persuasion_bonus: 5            # Persuasion success reward
  surrender_penalty: -3          # Surrender penalty
  diversity_bonus: 0.1           # Strategy diversity reward
```

### Actor-Critic Network
```yaml
policy_network:
  state_dim: 901                 # State space dimension
  hidden_size: 256
  num_strategies: 4              # Number of strategies
```

## rag.yaml - RAG Configuration

### Hybrid Retrieval
```yaml
hybrid_retrieval:
  enabled: true
  weights:
    vector_search: 0.7           # Vector retrieval weight
    bm25: 0.3                   # Keyword retrieval weight
```

### Reranking
```yaml
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_rerank: 20              # Reranking candidate count
  final_top_k: 5                # Final return count
```

### Performance Optimization
```yaml
optimization:
  batch_processing:
    enabled: true
    batch_size: 32
  gpu_acceleration:
    enabled: true
  index_optimization:
    type: "IVF"                 # Index type
```

## Configuration Update Process

### 1. Modify Configuration
Edit the corresponding YAML file, ensuring correct syntax.

### 2. Validate Configuration
```bash
python scripts/validate_configs.py
```

### 3. Reload
- Development mode: Automatic hot reload
- Production mode: Requires service restart

## Best Practices

### 1. Environment-specific Configuration
Create configurations for different environments:
```
configs/
   debate.yaml          # Base configuration
   debate.dev.yaml      # Development environment
   debate.prod.yaml     # Production environment
```

### 2. Sensitive Information
Don't store sensitive information in configs, use environment variables:
```yaml
api:
  openai:
    api_key_env: "OPENAI_API_KEY"  # Read from environment variable
```

### 3. Version Control
- Include configuration files in version control
- Use `.gitignore` to exclude local configurations

### 4. Configuration Validation
Validate configuration before startup:
```python
from utils.config_loader import ConfigLoader

# Load and validate configuration
config = ConfigLoader.load("debate")
ConfigLoader.validate(config)
```

## Debug Configuration

### Enable Debug Logging
```yaml
logging:
  level: "DEBUG"
  module_logging:
    rl: "DEBUG"
    gnn: "DEBUG"
    rag: "DEBUG"
```

### Performance Analysis
```yaml
monitoring:
  system_metrics:
    enabled: true
    interval: 10  # More frequent monitoring
```

## Configuration Examples

### Quick Test Configuration
```yaml
# debate.yaml
debate:
  max_rounds: 3  # Reduce rounds
  
# rl.yaml  
ppo:
  episodes: 100  # Reduce training episodes
  
# gnn.yaml
training:
  epochs: 10     # Reduce training epochs
```

### High Performance Configuration
```yaml
# system.yaml
resources:
  gpu:
    memory_fraction: 0.95
  cpu:
    max_threads: 16
    
# rag.yaml
optimization:
  batch_processing:
    batch_size: 64
  index_optimization:
    type: "HNSW"
```

##  Common Issues

### Q: Configuration loading failed
A: Check YAML syntax, ensure correct indentation.

### Q: Configuration not taking effect
A: Confirm configuration file path is correct and restart service.

### Q: Out of memory
A: Adjust batch size and memory limits:
```yaml
training:
  batch_size: 8  # Reduce batch size
resources:
  memory:
    max_usage_gb: 8  # Lower limit
```

---

For more detailed information, please refer to the dedicated documentation for each module.

---

## Chinese Version

# 

*[English](#configuration-guide) | *

 Social Debate AI 

##  

 YAML  `configs/` 

- **system.yaml** - 
- **debate.yaml** - 
- **gnn.yaml** - 
- **rl.yaml** - 
- **rag.yaml** - 

## system.yaml - 

### 
```yaml
version: "1.0.0"          # 
mode: "production"        # development/production/debug
```

### 
```yaml
modules:
  rl:
    enabled: true         #  RL 
    type: "ppo"          # RL ppo/dqn/a2c
  gnn:
    enabled: true
    type: "supervised"    # GNN supervised/unsupervised
  rag:
    enabled: true
    type: "hybrid"       # RAG simple/chroma/hybrid
```

### 
- GPU 
- CPU 
- 

## debate.yaml - 

### Agent 
```yaml
agent_configs:
  Agent_A:
    initial_stance: 0.8      #  (-1  1)
    initial_conviction: 0.7  #  (0  1)
    personality: "analytical"
    strategy_preference:     # 
      analytical: 0.4
      empathetic: 0.3
      defensive: 0.2
      aggressive: 0.1
```

### 
```yaml
victory_conditions:
  surrender_threshold: 0.4        # 
  stance_neutral_threshold: 0.2   # 
  consecutive_persuasion: 3       # 
```

### 
```yaml
strategy_fusion:
  method: "weighted_average"      # 
  weights:
    rl_policy: 0.4               # RL 
    gnn_suggestion: 0.3          # GNN 
    personality: 0.3             # 
```

##  gnn.yaml - GNN 

### 
```yaml
model:
  architecture: "PersuasionGNN"
  input_dim: 770                 # BERT(768) + (2)
  hidden_dim: 256
  num_layers: 3
  conv_type: "GraphSAGE"
```

### 
```yaml
task_weights:
  delta_prediction: 0.5          # Delta 
  quality_regression: 0.3        # 
  strategy_classification: 0.2    # 
```

### 
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001
```

## rl.yaml - PPO 

### PPO 
```yaml
ppo:
  episodes: 1000                 # 
  max_steps: 50                  # 
  learning_rate: 3e-4
  gamma: 0.99                    # 
  clip_epsilon: 0.2              # PPO 
```

### 
```yaml
environment:
  reward_scale: 1.0
  persuasion_bonus: 5            # 
  surrender_penalty: -3          # 
  diversity_bonus: 0.1           # 
```

### Actor-Critic 
```yaml
policy_network:
  state_dim: 901                 # 
  hidden_size: 256
  num_strategies: 4              # 
```

## rag.yaml - RAG 

### 
```yaml
hybrid_retrieval:
  enabled: true
  weights:
    vector_search: 0.7           # 
    bm25: 0.3                   # 
```

### 
```yaml
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_rerank: 20              # 
  final_top_k: 5                # 
```

### 
```yaml
optimization:
  batch_processing:
    enabled: true
    batch_size: 32
  gpu_acceleration:
    enabled: true
  index_optimization:
    type: "IVF"                 # 
```

##  

### 1. 
 YAML 

### 2. 
```bash
python scripts/validate_configs.py
```

### 3. 
- 
- 

## 

### 1. 

```
configs/
   debate.yaml          # 
   debate.dev.yaml      # 
   debate.prod.yaml     # 
```

### 2. 

```yaml
api:
  openai:
    api_key_env: "OPENAI_API_KEY"  # 
```

### 3. 
- 
-  `.gitignore` 

### 4. 

```python
from utils.config_loader import ConfigLoader

# 
config = ConfigLoader.load("debate")
ConfigLoader.validate(config)
```

## 

### 
```yaml
logging:
  level: "DEBUG"
  module_logging:
    rl: "DEBUG"
    gnn: "DEBUG"
    rag: "DEBUG"
```

### 
```yaml
monitoring:
  system_metrics:
    enabled: true
    interval: 10  # 
```

## 

### 
```yaml
# debate.yaml
debate:
  max_rounds: 3  # 
  
# rl.yaml  
ppo:
  episodes: 100  # 
  
# gnn.yaml
training:
  epochs: 10     # 
```

### 
```yaml
# system.yaml
resources:
  gpu:
    memory_fraction: 0.95
  cpu:
    max_threads: 16
    
# rag.yaml
optimization:
  batch_processing:
    batch_size: 64
  index_optimization:
    type: "HNSW"
```

##  

### Q: 
A:  YAML 

### Q: 
A: 

### Q: 
A: 
```yaml
training:
  batch_size: 8  # 
resources:
  memory:
    max_usage_gb: 8  # 
```

---

 