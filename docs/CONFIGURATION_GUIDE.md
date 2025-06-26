# Configuration Guide

*English | [中文](#chinese-version)*

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
  ├── debate.yaml          # Base configuration
  ├── debate.dev.yaml      # Development environment
  └── debate.prod.yaml     # Production environment
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

## 🚨 Common Issues

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

# 配置指南

*[English](#configuration-guide) | 中文*

本指南詳細說明 Social Debate AI 系統的所有配置檔案。

## 📁 配置檔案概覽

系統使用 YAML 格式的配置檔案，位於 `configs/` 目錄：

- **system.yaml** - 系統總體配置
- **debate.yaml** - 辯論系統核心配置
- **gnn.yaml** - 圖神經網路配置
- **rl.yaml** - 強化學習配置
- **rag.yaml** - 檢索增強生成配置

## system.yaml - 系統總體配置

### 基本設定
```yaml
version: "1.0.0"          # 系統版本
mode: "production"        # 運行模式：development/production/debug
```

### 模組管理
```yaml
modules:
  rl:
    enabled: true         # 是否啟用 RL 模組
    type: "ppo"          # RL 類型：ppo/dqn/a2c
  gnn:
    enabled: true
    type: "supervised"    # GNN 類型：supervised/unsupervised
  rag:
    enabled: true
    type: "hybrid"       # RAG 類型：simple/chroma/hybrid
```

### 資源配置
- GPU 記憶體分配
- CPU 線程數限制
- 記憶體使用上限

## debate.yaml - 辯論系統配置

### Agent 設定
```yaml
agent_configs:
  Agent_A:
    initial_stance: 0.8      # 初始立場 (-1 到 1)
    initial_conviction: 0.7  # 初始信念強度 (0 到 1)
    personality: "analytical"
    strategy_preference:     # 策略偏好權重
      analytical: 0.4
      empathetic: 0.3
      defensive: 0.2
      aggressive: 0.1
```

### 勝負判定
```yaml
victory_conditions:
  surrender_threshold: 0.4        # 投降的信念閾值
  stance_neutral_threshold: 0.2   # 中立立場閾值
  consecutive_persuasion: 3       # 連續被說服回合數
```

### 策略融合
```yaml
strategy_fusion:
  method: "weighted_average"      # 融合方法
  weights:
    rl_policy: 0.4               # RL 策略權重
    gnn_suggestion: 0.3          # GNN 建議權重
    personality: 0.3             # 個性偏好權重
```

## 🧠 gnn.yaml - GNN 配置

### 模型架構
```yaml
model:
  architecture: "PersuasionGNN"
  input_dim: 770                 # BERT(768) + 論證特徵(2)
  hidden_dim: 256
  num_layers: 3
  conv_type: "GraphSAGE"
```

### 多任務學習
```yaml
task_weights:
  delta_prediction: 0.5          # Delta 預測權重
  quality_regression: 0.3        # 品質評分權重
  strategy_classification: 0.2    # 策略分類權重
```

### 訓練參數
```yaml
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001
```

## rl.yaml - PPO 強化學習配置

### PPO 算法參數
```yaml
ppo:
  episodes: 1000                 # 訓練回合數
  max_steps: 50                  # 每回合最大步數
  learning_rate: 3e-4
  gamma: 0.99                    # 折扣因子
  clip_epsilon: 0.2              # PPO 裁剪參數
```

### 辯論環境
```yaml
environment:
  reward_scale: 1.0
  persuasion_bonus: 5            # 說服成功獎勵
  surrender_penalty: -3          # 投降懲罰
  diversity_bonus: 0.1           # 策略多樣性獎勵
```

### Actor-Critic 網路
```yaml
policy_network:
  state_dim: 901                 # 狀態空間維度
  hidden_size: 256
  num_strategies: 4              # 策略數量
```

## rag.yaml - RAG 配置

### 混合檢索
```yaml
hybrid_retrieval:
  enabled: true
  weights:
    vector_search: 0.7           # 向量檢索權重
    bm25: 0.3                   # 關鍵詞檢索權重
```

### 重排序
```yaml
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k_rerank: 20              # 重排序候選數
  final_top_k: 5                # 最終返回數
```

### 效能優化
```yaml
optimization:
  batch_processing:
    enabled: true
    batch_size: 32
  gpu_acceleration:
    enabled: true
  index_optimization:
    type: "IVF"                 # 索引類型
```

## 🔄 配置更新流程

### 1. 修改配置
編輯對應的 YAML 檔案，確保語法正確。

### 2. 驗證配置
```bash
python scripts/validate_configs.py
```

### 3. 重新載入
- 開發模式：自動熱重載
- 生產模式：需要重啟服務

## 最佳實踐

### 1. 環境特定配置
為不同環境創建配置：
```
configs/
  ├── debate.yaml          # 基礎配置
  ├── debate.dev.yaml      # 開發環境
  └── debate.prod.yaml     # 生產環境
```

### 2. 敏感資訊
不要在配置中儲存敏感資訊，使用環境變數：
```yaml
api:
  openai:
    api_key_env: "OPENAI_API_KEY"  # 從環境變數讀取
```

### 3. 版本控制
- 將配置檔案納入版本控制
- 使用 `.gitignore` 排除本地配置

### 4. 配置驗證
在啟動前驗證配置：
```python
from utils.config_loader import ConfigLoader

# 載入並驗證配置
config = ConfigLoader.load("debate")
ConfigLoader.validate(config)
```

## 調試配置

### 啟用調試日誌
```yaml
logging:
  level: "DEBUG"
  module_logging:
    rl: "DEBUG"
    gnn: "DEBUG"
    rag: "DEBUG"
```

### 效能分析
```yaml
monitoring:
  system_metrics:
    enabled: true
    interval: 10  # 更頻繁的監控
```

## 配置範例

### 快速測試配置
```yaml
# debate.yaml
debate:
  max_rounds: 3  # 減少回合數
  
# rl.yaml  
ppo:
  episodes: 100  # 減少訓練回合
  
# gnn.yaml
training:
  epochs: 10     # 減少訓練輪數
```

### 高效能配置
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

## 🚨 常見問題

### Q: 配置載入失敗
A: 檢查 YAML 語法，確保縮排正確。

### Q: 配置不生效
A: 確認配置檔案路徑正確，並重啟服務。

### Q: 記憶體不足
A: 調整批次大小和記憶體限制：
```yaml
training:
  batch_size: 8  # 減小批次
resources:
  memory:
    max_usage_gb: 8  # 降低限制
```

---

更多詳細資訊請參考各模組的專門文檔。 