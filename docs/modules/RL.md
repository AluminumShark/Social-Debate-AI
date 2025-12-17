# 🎮 RL Module Quick Reference

*English | [中文](#中文版本)*

> 📚 **For detailed explanation, see [LEARNING_NOTE §3: PPO Deep Dive](../LEARNING_NOTE.md#part-3-ppo-deep-dive-)**

---

## Overview

The RL (Reinforcement Learning) module uses **PPO (Proximal Policy Optimization)** algorithm with an **Actor-Critic** architecture to select optimal debate strategies.

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Input["📥 Input"]
        S["State (768-dim)"]
    end

    subgraph Network["🧠 PPO Network"]
        SH["Shared Layers"]
        subgraph Heads["Dual Heads"]
            A["Actor<br/>(Policy)"]
            C["Critic<br/>(Value)"]
        end
        SH --> A & C
    end

    subgraph Output["📊 Output"]
        P["Action Probs<br/>[4 strategies]"]
        V["State Value"]
    end

    Input --> Network
    A --> P
    C --> V

    style Input fill:#06b6d4,color:#fff
    style Network fill:#8b5cf6,color:#fff
    style Output fill:#10b981,color:#fff
```

---

## API Reference

### `PPONetwork`

```python
from src.rl.ppo_trainer import PPONetwork

network = PPONetwork(
    state_dim=768,     # State dimension
    action_dim=4,      # Number of strategies
    hidden_dim=256     # Hidden layer dimension
)
```

### `select_action()`

```python
action, log_prob, value = network.select_action(state_tensor)

# action: 0=aggressive, 1=defensive, 2=analytical, 3=empathetic
```

### `rl_select_strategy()` (LangGraph Tool)

```python
from src.orchestrator.debate_tools import rl_select_strategy

result = rl_select_strategy.invoke({
    "context": "Topic: AI regulation...",
    "social_context": [0.1, -0.2, ...]  # 128-dim
})

# Returns:
{
    'strategy': 'analytical',
    'quality_score': 0.75,
    'confidence': 0.8
}
```

---

## Strategies

| ID | Strategy | Description | Best When |
|----|----------|-------------|-----------|
| 0 | `aggressive` | Challenge opponent directly | Opponent has weak arguments |
| 1 | `defensive` | Strengthen own position | Under strong attack |
| 2 | `analytical` | Use logic and evidence | Building credibility |
| 3 | `empathetic` | Find common ground | Seeking consensus |

---

## Architecture Summary

| Component | Dimensions | Purpose |
|-----------|------------|---------|
| Shared Layer 1 | 768 → 256 | Feature extraction |
| Shared Layer 2 | 256 → 256 | Representation learning |
| Actor Head | 256 → 128 → 4 | Policy (action probabilities) |
| Critic Head | 256 → 128 → 1 | Value (state estimation) |

---

## Configuration

`configs/rl.yaml`:

```yaml
ppo:
  gamma: 0.99           # Discount factor
  gae_lambda: 0.95      # GAE parameter
  epsilon: 0.2          # Clipping parameter
  update_epochs: 4      # PPO update epochs
  
network:
  state_dim: 768
  action_dim: 4
  hidden_dim: 256
  
training:
  learning_rate: 0.0003
  batch_size: 64
```

---

## Key Files

| File | Description |
|------|-------------|
| `src/rl/policy_network.py` | Policy network for inference |
| `src/rl/ppo_trainer.py` | PPO trainer and environment |
| `src/rl/train_ppo.py` | Training script |

---

<a name="中文版本"></a>

# 🎮 RL 模組快速參考

*[English](#-rl-module-quick-reference) | 中文*

> 📚 **詳細說明請見 [LEARNING_NOTE §3: PPO 深度解析](../LEARNING_NOTE.md#part-3-ppo-deep-dive-)**

---

## 概述

RL（強化學習）模組使用 **PPO（近端策略優化）** 算法和 **Actor-Critic** 架構來選擇最佳辯論策略。

---

## API 參考

### `PPONetwork`

```python
from src.rl.ppo_trainer import PPONetwork

network = PPONetwork(
    state_dim=768,     # 狀態維度
    action_dim=4,      # 策略數量
    hidden_dim=256     # 隱藏層維度
)
```

### `select_action()`

```python
action, log_prob, value = network.select_action(state_tensor)

# action: 0=激進, 1=防守, 2=分析, 3=同理
```

---

## 策略

| ID | 策略 | 說明 | 適用情境 |
|----|------|------|----------|
| 0 | `aggressive` | 直接挑戰對手 | 對手論點薄弱時 |
| 1 | `defensive` | 鞏固自身立場 | 受到強烈攻擊時 |
| 2 | `analytical` | 使用邏輯和證據 | 建立可信度時 |
| 3 | `empathetic` | 尋找共同點 | 尋求共識時 |

---

## 架構摘要

| 組件 | 維度 | 用途 |
|------|------|------|
| 共享層 1 | 768 → 256 | 特徵提取 |
| 共享層 2 | 256 → 256 | 表示學習 |
| Actor Head | 256 → 128 → 4 | 策略（動作機率）|
| Critic Head | 256 → 128 → 1 | 價值（狀態估計）|

---

## 配置

`configs/rl.yaml`:

```yaml
ppo:
  gamma: 0.99           # 折扣因子
  gae_lambda: 0.95      # GAE 參數
  epsilon: 0.2          # 裁剪參數
  update_epochs: 4      # PPO 更新次數
  
network:
  state_dim: 768
  action_dim: 4
  hidden_dim: 256
```

---

## 關鍵文件

| 文件 | 說明 |
|------|------|
| `src/rl/policy_network.py` | 推理用策略網路 |
| `src/rl/ppo_trainer.py` | PPO 訓練器和環境 |
| `src/rl/train_ppo.py` | 訓練腳本 |
