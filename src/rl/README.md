# RL 策略模組（PPO）

## 概述

本模組使用 **Proximal Policy Optimization (PPO)** 算法訓練辯論策略。與舊版本的監督學習方法不同，新版本實現了真正的強化學習，通過與環境互動來學習最佳辯論策略。

## 主要特點

### 1. 真正的強化學習
- **環境互動**：模擬辯論環境，包含立場變化和回合限制
- **獎勵機制**：基於策略效果和說服成功度給予獎勵
- **探索與利用**：平衡嘗試新策略和使用已知好策略

### 2. PPO 算法優勢
- **穩定訓練**：通過限制策略更新幅度避免訓練崩潰
- **高效採樣**：重複使用經驗數據進行多次更新
- **Actor-Critic**：同時學習策略（Actor）和價值函數（Critic）

### 3. 辯論策略
系統支援四種辯論策略：
- `aggressive`：積極攻擊型 - 直接挑戰對方論點
- `defensive`：防禦反駁型 - 鞏固自己的論點
- `analytical`：分析論證型 - 理性分析各方觀點
- `empathetic`：同理說服型 - 理解對方立場

## 檔案結構

```
src/rl/
├── ppo_trainer.py      # PPO 訓練器和環境實現
├── train_ppo.py        # PPO 訓練主程式
├── policy_network.py   # 策略網路介面
└── __init__.py         # 模組初始化
```

## 使用方式

### 訓練 PPO 模型
```bash
# 完整訓練（1000 回合）
python -m src.rl.train_ppo --episodes 1000

# 快速訓練（100 回合）
python -m src.rl.train_ppo --episodes 100

# 自定義參數
python -m src.rl.train_ppo \
    --episodes 500 \
    --lr 1e-4 \
    --gamma 0.95 \
    --eps_clip 0.3
```

### 使用訓練好的策略
```python
from src.rl.policy_network import select_strategy, choose_snippet

# 選擇辯論策略
strategy = select_strategy(
    query="Climate change is not real",
    context="Discussing environmental policies",
    social_context=[0.3, -0.5],  # [我方立場, 對方立場]
    debate_history=[
        {'strategy': 'analytical', 'round': 1},
        {'strategy': 'aggressive', 'round': 2}
    ]
)
print(f"建議策略: {strategy}")

# 選擇最佳證據片段
snippets = [
    "Studies show global temperatures rising",
    "Economic impacts of climate policies",
    "Scientific consensus on climate change"
]
best_snippet = choose_snippet(
    query="Need evidence for climate change",
    snippets=snippets,
    strategy=strategy
)
```

## 環境設計

### 狀態空間
- 文本嵌入（768維）
- 當前立場（-1 到 1）
- 對手立場（-1 到 1）
- 回合進度
- 歷史策略

### 動作空間
- 4種辯論策略選擇

### 獎勵函數
- **策略匹配獎勵**：根據當前狀態選擇合適策略
- **說服獎勵**：成功改變對方立場
- **多樣性懲罰**：避免重複使用同一策略
- **終局獎勵**：成功說服對方或堅守立場

## 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| episodes | 1000 | 訓練回合數 |
| learning_rate | 3e-4 | 學習率 |
| gamma | 0.99 | 折扣因子 |
| eps_clip | 0.2 | PPO clipping 參數 |
| k_epochs | 4 | 每次更新的訓練輪數 |

## 模型架構

### PPO 網路
```
輸入層 (768) 
    ↓
共享層 (256) → ReLU → Dropout
    ↓
共享層 (256) → ReLU → Dropout
    ↓         ↓
Actor頭 (4)  Critic頭 (1)
(策略輸出)   (價值輸出)
```

## 性能指標

訓練過程會追蹤：
- 平均回合獎勵
- 策略分布熵（探索程度）
- 價值函數損失
- 策略改進程度

## 與系統整合

PPO 模組與其他組件的整合：
1. **Orchestrator**：使用 PPO 選擇的策略指導辯論
2. **GNN 模組**：結合社會影響力調整策略
3. **RAG 模組**：根據策略選擇合適的證據 