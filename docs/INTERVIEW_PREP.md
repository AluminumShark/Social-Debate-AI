# 🎯 Google 面試準備完整指南
## Social Debate AI - 多智能體社會辯論系統

> **這份文件的目標**：讓你徹底搞懂每個技術細節，自信地通過 Google 面試

---

# 📚 目錄

## Part 1: 專案基礎
- [1.1 專案是什麼？](#11-專案是什麼)
- [1.2 系統架構總覽](#12-系統架構總覽)
- [1.3 程式碼結構](#13-程式碼結構)

## Part 2: GNN 深度解析 ⭐
- [2.1 什麼是圖神經網路？](#21-什麼是圖神經網路)
- [2.2 GraphSAGE 原理詳解](#22-graphsage-原理詳解)
- [2.3 GAT (Graph Attention) 原理](#23-gat-graph-attention-原理)
- [2.4 本專案 GNN 架構逐行解析](#24-本專案-gnn-架構逐行解析)
- [2.5 GNN 訓練流程](#25-gnn-訓練流程)
- [2.6 GNN 推理流程](#26-gnn-推理流程)

## Part 3: PPO 深度解析 ⭐
- [3.1 強化學習基礎概念](#31-強化學習基礎概念)
- [3.2 Policy Gradient 方法](#32-policy-gradient-方法)
- [3.3 PPO 核心原理](#33-ppo-核心原理)
- [3.4 Actor-Critic 架構](#34-actor-critic-架構)
- [3.5 GAE (廣義優勢估計)](#35-gae-廣義優勢估計)
- [3.6 本專案 PPO 實作逐行解析](#36-本專案-ppo-實作逐行解析)

## Part 4: RAG 與 LangGraph
- [4.1 RAG 原理與實作](#41-rag-原理與實作)
- [4.2 LangGraph 狀態機](#42-langgraph-狀態機)

## Part 5: 面試準備
- [5.1 一分鐘自我介紹](#51-一分鐘自我介紹)
- [5.2 三個刁鑽問題](#52-三個刁鑽問題)
- [5.3 面試檢查清單](#53-面試檢查清單)

---

# Part 1: 專案基礎

## 1.1 專案是什麼？

### 一句話說明
> **Social Debate AI** = 3 個 AI 互相辯論，每個 AI 會：
> 1. 用 **GNN** 分析「誰容易被說服」
> 2. 用 **RL (PPO)** 選擇「攻擊/防守/分析/同理」策略
> 3. 用 **RAG** 查找支持論點的證據
> 4. 用 **GPT-4** 生成辯論發言

### 視覺化理解

```
┌─────────────────── 一場辯論的流程 ───────────────────┐
│                                                      │
│  議題: "AI 應該被政府監管嗎？"                        │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │              Agent A 的回合                      │ │
│  │                                                 │ │
│  │  Step 1: 平行分析 (同時進行)                    │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐          │ │
│  │  │   GNN   │ │   PPO   │ │   RAG   │          │ │
│  │  │         │ │         │ │         │          │ │
│  │  │ 分析:   │ │ 選擇:   │ │ 檢索:   │          │ │
│  │  │ B 容易  │ │ 用分析型│ │ 找到3條 │          │ │
│  │  │ 被說服  │ │ 策略    │ │ 相關證據│          │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘          │ │
│  │       └───────────┼───────────┘               │ │
│  │                   ▼                            │ │
│  │  Step 2: 融合結果                              │ │
│  │  → 最終策略: analytical                        │ │
│  │  → 最佳證據: "歐盟 AI Act..."                  │ │
│  │                   │                            │ │
│  │                   ▼                            │ │
│  │  Step 3: GPT-4 生成發言                        │ │
│  │  "根據歐盟的 AI Act，我認為..."                │ │
│  │                   │                            │ │
│  │                   ▼                            │ │
│  │  Step 4: 更新狀態                              │ │
│  │  • B 的信念: 0.7 → 0.65 (被說服一點)          │ │
│  │  • C 的立場: 0.0 → 0.1 (稍微偏支持)           │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  接下來: Agent B 的回合、Agent C 的回合...           │
│  直到: 達到最大回合數 或 有人投降                    │
└──────────────────────────────────────────────────────┘
```

---

## 1.2 系統架構總覽

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Web 介面 (Flask)                            │
│                     http://localhost:5000                           │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                           │
│              src/orchestrator/langgraph_orchestrator.py             │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                    StateGraph 狀態機                       │    │
│   │                                                           │    │
│   │   parallel_analysis ──► fuse_results ──► generate_response│    │
│   │          │                                      │         │    │
│   │          ▼                                      ▼         │    │
│   │   ┌─────────────┐                      update_states     │    │
│   │   │ 三個模組並行 │                           │           │    │
│   │   │ RL + GNN    │                           ▼           │    │
│   │   │    + RAG    │                    should_continue     │    │
│   │   └─────────────┘                      ↓      ↓         │    │
│   │                                     continue   END       │    │
│   └───────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                    │               │               │
          ┌─────────┘               │               └─────────┐
          ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   GNN 模組       │       │   RL 模組        │       │   RAG 模組       │
│ src/gnn/        │       │ src/rl/         │       │ src/rag/        │
│                 │       │                 │       │                 │
│ • GraphSAGE    │       │ • PPO 訓練器    │       │ • FAISS 索引   │
│ • GAT 注意力   │       │ • Actor-Critic  │       │ • OpenAI Embed │
│ • 多任務學習   │       │ • GAE 優勢估計  │       │ • Chroma DB    │
│                 │       │                 │       │                 │
│ 輸出:           │       │ 輸出:           │       │ 輸出:           │
│ • 說服成功率   │       │ • 策略選擇      │       │ • 相關證據     │
│ • 最佳策略     │       │ • 品質分數      │       │ • 信心分數     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 1.3 程式碼結構

```
Social-Debate-AI/
│
├── src/
│   ├── gnn/                      # 🧠 圖神經網路
│   │   ├── social_encoder.py     # GNN 模型定義 + 推理
│   │   └── train_supervised.py   # GNN 訓練腳本
│   │
│   ├── rl/                       # 🎮 強化學習
│   │   ├── policy_network.py     # 策略網路 (推理用)
│   │   └── ppo_trainer.py        # PPO 訓練器
│   │
│   ├── rag/                      # 🔍 知識檢索
│   │   ├── retriever.py          # 增強檢索器
│   │   └── simple_retriever.py   # 簡單檢索器
│   │
│   └── orchestrator/             # 🎯 編排器
│       ├── langgraph_orchestrator.py  # 主編排器
│       ├── debate_state.py            # 狀態定義
│       └── debate_tools.py            # 工具包裝
│
├── ui/                           # 🌐 Web 介面
│   └── app.py                    # Flask 應用
│
└── configs/                      # ⚙️ 配置
    ├── gnn.yaml
    ├── rl.yaml
    └── ...
```

---

# Part 2: GNN 深度解析 ⭐

## 2.1 什麼是圖神經網路？

### 從普通神經網路說起

**普通神經網路 (MLP)**：
```
輸入: 固定大小的向量 [x1, x2, x3, ...]
        │
        ▼
    全連接層
        │
        ▼
輸出: 固定大小的向量 [y1, y2, ...]

問題: 無法處理「關係」數據
```

**圖神經網路 (GNN)**：
```
輸入: 圖 = 節點 + 邊
      
      節點A ────── 節點B
        │    ╲      │
        │     ╲     │
        │      ╲    │
      節點C ────── 節點D
      
每個節點有特徵向量
邊表示關係

GNN 能學習:
1. 節點自己的特徵
2. 鄰居節點的特徵
3. 整個圖的結構
```

### 為什麼辯論需要 GNN？

```
辯論場景的圖結構:

   帖子(Post) ─────────► 回覆1(Reply)
       │                    │
       │                    ▼
       │              回覆1的回覆
       │
       └─────────────► 回覆2(Reply)
                           │
                           ▼
                      回覆2的回覆
                      (成功說服! ✓)

GNN 可以學習:
• 什麼樣的回覆容易成功說服？
• 在什麼「對話結構」下說服率高？
• 哪種「策略」對哪種人有效？
```

---

## 2.2 GraphSAGE 原理詳解

### 問題：如何從鄰居獲取信息？

**傳統方法 (GCN)**：
```
節點 i 的新特徵 = Σ (鄰居 j 的特徵) / 鄰居數量

問題: 計算量大，需要全圖數據
```

**GraphSAGE (SAmple and aggreGatE)**：
```
創新點：採樣 + 聚合

Step 1: 採樣 K 個鄰居（不用全部）
Step 2: 聚合鄰居特徵（mean/max/LSTM）
Step 3: 結合自己的特徵
Step 4: 非線性變換

優點: 可以處理大規模圖，可以泛化到新節點
```

### GraphSAGE 數學公式

```
第 k 層的更新:

h_N(v)^k = AGGREGATE_k({h_u^(k-1), ∀u ∈ N(v)})
                     ↑
                 鄰居聚合

h_v^k = σ(W^k · CONCAT(h_v^(k-1), h_N(v)^k))
        ↑         ↑              ↑
    激活函數   可學習權重    自己 + 鄰居拼接

其中:
• h_v^k = 節點 v 在第 k 層的特徵
• N(v) = 節點 v 的鄰居集合
• AGGREGATE = 聚合函數 (mean, max, etc.)
• W^k = 第 k 層的權重矩陣
• σ = 激活函數 (ReLU)
```

### 程式碼對應 (PyTorch Geometric)

```python
# src/gnn/social_encoder.py 第 19-21 行

self.conv1 = tgnn.SAGEConv(input_dim, hidden_dim)
#           ↑
#    PyTorch Geometric 的 GraphSAGE 層
#    input_dim = 768 (BERT 嵌入維度)
#    hidden_dim = 256

self.conv2 = tgnn.SAGEConv(hidden_dim, hidden_dim)
# 第二層: 256 → 256

self.conv3 = tgnn.SAGEConv(hidden_dim, 128)
# 第三層: 256 → 128
```

### SAGEConv 內部做了什麼？

```python
# SAGEConv 的簡化版本 (實際實作更複雜)

class SAGEConv:
    def forward(self, x, edge_index):
        # x: 所有節點的特徵 [num_nodes, input_dim]
        # edge_index: 邊的列表 [[源節點], [目標節點]]
        
        # Step 1: 對每個節點，聚合鄰居特徵
        neighbor_features = aggregate_neighbors(x, edge_index)
        # neighbor_features: [num_nodes, input_dim]
        
        # Step 2: 拼接自己和鄰居的特徵
        combined = concat(x, neighbor_features)
        # combined: [num_nodes, input_dim * 2]
        
        # Step 3: 線性變換
        output = linear_layer(combined)
        # output: [num_nodes, output_dim]
        
        return output
```

---

## 2.3 GAT (Graph Attention) 原理

### 問題：鄰居的重要性應該相同嗎？

```
GraphSAGE 的 mean 聚合:
  新特徵 = (鄰居1 + 鄰居2 + 鄰居3) / 3
  
問題: 每個鄰居的權重都是 1/3，但有些鄰居更重要！
```

### GAT 的解決方案：注意力機制

```
核心想法: 學習每個鄰居的「重要性權重」

  節點A 的新特徵 = α_B × 鄰居B + α_C × 鄰居C + α_D × 鄰居D
                  ↑
              學習出來的權重，加起來 = 1
              
例如:
  α_B = 0.5 (鄰居B 很重要)
  α_C = 0.3 (鄰居C 中等)
  α_D = 0.2 (鄰居D 不太重要)
```

### 注意力係數怎麼計算？

```
Step 1: 計算節點對的「注意力分數」

e_ij = LeakyReLU(a^T · [W·h_i || W·h_j])
       ↑          ↑    ↑         ↑
    激活函數   可學習向量 線性變換後拼接

Step 2: Softmax 歸一化

α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_k exp(e_ik)

Step 3: 加權聚合

h'_i = σ(Σ_j α_ij · W · h_j)
```

### 多頭注意力 (Multi-Head Attention)

```
想法: 用多組注意力，捕捉不同類型的關係

         ┌──── Head 1 ────┐
輸入 ────┼──── Head 2 ────┼──── 拼接/平均 ──── 輸出
         ├──── Head 3 ────┤
         └──── Head 4 ────┘

每個 head 學習不同的注意力模式
```

### 程式碼對應

```python
# src/gnn/social_encoder.py 第 24 行

self.attention = tgnn.GATConv(128, 128, heads=4, concat=False)
#                              ↑    ↑     ↑       ↑
#                          輸入維度 輸出維度 4個頭 不拼接(取平均)

# heads=4: 使用 4 個獨立的注意力頭
# concat=False: 最後對 4 個頭的結果取平均
#              如果 concat=True，會拼接成 128*4=512 維
```

---

## 2.4 本專案 GNN 架構逐行解析

### 完整架構圖

```
輸入: x [num_nodes, 768]  (BERT 嵌入)
      edge_index [2, num_edges]  (邊列表)

        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 1: SAGEConv(768 → 256) + ReLU + Dropout(0.3)           │
│                                                               │
│   x = F.relu(self.conv1(x, edge_index))  # [N, 768] → [N, 256]│
│   x = self.dropout(x)                                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 2: SAGEConv(256 → 256) + ReLU + Dropout(0.3)           │
│                                                               │
│   x = F.relu(self.conv2(x, edge_index))  # [N, 256] → [N, 256]│
│   x = self.dropout(x)                                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 3: SAGEConv(256 → 128) + ReLU (無 Dropout)             │
│                                                               │
│   x = F.relu(self.conv3(x, edge_index))  # [N, 256] → [N, 128]│
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ Layer 4: GATConv(128 → 128, heads=4) + Dropout(0.3)          │
│                                                               │
│   x = self.attention(x, edge_index)  # [N, 128] → [N, 128]   │
│   x = self.dropout(x)                                         │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│ 可選: Global Mean Pooling (如果有 batch 信息)                 │
│                                                               │
│   if batch is not None:                                       │
│       x = global_mean_pool(x, batch)  # [N, 128] → [B, 128]  │
└───────────────────────────────────────────────────────────────┘
        │
        ├──────────────────┬──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  delta_head  │   │ quality_head │   │strategy_head │
│              │   │              │   │              │
│ Linear(128,64)│  │ Linear(128,64)│  │ Linear(128,64)│
│ ReLU         │   │ ReLU         │   │ ReLU         │
│ Dropout(0.3) │   │ Dropout(0.3) │   │ Dropout(0.3) │
│ Linear(64,1) │   │ Linear(64,1) │   │ Linear(64,4) │
│              │   │              │   │              │
│ 輸出: [N,1]  │   │ 輸出: [N,1]  │   │ 輸出: [N,4]  │
│ 說服成功率   │   │ 品質分數     │   │ 策略機率     │
└──────────────┘   └──────────────┘   └──────────────┘
```

### 程式碼逐行解析

```python
# src/gnn/social_encoder.py

class PersuasionGNN(nn.Module):
    
    def __init__(self, input_dim=768, hidden_dim=256, num_strategies=4):
        super().__init__()
        
        # ====== 圖卷積層 ======
        # 這三層逐步從 768 維壓縮到 128 維
        # 同時聚合鄰居信息
        
        self.conv1 = tgnn.SAGEConv(input_dim, hidden_dim)
        # input_dim=768: BERT 的輸出維度
        # hidden_dim=256: 壓縮到 256 維
        
        self.conv2 = tgnn.SAGEConv(hidden_dim, hidden_dim)
        # 256 → 256，保持維度，繼續聚合
        
        self.conv3 = tgnn.SAGEConv(hidden_dim, 128)
        # 256 → 128，再次壓縮
        
        # ====== 注意力層 ======
        self.attention = tgnn.GATConv(128, 128, heads=4, concat=False)
        # 4 個注意力頭，學習不同的關注模式
        # concat=False: 4 個頭的結果取平均，而不是拼接
        
        # ====== 正則化 ======
        self.dropout = nn.Dropout(0.3)
        # 30% 的節點會被隨機置零，防止過擬合
        
        # ====== 多任務輸出頭 ======
        # 共享特徵提取，分別預測 3 個任務
        
        self.delta_head = nn.Sequential(
            nn.Linear(128, 64),     # 128 → 64
            nn.ReLU(),              # 激活函數
            nn.Dropout(0.3),        # 正則化
            nn.Linear(64, 1)        # 64 → 1 (二分類: 是否成功說服)
        )
        # delta = "改變觀點"，預測這個回覆是否能說服對方
        
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)        # 64 → 1 (回歸: 品質分數 0-1)
        )
        # 預測回覆的品質分數
        
        self.strategy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_strategies)  # 64 → 4 (四分類)
        )
        # 預測應該使用哪種策略
    
    def forward(self, x, edge_index, batch=None):
        """
        前向傳播
        
        Args:
            x: 節點特徵 [num_nodes, input_dim]
            edge_index: 邊列表 [2, num_edges]
            batch: 批次索引 (用於多圖) [num_nodes]
        """
        
        # ====== 第 1 層 GraphSAGE ======
        x = F.relu(self.conv1(x, edge_index))
        # x: [N, 768] → [N, 256]
        # 每個節點聚合了 1-hop 鄰居的信息
        
        x = self.dropout(x)
        # 隨機丟棄 30% 的特徵
        
        # ====== 第 2 層 GraphSAGE ======
        x = F.relu(self.conv2(x, edge_index))
        # x: [N, 256] → [N, 256]
        # 現在每個節點聚合了 2-hop 鄰居的信息
        
        x = self.dropout(x)
        
        # ====== 第 3 層 GraphSAGE ======
        x = F.relu(self.conv3(x, edge_index))
        # x: [N, 256] → [N, 128]
        # 3-hop 鄰居信息
        
        # ====== 注意力層 ======
        x = self.attention(x, edge_index)
        # 用注意力重新加權鄰居的重要性
        
        x = self.dropout(x)
        
        # ====== 圖級別池化 (可選) ======
        if batch is not None:
            x = tgnn.global_mean_pool(x, batch)
            # 如果有多個圖，對每個圖的所有節點取平均
            # [N, 128] → [B, 128] 其中 B 是圖的數量
        
        # ====== 多任務預測 ======
        delta_pred = self.delta_head(x)      # [N, 1] 或 [B, 1]
        quality_pred = self.quality_head(x)  # [N, 1] 或 [B, 1]
        strategy_pred = self.strategy_head(x) # [N, 4] 或 [B, 4]
        
        return {
            'delta': delta_pred,       # 說服成功率 (需要 sigmoid)
            'quality': quality_pred,   # 品質分數 (需要 sigmoid)
            'strategy': strategy_pred, # 策略機率 (需要 softmax)
            'embeddings': x            # 128 維嵌入向量
        }
```

---

## 2.5 GNN 訓練流程

### 數據準備

```python
# src/gnn/train_supervised.py

class PersuasionDataset:
    """
    數據來源: Reddit 的 ChangeMyView (CMV) 數據集
    CMV 是一個辯論論壇，用戶發帖表達觀點，
    如果被說服會給對方 "delta" (Δ) 標記
    """
    
    def __init__(self):
        # 使用 DistilBERT 編碼文本
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        
    def encode_text(self, text: str) -> np.ndarray:
        """
        將文本轉成 768 維向量
        
        "I think AI should be regulated" 
              ↓ DistilBERT
        [0.12, -0.34, 0.56, ...] (768 維)
        """
        inputs = self.tokenizer(text, truncation=True, max_length=512, 
                                return_tensors='pt')
        outputs = self.encoder(**inputs)
        
        # 取 [CLS] token 的輸出作為整句的表示
        return outputs.last_hidden_state[:, 0, :].numpy().squeeze()
        #                                ↑
        #                        第 0 個 token = [CLS]
    
    def build_graph(self):
        """
        構建圖數據
        
        圖的結構:
        • 節點: 每個帖子和回覆
        • 邊: 帖子 ↔ 回覆 (雙向)
        • 標籤:
          - delta: 是否成功說服 (0/1)
          - quality: 品質分數 (0-1)
          - strategy: 使用的策略 (0-3)
        """
        ...
```

### 訓練循環

```python
def train_gnn(epochs=50, hidden_dim=256, lr=0.001):
    # 載入數據
    data, stats = dataset.build_graph()
    
    # 創建模型
    model = PersuasionGNN(input_dim=768, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 前向傳播
        out = model(data.x, data.edge_index)
        
        # ====== 多任務損失 ======
        # 任務 1: 說服預測 (二分類)
        delta_loss = F.binary_cross_entropy_with_logits(
            out['delta'].squeeze(),     # 模型輸出
            data.y_delta                # 真實標籤 (0 或 1)
        )
        
        # 任務 2: 品質預測 (回歸)
        quality_loss = F.mse_loss(
            out['quality'].squeeze(),   # 模型輸出
            data.y_quality              # 真實標籤 (0-1)
        )
        
        # 任務 3: 策略分類 (四分類)
        strategy_loss = F.cross_entropy(
            out['strategy'],            # 模型輸出 [N, 4]
            data.y_strategy             # 真實標籤 (0, 1, 2, 或 3)
        )
        
        # 總損失 = 三個任務的加權和
        total_loss = delta_loss + quality_loss + strategy_loss
        
        # 反向傳播
        total_loss.backward()
        optimizer.step()
```

### 為什麼用多任務學習？

```
單任務學習:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 模型 A      │     │ 模型 B      │     │ 模型 C      │
│ 預測 delta  │     │ 預測 quality│     │ 預測 strategy│
└─────────────┘     └─────────────┘     └─────────────┘
  ↑                   ↑                   ↑
  獨立訓練             獨立訓練             獨立訓練
  
問題: 
• 3 個模型，3 倍參數量
• 沒有共享知識

多任務學習:
┌─────────────────────────────────────┐
│            共享特徵提取              │
│         (GraphSAGE + GAT)           │
└─────────────┬───────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ delta │ │quality│ │strategy│
│ head  │ │ head  │ │ head  │
└───────┘ └───────┘ └───────┘

優點:
• 參數共享，更高效
• 任務間共享知識
• 正則化效果（防止過擬合）
```

---

## 2.6 GNN 推理流程

### 推理時的問題

```python
# src/orchestrator/debate_tools.py 第 161-162 行

text_features = np.random.randn(768)  # ⚠️ 這是問題！
persuasion_pred = gnn.predict_persuasion(text_features, agent_id)

# 問題: 
# 訓練時用的是 DistilBERT 編碼的真實文本
# 推理時用的是隨機噪聲
# 這導致模型的預測毫無意義！
```

### 正確的推理流程應該是

```python
# 正確做法:

# 1. 獲取當前辯論上下文
context = f"Topic: {topic}. Last message: {last_message}"

# 2. 使用 DistilBERT 編碼
text_features = encoder.encode(context)  # [768]

# 3. 調用 GNN 預測
result = predict_persuasion(text_features, agent_id)

# 4. 結果
print(result)
# {
#     'delta_probability': 0.73,        # 有 73% 機率說服成功
#     'quality_score': 0.65,            # 品質分數 0.65
#     'best_strategy': 'analytical',    # 建議用分析型策略
#     'strategy_scores': {
#         'aggressive': 0.15,
#         'defensive': 0.20,
#         'analytical': 0.45,  # 最高
#         'empathetic': 0.20
#     }
# }
```

---

# Part 3: PPO 深度解析 ⭐

## 3.1 強化學習基礎概念

### RL 的核心要素

```
┌─────────────────────────────────────────────────────────────────┐
│                    強化學習框架                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        環境 (Environment)                        │
│                              ↓                                   │
│                         狀態 s_t                                 │
│                              ↓                                   │
│     ┌─────────────────────────────────────────────┐            │
│     │                智能體 (Agent)                │            │
│     │                                             │            │
│     │  觀察狀態 s_t ──► 策略 π(a|s) ──► 選擇動作 a_t│            │
│     │                                             │            │
│     └─────────────────────────────────────────────┘            │
│                              ↓                                   │
│                         執行動作 a_t                             │
│                              ↓                                   │
│     環境返回: 獎勵 r_t 和 新狀態 s_{t+1}                          │
│                                                                 │
│  目標: 找到最優策略 π*，使累積獎勵最大化                          │
│        max_π E[Σ γ^t r_t]                                       │
│              ↑                                                  │
│         折扣因子 (0 < γ < 1)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 本專案的 RL 對應

| RL 概念 | 本專案對應 |
|--------|-----------|
| **狀態 (State)** | 768 維向量：辯論上下文的 BERT 嵌入 |
| **動作 (Action)** | 4 種策略：aggressive, defensive, analytical, empathetic |
| **獎勵 (Reward)** | 說服成功 → 高獎勵，被反駁 → 低獎勵 |
| **策略 (Policy)** | 神經網路，輸入狀態，輸出動作機率 |
| **環境 (Environment)** | 辯論模擬器 |

---

## 3.2 Policy Gradient 方法

### 策略梯度的直覺

```
目標: 提高「好動作」的機率，降低「壞動作」的機率

假設:
• 狀態 s 下選了動作 a，獲得獎勵 r = +1 (好)
• 策略應該調整，讓 π(a|s) 增加

• 狀態 s 下選了動作 a，獲得獎勵 r = -1 (壞)
• 策略應該調整，讓 π(a|s) 減少
```

### 策略梯度公式

```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q(s,a)]
            ↑           ↑         ↑
      對參數求導    動作機率   動作價值
      
解讀:
• ∇_θ log π_θ(a|s): 告訴我們「怎麼調整才能增加動作 a 的機率」
• Q(s,a): 這個動作有多好
• 相乘: 好動作的方向會被放大，壞動作的方向會被縮小
```

### Policy Gradient 的問題

```
問題 1: 高方差
  獎勵 r 的波動會導致梯度估計不穩定
  
問題 2: 樣本效率低
  每次更新後，舊數據就不能用了
  
問題 3: 更新步長難調
  步子太大 → 策略崩潰
  步子太小 → 學習太慢
```

---

## 3.3 PPO 核心原理

### PPO 解決了什麼問題？

```
傳統 Policy Gradient:
  θ_new = θ_old + α · ∇_θ J(θ)
  
問題: 一步更新可能讓策略變化太大，導致性能驟降

PPO 的解決方案:
  限制新舊策略的差異，確保「信任區域」內更新
```

### PPO 的核心公式

```
PPO Objective:

L^CLIP(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
                  ↑           ↑                    ↑
              機率比      裁剪函數              優勢函數

其中:
• r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
           ↑                  ↑
        新策略機率         舊策略機率
        
• A_t = 優勢函數 (這個動作比平均水平好多少)

• ε = 0.2 (裁剪參數)
```

### 裁剪機制詳解

```
假設 ε = 0.2，優勢 A = +5 (好動作)

情況 1: r = 1.0 (新舊策略相同)
  min(1.0 × 5, clip(1.0, 0.8, 1.2) × 5)
= min(5, 1.0 × 5)
= min(5, 5) = 5
→ 正常更新

情況 2: r = 1.5 (新策略機率增加了 50%)
  min(1.5 × 5, clip(1.5, 0.8, 1.2) × 5)
= min(7.5, 1.2 × 5)
= min(7.5, 6) = 6  ← 被裁剪了！
→ 限制更新幅度，不讓策略變化太大

情況 3: r = 0.7 (新策略機率降低了 30%)
  min(0.7 × 5, clip(0.7, 0.8, 1.2) × 5)
= min(3.5, 0.8 × 5)
= min(3.5, 4) = 3.5  ← 不裁剪，允許降低
→ 對於好動作，不會懲罰它降低機率
```

### 視覺化理解

```
對於好動作 (A > 0):
                    裁剪區域
                ┌─────────────┐
                │             │
    ────────────┤             ├────────────
        不裁剪  │   1-ε  1  1+ε  │  裁剪
                │             │
                └─────────────┘
                      ↑
                機率比 r_t(θ)

目的: 允許增加好動作的機率，但不要增加太多 (>1.2倍)

對於壞動作 (A < 0):
                    裁剪區域
                ┌─────────────┐
                │             │
    ────────────┤             ├────────────
        裁剪    │   1-ε  1  1+ε  │  不裁剪
                │             │
                └─────────────┘

目的: 允許降低壞動作的機率，但不要降低太多 (<0.8倍)
```

---

## 3.4 Actor-Critic 架構

### 什麼是 Actor-Critic？

```
Actor (演員): 決定「做什麼動作」
             π(a|s) → 動作機率分佈
             
Critic (評論家): 評估「這個狀態值多少」
                 V(s) → 狀態價值
                 
兩者配合:
• Actor 負責探索和決策
• Critic 負責評估和指導

優點:
• 比純 Policy Gradient 方差更低
• 比純 Value-based 方法更靈活
```

### 本專案的 Actor-Critic 網路

```
                    輸入: 狀態 s (768 維)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        共享層 (Shared)                          │
│                                                                 │
│   Linear(768 → 256) → ReLU → Linear(256 → 256) → ReLU → Drop   │
│                                                                 │
│   功能: 提取狀態的高級特徵                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     Actor Head (策略)    │     │    Critic Head (價值)   │
│                         │     │                         │
│ Linear(256 → 128) → ReLU│     │ Linear(256 → 128) → ReLU│
│ Linear(128 → 4) → Softmax│    │ Linear(128 → 1)         │
│                         │     │                         │
│ 輸出: [0.1, 0.2, 0.5, 0.2]│    │ 輸出: 2.5               │
│       動作機率分佈       │     │       狀態價值           │
└─────────────────────────┘     └─────────────────────────┘
```

---

## 3.5 GAE (廣義優勢估計)

### 什麼是優勢函數？

```
優勢函數: A(s, a) = Q(s, a) - V(s)
                    ↑         ↑
              動作價值    狀態價值
              
解讀: 
• A > 0: 這個動作比平均水平好
• A < 0: 這個動作比平均水平差
• A = 0: 這個動作就是平均水平

為什麼用優勢而不是直接用 Q？
• 減少方差！
• 即使兩個狀態的 Q 值差 1000 倍，優勢可能都接近 0
```

### GAE 公式

```
傳統優勢估計:
  A_t = r_t + γV(s_{t+1}) - V(s_t)
  
問題: 只看一步，估計不穩定

GAE (Generalized Advantage Estimation):
  A_t^GAE = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
  
其中:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD 誤差)
  γ = 0.99  (折扣因子)
  λ = 0.95  (GAE 參數)
  
解讀:
• λ=0: 只看一步 (高偏差，低方差)
• λ=1: 看到終點 (低偏差，高方差)
• λ=0.95: 平衡，實踐中效果好
```

### 程式碼實作

```python
# src/rl/ppo_trainer.py 第 198-228 行

def _calculate_advantages(self, transitions):
    """計算 GAE 優勢"""
    
    # Step 1: 計算 Return (從後往前累積)
    returns = []
    G = 0
    for transition in reversed(transitions):
        G = transition.reward + self.gamma * G
        #                       ↑
        #                  γ = 0.99
        returns.insert(0, G)
    
    # Step 2: 計算 GAE 優勢
    values = [t.value for t in transitions] + [0]  # 加上終止狀態的 V=0
    advantages = []
    
    gae = 0
    for i in reversed(range(len(transitions))):
        # TD 誤差
        delta = (transitions[i].reward + 
                self.gamma * values[i + 1] * (1 - transitions[i].done) - 
                values[i])
        # δ_t = r_t + γV(s_{t+1}) - V(s_t)
        #       如果是終止狀態，V(s_{t+1}) = 0
        
        # GAE 累積
        gae = delta + self.gamma * self.gae_lambda * (1 - transitions[i].done) * gae
        #            ↑            ↑
        #       γ = 0.99     λ = 0.95
        
        advantages.insert(0, gae)
    
    # Step 3: 標準化優勢 (減少方差)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #             ↑                                  ↑
    #          減去平均                           除以標準差
    
    return advantages
```

---

## 3.6 本專案 PPO 實作逐行解析

### 網路架構

```python
# src/rl/ppo_trainer.py 第 27-72 行

class PPONetwork(nn.Module):
    """PPO 的 Actor-Critic 網路"""
    
    def __init__(self, state_dim=768, action_dim=4, hidden_dim=256):
        super().__init__()
        
        # ====== 共享層 ======
        # 兩個分支共享這部分，節省參數，共享特徵
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 768 → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # 256 → 256
            nn.ReLU(),
            nn.Dropout(0.1)                    # 10% Dropout
        )
        
        # ====== Actor 分支 (策略網路) ======
        # 輸出: 動作機率分佈 [p(a=0), p(a=1), p(a=2), p(a=3)]
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),  # 128 → 4
            nn.Softmax(dim=-1)                       # 轉成機率
        )
        
        # ====== Critic 分支 (價值網路) ======
        # 輸出: 狀態價值 V(s) (標量)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)            # 128 → 1
        )
    
    def forward(self, state):
        """前向傳播"""
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)   # [batch, 4]
        state_value = self.critic(shared_features)   # [batch, 1]
        return action_probs, state_value
    
    def select_action(self, state):
        """選擇動作 (用於收集經驗)"""
        action_probs, state_value = self.forward(state)
        
        # 創建類別分佈
        dist = Categorical(action_probs)
        # Categorical: 離散動作的機率分佈
        # 例如 [0.1, 0.2, 0.5, 0.2] → 有 50% 機率選動作 2
        
        # 從分佈中採樣
        action = dist.sample()
        # 例如可能採樣到 action = 2
        
        # 返回: 動作, log機率, 狀態價值
        return action.item(), dist.log_prob(action), state_value.squeeze()
        #      ↑                ↑                    ↑
        #      2         log(0.5) = -0.69          2.5
```

### 訓練循環

```python
# src/rl/ppo_trainer.py 第 230-273 行

def update_policy(self, trajectories):
    """使用 PPO 更新策略"""
    
    # 準備數據
    states = torch.stack([t.state for t in trajectories])      # [N, 768]
    actions = torch.tensor([t.action for t in trajectories])   # [N]
    old_log_probs = torch.tensor([t.log_prob for t in trajectories])  # [N]
    returns = torch.tensor([t.return_value for t in trajectories])    # [N]
    advantages = torch.tensor([t.advantage for t in trajectories])    # [N]
    
    # ====== PPO 更新 (多個 epoch) ======
    for _ in range(self.update_epochs):  # update_epochs = 4
        # 前向傳播
        action_probs, values = self.network(states)
        dist = Categorical(action_probs)
        
        # 計算新的 log 機率
        new_log_probs = dist.log_prob(actions)
        
        # ====== 計算機率比 ======
        ratio = torch.exp(new_log_probs - old_log_probs)
        # ratio = π_new(a|s) / π_old(a|s)
        # exp(log_new - log_old) = exp(log(new/old)) = new/old
        
        # ====== PPO Clipped Objective ======
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        #       ↑                    ↑
        #      裁剪               ε = 0.2
        
        actor_loss = -torch.min(surr1, surr2).mean()
        #            ↑
        #       取負是因為要最大化目標，但優化器做最小化
        
        # ====== Critic 損失 ======
        critic_loss = F.mse_loss(values.squeeze(), returns)
        # 讓 V(s) 逼近真實的 Return
        
        # ====== 熵正則化 ======
        entropy = dist.entropy().mean()
        # 熵越大 → 策略越隨機 → 鼓勵探索
        
        # ====== 總損失 ======
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        #                         ↑                   ↑
        #                    critic 權重 0.5     熵 bonus 0.01
        
        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        
        self.optimizer.step()
```

### 辯論環境 (有問題！)

```python
# src/rl/ppo_trainer.py 第 74-134 行

class DebateEnvironment:
    """辯論環境 (⚠️ 這是一個簡化的模擬環境)"""
    
    def __init__(self):
        self.strategies = ['aggressive', 'defensive', 'analytical', 'empathetic']
        
    def reset(self):
        """重置環境"""
        self.current_stance = 0.0   # 立場
        self.conviction = 0.7       # 信念
        self.round = 0
        self.max_rounds = 5
        
        # ⚠️ 問題: 用隨機向量代替真實的辯論狀態
        self.state = torch.randn(768)
        return self.state
    
    def step(self, action):
        """執行動作"""
        reward = self._calculate_reward(action)
        
        self.round += 1
        self.state = torch.randn(768)  # ⚠️ 問題: 又是隨機向量
        
        done = self.round >= self.max_rounds
        return self.state, reward, done
    
    def _calculate_reward(self, action):
        """計算獎勵 (⚠️ 問題: 硬編碼!)"""
        strategy = self.strategies[action]
        
        # 這些獎勵是硬編碼的，與真實辯論效果無關!
        if strategy == 'analytical':
            base_reward = 0.8   # 永遠 0.8
        elif strategy == 'empathetic':
            base_reward = 0.7   # 永遠 0.7
        elif strategy == 'defensive':
            base_reward = 0.5   # 永遠 0.5
        elif strategy == 'aggressive':
            base_reward = random.choice([0.3, 0.9])  # 隨機！
        
        return np.clip(base_reward + np.random.normal(0, 0.1), 0, 1)

# ⚠️ 這個環境的問題:
# 1. 狀態是隨機噪聲，不是真實的辯論上下文
# 2. 獎勵是硬編碼的，不是基於真實辯論效果
# 3. 模型學到的只是「選 analytical 得 0.8 分」
#    而不是「根據情況選最優策略」
```

---

# Part 4: RAG 與 LangGraph

## 4.1 RAG 原理與實作

### RAG 流程圖

```
用戶問題: "AI 監管的成功案例？"
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 文本向量化 (Embedding)                                  │
│                                                                 │
│ "AI 監管的成功案例？" ──► OpenAI Embedding ──► [0.12, -0.3, ...]│
│                                                    (1536 維)    │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 向量搜索 (FAISS)                                        │
│                                                                 │
│ 查詢向量 ──► 在知識庫中找最相似的文檔                            │
│                                                                 │
│ 知識庫:                                                         │
│ • 文檔1: 歐盟 AI Act... (相似度: 0.92) ✓                       │
│ • 文檔2: 中國算法治理... (相似度: 0.85) ✓                      │
│ • 文檔3: 美國 FTC 案例... (相似度: 0.78) ✓                     │
│ • 文檔4: 無關內容... (相似度: 0.23) ✗                          │
└─────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 生成增強 (Augmented Generation)                         │
│                                                                 │
│ Prompt:                                                         │
│ "根據以下資料回答問題:                                          │
│  資料1: 歐盟 AI Act 於 2024 年...                               │
│  資料2: 中國算法治理要求...                                      │
│  問題: AI 監管的成功案例？"                                      │
│                                                                 │
│ GPT-4 生成: "根據歐盟的 AI Act，成功案例包括..."                │
└─────────────────────────────────────────────────────────────────┘
```

### 本專案 RAG 實作

```python
# src/rag/retriever.py

class EnhancedRetriever:
    def __init__(self):
        # 使用 OpenAI 的 Embedding 模型
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # 向量資料庫: Chroma (基於 FAISS)
        self.stores = {}
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        檢索最相關的 k 個文檔
        
        Args:
            query: 查詢字串
            k: 返回的文檔數量
            
        Returns:
            [{content, similarity_score, metadata}, ...]
        """
        # 在向量資料庫中搜索
        docs = store.similarity_search_with_score(query, k=k*2)
        
        # 過濾和排序
        results = []
        for doc, score in docs:
            results.append({
                'content': doc.page_content,
                'similarity_score': float(1 - score),  # 轉成相似度
                'metadata': doc.metadata
            })
        
        return results[:k]
```

---

## 4.2 LangGraph 狀態機

### 什麼是 StateGraph？

```
傳統程式流程:
    if condition1:
        if condition2:
            ...
    else:
        ...
    
問題: 複雜、難維護、難視覺化

StateGraph 方式:
    定義狀態 → 定義節點 → 定義邊 → 自動執行
    
優點:
• 宣告式，清晰易懂
• 可視覺化
• 內建狀態管理
• 支持並行執行
```

### 本專案的 StateGraph

```python
# src/orchestrator/langgraph_orchestrator.py

def _build_graph(self) -> StateGraph:
    workflow = StateGraph(DebateState)
    
    # 添加節點
    workflow.add_node("parallel_analysis", self._parallel_analysis_node)
    workflow.add_node("fuse_results", self._fuse_results_node)
    workflow.add_node("generate_response", self._generate_response_node)
    workflow.add_node("update_states", self._update_states_node)
    workflow.add_node("advance_turn", self._advance_turn_node)
    
    # 定義邊 (流程)
    workflow.set_entry_point("parallel_analysis")
    workflow.add_edge("parallel_analysis", "fuse_results")
    workflow.add_edge("fuse_results", "generate_response")
    workflow.add_edge("generate_response", "update_states")
    
    # 條件邊 (分支)
    workflow.add_conditional_edges(
        "update_states",
        self._should_continue,  # 判斷函數
        {
            "next_speaker": "advance_turn",
            "next_round": "advance_turn", 
            "end": END
        }
    )
    
    return workflow
```

### 狀態累積的魔法

```python
# src/orchestrator/debate_state.py

class DebateState(TypedDict):
    topic: str
    current_round: int
    
    # 魔法在這裡! 👇
    history: Annotated[List[Dict], operator.add]
    #        ↑                     ↑
    #   類型註解             累積方式: 列表相加

# 使用時:
# 節點返回 {"history": [new_response]}
# LangGraph 自動執行 history = history + [new_response]

# 為什麼這很有用？
# 不用手動管理列表，不會覆蓋之前的歷史
# 每個節點只關心「新增什麼」，不用知道「之前有什麼」
```

---

# Part 5: 面試準備

## 5.1 一分鐘自我介紹

### English Version

```
Hi, I'm [Your Name], and I'd like to share Social Debate AI.

[Problem]
Traditional debate systems use static rules that can't model how real 
debates evolve—the dynamic social influence and adaptive strategies.

[Solution]
I built a multi-agent debate system integrating:
• GNN (GraphSAGE + GAT) to predict persuasion success
• PPO reinforcement learning for dynamic strategy selection
• RAG with FAISS for evidence retrieval
• LangGraph for declarative workflow orchestration

[Key Challenge]
Fusing three ML models with different output semantics required a 
careful strategy fusion layer with conditional routing. I also used 
LangGraph's annotated state schema for automatic history accumulation.

[Results]
The system runs real-time debates with GPT-4, has a Flask web UI, 
and comprehensive tests. I'm proud of how cleanly the architecture 
separates concerns while enabling complex orchestration.
```

### 中文版本

```
你好，我是[你的名字]，我想分享 Social Debate AI 這個專案。

【問題】
傳統辯論系統使用靜態規則，無法模擬真實辯論中的
動態社會影響力和自適應策略。

【解決方案】
我建立了一個整合以下技術的多智能體辯論系統：
• GNN (GraphSAGE + GAT) 預測說服成功率
• PPO 強化學習進行動態策略選擇
• RAG + FAISS 進行證據檢索
• LangGraph 進行宣告式工作流編排

【關鍵挑戰】
融合三個輸出語義不同的 ML 模型，需要設計帶有條件路由的
策略融合層。我還使用 LangGraph 的 annotated state schema
實現歷史紀錄的自動累積。

【成果】
系統支援與 GPT-4 的即時辯論，有 Flask Web 介面和完整測試。
我很自豪這個架構在實現複雜編排的同時，能清晰地分離關注點。
```

---

## 5.2 三個刁鑽問題

### 問題 1: GNN Training-Serving Skew

**問題：** 你的 GNN 在推理時輸入隨機向量 `np.random.randn(768)`，
但訓練時用的是 BERT 編碼。模型怎麼可能有意義？

**答案：**
```
【承認】
You're right. This is a training-serving skew issue.

【根因】
訓練時用 DistilBERT 編碼真實文本
推理時用隨機噪聲
導致 GNN 的預測毫無意義

【修復】
添加 TextEncoder 單例類，推理時也使用 DistilBERT 編碼

【學習】
這提醒我要在開發初期就建立 end-to-end 測試，
確保訓練和推理的 feature pipeline 一致。
```

### 問題 2: PPO Sim2Real Gap

**問題：** 你的 RL 訓練環境用硬編碼獎勵 (analytical=0.8)，
但真實系統用關鍵字計數評估。這個 policy 怎麼可能有效？

**答案：**
```
【承認】
You've identified a critical limitation. The training environment
is entirely synthetic.

【根因】
訓練: 策略 → 硬編碼獎勵
真實: 策略 → LLM → 關鍵字評估 → 獎勵
因果鏈完全不同

【修復選項】
1. Online RL: 用真實 LLM 訓練 (昂貴)
2. Offline RL: 從日誌學習
3. 監督學習: 更簡單，從成功案例學習

【務實選擇】
鑒於 API 成本，我會先用監督學習，
等有足夠日誌後再遷移到 Offline RL。
```

### 問題 3: Event Loop 問題

**問題：** 你每次調用 `_parallel_analysis_node` 都創建新的 event loop，
如果從 FastAPI 調用會發生什麼？

**答案：**
```
【承認】
This is a concurrency anti-pattern.

【問題】
1. 效能開銷: 每次創建/銷毀 event loop
2. 衝突: FastAPI 已有 event loop，會觸發 RuntimeError
3. 無用: 我只用 ThreadPoolExecutor，根本不需要 event loop

【修復】
移除 event loop 創建，直接用 concurrent.futures.wait()

【學習】
這提醒我要理解 async/threading 的區別，
以及考慮程式碼在不同 context 下的行為。
```

---

## 5.3 面試檢查清單

### 必須能回答的問題

- [ ] 專案解決什麼問題？
- [ ] GNN 的 GraphSAGE 和 GAT 有什麼區別？
- [ ] PPO 的 clipping 機制為什麼有效？
- [ ] GAE 的 λ 參數有什麼作用？
- [ ] LangGraph 的 `operator.add` 是什麼意思？
- [ ] 為什麼用多任務學習？

### 必須承認的問題

- [ ] GNN 推理時的 placeholder 問題
- [ ] PPO 訓練環境的 Sim2Real gap
- [ ] Event loop 的並發問題

### 加分項目

- [ ] 能畫出 GNN 的前向傳播流程
- [ ] 能寫出 PPO 的損失函數
- [ ] 能解釋 GAE 的數學推導
- [ ] 能提出具體的修復方案

---

# 📖 術語對照表

| 術語 | 英文 | 解釋 |
|-----|------|------|
| GNN | Graph Neural Network | 處理圖結構數據的神經網路 |
| GraphSAGE | Sample and Aggregate | 採樣+聚合的圖卷積方法 |
| GAT | Graph Attention Network | 用注意力機制的圖神經網路 |
| PPO | Proximal Policy Optimization | 近端策略優化，一種 RL 算法 |
| GAE | Generalized Advantage Estimation | 廣義優勢估計 |
| Actor-Critic | - | 策略+價值的雙網路架構 |
| RAG | Retrieval Augmented Generation | 檢索增強生成 |
| FAISS | Facebook AI Similarity Search | Facebook 的向量搜索庫 |
| LangGraph | - | LangChain 的圖工作流框架 |
| Embedding | - | 將文本轉成向量的技術 |
| Softmax | - | 將向量轉成機率分佈 |
| Cross-Entropy | - | 分類任務的損失函數 |

---

*Generated for Google Interview Preparation*  
*Project: Social Debate AI*  
*Last Updated: December 2024*
