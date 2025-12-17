# 📚 深度學習與系統設計學習筆記
## Social Debate AI - 多智能體社交辯論系統

> **狀態**：進階系統設計與機器學習技術筆記。
> **核心**：GNN (GraphSAGE/GAT)、RL (PPO)、RAG、LangGraph。
> **目標**：從零到精通 - 釐清概念、數學直覺與程式碼實作。

---

# 📑 目錄

## 第一部分：系統設計 (The "Big Picture")
- [1.1 高層架構設計](#11-高層架構設計)
- [1.2 架構取捨與決策 (Trade-offs)](#12-架構取捨與決策-trade-offs)

## 第二部分：圖神經網路 (GNN) ⭐
- [2.1 核心概念：訊息傳遞 (Message Passing)](#21-核心概念訊息傳遞-message-passing)
- [2.2 Inductive vs. Transductive (核心概念辨析)](#22-inductive-vs-transductive-核心概念辨析)
- [2.3 GraphSAGE：擴展性之王](#23-graphsage擴展性之王)
- [2.4 GAT：注意力機制](#24-gat注意力機制)
- [2.5 實作細節：多任務學習](#25-實作細節多任務學習)
- [2.6 專案實作細節](#26-專案實作細節)
- [2.7 手把手 GNN 實作指南](#27-手把手-gnn-實作指南)

## 第三部分：強化學習 (PPO) ⭐
- [3.1 RL 基礎與策略梯度](#31-rl-基礎與策略梯度)
- [3.2 PPO 數學原理 (白話版)](#32-ppo-數學原理-白話版)
- [3.3 Actor-Critic 架構實作](#33-actor-critic-架構實作)
- [3.4 GAE：降低變異數](#34-gae降低變異數)
- [3.5 獎勵工程 (Reward Engineering)](#35-獎勵工程-reward-engineering)
- [3.6 專案實作細節](#36-專案實作細節)
- [3.7 手把手 PPO 實作指南](#37-手把手-ppo-實作指南)

## 第四部分：RAG 與 編排系統
- [4.1 RAG 系統設計](#41-rag-系統設計)
- [4.2 LangGraph：LLM 的狀態機](#42-langgraphllm-的狀態機)
- [4.3 專案實作細節](#43-專案實作細節)
- [4.4 手把手編排器實作指南](#44-手把手編排器實作指南)

## 第五部分：重點概念總結 (Key Concepts)
- [5.1 常見問題解析 (FAQ)](#51-常見問題解析-faq)
- [5.2 核心重點回顧](#52-核心重點回顧)

---

# 第一部分：系統設計

## 1.1 高層架構設計

我們正在構建一個 **多智能體系統 (Multi-Agent System)**，這些 Agent 不僅僅是「聊天」，而是有策略地「辯論」。

*   **輸入**：用戶設定的辯論題目。
*   **編排器 (Orchestrator)**：**LangGraph** (狀態機)，負責管理辯論的流程循環。
*   **大腦 (並行執行)**：
    *   **GNN**：「我該針對誰？誰容易被說服？」(社交動態分析)。
    *   **RL (PPO)**：「我該用什麼策略？」(激進/邏輯/防守等)。
    *   **RAG**：「我需要什麼事實證據？」(向量資料庫檢索)。
*   **輸出**：**GPT-4** 根據 {目標, 策略, 事實} 生成最終文本。

## 1.2 架構取捨與決策 (Trade-offs)

**Q: 為什麼不直接讓 GPT-4 自己辯論？**
*   **A**: LLM 是無狀態且容易漂移的 (Drift)。它們不會為了「長期說服目標」而優化。我們需要 **RL** 來注入「目標導向」(贏得辯論)，並需要 **GNN** 來理解「場域」(社交圖譜)。

**Q: 為什麼選用 LangGraph 而不是寫簡單的 Python 迴圈？**
*   **A**:
    *   **持久化 (Persistence)**：支持暫停/恢復辯論 (Human-in-the-loop)。
    *   **循環圖 (Cyclic Graphs)**：辯論是循環過程，不是 DAG (有向無環圖)。LangGraph 原生支持循環。
    *   **狀態管理**：透過 `Annotated[List, operator.add]` 自動處理歷史訊息的追加 (Append)。

---

# 第二部分：圖神經網路 (GNN) ⭐

## 2.1 核心概念：訊息傳遞 (Message Passing)

標準神經網路 (MLP) 假設數據是獨立的 (i.i.d)。GNN 假設數據是 **連接的 (Connected)**。

**通用框架**：
1.  **訊息 (Message)**：節點從鄰居收集資訊。
2.  **聚合 (Aggregate)**：將資訊摘要 (Sum/Mean/Max)。
3.  **更新 (Update)**：結合自身狀態與鄰居資訊來更新自己。

## 2.2 Inductive vs. Transductive (核心概念辨析)

*   **Transductive (直推式，如 GCN)**：訓練時需要看到 *整張圖*。如果有新節點加入，必須重新訓練模型。這對動態辯論（不斷有新回覆）來說是不可行的。
*   **Inductive (歸納式，如 GraphSAGE)**：學習的是一個「聚合函數」(Aggregator)。可以對 **未見過的節點** (Unseen nodes) 進行推理。**這是我們選擇 GraphSAGE 的原因。**

## 2.3 GraphSAGE：擴展性之王

GraphSAGE = **S**ample (採樣) and **Agg**regat**e** (聚合)。

### 數學公式
$$ h_v^k = \sigma(W^k \cdot \text{CONCAT}(h_v^{k-1}, \text{AGG}\{h_u^{k-1}, \forall u \in N(v)\})) $$

*   $h_v^k$：節點 $v$ 在第 $k$ 層的特徵。
*   $N(v)$：節點 $v$ 的鄰居集合。
*   $\text{AGG}$：聚合函數 (Mean, Max, LSTM)。

### PyTorch Geometric 實作
```python
# SAGEConv 內部運作：W_1 * x + W_2 * mean(neighbors)
self.conv1 = tgnn.SAGEConv(in_channels=768, out_channels=256)
```

## 2.4 GAT：注意力機制

GraphSAGE 對所有鄰居一視同仁（或固定權重）。**GAT** 則學習 *誰更重要*。

### 直覺理解
「在一場辯論中，一個『意見領袖』的回覆比路人的回覆更重要。」

### 數學公式 (注意力係數)
$$ \alpha_{ij} = \text{softmax}_j( \text{LeakyReLU}( \vec{a}^T [W h_i || W h_j] ) ) $$

1.  特徵轉換 ($W h$)。
2.  串接 (Concat) 節點 $i$ 和鄰居 $j$。
3.  通過注意力向量 $\vec{a}$ 計算分數。
4.  通過 Softmax 歸一化。

### 多頭注意力 (Multi-Head Attention)
獨立執行 $K$ 次上述過程，然後將結果平均或串接，以穩定學習效果。

```python
# heads=4, concat=False -> 輸出維度維持 hidden_dim (取平均)
self.attention = tgnn.GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
```

## 2.5 實作細節：多任務學習

我們不訓練三個分開的模型，而是使用 **一個編碼器，多個輸出頭**。

*   **骨幹 (Backbone)**：SAGE -> SAGE -> SAGE -> GAT。
*   **輸出頭 (Heads)**：
    1.  `delta_head` (二元分類)：說服會成功嗎？
    2.  `quality_head` (回歸)：品質分數 0-1。
    3.  `strategy_head` (多分類)：這是哪種策略？

**為什麼？** 骨幹學習的是「理解辯論局勢」，這對所有任務都是通用的。參數共享 = 更好的泛化能力 (Generalization)。

## 2.6 專案實作細節

### 技術棧 (Tech Stack)
*   **框架**：使用 `PyTorch Geometric` (PyG) 進行圖神經網路運算。這比手寫 Sparse Matrix 乘法快得多。
*   **特徵嵌入**：使用 `HuggingFace Transformers` 的 `DistilBERT` (768 維) 將文本轉為向量。選用 DistilBERT 是為了速度與效能的平衡。

### 關鍵參數 (Configurations)
*   **層數設計**：3 層 SAGEConv (壓縮特徵) + 1 層 GATConv (4 頭注意力，用於最終決策)。
*   **維度變換**：768 (輸入) → 256 → 256 → 128 (共享潛在空間)。
*   **Dropout**：設為 0.3。由於辯論圖通常較小（<100 個節點），防止過擬合非常重要。
*   **損失函數 (Loss)**：
    ```python
    loss = 0.5 * BCE(delta) + 0.3 * MSE(quality) + 0.2 * CrossEntropy(strategy)
    ```
    *設計理由*：最重要的是預測「能否說服」(delta)，所以權重最高 (0.5)。

### 數據流 (Data Flow)
1.  **原始文本** → DistilBERT Tokenizer → Model → 取 `[CLS]` Token 向量 (768維)。
2.  **圖構建**：
    *   節點特徵 `x`: `[num_nodes, 768]`。
    *   邊索引 `edge_index`: `[2, num_edges]` (稀疏鄰接矩陣格式)。
3.  **前向傳播**：`x, edge_index` → GNN Layers → `h` (128維)。
4.  **輸出頭**：`h` → Linear Layers → 3 個不同的預測結果。

## 2.7 手把手 GNN 實作指南

**目標**：從頭開始構建 `PersuasionGNN` 類別。

### 步驟 1: 引入依賴
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
```

### 步驟 2: 定義類別與初始化
```python
class PersuasionGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        
        # 1. GraphSAGE 層 (負責「聚合」資訊)
        # 壓縮 768 -> 256
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        # 保持 256 -> 256
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # 壓縮 256 -> 128
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        # 2. GAT 層 (負責「精煉」資訊)
        # 128 -> 128, 4 個頭取平均
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 2, 
                                heads=4, concat=False)
                                
        # 3. 任務頭 (負責「預測」)
        # 任務 A: 會被說服嗎？(二元分類)
        self.delta_head = nn.Linear(hidden_dim // 2, 1)
        # 任務 B: 品質如何？(回歸)
        self.quality_head = nn.Linear(hidden_dim // 2, 1)
        # 任務 C: 這是什麼策略？(多分類)
        self.strategy_head = nn.Linear(hidden_dim // 2, 4)
```

### 步驟 3: 定義前向傳播
```python
    def forward(self, x, edge_index):
        # x: [節點數, 768], edge_index: [2, 邊數]
        
        # 第 1 層: SAGE + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 第 2 層: SAGE + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # 第 3 層: SAGE + ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # 第 4 層: GAT (注意力機制)
        x = self.attention(x, edge_index)
        
        # 多任務輸出
        return {
            'delta': torch.sigmoid(self.delta_head(x)), # 0-1 機率
            'quality': torch.sigmoid(self.quality_head(x)), # 0-1 分數
            'strategy': self.strategy_head(x) # 4 個類別的 Logits
        }
```

---

# 第三部分：強化學習 (PPO) ⭐

**目標**：訓練 Agent 在特定 *情境 (Context/State)* 下選擇最佳 *策略 (Action)*，以最大化 *說服效果 (Reward)*。

## 3.1 RL 基礎與策略梯度

*   **策略 (Policy) $\pi(a|s)$**：大腦。給定狀態 $s$，輸出動作 $a$ 的機率。
*   **目標**：最大化預期回報 $J(\theta) = E[\sum r]$。

**原始策略梯度 (Vanilla Policy Gradient)**：
$$ \nabla J(\theta) = E [ \nabla \log \pi_\theta(a|s) \cdot A(s,a) ] $$
*   「如果優勢 (Advantage) > 0，就提高該動作機率；反之則降低。」
*   **問題**：變異數極大。一次糟糕的更新可能毀掉整個策略。

## 3.2 PPO 數學原理 (白話版)

**PPO (Proximal Policy Optimization)** 的核心在於 **穩定性**。

### 機率比率 (The Ratio)
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$
*   比率 = 1：新舊策略無變化。
*   比率 > 1：該動作在新策略中更有可能發生。

### 截斷目標函數 (The Clipped Objective - 秘方所在)
$$ L^{CLIP} = \min( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t ) $$

*   如果動作是好的 ($A > 0$)：不要讓機率增加太多 (限制在 $1+\epsilon$)。
*   如果動作是壞的 ($A < 0$)：不要讓機率減少太多 (限制在 $1-\epsilon$)。
*   **為什麼？** 防止策略「掉下懸崖」，避免更新到一個無法恢復的糟糕區域。通常 $\epsilon=0.2$。

## 3.3 Actor-Critic 架構實作

我們需要兩個網路（通常共享底層）：
1.  **Actor (策略網路)**：輸出 4 種策略的 Logits。
2.  **Critic (價值網路)**：輸出純量價值 $V(s)$。

```python
class PPONetwork(nn.Module):
    def __init__(self):
        # 共享特徵提取層
        self.shared = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        
        # Actor: State -> Action Probs
        self.actor = nn.Linear(256, 4) 
        
        # Critic: State -> Value Estimate
        self.critic = nn.Linear(256, 1)
```

## 3.4 GAE：降低變異數

**廣義優勢估計 (GAE)** 在以下兩者間取得平衡：
*   **TD-Error (看一步)**：低變異，高偏差。
*   **Monte Carlo (看到底)**：高變異，低偏差。

公式：
$$ A_t^{GAE} = \sum (\gamma \lambda)^l \delta_{t+l} $$
*   $\lambda$ 控制權衡。通常 $\lambda=0.95$。

## 3.5 獎勵工程 (The "Dark Art")

RL 的效果完全取決於獎勵函數。
*   **稀疏獎勵 (Sparse)**：只有辯論結束贏了才給 +1。（很難學，Agent 就像無頭蒼蠅）。
*   **密集獎勵 (Dense / Shaping)**：
    *   引用證據 +0.1。
    *   對手立場鬆動 > 0.1 給 +0.2。
    *   重複語句 -0.1。

## 3.6 專案實作細節

### 技術選擇 (Tech Stack)
*   **手刻 PPO (Custom PPO)**：沒有使用 `stable-baselines3` (SB3)。
    *   *為什麼？* SB3 假設環境是標準的 Gym 介面 (numpy array)，但我們的 `DebateState` 包含複雜的語意資訊，且需要將 `DistilBERT` 嵌入過程整合到模型中，手刻 PyTorch 更有彈性。

### 關鍵超參數 (Key Hyperparameters)
*   **Learning Rate**: `3e-4` (Adam 優化器的標準起點)。
*   **Gamma (折扣因子)**: `0.99` (辯論是長期的，未來的獎勵很重要)。
*   **GAE Lambda**: `0.95` (在偏差與變異數之間取得平衡)。
*   **Clip Epsilon**: `0.2` (標準 PPO 參數，不讓策略更新太激進)。
*   **Update Epochs**: `4` (每個 batch 的數據重複利用 4 次來更新)。

### 狀態表示 (State Representation)
*   **輸入**：`[768]` 維向量，來自當前對話歷史的 BERT Embedding。
*   **處理變長輸入**：我們取最近 3 回合對話的 `[CLS]` Token 平均值 + 題目 Embedding，組合成固定長度的 Context Vector。

### 訓練循環 (Training Loop)
1.  **Rollout**：使用當前策略跑 `N` 回合辯論。
2.  **存儲**：將 `(state, action, reward, next_state, log_prob)` 存入 Buffer。
3.  **計算優勢**：從最後一步倒推計算 GAE。
4.  **更新**：在 PPO Loss 上執行 SGD 更新 `K` 個 Epochs。
5.  **清空 Buffer**：PPO 是 On-policy 算法，更新後舊數據必須丟棄。

## 3.7 手把手 PPO 實作指南

**目標**：實作 PPO 的核心 `update` 函數。

### 步驟 1: 計算 GAE (廣義優勢估計)
```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    輸入: 獎勵列表, 價值列表 等等
    輸出: 優勢列表 (Advantages)
    """
    advantages = []
    gae = 0
    
    # 從最後一步開始往前迭代
    for i in reversed(range(len(rewards))):
        # 1. 計算 TD Error (delta)
        # delta = r + γ * V(s') - V(s)
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        
        # 2. 累積 GAE
        # gae = delta + γ * λ * gae_prev
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        
        advantages.insert(0, gae)
        
    return torch.tensor(advantages)
```

### 步驟 2: PPO 損失函數
```python
def ppo_loss(old_log_probs, states, actions, advantages, returns):
    # 1. 用當前策略計算新的機率
    logits, values = model(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # 2. 計算比率 (Ratio: π_new / π_old)
    # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 3. 計算 Surrogate Objectives
    # Obj1: 未截斷的
    surr1 = ratio * advantages
    # Obj2: 截斷的 (PPO 的魔法!)
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    
    # 4. 策略損失 (取最小值的負數，因為要最大化)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 5. 價值損失 (預測值與實際回報的 MSE)
    value_loss = F.mse_loss(values.squeeze(), returns)
    
    # 6. 總損失
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss
```

---

# 第四部分：RAG 與 編排系統

## 4.1 RAG 系統設計

**檢索增強生成 (Retrieval Augmented Generation)** 模式：

1.  **切塊 (Chunking)**：長文切分 (512 tokens)。保留重疊 (Overlap 50 tokens) 以維持語意連貫。
2.  **嵌入 (Embedding)**：OpenAI `text-embedding-3-small`。
3.  **索引 (Indexing)**：FAISS (Facebook AI Similarity Search) 進行毫秒級近鄰搜索。
4.  **檢索 (Retrieval)**：Query -> Vector -> Top-K chunks。
5.  **生成 (Generation)**：Prompt = `Context: {chunks} Question: {q}`。

**優化技巧**：使用 **Cross-Encoder** 對 Top-K 結果進行重排序 (Re-ranking)，以提高精準度。

## 4.2 LangGraph：LLM 的狀態機

LangGraph 將應用程式視為由 `Nodes` (節點) 和 `Edges` (邊) 組成的圖。

**狀態定義 (Schema)**：
```python
class DebateState(TypedDict):
    history: Annotated[List[BaseMessage], operator.add] 
    # operator.add 是魔法所在！它的意思是：
    # new_state['history'] = old_state['history'] + returned_history (自動追加)
```

**工作流**：
1.  **並行節點**：在 ThreadPool 中同時執行 GNN/RL/RAG。
2.  **融合節點**：將結果組合成 Prompt。
3.  **生成節點**：呼叫 LLM。
4.  **條件邊**：檢查 `if rounds > max` 則 `END`。

## 4.3 專案實作細節

### RAG 實作
*   **庫**：`LangChain` + `FAISS`。
*   **向量存儲**：開發時使用內存版 FAISS，並將索引序列化為 `.index` 文件存儲在硬碟。
*   **檢索策略**：
    *   `Similarity Search`：先抓取前 10 個最相似片段。
    *   `MMR (Maximal Marginal Relevance)`：**關鍵！** 用於確保證據的「多樣性」。我們不希望抓到 10 個內容一模一樣的片段，MMR 會在「相關性」與「新穎性」之間權衡。

### LangGraph 實作
*   **異步/同步橋接 (Async/Sync Bridge)**：LangGraph 本身是異步的 (`async def`)，但 PyTorch 模型推論往往是同步且阻塞的 (Blocking)。
    *   *解決方案*：在 `_parallel_analysis_node` 中使用 `concurrent.futures.ThreadPoolExecutor` 將 GNN/RL/RAG 的計算放到執行緒池中，避免阻塞主 Event Loop，這在高並發 API 服務中至關重要。
*   **工具綁定 (Tool Binding)**：
    *   GNN, RL, RAG 都被封裝為 `LangChain Tools` (`@tool` 裝飾器)。
    *   這保留了未來擴展的可能性：可以讓 LLM 自己決定「是否需要調用 GNN」，而不僅僅是硬編碼在流程中。

## 4.4 手把手編排器實作指南

**目標**：構建 LangGraph 工作流。

### 步驟 1: 定義狀態
```python
from typing import Annotated, TypedDict, List
import operator

class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[str], operator.add] # 自動追加
    round: int
```

### 步驟 2: 定義節點
```python
def parallel_analysis(state: DebateState):
    # 執行分析 (簡化版)
    # 實際代碼中，這裡要用 ThreadPoolExecutor!
    return {"analysis_results": "..."}

def generate_response(state: DebateState):
    # 呼叫 LLM
    response = llm.invoke(state['topic'])
    # 只返回「新的訊息」
    return {
        "messages": [response.content], 
        "round": state['round'] + 1
    }
```

### 步驟 3: 構建圖
```python
from langgraph.graph import StateGraph, END

# 1. 初始化圖
workflow = StateGraph(DebateState)

# 2. 添加節點
workflow.add_node("analyze", parallel_analysis)
workflow.add_node("respond", generate_response)

# 3. 添加邊
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "respond")

# 4. 條件邊
def check_end(state):
    if state['round'] > 5:
        return END
    return "analyze"

workflow.add_conditional_edges("respond", check_end)

# 5. 編譯
app = workflow.compile()
```

---

# 第五部分：重點概念總結 (Key Concepts)

## 5.1 常見問題解析 (FAQ)

**Q: 為什麼選擇 PPO 而不是 DQN？**
*   **A**: DQN 只能處理離散動作且是 Value-based。PPO 是 Actor-Critic 架構，能處理連續或離散動作，且能學習隨機策略 (Stochastic Policy)，這對於需要多樣性的辯論場景更合適，且訓練更穩定。

**Q: 解釋 GraphSAGE 和 GCN 的區別。**
*   **A**: GCN 需要整張圖的拉普拉斯矩陣 (Transductive)。GraphSAGE 學習的是聚合函數 (Inductive)，允許它在不重新訓練的情況下處理新節點（即新的辯論回覆）。

**Q: 如何解決 RAG 的 Context Window 限制？**
*   **A**: 1. 更好的切塊策略。 2. 重排序 (檢索 50 個，精選 Top 5)。 3. Map-Reduce 摘要技術（針對廣泛查詢）。

**Q: 為什麼 GNN 要用多任務學習？**
*   **A**: 預測「說服成功率」和「品質分數」依賴於相同的底層特徵（論點邏輯、情感等）。共享底層參數可以起到正則化的作用，提高特徵的魯棒性。

## 5.2 核心重點回顧

1.  **GNN** 理解論點的 **結構 (Structure)**。
2.  **RL** 優化長期的 **策略 (Strategy)**。
3.  **RAG** 提供辯論的 **事實 (Fact)**。
4.  **LangGraph** 編排整體的 **流程 (Flow)**。

---
*最後更新：2025 年 12 月*
