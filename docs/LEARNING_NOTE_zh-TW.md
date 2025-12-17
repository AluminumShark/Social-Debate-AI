# 📚 深度學習與系統設計學習筆記
## Social Debate AI - 多智能體社交辯論系統

> **狀態**：Google 等級系統設計與 ML 面試準備筆記。
> **核心**：GNN (GraphSAGE/GAT)、RL (PPO)、RAG、LangGraph。
> **目標**：從零到精通 - 釐清概念、數學直覺與程式碼實作。

---

# 📑 目錄

## 第一部分：系統設計 (The "Big Picture")
- [1.1 高層架構設計](#11-高層架構設計)
- [1.2 架構取捨與決策 (Trade-offs)](#12-架構取捨與決策-trade-offs)

## 第二部分：圖神經網路 (GNN) ⭐
- [2.1 核心概念：訊息傳遞 (Message Passing)](#21-核心概念訊息傳遞-message-passing)
- [2.2 Inductive vs. Transductive (面試必考)](#22-inductive-vs-transductive-面試必考)
- [2.3 GraphSAGE：擴展性之王](#23-graphsage擴展性之王)
- [2.4 GAT：注意力機制](#24-gat注意力機制)
- [2.5 實作細節：多任務學習](#25-實作細節多任務學習)

## 第三部分：強化學習 (PPO) ⭐
- [3.1 RL 基礎與策略梯度](#31-rl-基礎與策略梯度)
- [3.2 PPO 數學原理 (白話版)](#32-ppo-數學原理-白話版)
- [3.3 Actor-Critic 架構實作](#33-actor-critic-架構實作)
- [3.4 GAE：降低變異數](#34-gae降低變異數)
- [3.5 獎勵工程 (Reward Engineering)](#35-獎勵工程-reward-engineering)

## 第四部分：RAG 與 編排系統
- [4.1 RAG 系統設計](#41-rag-系統設計)
- [4.2 LangGraph：LLM 的狀態機](#42-langgraphllm-的狀態機)

## 第五部分：面試速查表 (Cheat Sheet)
- [5.1 常見面試題](#51-常見面試題)
- [5.2 核心重點總結](#52-核心重點總結)

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

## 2.2 Inductive vs. Transductive (面試必考)

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

---

# 第五部分：面試速查表 (Cheat Sheet)

## 5.1 常見面試題

**Q: 為什麼選擇 PPO 而不是 DQN？**
*   **A**: DQN 只能處理離散動作且是 Value-based。PPO 是 Actor-Critic 架構，能處理連續或離散動作，且能學習隨機策略 (Stochastic Policy)，這對於需要多樣性的辯論場景更合適，且訓練更穩定。

**Q: 解釋 GraphSAGE 和 GCN 的區別。**
*   **A**: GCN 需要整張圖的拉普拉斯矩陣 (Transductive)。GraphSAGE 學習的是聚合函數 (Inductive)，允許它在不重新訓練的情況下處理新節點（即新的辯論回覆）。

**Q: 如何解決 RAG 的 Context Window 限制？**
*   **A**: 1. 更好的切塊策略。 2. 重排序 (檢索 50 個，精選 Top 5)。 3. Map-Reduce 摘要技術（針對廣泛查詢）。

**Q: 為什麼 GNN 要用多任務學習？**
*   **A**: 預測「說服成功率」和「品質分數」依賴於相同的底層特徵（論點邏輯、情感等）。共享底層參數可以起到正則化的作用，提高特徵的魯棒性。

## 5.2 核心重點總結

1.  **GNN** 理解論點的 **結構 (Structure)**。
2.  **RL** 優化長期的 **策略 (Strategy)**。
3.  **RAG** 提供辯論的 **事實 (Fact)**。
4.  **LangGraph** 編排整體的 **流程 (Flow)**。

---
*最後更新：2025 年 12 月*
