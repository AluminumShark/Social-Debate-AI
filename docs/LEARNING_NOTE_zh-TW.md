#  
## Social Debate AI - 

> ****
> ****GNN (GraphSAGE/GAT)RL (PPO)RAGLangGraph
> **** - 

---

#  

##  (The "Big Picture")
- [1.1 ](#11-)
- [1.2  (Trade-offs)](#12--trade-offs)

##  (GNN) 
- [2.1  (Message Passing)](#21--message-passing)
- [2.2 Inductive vs. Transductive ()](#22-inductive-vs-transductive-)
- [2.3 GraphSAGE](#23-graphsage)
- [2.4 GAT](#24-gat)
- [2.5 ](#25-)
- [2.6 ](#26-)
- [2.7  GNN ](#27--gnn-)

##  (PPO) 
- [3.1 RL ](#31-rl-)
- [3.2 PPO  ()](#32-ppo--)
- [3.3 Actor-Critic ](#33-actor-critic-)
- [3.4 GAE](#34-gae)
- [3.5  (Reward Engineering)](#35--reward-engineering)
- [3.6 ](#36-)
- [3.7  PPO ](#37--ppo-)

## RAG  
- [4.1 RAG ](#41-rag-)
- [4.2 LangGraphLLM ](#42-langgraphllm-)
- [4.3 ](#43-)
- [4.4 ](#44-)

##  (Key Concepts)
- [5.1  (FAQ)](#51--faq)
- [5.2 ](#52-)

---

# 

## 1.1 

 ** (Multi-Agent System)** Agent 

*   ****
*   ** (Orchestrator)****LangGraph** ()
*   ** ()**
    *   **GNN**()
    *   **RL (PPO)**(//)
    *   **RAG**()
*   ******GPT-4**  {, , } 

## 1.2  (Trade-offs)

**Q:  GPT-4 **
*   **A**: LLM  (Drift) **RL** () **GNN** ()

**Q:  LangGraph  Python **
*   **A**:
    *   ** (Persistence)**/ (Human-in-the-loop)
    *   ** (Cyclic Graphs)** DAG ()LangGraph 
    *   **** `Annotated[List, operator.add]`  (Append)

---

#  (GNN) 

## 2.1  (Message Passing)

 (MLP)  (i.i.d)GNN  ** (Connected)**

****
1.  ** (Message)**
2.  ** (Aggregate)** (Sum/Mean/Max)
3.  ** (Update)**

## 2.2 Inductive vs. Transductive ()

*   **Transductive ( GCN)** **
*   **Inductive ( GraphSAGE)**(Aggregator) **** (Unseen nodes) ** GraphSAGE **

## 2.3 GraphSAGE

GraphSAGE = **S**ample () and **Agg**regat**e** ()

### 
$$ h_v^k = \sigma(W^k \cdot \text{CONCAT}(h_v^{k-1}, \text{AGG}\{h_u^{k-1}, \forall u \in N(v)\})) $$

*   $h_v^k$ $v$  $k$ 
*   $N(v)$ $v$ 
*   $\text{AGG}$ (Mean, Max, LSTM)

### PyTorch Geometric 
```python
# SAGEConv W_1 * x + W_2 * mean(neighbors)
self.conv1 = tgnn.SAGEConv(in_channels=768, out_channels=256)
```

## 2.4 GAT

GraphSAGE **GAT**  **

### 


###  ()
$$ \alpha_{ij} = \text{softmax}_j( \text{LeakyReLU}( \vec{a}^T [W h_i || W h_j] ) ) $$

1.   ($W h$)
2.   (Concat)  $i$  $j$
3.   $\vec{a}$ 
4.   Softmax 

###  (Multi-Head Attention)
 $K$ 

```python
# heads=4, concat=False ->  hidden_dim ()
self.attention = tgnn.GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
```

## 2.5 

 ****

*   ** (Backbone)**SAGE -> SAGE -> SAGE -> GAT
*   ** (Heads)**
    1.  `delta_head` ()
    2.  `quality_head` () 0-1
    3.  `strategy_head` ()

****  =  (Generalization)

## 2.6 

###  (Tech Stack)
*   **** `PyTorch Geometric` (PyG)  Sparse Matrix 
*   **** `HuggingFace Transformers`  `DistilBERT` (768 )  DistilBERT 

###  (Configurations)
*   ****3  SAGEConv () + 1  GATConv (4 )
*   ****768 () → 256 → 256 → 128 ()
*   **Dropout** 0.3<100 
*   ** (Loss)**
    ```python
    loss = 0.5 * BCE(delta) + 0.3 * MSE(quality) + 0.2 * CrossEntropy(strategy)
    ```
    **(delta) (0.5)

###  (Data Flow)
1.  **** → DistilBERT Tokenizer → Model →  `[CLS]` Token  (768)
2.  ****
    *    `x`: `[num_nodes, 768]`
    *    `edge_index`: `[2, num_edges]` ()
3.  ****`x, edge_index` → GNN Layers → `h` (128)
4.  ****`h` → Linear Layers → 3 

## 2.7  GNN 

**** `PersuasionGNN` 

###  1: 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
```

###  2: 
```python
class PersuasionGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        
        # 1. GraphSAGE  ()
        #  768 -> 256
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        #  256 -> 256
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        #  256 -> 128
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        # 2. GAT  ()
        # 128 -> 128, 4 
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 2, 
                                heads=4, concat=False)
                                
        # 3.  ()
        #  A: ()
        self.delta_head = nn.Linear(hidden_dim // 2, 1)
        #  B: ()
        self.quality_head = nn.Linear(hidden_dim // 2, 1)
        #  C: ()
        self.strategy_head = nn.Linear(hidden_dim // 2, 4)
```

###  3: 
```python
    def forward(self, x, edge_index):
        # x: [, 768], edge_index: [2, ]
        
        #  1 : SAGE + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        #  2 : SAGE + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        #  3 : SAGE + ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        #  4 : GAT ()
        x = self.attention(x, edge_index)
        
        # 
        return {
            'delta': torch.sigmoid(self.delta_head(x)), # 0-1 
            'quality': torch.sigmoid(self.quality_head(x)), # 0-1 
            'strategy': self.strategy_head(x) # 4  Logits
        }
```

---

#  (PPO) 

**** Agent  * (Context/State)*  * (Action)* * (Reward)*

## 3.1 RL 

*   ** (Policy) $\pi(a|s)$** $s$ $a$ 
*   **** $J(\theta) = E[\sum r]$

** (Vanilla Policy Gradient)**
$$ \nabla J(\theta) = E [ \nabla \log \pi_\theta(a|s) \cdot A(s,a) ] $$
*    (Advantage) > 0
*   ****

## 3.2 PPO  ()

**PPO (Proximal Policy Optimization)**  ****

###  (The Ratio)
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$
*    = 1
*    > 1

###  (The Clipped Objective - )
$$ L^{CLIP} = \min( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t ) $$

*    ($A > 0$) ( $1+\epsilon$)
*    ($A < 0$) ( $1-\epsilon$)
*   ****  $\epsilon=0.2$

## 3.3 Actor-Critic 


1.  **Actor ()** 4  Logits
2.  **Critic ()** $V(s)$

```python
class PPONetwork(nn.Module):
    def __init__(self):
        # 
        self.shared = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        
        # Actor: State -> Action Probs
        self.actor = nn.Linear(256, 4) 
        
        # Critic: State -> Value Estimate
        self.critic = nn.Linear(256, 1)
```

## 3.4 GAE

** (GAE)** 
*   **TD-Error ()**
*   **Monte Carlo ()**


$$ A_t^{GAE} = \sum (\gamma \lambda)^l \delta_{t+l} $$
*   $\lambda$  $\lambda=0.95$

## 3.5  (The "Dark Art")

RL 
*   ** (Sparse)** +1Agent 
*   ** (Dense / Shaping)**
    *    +0.1
    *    > 0.1  +0.2
    *    -0.1

## 3.6 

###  (Tech Stack)
*   ** PPO (Custom PPO)** `stable-baselines3` (SB3)
    *   ** SB3  Gym  (numpy array) `DebateState`  `DistilBERT`  PyTorch 

###  (Key Hyperparameters)
*   **Learning Rate**: `3e-4` (Adam )
*   **Gamma ()**: `0.99` ()
*   **GAE Lambda**: `0.95` ()
*   **Clip Epsilon**: `0.2` ( PPO )
*   **Update Epochs**: `4` ( batch  4 )

###  (State Representation)
*   ****`[768]`  BERT Embedding
*   **** 3  `[CLS]` Token  +  Embedding Context Vector

###  (Training Loop)
1.  **Rollout** `N` 
2.  **** `(state, action, reward, next_state, log_prob)`  Buffer
3.  **** GAE
4.  **** PPO Loss  SGD  `K`  Epochs
5.  ** Buffer**PPO  On-policy 

## 3.7  PPO 

**** PPO  `update` 

###  1:  GAE ()
```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    : ,  
    :  (Advantages)
    """
    advantages = []
    gae = 0
    
    # 
    for i in reversed(range(len(rewards))):
        # 1.  TD Error (delta)
        # delta = r + γ * V(s') - V(s)
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        
        # 2.  GAE
        # gae = delta + γ * λ * gae_prev
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        
        advantages.insert(0, gae)
        
    return torch.tensor(advantages)
```

###  2: PPO 
```python
def ppo_loss(old_log_probs, states, actions, advantages, returns):
    # 1. 
    logits, values = model(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # 2.  (Ratio: π_new / π_old)
    # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 3.  Surrogate Objectives
    # Obj1: 
    surr1 = ratio * advantages
    # Obj2:  (PPO !)
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    
    # 4.  ()
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 5.  ( MSE)
    value_loss = F.mse_loss(values.squeeze(), returns)
    
    # 6. 
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss
```

---

# RAG  

## 4.1 RAG 

** (Retrieval Augmented Generation)** 

1.  ** (Chunking)** (512 tokens) (Overlap 50 tokens) 
2.  ** (Embedding)**OpenAI `text-embedding-3-small`
3.  ** (Indexing)**FAISS (Facebook AI Similarity Search) 
4.  ** (Retrieval)**Query -> Vector -> Top-K chunks
5.  ** (Generation)**Prompt = `Context: {chunks} Question: {q}`

**** **Cross-Encoder**  Top-K  (Re-ranking)

## 4.2 LangGraphLLM 

LangGraph  `Nodes` ()  `Edges` () 

** (Schema)**
```python
class DebateState(TypedDict):
    history: Annotated[List[BaseMessage], operator.add] 
    # operator.add 
    # new_state['history'] = old_state['history'] + returned_history ()
```

****
1.  **** ThreadPool  GNN/RL/RAG
2.  **** Prompt
3.  **** LLM
4.  **** `if rounds > max`  `END`

## 4.3 

### RAG 
*   ****`LangChain` + `FAISS`
*   **** FAISS `.index` 
*   ****
    *   `Similarity Search` 10 
    *   `MMR (Maximal Marginal Relevance)`****  10 MMR 

### LangGraph 
*   **/ (Async/Sync Bridge)**LangGraph  (`async def`) PyTorch  (Blocking)
    *   ** `_parallel_analysis_node`  `concurrent.futures.ThreadPoolExecutor`  GNN/RL/RAG  Event Loop API 
*   ** (Tool Binding)**
    *   GNN, RL, RAG  `LangChain Tools` (`@tool` )
    *    LLM  GNN

## 4.4 

**** LangGraph 

###  1: 
```python
from typing import Annotated, TypedDict, List
import operator

class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[str], operator.add] # 
    round: int
```

###  2: 
```python
def parallel_analysis(state: DebateState):
    #  ()
    #  ThreadPoolExecutor!
    return {"analysis_results": "..."}

def generate_response(state: DebateState):
    #  LLM
    response = llm.invoke(state['topic'])
    # 
    return {
        "messages": [response.content], 
        "round": state['round'] + 1
    }
```

###  3: 
```python
from langgraph.graph import StateGraph, END

# 1. 
workflow = StateGraph(DebateState)

# 2. 
workflow.add_node("analyze", parallel_analysis)
workflow.add_node("respond", generate_response)

# 3. 
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "respond")

# 4. 
def check_end(state):
    if state['round'] > 5:
        return END
    return "analyze"

workflow.add_conditional_edges("respond", check_end)

# 5. 
app = workflow.compile()
```

---

#  (Key Concepts)

## 5.1  (FAQ)

**Q:  PPO  DQN**
*   **A**: DQN  Value-basedPPO  Actor-Critic  (Stochastic Policy)

**Q:  GraphSAGE  GCN **
*   **A**: GCN  (Transductive)GraphSAGE  (Inductive)

**Q:  RAG  Context Window **
*   **A**: 1.  2.  ( 50  Top 5) 3. Map-Reduce 

**Q:  GNN **
*   **A**: 

## 5.2 

1.  **GNN**  ** (Structure)**
2.  **RL**  ** (Strategy)**
3.  **RAG**  ** (Fact)**
4.  **LangGraph**  ** (Flow)**

---
*2025  12 *
