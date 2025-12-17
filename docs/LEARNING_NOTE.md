# 📚 Deep Learning & System Design Study Notes
## Social Debate AI - Multi-Agent Social Debate System

> **Status**: Advanced System Design & Machine Learning Technical Notes.
> **Focus**: GNN (GraphSAGE/GAT), RL (PPO), RAG, LangGraph.
> **Goal**: From zero to mastery - clear concepts, math intuition, and code implementation.

---

# 📑 Table of Contents

## Part 1: System Design (The "Big Picture")
- [1.1 High-Level Architecture](#11-high-level-architecture)
- [1.2 Why this Architecture? (Trade-offs)](#12-why-this-architecture-trade-offs)

## Part 2: Graph Neural Networks (GNN) ⭐
- [2.1 Core Concept: Message Passing](#21-core-concept-message-passing)
- [2.2 Inductive vs. Transductive (Core Concept)](#22-inductive-vs-transductive-core-concept)
- [2.3 GraphSAGE: Scalability King](#23-graphsage-scalability-king)
- [2.4 GAT: Attention Mechanism](#24-gat-attention-mechanism)
- [2.5 Implementation: Multi-Task Learning](#25-implementation-multi-task-learning)
- [2.6 Project Implementation Details](#26-project-implementation-details)
- [2.7 Step-by-Step GNN Coding Guide](#27-step-by-step-gnn-coding-guide)

## Part 3: Reinforcement Learning (PPO) ⭐
- [3.1 RL Basics & Policy Gradient](#31-rl-basics--policy-gradient)
- [3.2 The Math of PPO (Simplified)](#32-the-math-of-ppo-simplified)
- [3.3 Actor-Critic Implementation](#33-actor-critic-implementation)
- [3.4 GAE: Reducing Variance](#34-gae-reducing-variance)
- [3.5 Reward Engineering (The "Dark Art")](#35-reward-engineering-the-dark-art)
- [3.6 Project Implementation Details](#36-project-implementation-details)
- [3.7 Step-by-Step PPO Coding Guide](#37-step-by-step-ppo-coding-guide)

## Part 4: RAG & Orchestration
- [4.1 RAG System Design](#41-rag-system-design)
- [4.2 LangGraph: State Machines for LLMs](#42-langgraph-state-machines-for-llms)
- [4.3 Project Implementation Details](#43-project-implementation-details)
- [4.4 Step-by-Step Orchestration Coding Guide](#44-step-by-step-orchestration-coding-guide)

## Part 5: Key Concepts Summary
- [5.1 FAQ & Deep Dive](#51-faq--deep-dive)
- [5.2 Key Takeaways](#52-key-takeaways)

---

# Part 1: System Design

## 1.1 High-Level Architecture

We are building a **Multi-Agent System** where agents don't just "chat", they "strategize".

*   **Input**: User topic.
*   **Orchestrator**: **LangGraph** (State Machine). Manages the loop.
*   **Brain (Parallel Execution)**:
    *   **GNN**: "Who should I target?" (Social dynamics).
    *   **RL (PPO)**: "What strategy works?" (Aggressive/Logical/etc.).
    *   **RAG**: "What facts do I need?" (Vector DB).
*   **Output**: **GPT-4** generates text based on {Target, Strategy, Facts}.

## 1.2 Why this Architecture? (Trade-offs)

**Q: Why not just ask GPT-4 to "debate"?**
*   **A**: LLMs are stateless and drift. They don't optimize for long-term persuasion goals. We need **RL** to inject a "goal" (winning the debate) and **GNN** to understand the "room" (social graph).

**Q: Why LangGraph instead of simple Python loops?**
*   **A**:
    *   **Persistence**: We can pause/resume debates (Human-in-the-loop).
    *   **Cyclic Graphs**: Debates are loops, not DAGs. LangGraph handles cycles natively.
    *   **State Management**: `Annotated[List, operator.add]` handles history append automatically.

---

# Part 2: Graph Neural Networks (GNN) ⭐

## 2.1 Core Concept: Message Passing

Standard NN (MLP) assumes independent data (i.i.d). GNN assumes data is **connected**.

**The Framework**:
1.  **Message**: Node gathers info from neighbors.
2.  **Aggregate**: Summarize info (Sum/Mean/Max).
3.  **Update**: Update own state based on self + neighbors.

## 2.2 Inductive vs. Transductive (Core Concept)

*   **Transductive (e.g., GCN)**: Requires *all* nodes during training. Can't handle new nodes without retraining. Bad for dynamic debates.
*   **Inductive (e.g., GraphSAGE)**: Learns a *function* to aggregate neighbors. Can handle **unseen nodes** (new replies). **We use this.**

## 2.3 GraphSAGE: Scalability King

GraphSAGE = **S**ample and **Agg**regat**e**.

### The Math
$$ h_v^k = \sigma(W^k \cdot \text{CONCAT}(h_v^{k-1}, \text{AGG}\{h_u^{k-1}, \forall u \in N(v)\})) $$

*   $h_v^k$: Feature of node $v$ at layer $k$.
*   $N(v)$: Neighbors of $v$.
*   $\text{AGG}$: Mean, Max, or LSTM.

### PyTorch Geometric Implementation
```python
# SAGEConv applies: W_1 * x + W_2 * mean(neighbors)
self.conv1 = tgnn.SAGEConv(in_channels=768, out_channels=256)
```

## 2.4 GAT: Attention Mechanism

GraphSAGE treats all neighbors equally (or fixed weights). **GAT** learns *who matters*.

### The Intuition
"In a debate, a reply from a 'Influencer' matters more than a random user."

### The Math (Attention Coefficient)
$$ \alpha_{ij} = \text{softmax}_j( \text{LeakyReLU}( \vec{a}^T [W h_i || W h_j] ) ) $$

1.  Transform features ($W h$).
2.  Concat node $i$ and neighbor $j$.
3.  Calculate score via attention vector $\vec{a}$.
4.  Normalize via Softmax.

### Multi-Head Attention
Run the process $K$ times independently and average/concat results to stabilize learning.

```python
# heads=4, concat=False -> Output dim is hidden_dim (averaged)
self.attention = tgnn.GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
```

## 2.5 Implementation: Multi-Task Learning

We don't train separate models. We use **one encoder, multiple heads**.

*   **Backbone**: SAGE -> SAGE -> SAGE -> GAT.
*   **Heads**:
    1.  `delta_head` (Binary): Will persuasion succeed?
    2.  `quality_head` (Regression): Score 0-1.
    3.  `strategy_head` (Classification): Which strategy?

**Why?** The backbone learns "Understanding Debate State", which is useful for ALL tasks. Shared parameters = Better generalization.

## 2.6 Project Implementation Details

### Tech Stack
*   **Framework**: `PyTorch Geometric` (PyG) for efficient graph operations.
*   **Embedding**: `DistilBERT` (768-dim) from `HuggingFace Transformers` for node features.

### Specific Configurations
*   **Layers**: 3x SAGEConv + 1x GATConv (4 heads).
*   **Dimensions**: 768 (Input) → 256 → 256 → 128 (Shared Latent).
*   **Dropout**: 0.3 to prevent overfitting on small graphs.
*   **Loss Function**:
    ```python
    loss = 0.5 * BCE(delta) + 0.3 * MSE(quality) + 0.2 * CrossEntropy(strategy)
    ```
    *Reasoning*: Persuasion prediction (delta) is the primary goal, hence highest weight.

### Data Flow
1.  **Raw Text** → DistilBERT Tokenizer → Model → `[CLS]` Token Vector (768-dim).
2.  **Graph Construction**:
    *   Node Features `x`: `[num_nodes, 768]`.
    *   Edge Index `edge_index`: `[2, num_edges]` (sparse adjacency matrix).
3.  **Forward Pass**: `x, edge_index` → GNN Layers → `h` (128-dim).
4.  **Heads**: `h` → Linear Layers → 3 Outputs.

## 2.7 Step-by-Step GNN Coding Guide

**Goal**: Build `PersuasionGNN` class from scratch.

### Step 1: Imports
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
```

### Step 2: Define the Class & Init
```python
class PersuasionGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        
        # 1. GraphSAGE Layers (The "Aggregator")
        # Compresses 768 -> 256
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        # Keeps 256 -> 256
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        # Compresses 256 -> 128
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)
        
        # 2. GAT Layer (The "Refiner")
        # 128 -> 128, 4 heads averaged
        self.attention = GATConv(hidden_dim // 2, hidden_dim // 2, 
                                heads=4, concat=False)
                                
        # 3. Task Heads (The "Predictors")
        # Task A: Will they be persuaded? (Binary)
        self.delta_head = nn.Linear(hidden_dim // 2, 1)
        # Task B: What is the quality? (Regression)
        self.quality_head = nn.Linear(hidden_dim // 2, 1)
        # Task C: Which strategy is this? (Classification)
        self.strategy_head = nn.Linear(hidden_dim // 2, 4)
```

### Step 3: Define Forward Pass
```python
    def forward(self, x, edge_index):
        # x: [num_nodes, 768], edge_index: [2, num_edges]
        
        # Layer 1: SAGE + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 2: SAGE + ReLU + Dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Layer 3: SAGE + ReLU
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Layer 4: GAT (Attention)
        x = self.attention(x, edge_index)
        
        # Multi-task Outputs
        return {
            'delta': torch.sigmoid(self.delta_head(x)), # 0-1 prob
            'quality': torch.sigmoid(self.quality_head(x)), # 0-1 score
            'strategy': self.strategy_head(x) # Logits for 4 classes
        }
```

---

# Part 3: Reinforcement Learning (PPO) ⭐

**Goal**: Train an agent to pick the best *Strategy* (Action) given the *Context* (State) to maximize *Persuasion* (Reward).

## 3.1 RL Basics & Policy Gradient

*   **Policy $\pi(a|s)$**: The brain. Given state $s$, output probs for actions $a$.
*   **Objective**: Maximize expected return $J(\theta) = E[\sum r]$.

**Vanilla Policy Gradient**:
$$ \nabla J(\theta) = E [ \nabla \log \pi_\theta(a|s) \cdot A(s,a) ] $$
*   "Push prob up if Advantage > 0, down if Advantage < 0."
*   **Problem**: High variance. One bad update ruins the policy.

## 3.2 The Math of PPO (Simplified)

**PPO (Proximal Policy Optimization)** is about **Stability**.

### The Ratio
$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} $$
*   Ratio = 1: No change.
*   Ratio > 1: Action more likely now.

### The Clipped Objective (The "Secret Sauce")
$$ L^{CLIP} = \min( r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t ) $$

*   If action is good ($A > 0$): Don't increase prob too much (limit to $1+\epsilon$).
*   If action is bad ($A < 0$): Don't decrease prob too much (limit to $1-\epsilon$).
*   **Why?** Prevents "falling off a cliff" into a bad policy region that we can't recover from.

## 3.3 Actor-Critic Implementation

We need two networks (often sharing layers):
1.  **Actor**: Outputs logits for 4 strategies.
2.  **Critic**: Outputs scalar Value $V(s)$.

```python
class PPONetwork(nn.Module):
    def __init__(self):
        # Shared feature extractor
        self.shared = nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        
        # Actor: State -> Action Probs
        self.actor = nn.Linear(256, 4) 
        
        # Critic: State -> Value Estimate
        self.critic = nn.Linear(256, 1)
```

## 3.4 GAE: Reducing Variance

**Generalized Advantage Estimation (GAE)** balances:
*   **TD-Error (1-step)**: Low variance, High bias.
*   **Monte Carlo (All-steps)**: High variance, Low bias.

Formula:
$$ A_t^{GAE} = \sum (\gamma \lambda)^l \delta_{t+l} $$
*   $\lambda$ controls the trade-off. Usually $\lambda=0.95$.

## 3.5 Reward Engineering (The "Dark Art")

RL is only as good as the reward.
*   **Sparse Reward**: +1 only at end of debate. (Hard to learn).
*   **Dense Reward (Shaping)**:
    *   +0.1 for citing evidence.
    *   +0.2 if opponent stance shifts > 0.1.
    *   -0.1 for repeating sentences.

## 3.6 Project Implementation Details

### Tech Stack
*   **Custom PPO Implementation**: Built from scratch using PyTorch (no `stable-baselines3`).
    *   *Why?* To handle custom `DebateState` encoding and integrating `DistilBERT` embeddings directly into the state space.

### Key Hyperparameters
*   **Learning Rate**: `3e-4` (Standard Adam default).
*   **Gamma (Discount)**: `0.99` (Values future rewards highly).
*   **GAE Lambda**: `0.95` (Balances bias/variance).
*   **Clip Epsilon**: `0.2` (Standard PPO clip range).
*   **Update Epochs**: `4` (Reuse data 4 times per batch).

### State Representation
*   **Input**: `[768]` vector from BERT embedding of current dialogue context.
*   **Handling Variable Length**: We use the `[CLS]` token of the last 3 turns + Topic embedding to create a fixed-size context vector.

### Training Loop
1.  **Rollout**: Run debate for `N` turns using current policy.
2.  **Store**: Save `(state, action, reward, next_state, log_prob)` in buffer.
3.  **Compute Advantage**: Use GAE formula backwards from last step.
4.  **Update**: Run SGD on PPO loss function for `K` epochs.
5.  **Clear Buffer**: On-policy algorithms must discard old data.

## 3.7 Step-by-Step PPO Coding Guide

**Goal**: Implement the core `update` function of PPO.

### Step 1: Compute GAE (Generalized Advantage Estimation)
```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Input: lists of rewards, values, etc.
    Output: list of advantages
    """
    advantages = []
    gae = 0
    
    # Iterate backwards (from last step to first)
    for i in reversed(range(len(rewards))):
        # 1. Calculate TD Error (delta)
        # delta = r + γ * V(s') - V(s)
        delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
        
        # 2. Accumulate GAE
        # gae = delta + γ * λ * gae_prev
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        
        advantages.insert(0, gae)
        
    return torch.tensor(advantages)
```

### Step 2: The PPO Loss Function
```python
def ppo_loss(old_log_probs, states, actions, advantages, returns):
    # 1. Get new probabilities from current policy
    logits, values = model(states)
    dist = Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    
    # 2. Calculate Ratio (π_new / π_old)
    # log(a/b) = log(a) - log(b) => a/b = exp(log(a) - log(b))
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # 3. Calculate Surrogate Objectives
    # Obj1: Unclipped
    surr1 = ratio * advantages
    # Obj2: Clipped (The PPO magic!)
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
    
    # 4. Policy Loss (Maximize obj => Minimize negative obj)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 5. Value Loss (MSE between predicted value and actual return)
    value_loss = F.mse_loss(values.squeeze(), returns)
    
    # 6. Total Loss
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss
```

---

# Part 4: RAG & Orchestration

## 4.1 RAG System Design

**Retrieval Augmented Generation** pattern:

1.  **Chunking**: Split long docs (512 tokens). Overlap (50 tokens) to preserve context.
2.  **Embedding**: OpenAI `text-embedding-3-small`.
3.  **Indexing**: FAISS (Facebook AI Similarity Search) for millisecond-level NN search.
4.  **Retrieval**: Query -> Vector -> Top-K chunks.
5.  **Generation**: Prompt = `Context: {chunks} Question: {q}`.

**Optimization**: Use **Cross-Encoder** for re-ranking top-K results to improve precision.

## 4.2 LangGraph: State Machines for LLMs

LangGraph treats your app as a graph `Nodes` and `Edges`.

**State Schema**:
```python
class DebateState(TypedDict):
    history: Annotated[List[BaseMessage], operator.add] 
    # operator.add is Magic! It means:
    # new_state['history'] = old_state['history'] + returned_history
```

**Workflow**:
1.  **Parallel Node**: Run GNN/RL/RAG in thread pool.
2.  **Fusion Node**: Combine results into a prompt.
3.  **Generation Node**: Call LLM.
4.  **Conditional Edge**: Check `if rounds > max` then `END`.

## 4.3 Project Implementation Details

### RAG Implementation
*   **Library**: `LangChain` + `FAISS`.
*   **Vector Store**: In-memory FAISS index, saved to disk as `.index` file.
*   **Retriever**:
    *   `Similarity Search`: Top 10 results.
    *   `MMR (Maximal Marginal Relevance)`: Used to ensure diversity in evidence (don't return 10 identical facts).

### LangGraph Implementation
*   **Async/Sync Bridge**: The graph runs asynchronously (`async def`), but some ML models (PyTorch) are blocking.
    *   *Solution*: Used `concurrent.futures.ThreadPoolExecutor` inside `_parallel_analysis_node` to prevent blocking the event loop.
*   **Tool Binding**:
    *   GNN, RL, and RAG are wrapped as `LangChain Tools` (`@tool` decorator).
    *   This allows potential future expansion where the LLM *decides* which tool to call (ReAct pattern), though currently hard-coded in the parallel node.

## 4.4 Step-by-Step Orchestration Coding Guide

**Goal**: Build the LangGraph workflow.

### Step 1: Define State
```python
from typing import Annotated, TypedDict, List
import operator

class DebateState(TypedDict):
    topic: str
    messages: Annotated[List[str], operator.add] # Auto-append
    round: int
```

### Step 2: Define Nodes
```python
def parallel_analysis(state: DebateState):
    # Run analysis (simplified)
    # In real code, use ThreadPoolExecutor here!
    return {"analysis_results": "..."}

def generate_response(state: DebateState):
    # Call LLM
    response = llm.invoke(state['topic'])
    # Return JUST the new message
    return {
        "messages": [response.content], 
        "round": state['round'] + 1
    }
```

### Step 3: Build Graph
```python
from langgraph.graph import StateGraph, END

# 1. Init Graph
workflow = StateGraph(DebateState)

# 2. Add Nodes
workflow.add_node("analyze", parallel_analysis)
workflow.add_node("respond", generate_response)

# 3. Add Edges
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "respond")

# 4. Conditional Edge
def check_end(state):
    if state['round'] > 5:
        return END
    return "analyze"

workflow.add_conditional_edges("respond", check_end)

# 5. Compile
app = workflow.compile()
```

---

# Part 5: Key Concepts Summary

## 5.1 FAQ & Deep Dive

**Q: Why use PPO over DQN?**
*   **A**: DQN only works for discrete actions and value-based. PPO is Actor-Critic, works for both, and handles stochastic policies (better for debate variety) with greater stability.

**Q: Explain the difference between GraphSAGE and GCN.**
*   **A**: GCN requires the full graph Laplacian (transductive). GraphSAGE learns an aggregation function (inductive), allowing it to handle new nodes (new debate replies) without retraining.

**Q: How do you handle the "context window" limit in RAG?**
*   **A**: 1. Better Chunking. 2. Re-ranking (retrieve 50, rank top 5). 3. Map-Reduce summarization for broad queries.

**Q: Why Multi-Task Learning for GNN?**
*   **A**: Predicting "Persuasion success" and "Quality score" relies on the same underlying features (argument logic, sentiment). Sharing layers acts as regularization and improves feature robustness.

## 5.2 Key Takeaways

1.  **GNN** understands the *structure* of arguments.
2.  **RL** optimizes the *strategy* over time.
3.  **RAG** grounds the debate in *fact*.
4.  **LangGraph** orchestrates the *flow*.

---
*Last Updated: December 2025*
