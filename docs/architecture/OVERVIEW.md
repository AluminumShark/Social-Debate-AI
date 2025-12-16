# System Architecture Overview

## High-Level Architecture

Social Debate AI is a multi-agent debate simulation system that integrates three AI/ML modules:
- **RAG** (Retrieval Augmented Generation) - Evidence retrieval
- **GNN** (Graph Neural Networks) - Social influence analysis
- **RL** (Reinforcement Learning) - Strategy selection

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                             │
│                     (Flask + Bootstrap 5)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                    LangGraph Orchestrator                        │
│              (StateGraph-based workflow engine)                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Parallel Analysis                       │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │    │
│  │  │   RL    │  │   GNN   │  │   RAG   │                  │    │
│  │  │ Strategy│  │ Social  │  │Evidence │                  │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                  │    │
│  │       └───────────┬┴───────────┘                        │    │
│  │                   ▼                                      │    │
│  │            Result Fusion                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Response Generation (LLM)                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               State Update & Control                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      Debate Agents                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Agent A    │  │  Agent B    │  │  Agent C    │              │
│  │  (Support)  │  │  (Oppose)   │  │  (Neutral)  │              │
│  │  Stance:+0.8│  │  Stance:-0.6│  │  Stance:0.0 │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. Web Interface Layer
- **Technology**: Flask + Bootstrap 5
- **Features**: Modern responsive UI, real-time updates
- **API**: RESTful endpoints for debate control

### 2. Orchestration Layer
- **LangGraph Orchestrator** (v0.2.0+): StateGraph-based workflow
- **Legacy Orchestrator**: Manual async orchestration (fallback)

### 3. Analysis Modules

#### RAG (Retrieval Augmented Generation)
- Vector database: FAISS
- Evidence retrieval and ranking
- Context-aware information extraction

#### GNN (Graph Neural Network)
- Architecture: GraphSAGE + GAT
- Predicts persuasion success probability
- Analyzes social influence patterns

#### RL (Reinforcement Learning)
- Algorithm: PPO (Proximal Policy Optimization)
- 4 strategies: aggressive, defensive, analytical, empathetic
- Dynamic strategy selection based on context

### 4. Agent Layer
- 3 AI agents with distinct personalities
- Stance and conviction tracking
- Surrender detection system

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | Bootstrap 5, JavaScript |
| Backend | Flask, Python 3.10+ |
| Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT-3.5/4 |
| ML Framework | PyTorch, PyTorch Geometric |
| Vector DB | FAISS |
| Package Manager | uv |

## Data Flow

1. User submits debate topic via Web UI
2. Orchestrator initializes agent states
3. For each turn:
   - Run parallel analysis (RL + GNN + RAG)
   - Fuse analysis results
   - Generate response via LLM
   - Evaluate response effects
   - Update agent states
4. Check end conditions (surrender/max rounds)
5. Generate final summary and verdict

