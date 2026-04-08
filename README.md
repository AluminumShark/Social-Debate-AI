# Social Debate AI

<p align="center">
  <strong>A Multi-Agent Debate System Powered by Deep Learning</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-764ABC?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## Overview

**Social Debate AI** is an intelligent multi-agent debate simulation system that leverages cutting-edge deep learning technologies. It orchestrates dynamic debates between AI agents with distinct personalities and stances, using **LangGraph** for workflow management, **RAG** for evidence retrieval, **GNN** for social dynamics modeling, and **RL** for strategic decision-making.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Debate** | 3 AI agents with unique stances (Support / Oppose / Neutral) engage in dynamic debates |
| **LangGraph Orchestration** | Declarative state-graph workflow with parallel analysis pipelines |
| **RAG Evidence Retrieval** | FAISS-powered vector search for relevant evidence and citations |
| **GNN Social Modeling** | Graph neural networks predict persuasion success and social influence |
| **RL Strategy Learning** | PPO-based reinforcement learning with 4 adaptive debate strategies |
| **Modern Web Interface** | Flask + Bootstrap 5 responsive UI for real-time debate visualization |

---

## Dataset

This project uses the **ChangeMyView (CMV) Corpus** from [Cornell ConvoKit](https://convokit.cornell.edu/documentation/changemyview.html) as training data.

| Attribute | Description |
|-----------|-------------|
| **Source** | [Cornell ConvoKit - ChangeMyView Corpus](https://convokit.cornell.edu/documentation/changemyview.html) |
| **Dataset Page** | [ConvoKit Datasets](https://convokit.cornell.edu/documentation/datasets.html) |
| **Origin** | Reddit r/changemyview subreddit |
| **Key Feature** | Contains "delta" (Δ) annotations marking successful persuasion |
| **Usage** | Training GNN persuasion prediction, RAG evidence retrieval, RL strategy learning |

> **Note**: The CMV dataset contains posts where users present their viewpoints and others attempt to change their minds. When someone's view is successfully changed, they award a "delta" (Δ) to the convincing argument.

---

## System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart TB
    classDef web fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;
    classDef orch fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef brain fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef gen fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef agent fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;

    subgraph Presentation["Presentation Layer"]
        direction TB
        UI["Bootstrap 5 UI"]:::web
        API["Flask API"]:::web
        UI <--> API
    end

    subgraph Core["Orchestration Core"]
        direction TB
        LG["LangGraph Engine<br/>(State Management)"]:::orch
    end

    subgraph Intelligence["Intelligence Modules"]
        direction LR
        RL["RL Strategy<br/>(PPO Policy)"]:::brain
        GNN["GNN Social<br/>(GraphSAGE)"]:::brain
        RAG["RAG Evidence<br/>(FAISS)"]:::brain
    end

    subgraph Generation["Generation Layer"]
        direction TB
        Fusion["Result Fusion"]:::gen
        LLM["LLM Inference<br/>(GPT-3.5/4)"]:::gen
        Fusion --> LLM
    end

    subgraph Agents["Debate Agents"]
        direction LR
        A1["Agent A<br/>(Support)"]:::agent
        A2["Agent B<br/>(Oppose)"]:::agent
        A3["Agent C<br/>(Neutral)"]:::agent
    end

    API <==> LG
    LG ==> Intelligence
    Intelligence ==> Fusion
    LLM ==> Agents
    Agents -.->|"State Update"| LG

    style Presentation fill:#f0f9ff,stroke:#bae6fd,color:#0369a1
    style Core fill:#faf5ff,stroke:#e9d5ff,color:#6b21a8
    style Intelligence fill:#f0fdf4,stroke:#bbf7d0,color:#15803d
    style Generation fill:#fff7ed,stroke:#fed7aa,color:#c2410c
    style Agents fill:#fef2f2,stroke:#fecaca,color:#b91c1c
```

---

## LangGraph Workflow

The debate flow is managed by a declarative **StateGraph**:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#818cf8'}}}%%
flowchart TB
    classDef start fill:#10b981,stroke:#059669,stroke-width:2px,color:white;
    classDef process fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#9a3412;
    classDef endNode fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white;

    Start(("Start")):::start
    End(("End")):::endNode

    subgraph Cycle["Debate Cycle"]
        direction TB
        Analyze["Parallel Analysis<br/>(RL + GNN + RAG)"]:::process
        Fuse["Result Fusion"]:::process
        Gen["Response Generation"]:::process
        Update["State Update"]:::process

        Analyze --> Fuse --> Gen --> Update
    end

    Check{"Continue?"}:::decision

    Start --> Analyze
    Update --> Check
    Check -->|"Yes (Next Turn)"| Analyze
    Check -->|"No (Max Rounds/Surrender)"| End

    linkStyle default stroke:#6366f1,stroke-width:2px;
```

### State Schema

| DebateState | AgentState |
|------------|------------|
| `topic` `current_round` `max_rounds` | `agent_id` `current_stance` |
| `agent_states` `history` | `conviction` `persuasion_history` |
| `rl_result` `gnn_result` `rag_result` | `attack_history` `has_surrendered` |

---

## Debate Strategies

The RL module selects from **4 adaptive strategies**:

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Matrix["Strategy Matrix"]
        direction TB

        subgraph Row1[" "]
            direction LR
            S1["Aggressive<br/>(Challenge & Critique)"]:::red
            S2["Defensive<br/>(Consolidate & Protect)"]:::blue
        end

        subgraph Row2[" "]
            direction LR
            S3["Analytical<br/>(Logic & Evidence)"]:::purple
            S4["Empathetic<br/>(Connect & Persuade)"]:::green
        end
    end

    classDef red fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;
    classDef blue fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a;
    classDef purple fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef green fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;

    style Matrix fill:#ffffff,stroke:#e5e7eb,color:#374151
    style Row1 fill:transparent,stroke:transparent
    style Row2 fill:transparent,stroke:transparent
```

---

## Quick Start

### Prerequisites

- **Python** 3.10+
- **CUDA** 11.8+ (optional, for GPU acceleration)
- **RAM** 8GB+
- **OpenAI API Key**

### Installation (using uv - Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI

# 2. Install uv package manager
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create environment and install dependencies
uv sync

# 4. Configure environment
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Alternative: pip Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Using uv
uv run python ui/app.py

# Or with activated venv
python ui/app.py
```

Open http://localhost:5000 to start debating!

---

## Project Structure

```
Social-Debate-AI/
├── src/                          # Core source code
│   ├── agents/                   # Agent implementations
│   │   ├── base_agent.py         # Base agent class
│   │   ├── agent_a.py            # Support agent
│   │   ├── agent_b.py            # Oppose agent
│   │   └── agent_c.py            # Neutral agent
│   ├── orchestrator/             # LangGraph orchestration
│   │   ├── langgraph_orchestrator.py  # Main orchestrator
│   │   ├── debate_state.py       # State schema
│   │   └── debate_tools.py       # Tool wrappers
│   ├── rag/                      # Retrieval-Augmented Generation
│   │   ├── retriever.py          # Enhanced retriever
│   │   └── simple_retriever.py   # Lightweight retriever
│   ├── gnn/                      # Graph Neural Network
│   │   ├── social_encoder.py     # Social graph encoder
│   │   └── train_supervised.py   # Training script
│   ├── rl/                       # Reinforcement Learning
│   │   ├── policy_network.py     # PPO policy network
│   │   └── ppo_trainer.py        # PPO trainer
│   └── dialogue/                 # Dialogue management
├── ui/                           # Web application
│   ├── app.py                    # Flask server
│   ├── static/                   # CSS & JavaScript
│   └── templates/                # HTML templates
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── configs/                      # Configuration files
│   ├── debate.yaml               # Debate parameters
│   ├── rag.yaml                  # RAG settings
│   ├── gnn.yaml                  # GNN settings
│   └── rl.yaml                   # RL settings
├── docs/                         # Documentation
├── pyproject.toml                # Project configuration
└── uv.lock                       # Dependency lock file
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_debate_state.py

# Run with coverage report
uv run pytest --cov=src
```

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
USE_LANGGRAPH=true    # Enable LangGraph orchestrator (default: true)
```

### Configuration Files

| File | Description |
|------|-------------|
| `configs/debate.yaml` | Debate rounds, timing, agent settings |
| `configs/rag.yaml` | Vector DB, embedding model, retrieval params |
| `configs/gnn.yaml` | Graph structure, hidden dimensions |
| `configs/rl.yaml` | PPO hyperparameters, reward design |
| `configs/system.yaml` | Global system settings |

---

## Model Training

```bash
# Train all models
uv run python train_all.py --all

# Train individual models
uv run python train_all.py --gnn    # GNN social encoder
uv run python train_all.py --rl     # RL policy network
uv run python train_all.py --rag    # Build RAG index
```

---

## Tech Stack

```mermaid
%%{init: {'theme': 'base'}}%%
block-beta
    columns 5

    block:orch:1
        columns 1
        A["LangGraph"]
        B["LangChain"]
    end

    block:ml:1
        columns 1
        C["PyTorch"]
        D["PyG"]
        E["FAISS"]
    end

    block:llm:1
        columns 1
        F["OpenAI<br/>GPT-3.5/4"]
    end

    block:web:1
        columns 1
        G["Flask"]
        H["Bootstrap 5"]
    end

    block:tools:1
        columns 1
        I["uv"]
        J["pytest"]
    end

    style orch fill:#818cf8,color:#fff
    style ml fill:#fb923c,color:#fff
    style llm fill:#4ade80,color:#fff
    style web fill:#22d3ee,color:#fff
    style tools fill:#f472b6,color:#fff
```

---

## Documentation

### Core Learning Resource
- **[LEARNING_NOTE.md](docs/LEARNING_NOTE.md)** — Comprehensive deep learning study notes (GNN, PPO, RAG, LangGraph)

### Architecture
- [System Overview](docs/architecture/OVERVIEW.md) - High-level architecture
- [LangGraph Orchestration](docs/architecture/LANGGRAPH.md) - Workflow engine
- [Data Flow](docs/architecture/DATA_FLOW.md) - State management

### Guides
- [Quick Start Guide](docs/guides/QUICKSTART.md) - Get running in 5 minutes
- [Configuration Guide](docs/guides/CONFIGURATION.md) - System configuration
- [Training Guide](docs/guides/TRAINING.md) - Model training
- [Deployment Guide](docs/guides/DEPLOYMENT.md) - Production deployment

### API & Modules
- [REST API Reference](docs/api/REST_API.md) - Flask API endpoints
- [RAG Module](docs/modules/RAG.md) - Evidence retrieval
- [GNN Module](docs/modules/GNN.md) - Social analysis
- [RL Module](docs/modules/RL.md) - Strategy selection
- [Scoring System](docs/modules/SCORING.md) - Victory determination

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`uv run pytest`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
