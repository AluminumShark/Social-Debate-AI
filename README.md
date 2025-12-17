# 🎭 Social Debate AI

<p align="center">
  <strong>A Multi-Agent Debate System Powered by Deep Learning</strong>
</p>

<p align="center">
  <a href="#english-version">English</a> | <a href="#中文版本">中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-764ABC?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

<a name="english-version"></a>

## 📖 Overview

**Social Debate AI** is an intelligent multi-agent debate simulation system that leverages cutting-edge deep learning technologies. It orchestrates dynamic debates between AI agents with distinct personalities and stances, using **LangGraph** for workflow management, **RAG** for evidence retrieval, **GNN** for social dynamics modeling, and **RL** for strategic decision-making.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Agent Debate** | 3 AI agents with unique stances (Support / Oppose / Neutral) engage in dynamic debates |
| 🔄 **LangGraph Orchestration** | Declarative state-graph workflow with parallel analysis pipelines |
| 📚 **RAG Evidence Retrieval** | FAISS-powered vector search for relevant evidence and citations |
| 🕸️ **GNN Social Modeling** | Graph neural networks predict persuasion success and social influence |
| 🎮 **RL Strategy Learning** | PPO-based reinforcement learning with 4 adaptive debate strategies |
| 🌐 **Modern Web Interface** | Flask + Bootstrap 5 responsive UI for real-time debate visualization |

---

## 🏗️ System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart TB
    %% Nodes Configuration
    classDef web fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;
    classDef orch fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef brain fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef gen fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef agent fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;

    subgraph Presentation["🌐 Presentation Layer"]
        direction TB
        UI["🖥️ Bootstrap 5 UI"]:::web
        API["⚡ Flask API"]:::web
        UI <--> API
    end

    subgraph Core["⚙️ Orchestration Core"]
        direction TB
        LG["📊 LangGraph Engine<br/>(State Management)"]:::orch
    end

    subgraph Intelligence["🧠 Intelligence Modules"]
        direction LR
        RL["🎮 RL Strategy<br/>(PPO Policy)"]:::brain
        GNN["🕸️ GNN Social<br/>(GraphSAGE)"]:::brain
        RAG["📚 RAG Evidence<br/>(FAISS)"]:::brain
    end

    subgraph Generation["🔮 Generation Layer"]
        direction TB
        Fusion["🔗 Result Fusion"]:::gen
        LLM["🤖 LLM Inference<br/>(GPT-3.5/4)"]:::gen
        Fusion --> LLM
    end

    subgraph Agents["👥 Debate Agents"]
        direction LR
        A1["🟢 Agent A<br/>(Support)"]:::agent
        A2["🔴 Agent B<br/>(Oppose)"]:::agent
        A3["🟡 Agent C<br/>(Neutral)"]:::agent
    end

    %% Data Flow Connections
    API <==> LG
    LG ==> Intelligence
    Intelligence ==> Fusion
    LLM ==> Agents
    Agents -.->|"State Update"| LG

    %% Styling
    style Presentation fill:#f0f9ff,stroke:#bae6fd,color:#0369a1
    style Core fill:#faf5ff,stroke:#e9d5ff,color:#6b21a8
    style Intelligence fill:#f0fdf4,stroke:#bbf7d0,color:#15803d
    style Generation fill:#fff7ed,stroke:#fed7aa,color:#c2410c
    style Agents fill:#fef2f2,stroke:#fecaca,color:#b91c1c
```

---

## 🔄 LangGraph Workflow

The debate flow is managed by a declarative **StateGraph**:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#818cf8'}}}%%
flowchart TB
    %% Define styles
    classDef start fill:#10b981,stroke:#059669,stroke-width:2px,color:white;
    classDef process fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#9a3412;
    classDef endNode fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white;

    Start(("🚀 Start")):::start
    End(("🏁 End")):::endNode

    subgraph Cycle["🔄 Debate Cycle"]
        direction TB
        Analyze["⚡ Parallel Analysis<br/>(RL + GNN + RAG)"]:::process
        Fuse["🔗 Result Fusion"]:::process
        Gen["💬 Response Generation"]:::process
        Update["📝 State Update"]:::process
        
        Analyze --> Fuse --> Gen --> Update
    end

    Check{"❓ Continue?"}:::decision

    Start --> Analyze
    Update --> Check
    Check -->|"Yes (Next Turn)"| Analyze
    Check -->|"No (Max Rounds/Surrender)"| End

    linkStyle default stroke:#6366f1,stroke-width:2px;
```

### 📋 State Schema

| DebateState | AgentState |
|------------|------------|
| `topic` `current_round` `max_rounds` | `agent_id` `current_stance` |
| `agent_states` `history` | `conviction` `persuasion_history` |
| `rl_result` `gnn_result` `rag_result` | `attack_history` `has_surrendered` |

---

## 🎮 Debate Strategies

The RL module selects from **4 adaptive strategies**:

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Matrix["🎯 Strategy Matrix"]
        direction TB
        
        subgraph Row1[" "]
            direction LR
            S1["🔥 Aggressive<br/>(Challenge & Critique)"]:::red
            S2["🛡️ Defensive<br/>(Consolidate & Protect)"]:::blue
        end
        
        subgraph Row2[" "]
            direction LR
            S3["🔬 Analytical<br/>(Logic & Evidence)"]:::purple
            S4["💚 Empathetic<br/>(Connect & Persuade)"]:::green
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

## 🚀 Quick Start

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

🌐 Open http://localhost:5000 to start debating!

---

## 📁 Project Structure

```
Social-Debate-AI/
├── src/                          # Core source code
│   ├── agents/                   # Agent implementations
│   │   ├── base_agent.py        # Base agent class
│   │   ├── agent_a.py           # Support agent
│   │   ├── agent_b.py           # Oppose agent
│   │   └── agent_c.py           # Neutral agent
│   ├── orchestrator/            # LangGraph orchestration
│   │   ├── langgraph_orchestrator.py  # Main orchestrator
│   │   ├── debate_state.py      # State schema
│   │   └── debate_tools.py      # Tool wrappers
│   ├── rag/                     # Retrieval-Augmented Generation
│   │   ├── retriever.py         # Enhanced retriever
│   │   └── simple_retriever.py  # Lightweight retriever
│   ├── gnn/                     # Graph Neural Network
│   │   ├── social_encoder.py    # Social graph encoder
│   │   └── train_supervised.py  # Training script
│   ├── rl/                      # Reinforcement Learning
│   │   ├── policy_network.py    # PPO policy network
│   │   └── ppo_trainer.py       # PPO trainer
│   └── dialogue/                # Dialogue management
├── ui/                          # Web application
│   ├── app.py                   # Flask server
│   ├── static/                  # CSS & JavaScript
│   └── templates/               # HTML templates
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── configs/                     # Configuration files
│   ├── debate.yaml              # Debate parameters
│   ├── rag.yaml                 # RAG settings
│   ├── gnn.yaml                 # GNN settings
│   └── rl.yaml                  # RL settings
├── docs/                        # Documentation
├── pyproject.toml               # Project configuration
└── uv.lock                      # Dependency lock file
```

---

## 🧪 Testing

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

## ⚙️ Configuration

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

## 🏋️ Model Training

```bash
# Train all models
uv run python train_all.py --all

# Train individual models
uv run python train_all.py --gnn    # GNN social encoder
uv run python train_all.py --rl     # RL policy network
uv run python train_all.py --rag    # Build RAG index
```

---

## 🛠️ Tech Stack

```mermaid
%%{init: {'theme': 'base'}}%%
block-beta
    columns 5
    
    block:orch:1
        columns 1
        A["🔄 LangGraph"]
        B["🔗 LangChain"]
    end
    
    block:ml:1
        columns 1
        C["🔥 PyTorch"]
        D["📊 PyG"]
        E["🔍 FAISS"]
    end
    
    block:llm:1
        columns 1
        F["🤖 OpenAI<br/>GPT-3.5/4"]
    end
    
    block:web:1
        columns 1
        G["🌐 Flask"]
        H["🎨 Bootstrap 5"]
    end
    
    block:tools:1
        columns 1
        I["📦 uv"]
        J["🧪 pytest"]
    end

    style orch fill:#818cf8,color:#fff
    style ml fill:#fb923c,color:#fff
    style llm fill:#4ade80,color:#fff
    style web fill:#22d3ee,color:#fff
    style tools fill:#f472b6,color:#fff
```

---

## 📚 Documentation

### 📖 Core Learning Resource
- **[LEARNING_NOTE.md](docs/LEARNING_NOTE.md)** 📚 Comprehensive deep learning study notes (GNN, PPO, RAG, LangGraph)

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

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`uv run pytest`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<a name="中文版本"></a>

# 🎭 Social Debate AI

<p align="center">
  <strong>基於深度學習的多智能體辯論系統</strong>
</p>

<p align="center">
  <a href="#english-version">English</a> | <a href="#中文版本">中文</a>
</p>

---

## 📖 概述

**Social Debate AI** 是一個智能多智能體辯論模擬系統，結合尖端深度學習技術。系統編排具有不同個性和立場的 AI 智能體進行動態辯論，使用 **LangGraph** 進行工作流管理、**RAG** 進行證據檢索、**GNN** 進行社交動態建模，以及 **RL** 進行策略決策。

### ✨ 核心特色

| 特色 | 說明 |
|------|------|
| 🤖 **多智能體辯論** | 3 個具有獨特立場（支持/反對/中立）的 AI 智能體進行動態辯論 |
| 🔄 **LangGraph 編排** | 宣告式狀態圖工作流，支援並行分析管線 |
| 📚 **RAG 證據檢索** | 基於 FAISS 的向量搜索，檢索相關證據和引用 |
| 🕸️ **GNN 社交建模** | 圖神經網路預測說服成功率和社交影響力 |
| 🎮 **RL 策略學習** | 基於 PPO 的強化學習，4 種自適應辯論策略 |
| 🌐 **現代化 Web 介面** | Flask + Bootstrap 5 響應式 UI，即時辯論視覺化 |

---

## 🏗️ 系統架構

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart TB
    %% Nodes Configuration
    classDef web fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;
    classDef orch fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef brain fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef gen fill:#ffedd5,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef agent fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;

    subgraph Presentation["🌐 展示層"]
        direction TB
        UI["🖥️ Bootstrap 5 UI"]:::web
        API["⚡ Flask API"]:::web
        UI <--> API
    end

    subgraph Core["⚙️ 編排核心"]
        direction TB
        LG["📊 LangGraph 引擎<br/>(狀態管理)"]:::orch
    end

    subgraph Intelligence["🧠 智能分析模組"]
        direction LR
        RL["🎮 RL 策略<br/>(PPO 決策)"]:::brain
        GNN["🕸️ GNN 社交<br/>(GraphSAGE)"]:::brain
        RAG["📚 RAG 證據<br/>(FAISS)"]:::brain
    end

    subgraph Generation["🔮 生成層"]
        direction TB
        Fusion["🔗 結果融合"]:::gen
        LLM["🤖 LLM 推理<br/>(GPT-3.5/4)"]:::gen
        Fusion --> LLM
    end

    subgraph Agents["👥 辯論智能體"]
        direction LR
        A1["🟢 智能體 A<br/>(支持方)"]:::agent
        A2["🔴 智能體 B<br/>(反對方)"]:::agent
        A3["🟡 智能體 C<br/>(中立方)"]:::agent
    end

    %% Data Flow Connections
    API <==> LG
    LG ==> Intelligence
    Intelligence ==> Fusion
    LLM ==> Agents
    Agents -.->|"狀態更新"| LG

    %% Styling
    style Presentation fill:#f0f9ff,stroke:#bae6fd,color:#0369a1
    style Core fill:#faf5ff,stroke:#e9d5ff,color:#6b21a8
    style Intelligence fill:#f0fdf4,stroke:#bbf7d0,color:#15803d
    style Generation fill:#fff7ed,stroke:#fed7aa,color:#c2410c
    style Agents fill:#fef2f2,stroke:#fecaca,color:#b91c1c
```

---

## 🔄 LangGraph 工作流程

辯論流程由宣告式 **StateGraph** 管理：

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#818cf8'}}}%%
flowchart TB
    %% Define styles
    classDef start fill:#10b981,stroke:#059669,stroke-width:2px,color:white;
    classDef process fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e3a8a;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#9a3412;
    classDef endNode fill:#ef4444,stroke:#b91c1c,stroke-width:2px,color:white;

    Start(("🚀 開始")):::start
    End(("🏁 結束")):::endNode

    subgraph Cycle["🔄 辯論循環"]
        direction TB
        Analyze["⚡ 並行分析<br/>(RL + GNN + RAG)"]:::process
        Fuse["🔗 結果融合"]:::process
        Gen["💬 生成回應"]:::process
        Update["📝 更新狀態"]:::process
        
        Analyze --> Fuse --> Gen --> Update
    end

    Check{"❓ 繼續?"}:::decision

    Start --> Analyze
    Update --> Check
    Check -->|"是 (下一回合)"| Analyze
    Check -->|"否 (最大回合/投降)"| End

    linkStyle default stroke:#6366f1,stroke-width:2px;
```

---

## 🎮 辯論策略

RL 模組根據辯論情境從 4 種自適應策略中選擇：

```mermaid
%%{init: {'theme': 'base'}}%%
graph TB
    subgraph Matrix["🎯 策略矩陣"]
        direction TB
        
        subgraph Row1[" "]
            direction LR
            S1["🔥 激進策略<br/>(挑戰與批判)"]:::red
            S2["🛡️ 防守策略<br/>(鞏固與防護)"]:::blue
        end
        
        subgraph Row2[" "]
            direction LR
            S3["🔬 分析策略<br/>(邏輯與證據)"]:::purple
            S4["💚 同理策略<br/>(連結與說服)"]:::green
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

## 🚀 快速開始

### 環境需求

- **Python** 3.10+
- **CUDA** 11.8+（可選，用於 GPU 加速）
- **RAM** 8GB+
- **OpenAI API Key**

### 安裝（使用 uv - 推薦）

```bash
# 1. 克隆儲存庫
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI

# 2. 安裝 uv 套件管理器
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 創建環境並安裝依賴
uv sync

# 4. 設定環境變數
cp env.example .env
# 編輯 .env 並添加您的 OPENAI_API_KEY
```

### 替代方案：pip 安裝

```bash
# 創建虛擬環境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 運行應用程式

```bash
# 使用 uv
uv run python ui/app.py

# 或使用已啟動的虛擬環境
python ui/app.py
```

🌐 開啟 http://localhost:5000 開始辯論！

---

## 📁 專案結構

```
Social-Debate-AI/
├── src/                          # 核心源碼
│   ├── agents/                   # 智能體實現
│   ├── orchestrator/             # LangGraph 編排
│   ├── rag/                      # 檢索增強生成
│   ├── gnn/                      # 圖神經網路
│   ├── rl/                       # 強化學習
│   └── dialogue/                 # 對話管理
├── ui/                           # Web 應用程式
├── tests/                        # 測試套件
├── configs/                      # 配置文件
├── docs/                         # 文檔
├── pyproject.toml                # 專案配置
└── uv.lock                       # 依賴鎖定文件
```

---

## 🧪 測試

```bash
# 運行所有測試
uv run pytest

# 詳細輸出
uv run pytest -v

# 運行特定測試文件
uv run pytest tests/unit/test_debate_state.py

# 生成覆蓋率報告
uv run pytest --cov=src
```

---

## ⚙️ 配置

### 環境變數

```bash
# 必要
OPENAI_API_KEY=sk-...

# 可選
USE_LANGGRAPH=true    # 啟用 LangGraph 編排器（預設：true）
```

### 配置文件

| 文件 | 說明 |
|------|------|
| `configs/debate.yaml` | 辯論回合數、時間、智能體設定 |
| `configs/rag.yaml` | 向量資料庫、嵌入模型、檢索參數 |
| `configs/gnn.yaml` | 圖結構、隱藏層維度 |
| `configs/rl.yaml` | PPO 超參數、獎勵設計 |
| `configs/system.yaml` | 全域系統設定 |

---

## 🏋️ 模型訓練

```bash
# 訓練所有模型
uv run python train_all.py --all

# 訓練單個模型
uv run python train_all.py --gnn    # GNN 社交編碼器
uv run python train_all.py --rl     # RL 策略網路
uv run python train_all.py --rag    # 建立 RAG 索引
```

---

## 🛠️ 技術棧

```mermaid
%%{init: {'theme': 'base'}}%%
block-beta
    columns 5
    
    block:orch:1
        columns 1
        A["🔄 LangGraph"]
        B["🔗 LangChain"]
    end
    
    block:ml:1
        columns 1
        C["🔥 PyTorch"]
        D["📊 PyG"]
        E["🔍 FAISS"]
    end
    
    block:llm:1
        columns 1
        F["🤖 OpenAI<br/>GPT-3.5/4"]
    end
    
    block:web:1
        columns 1
        G["🌐 Flask"]
        H["🎨 Bootstrap 5"]
    end
    
    block:tools:1
        columns 1
        I["📦 uv"]
        J["🧪 pytest"]
    end

    style orch fill:#818cf8,color:#fff
    style ml fill:#fb923c,color:#fff
    style llm fill:#4ade80,color:#fff
    style web fill:#22d3ee,color:#fff
    style tools fill:#f472b6,color:#fff
```

---

## 📚 文檔導覽

### 📖 核心學習資源
- **[LEARNING_NOTE.md](docs/LEARNING_NOTE.md)** 📚 完整深度學習筆記（GNN、PPO、RAG、LangGraph）

### 架構
- [系統概覽](docs/architecture/OVERVIEW.md) - 高層架構
- [LangGraph 編排](docs/architecture/LANGGRAPH.md) - 工作流引擎
- [資料流](docs/architecture/DATA_FLOW.md) - 狀態管理

### 指南
- [快速開始指南](docs/guides/QUICKSTART.md) - 5 分鐘啟動系統
- [配置指南](docs/guides/CONFIGURATION.md) - 系統配置
- [訓練指南](docs/guides/TRAINING.md) - 模型訓練
- [部署指南](docs/guides/DEPLOYMENT.md) - 生產部署

### API 與模組
- [REST API 參考](docs/api/REST_API.md) - Flask API 端點
- [RAG 模組](docs/modules/RAG.md) - 證據檢索
- [GNN 模組](docs/modules/GNN.md) - 社交分析
- [RL 模組](docs/modules/RL.md) - 策略選擇
- [評分系統](docs/modules/SCORING.md) - 勝負判定

---

## 🤝 貢獻

1. Fork 本儲存庫
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 運行測試 (`uv run pytest`)
4. 提交變更 (`git commit -m 'Add AmazingFeature'`)
5. 推送至分支 (`git push origin feature/AmazingFeature`)
6. 開啟 Pull Request

---

## 📄 授權

本專案採用 **MIT 授權** - 詳見 [LICENSE](LICENSE) 文件。

---

<p align="center">
  如果這個專案對您有幫助，請給我們一個 ⭐ Star！
</p>
