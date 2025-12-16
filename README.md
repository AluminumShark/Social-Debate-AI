# Social Debate AI

*English | [中文](#chinese-version)*

A deep learning-based multi-agent social debate system that integrates RAG, GNN, and RL technologies with **LangGraph orchestration** for intelligent debate simulation.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)

## What's New in v0.2.0

- 🚀 **LangGraph Orchestration** - Replaced manual async orchestration with declarative StateGraph
- 🔧 **uv Package Management** - Fast, modern Python package management
- 📦 **Modular Tools** - RAG/GNN/RL wrapped as LangGraph tools
- 📚 **Restructured Documentation** - Organized docs with clear hierarchy
- ✅ **Structured Tests** - pytest-based test suite in `tests/`

## Key Features

- **Multi-Agent Debate** - 3 AI Agents with different stances and personalities engage in dynamic debates
- **LangGraph Workflow** - Declarative graph-based orchestration with parallel analysis
- **RAG Enhancement** - FAISS-based vector retrieval for evidence
- **GNN Social Network** - Supervised learning to predict persuasion success rate
- **RL Strategy Learning** - PPO reinforcement learning with 4 dynamic debate strategies
- **Web Interface** - Modern Flask + Bootstrap 5 responsive interface

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                 LangGraph Orchestrator                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Parallel Analysis                       │    │
│  │    ┌────────┐  ┌────────┐  ┌────────┐              │    │
│  │    │   RL   │  │  GNN   │  │  RAG   │              │    │
│  │    └────┬───┘  └────┬───┘  └────┬───┘              │    │
│  │         └───────────┼───────────┘                   │    │
│  │                     ▼                               │    │
│  │              Result Fusion                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│              Response Generation (LLM)                       │
│                         │                                    │
│                         ▼                                    │
│              State Update & Control                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│  Agent A (Support)  │  Agent B (Oppose)  │  Agent C (Neutral)│
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Requirements
- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM
- OpenAI API Key

### Installation with uv (Recommended)

```bash
# 1. Clone the project
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI

# 2. Install uv (if not already installed)
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create environment and install dependencies
uv sync

# 4. Set up environment variables
cp env.example .env
# Edit .env file, add your OPENAI_API_KEY
```

### Alternative: pip installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# With uv
uv run python ui/app.py

# Or with activated venv
python ui/app.py
```

Visit http://localhost:5000 to start debating!

## Project Structure

```
Social-Debate-AI/
├── src/                      # Core modules
│   ├── agents/              # Agent implementations
│   ├── orchestrator/        # LangGraph orchestrator (NEW)
│   │   ├── langgraph_orchestrator.py
│   │   ├── debate_state.py
│   │   ├── debate_tools.py
│   │   └── parallel_orchestrator.py (legacy)
│   ├── rag/                 # RAG retrieval system
│   ├── gnn/                 # GNN social network
│   ├── rl/                  # RL strategy learning
│   ├── dialogue/            # Dialogue management
│   └── gpt_interface/       # GPT API interface
├── ui/                       # Flask web application
├── tests/                    # Test suite (NEW)
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docs/                     # Documentation (Restructured)
│   ├── architecture/       # System architecture
│   ├── guides/             # User guides
│   ├── api/                # API reference
│   └── modules/            # Module documentation
├── configs/                  # Configuration files
├── data/                     # Data and models
├── pyproject.toml           # Project configuration
└── uv.lock                  # Dependency lock file
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/unit/test_debate_state.py

# Run with coverage
uv run pytest --cov=src
```

## Documentation

### Architecture
- [System Overview](docs/architecture/OVERVIEW.md) - High-level architecture
- [LangGraph Orchestration](docs/architecture/LANGGRAPH.md) - **NEW** Graph-based workflow
- [Data Flow](docs/architecture/DATA_FLOW.md) - State management

### Guides
- [Quick Start](docs/guides/QUICKSTART.md) - 5-minute tutorial
- [Configuration](docs/guides/CONFIGURATION.md) - System configuration
- [Training](docs/guides/TRAINING.md) - Model training
- [Deployment](docs/guides/DEPLOYMENT.md) - Production deployment

### API & Modules
- [REST API](docs/api/REST_API.md) - Flask API reference
- [RAG Module](docs/modules/RAG.md) - Retrieval system
- [GNN Module](docs/modules/GNN.md) - Graph neural network
- [RL Module](docs/modules/RL.md) - Reinforcement learning
- [Scoring System](docs/modules/SCORING.md) - Debate evaluation

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
USE_LANGGRAPH=true  # Use LangGraph orchestrator (default: true)
```

### Config Files

Located in `configs/`:
- `debate.yaml` - Debate parameters
- `rag.yaml` - RAG system configuration
- `gnn.yaml` - GNN model configuration
- `rl.yaml` - RL training configuration
- `system.yaml` - System settings

## Training Models

```bash
# Train all models
uv run python train_all.py --all

# Individual training
uv run python train_all.py --gnn    # GNN social network
uv run python train_all.py --rl     # RL strategy model
uv run python train_all.py --rag    # RAG index
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph, LangChain |
| LLM | OpenAI GPT-3.5/4 |
| ML Framework | PyTorch, PyTorch Geometric |
| Vector DB | FAISS |
| Web | Flask, Bootstrap 5 |
| Package Manager | uv |

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Run tests (`uv run pytest`)
4. Commit your changes (`git commit -m 'Add AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<a name="chinese-version"></a>
# Social Debate AI

*[English](#social-debate-ai) | 中文*

基於深度學習的多智能體社會辯論系統，整合 RAG、GNN、RL 技術，並使用 **LangGraph 編排** 實現智能辯論模擬。

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)

## v0.2.0 新功能

- 🚀 **LangGraph 編排** - 以宣告式 StateGraph 取代手動 async 編排
- 🔧 **uv 套件管理** - 快速、現代的 Python 套件管理
- 📦 **模組化工具** - RAG/GNN/RL 封裝為 LangGraph tools
- 📚 **重構文檔** - 清晰的文檔結構
- ✅ **結構化測試** - 基於 pytest 的測試套件

## 核心特色

- **多智能體辯論** - 3 個具有不同立場和性格的 AI Agent 進行動態辯論
- **LangGraph 工作流** - 宣告式圖形編排，並行分析
- **RAG 檢索增強** - 基於 FAISS 向量檢索證據
- **GNN 社會網路** - 監督式學習預測說服成功率
- **RL 策略學習** - PPO 強化學習，4 種辯論策略動態選擇
- **Web 介面** - 現代化的 Flask + Bootstrap 5 響應式界面

## 快速開始

### 環境要求
- Python 3.10+
- CUDA 11.8+ (可選，用於 GPU 加速)
- 8GB+ RAM
- OpenAI API Key

### 使用 uv 安裝（推薦）

```bash
# 1. 克隆專案
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI

# 2. 安裝 uv（如果尚未安裝）
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 創建環境並安裝依賴
uv sync

# 4. 設置環境變數
cp env.example .env
# 編輯 .env 文件，添加您的 OPENAI_API_KEY
```

### 運行應用

```bash
# 使用 uv
uv run python ui/app.py
```

訪問 http://localhost:5000 開始辯論！

## 專案結構

```
Social-Debate-AI/
├── src/                      # 核心模組
│   ├── agents/              # Agent 實現
│   ├── orchestrator/        # LangGraph 編排器（新）
│   │   ├── langgraph_orchestrator.py
│   │   ├── debate_state.py
│   │   ├── debate_tools.py
│   │   └── parallel_orchestrator.py (舊版)
│   ├── rag/                 # RAG 檢索系統
│   ├── gnn/                 # GNN 社會網路
│   ├── rl/                  # RL 策略學習
│   └── ...
├── ui/                       # Flask Web 應用
├── tests/                    # 測試套件（新）
│   ├── unit/               # 單元測試
│   └── integration/        # 整合測試
├── docs/                     # 文檔（重構）
│   ├── architecture/       # 系統架構
│   ├── guides/             # 使用指南
│   ├── api/                # API 參考
│   └── modules/            # 模組文檔
├── configs/                  # 配置文件
├── pyproject.toml           # 專案配置
└── uv.lock                  # 依賴鎖定文件
```

## 運行測試

```bash
# 運行所有測試
uv run pytest

# 詳細輸出
uv run pytest -v

# 運行特定測試
uv run pytest tests/unit/test_debate_state.py
```

## 文檔導覽

### 架構
- [系統概覽](docs/architecture/OVERVIEW.md) - 高層架構
- [LangGraph 編排](docs/architecture/LANGGRAPH.md) - **新** 圖形工作流
- [資料流](docs/architecture/DATA_FLOW.md) - 狀態管理

### 指南
- [快速開始](docs/guides/QUICKSTART.md) - 5 分鐘教程
- [配置指南](docs/guides/CONFIGURATION.md) - 系統配置
- [訓練指南](docs/guides/TRAINING.md) - 模型訓練
- [部署指南](docs/guides/DEPLOYMENT.md) - 生產部署

### API 與模組
- [REST API](docs/api/REST_API.md) - Flask API 參考
- [RAG 模組](docs/modules/RAG.md) - 檢索系統
- [GNN 模組](docs/modules/GNN.md) - 圖神經網路
- [RL 模組](docs/modules/RL.md) - 強化學習
- [評分系統](docs/modules/SCORING.md) - 辯論評估

## 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件

---

如果這個專案對您有幫助，請給我們一個 ⭐ Star！
