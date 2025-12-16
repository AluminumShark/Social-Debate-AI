# Quick Start Guide

*English | [中文](#chinese-version)*

Get Social Debate AI up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- 8GB+ RAM
- Git
- OpenAI API Key (optional, for full functionality)

## Installation Steps

### 1. Clone the Project
```bash
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI
```

### 2. Install uv (Recommended)

uv is a fast, modern Python package manager.

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Environment and Install Dependencies

```bash
# Using uv (recommended, ~30 seconds)
uv sync

# Or using pip (alternative)
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 4. Set API Key

```bash
# Create .env file
cp env.example .env

# Edit .env file, add your OpenAI API Key
# OPENAI_API_KEY=sk-...
```

## Quick Run

### Method 1: Web UI (Recommended)

```bash
# Using uv
uv run python ui/app.py

# Or with activated venv
python ui/app.py
```

Open your browser and visit http://localhost:5000

### Method 2: Run Tests First

```bash
# Verify installation
uv run pytest tests/ -v
```

## Using Web UI

### 1. Initialize System
- Open http://localhost:5000
- System will auto-initialize with LangGraph orchestrator

### 2. Set Debate Topic
Enter your discussion topic, for example:
- "Should artificial intelligence be regulated by government?"
- "Is universal basic income feasible?"
- "Is social media's impact positive or negative?"

### 3. Start Debate
- Click "Start Debate" or "Next Round" button
- Observe the debate process between three agents:
  - **Agent A** (Support): Stance +0.8
  - **Agent B** (Oppose): Stance -0.6
  - **Agent C** (Neutral): Stance 0.0
- Watch real-time stance and conviction changes

### 4. Analyze Results
- System automatically determines victory
- Export complete debate records
- View detailed scoring breakdown

## Architecture (LangGraph)

The system uses LangGraph for orchestration:

```
parallel_analysis → fuse_results → generate_response → update_states
       ↓                                                      ↓
  ┌────┴────┐                                           should_continue
  │ RL/GNN  │                                                 ↓
  │  /RAG   │                                      next_speaker / end
  └─────────┘
```

See [LangGraph Architecture](../architecture/LANGGRAPH.md) for details.

## Training Models (Optional)

```bash
# Train all models
uv run python train_all.py --all

# Individual training
uv run python train_all.py --gnn    # GNN social network
uv run python train_all.py --rl     # RL strategy model
uv run python train_all.py --rag    # RAG index
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM |
| `USE_LANGGRAPH` | No | Use LangGraph orchestrator (default: true) |

*Required for full functionality; system works with fallback without it.

### Config Files

Located in `configs/`:
- `debate.yaml` - Debate parameters
- `rag.yaml` - RAG system configuration
- `gnn.yaml` - GNN model configuration
- `rl.yaml` - RL training configuration
- `system.yaml` - System settings

## FAQ

### Q: Can I run without GPU?
A: Yes! The system automatically uses CPU. Training will be slower, but inference speed impact is minimal.

### Q: Is OpenAI API Key required?
A: Recommended but not required. Without API Key, the system uses fallback responses with limited functionality.

### Q: How to change debate parameters?
A: Edit `configs/debate.yaml` file to adjust rounds, agent count, etc.

### Q: How to use legacy orchestrator?
A: Set environment variable `USE_LANGGRAPH=false` before running.

### Q: System using too much memory?
A: You can reduce batch size in config files or use `--demo` mode for training.

## Next Steps

- Check [LangGraph Architecture](../architecture/LANGGRAPH.md) to understand the new orchestration
- Check [Training Guide](TRAINING.md) to learn model training
- Check [REST API](../api/REST_API.md) to integrate into your applications
- Check [Deployment Guide](DEPLOYMENT.md) for production deployment

## Need Help?

- Submit [GitHub Issue](https://github.com/your-username/Social-Debate-AI/issues)
- Check [Complete Documentation](../README.md)

---

<a name="chinese-version"></a>
# 快速開始指南

*[English](#quick-start-guide) | 中文*

5 分鐘內啟動並運行 Social Debate AI！

## 前置要求

- Python 3.10+
- 8GB+ RAM
- Git
- OpenAI API Key（可選，完整功能需要）

## 安裝步驟

### 1. 克隆專案
```bash
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI
```

### 2. 安裝 uv（推薦）

uv 是一個快速、現代的 Python 套件管理器。

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 創建環境並安裝依賴

```bash
# 使用 uv（推薦，約 30 秒）
uv sync

# 或使用 pip（替代方案）
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 4. 設置 API Key

```bash
# 創建 .env 文件
cp env.example .env

# 編輯 .env 文件，添加您的 OpenAI API Key
# OPENAI_API_KEY=sk-...
```

## 快速運行

### 方式一：Web UI（推薦）

```bash
# 使用 uv
uv run python ui/app.py

# 或使用已啟動的虛擬環境
python ui/app.py
```

打開瀏覽器訪問 http://localhost:5000

### 方式二：先運行測試

```bash
# 驗證安裝
uv run pytest tests/ -v
```

## 使用 Web UI

### 1. 初始化系統
- 打開 http://localhost:5000
- 系統會自動使用 LangGraph 編排器初始化

### 2. 設置辯論主題
輸入您想討論的主題，例如：
- "人工智慧是否應該被政府監管？"
- "基本收入是否可行？"
- "社交媒體的影響是正面還是負面？"

### 3. 開始辯論
- 點擊「開始辯論」或「下一回合」按鈕
- 觀察三個 Agent 的辯論過程：
  - **Agent A**（支持）：立場 +0.8
  - **Agent B**（反對）：立場 -0.6
  - **Agent C**（中立）：立場 0.0
- 查看實時的立場和信念變化

### 4. 分析結果
- 系統會自動判定勝負
- 可以導出完整的辯論記錄
- 查看詳細的評分細節

## 架構（LangGraph）

系統使用 LangGraph 進行編排：

```
parallel_analysis → fuse_results → generate_response → update_states
       ↓                                                      ↓
  ┌────┴────┐                                           should_continue
  │ RL/GNN  │                                                 ↓
  │  /RAG   │                                      next_speaker / end
  └─────────┘
```

詳見 [LangGraph 架構](../architecture/LANGGRAPH.md)。

## 訓練模型（可選）

```bash
# 訓練所有模型
uv run python train_all.py --all

# 個別訓練
uv run python train_all.py --gnn    # GNN 社會網路
uv run python train_all.py --rl     # RL 策略模型
uv run python train_all.py --rag    # RAG 索引
```

## 配置

### 環境變數

| 變數 | 必須 | 說明 |
|------|------|------|
| `OPENAI_API_KEY` | 是* | OpenAI API key |
| `USE_LANGGRAPH` | 否 | 使用 LangGraph 編排器（預設：true）|

*完整功能需要；沒有時系統會使用備用回應。

### 配置文件

位於 `configs/` 目錄：
- `debate.yaml` - 辯論參數
- `rag.yaml` - RAG 系統配置
- `gnn.yaml` - GNN 模型配置
- `rl.yaml` - RL 訓練配置
- `system.yaml` - 系統設置

## 常見問題

### Q: 沒有 GPU 可以運行嗎？
A: 可以！系統會自動使用 CPU。訓練會慢一些，但推理速度影響不大。

### Q: 必須要 OpenAI API Key 嗎？
A: 推薦但不是必須。沒有 API Key 時系統會使用備用回應，功能有所限制。

### Q: 如何更改辯論參數？
A: 編輯 `configs/debate.yaml` 文件，可以調整回合數、Agent 數量等。

### Q: 如何使用舊版編排器？
A: 在運行前設置環境變數 `USE_LANGGRAPH=false`。

### Q: 系統佔用太多記憶體？
A: 可以在配置文件中減小批次大小，或訓練時使用 `--demo` 模式。

## 下一步

- 查看 [LangGraph 架構](../architecture/LANGGRAPH.md) 了解新的編排方式
- 查看 [訓練指南](TRAINING.md) 了解如何訓練模型
- 查看 [REST API](../api/REST_API.md) 了解如何集成到您的應用
- 查看 [部署指南](DEPLOYMENT.md) 了解生產環境部署

## 需要幫助？

- 提交 [GitHub Issue](https://github.com/your-username/Social-Debate-AI/issues)
- 查看 [完整文檔](../README.md)

---

恭喜！您已經成功設置 Social Debate AI。開始探索智能辯論的世界吧！
