# Quick Start Guide

*English | [](#chinese-version)*

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
                                             should_continue
   RL/GNN                                                   ↓
    /RAG                                         next_speaker / end
  
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
# 

*[English](#quick-start-guide) | *

5  Social Debate AI

## 

- Python 3.10+
- 8GB+ RAM
- Git
- OpenAI API Key

## 

### 1. 
```bash
git clone https://github.com/your-username/Social-Debate-AI.git
cd Social-Debate-AI
```

### 2.  uv

uv  Python 

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 

```bash
#  uv 30 
uv sync

#  pip
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 4.  API Key

```bash
#  .env 
cp env.example .env

#  .env  OpenAI API Key
# OPENAI_API_KEY=sk-...
```

## 

### Web UI

```bash
#  uv
uv run python ui/app.py

# 
python ui/app.py
```

 http://localhost:5000

### 

```bash
# 
uv run pytest tests/ -v
```

##  Web UI

### 1. 
-  http://localhost:5000
-  LangGraph 

### 2. 

- ""
- ""
- ""

### 3. 
- 
-  Agent 
  - **Agent A** +0.8
  - **Agent B** -0.6
  - **Agent C** 0.0
- 

### 4. 
- 
- 
- 

## LangGraph

 LangGraph 

```
parallel_analysis → fuse_results → generate_response → update_states
       ↓                                                      ↓
                                             should_continue
   RL/GNN                                                   ↓
    /RAG                                         next_speaker / end
  
```

 [LangGraph ](../architecture/LANGGRAPH.md)

## 

```bash
# 
uv run python train_all.py --all

# 
uv run python train_all.py --gnn    # GNN 
uv run python train_all.py --rl     # RL 
uv run python train_all.py --rag    # RAG 
```

## 

### 

|  |  |  |
|------|------|------|
| `OPENAI_API_KEY` | * | OpenAI API key |
| `USE_LANGGRAPH` |  |  LangGraph true|

*

### 

 `configs/` 
- `debate.yaml` - 
- `rag.yaml` - RAG 
- `gnn.yaml` - GNN 
- `rl.yaml` - RL 
- `system.yaml` - 

## 

### Q:  GPU 
A:  CPU

### Q:  OpenAI API Key 
A:  API Key 

### Q: 
A:  `configs/debate.yaml` Agent 

### Q: 
A:  `USE_LANGGRAPH=false`

### Q: 
A:  `--demo` 

## 

-  [LangGraph ](../architecture/LANGGRAPH.md) 
-  [](TRAINING.md) 
-  [REST API](../api/REST_API.md) 
-  [](DEPLOYMENT.md) 

## 

-  [GitHub Issue](https://github.com/your-username/Social-Debate-AI/issues)
-  [](../README.md)

---

 Social Debate AI
