# ⚙️ LangGraph Orchestration Architecture

*English | [中文](#中文版本)*

---

## Overview

Version 0.2.0 introduces a **LangGraph-based orchestrator** that replaces manual async orchestration with a declarative, graph-based workflow.

---

## Why LangGraph?

| Aspect | Before (Manual) | After (LangGraph) |
|--------|-----------------|-------------------|
| Parallel Execution | Manual asyncio + ThreadPoolExecutor | Built-in parallel branches |
| State Management | Manual dict + dataclass | Automatic via StateGraph |
| Flow Control | Hardcoded if/else + while | Declarative graph + conditional edges |
| Visualization | None | `graph.get_graph().draw_png()` |
| Checkpointing | None | Built-in memory persistence |
| Tool Calling | Manual function calls | ToolNode + Agent autonomous decisions |

---

## Graph Structure

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Entry["🚀 Entry"]
        PA["parallel_analysis<br/>(RL + GNN + RAG)"]
    end

    subgraph Process["⚙️ Processing"]
        FR["fuse_results"]
        GR["generate_response<br/>(LLM)"]
        US["update_states"]
    end

    subgraph Decision["🔀 Decision"]
        SC{"should_continue"}
    end

    subgraph Continue["🔄 Continue"]
        AT["advance_turn"]
    end

    subgraph End["🏁 End"]
        E["END"]
    end

    PA --> FR --> GR --> US --> SC
    SC -->|next_speaker| AT
    SC -->|next_round| AT
    SC -->|end| E
    AT --> PA

    style Entry fill:#06b6d4,color:#fff
    style Process fill:#8b5cf6,color:#fff
    style Decision fill:#f59e0b,color:#fff
    style Continue fill:#10b981,color:#fff
    style End fill:#ef4444,color:#fff
```

---

## State Schema

```mermaid
%%{init: {'theme': 'base'}}%%
classDiagram
    class DebateState {
        +str topic
        +int current_round
        +int max_rounds
        +List~str~ agent_order
        +int current_speaker_index
        +Dict agent_states
        +List~Dict~ history
        +List~Dict~ round_history
        +Dict rl_result
        +Dict gnn_result
        +Dict rag_result
        +Dict fused_result
        +str current_response
        +Dict response_effects
        +bool debate_ended
        +str end_reason
        +str winner
    }

    class AgentState {
        +str agent_id
        +float current_stance
        +float conviction
        +List~float~ social_context
        +List~float~ persuasion_history
        +List~float~ attack_history
        +bool has_surrendered
    }

    DebateState "1" *-- "*" AgentState : contains
```

```python
class DebateState(TypedDict):
    # Core debate info
    topic: str
    current_round: int
    max_rounds: int
    
    # Agent management
    agent_order: List[str]
    current_speaker_index: int
    agent_states: Dict[str, Any]
    
    # History tracking (with operator.add for accumulation)
    history: Annotated[List[Dict], operator.add]
    round_history: List[Dict]
    
    # Analysis results
    rl_result: Optional[Dict]
    gnn_result: Optional[Dict]
    rag_result: Optional[Dict]
    fused_result: Optional[Dict]
    
    # Control flags
    debate_ended: bool
    end_reason: Optional[str]
    winner: Optional[str]
```

---

## Tools

The system defines four LangGraph tools:

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Tools["🔧 LangGraph Tools"]
        direction TB
        T1["rl_select_strategy"]
        T2["gnn_analyze_social"]
        T3["rag_retrieve_evidence"]
        T4["evaluate_response_effects"]
    end

    subgraph Output["📊 Outputs"]
        O1["strategy + confidence"]
        O2["influence + prediction"]
        O3["evidence pool"]
        O4["persuasion score"]
    end

    T1 --> O1
    T2 --> O2
    T3 --> O3
    T4 --> O4

    style Tools fill:#8b5cf6,color:#fff
    style Output fill:#10b981,color:#fff
```

| Tool | Purpose | Returns |
|------|---------|---------|
| `rl_select_strategy` | Select optimal debate strategy | strategy, quality_score, confidence |
| `gnn_analyze_social` | Analyze social influence | influence_score, persuasion_prediction, stance_trend |
| `rag_retrieve_evidence` | Retrieve supporting evidence | evidence_pool, best_evidence, evidence_types |
| `evaluate_response_effects` | Evaluate response impact | persuasion_score, attack_score, evidence_score |

---

## Node Implementation

### Parallel Analysis Node

```python
def _parallel_analysis_node(self, state: DebateState) -> Dict:
    # Execute RL, GNN, RAG in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        rl_future = executor.submit(rl_select_strategy.invoke, {...})
        gnn_future = executor.submit(gnn_analyze_social.invoke, {...})
        rag_future = executor.submit(rag_retrieve_evidence.invoke, {...})
    
        return {
            "rl_result": rl_future.result(),
            "gnn_result": gnn_future.result(),
            "rag_result": rag_future.result()
        }
```

### Conditional Routing

```python
def _should_continue(self, state: DebateState) -> Literal["next_speaker", "next_round", "end"]:
    # Check surrender conditions
    for agent_state in state["agent_states"].values():
        if agent_state.get('has_surrendered'):
            return "end"
    
    # Check round completion
    if current_speaker_index >= len(agent_order):
        if current_round >= max_rounds:
            return "end"
        return "next_round"
    
    return "next_speaker"
```

---

## Usage

### Basic Usage

```python
from src.orchestrator import create_langgraph_orchestrator

orchestrator = create_langgraph_orchestrator(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

results = orchestrator.run_debate(
    topic="Should AI be regulated?",
    agent_configs=[
        {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
        {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.7}
    ],
    max_rounds=5
)
```

### Step-by-Step Execution

```python
# For UI integration with round-by-round control
state = create_initial_state(topic, agent_configs, max_rounds)

for event in orchestrator.compiled_graph.stream(state):
    # Process each step
    print(event)
```

---

## Configuration

### Environment Variables

```bash
# Use LangGraph orchestrator (default: true)
USE_LANGGRAPH=true

# OpenAI API key (required for LLM)
OPENAI_API_KEY=sk-...
```

### Fallback Behavior

If LangGraph initialization fails, the system automatically falls back to the legacy `ParallelOrchestrator`.

---

## Comparison with Legacy Orchestrator

| Metric | Legacy | LangGraph |
|--------|--------|-----------|
| Lines of Code | ~900 | ~400 |
| State Management | Manual | Automatic |
| Debugging | Difficult | Graph visualization |
| Extensibility | Low | High |

### Benefits

1. **Clearer State Transitions**: Declarative graph makes flow explicit
2. **Built-in Parallelism**: No manual async management
3. **Better Debugging**: Graph visualization available
4. **Extensibility**: Easy to add new nodes/tools
5. **Memory/Checkpointing**: Built-in support for state persistence

---

## File Structure

```
src/orchestrator/
├── __init__.py                 # Module exports
├── parallel_orchestrator.py    # Legacy orchestrator (fallback)
├── langgraph_orchestrator.py   # LangGraph orchestrator
├── debate_state.py             # State schema definitions
└── debate_tools.py             # LangGraph tools
```

---

<a name="中文版本"></a>

# ⚙️ LangGraph 編排架構

*[English](#-langgraph-orchestration-architecture) | 中文*

---

## 概述

v0.2.0 引入了基於 **LangGraph 的編排器**，用聲明式、圖形化的工作流取代手動異步編排。

---

## 為什麼選擇 LangGraph？

| 方面 | 之前（手動）| 之後（LangGraph）|
|------|------------|------------------|
| 並行執行 | 手動 asyncio + ThreadPoolExecutor | 內建並行分支 |
| 狀態管理 | 手動 dict + dataclass | StateGraph 自動管理 |
| 流程控制 | 硬編碼 if/else + while | 聲明式圖 + 條件邊 |
| 視覺化 | 無 | `graph.get_graph().draw_png()` |
| 檢查點 | 無 | 內建記憶體持久化 |
| 工具調用 | 手動函數調用 | ToolNode + Agent 自主決策 |

---

## 圖結構

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Entry["🚀 入口"]
        PA["parallel_analysis<br/>(RL + GNN + RAG)"]
    end

    subgraph Process["⚙️ 處理"]
        FR["fuse_results"]
        GR["generate_response<br/>(LLM)"]
        US["update_states"]
    end

    subgraph Decision["🔀 決策"]
        SC{"should_continue"}
    end

    subgraph Continue["🔄 繼續"]
        AT["advance_turn"]
    end

    subgraph End["🏁 結束"]
        E["END"]
    end

    PA --> FR --> GR --> US --> SC
    SC -->|next_speaker| AT
    SC -->|next_round| AT
    SC -->|end| E
    AT --> PA

    style Entry fill:#06b6d4,color:#fff
    style Process fill:#8b5cf6,color:#fff
    style Decision fill:#f59e0b,color:#fff
    style Continue fill:#10b981,color:#fff
    style End fill:#ef4444,color:#fff
```

---

## 狀態模式

```python
class DebateState(TypedDict):
    # 核心辯論資訊
    topic: str                    # 辯論主題
    current_round: int            # 當前回合
    max_rounds: int               # 最大回合數
    
    # Agent 管理
    agent_order: List[str]        # Agent 順序
    current_speaker_index: int    # 當前發言者索引
    agent_states: Dict[str, Any]  # 所有 Agent 狀態
    
    # 歷史追蹤（使用 operator.add 累積）
    history: Annotated[List[Dict], operator.add]
    round_history: List[Dict]
    
    # 分析結果
    rl_result: Optional[Dict]
    gnn_result: Optional[Dict]
    rag_result: Optional[Dict]
    fused_result: Optional[Dict]
    
    # 控制標誌
    debate_ended: bool
    end_reason: Optional[str]
    winner: Optional[str]
```

---

## 工具

系統定義四個 LangGraph 工具：

| 工具 | 用途 | 返回值 |
|------|------|--------|
| `rl_select_strategy` | 選擇最佳辯論策略 | strategy, quality_score, confidence |
| `gnn_analyze_social` | 分析社交影響力 | influence_score, persuasion_prediction, stance_trend |
| `rag_retrieve_evidence` | 檢索支持證據 | evidence_pool, best_evidence, evidence_types |
| `evaluate_response_effects` | 評估回應影響 | persuasion_score, attack_score, evidence_score |

---

## 使用方式

### 基本使用

```python
from src.orchestrator import create_langgraph_orchestrator

orchestrator = create_langgraph_orchestrator(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

results = orchestrator.run_debate(
    topic="AI 是否應該被監管？",
    agent_configs=[
        {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
        {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.7}
    ],
    max_rounds=5
)
```

---

## 與舊版編排器的比較

| 指標 | 舊版 | LangGraph |
|------|------|-----------|
| 程式碼行數 | ~900 | ~400 |
| 狀態管理 | 手動 | 自動 |
| 除錯 | 困難 | 圖形視覺化 |
| 可擴展性 | 低 | 高 |

### 優勢

1. **更清晰的狀態轉換**：聲明式圖使流程明確
2. **內建並行處理**：無需手動異步管理
3. **更好的除錯**：可用圖形視覺化
4. **可擴展性**：易於添加新節點/工具
5. **記憶體/檢查點**：內建狀態持久化支援

---

## 文件結構

```
src/orchestrator/
├── __init__.py                 # 模組導出
├── parallel_orchestrator.py    # 舊版編排器（備用）
├── langgraph_orchestrator.py   # LangGraph 編排器
├── debate_state.py             # 狀態模式定義
└── debate_tools.py             # LangGraph 工具
```
