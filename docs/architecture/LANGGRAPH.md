#  LangGraph Orchestration Architecture

*English | [](#)*

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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart TB
    %% Styles
    classDef entry fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;
    classDef process fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef continue fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef endNode fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;

    subgraph Entry[" Entry"]
        PA["parallel_analysis<br/>(RL + GNN + RAG)"]:::entry
    end

    subgraph Process[" Processing"]
        FR["fuse_results"]:::process
        GR["generate_response<br/>(LLM)"]:::process
        US["update_states"]:::process
    end

    subgraph Decision[" Decision"]
        SC{"should_continue"}:::decision
    end

    subgraph Continue[" Continue"]
        AT["advance_turn"]:::continue
    end

    subgraph End[" End"]
        E["END"]:::endNode
    end

    PA --> FR --> GR --> US --> SC
    SC -->|next_speaker| AT
    SC -->|next_round| AT
    SC -->|end| E
    AT --> PA
```

---

## State Schema

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
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
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart LR
    %% Styles
    classDef tools fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef output fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;

    subgraph Tools[" LangGraph Tools"]
        direction TB
        T1["rl_select_strategy"]:::tools
        T2["gnn_analyze_social"]:::tools
        T3["rag_retrieve_evidence"]:::tools
        T4["evaluate_response_effects"]:::tools
    end

    subgraph Output[" Outputs"]
        O1["strategy + confidence"]:::output
        O2["influence + prediction"]:::output
        O3["evidence pool"]:::output
        O4["persuasion score"]:::output
    end

    T1 --> O1
    T2 --> O2
    T3 --> O3
    T4 --> O4
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
 __init__.py                 # Module exports
 parallel_orchestrator.py    # Legacy orchestrator (fallback)
 langgraph_orchestrator.py   # LangGraph orchestrator
 debate_state.py             # State schema definitions
 debate_tools.py             # LangGraph tools
```

---

<a name=""></a>

#  LangGraph 

*[English](#-langgraph-orchestration-architecture) | *

---

## 

v0.2.0  **LangGraph **

---

##  LangGraph

|  | | LangGraph|
|------|------------|------------------|
|  |  asyncio + ThreadPoolExecutor |  |
|  |  dict + dataclass | StateGraph  |
|  |  if/else + while |  +  |
|  |  | `graph.get_graph().draw_png()` |
|  |  |  |
|  |  | ToolNode + Agent  |

---

## 

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4f46e5', 'primaryTextColor': '#fff', 'primaryBorderColor': '#4338ca', 'lineColor': '#6366f1', 'secondaryColor': '#10b981', 'tertiaryColor': '#f59e0b'}}}%%
flowchart TB
    %% Styles
    classDef entry fill:#e0f2fe,stroke:#0ea5e9,stroke-width:2px,color:#0c4a6e;
    classDef process fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:2px,color:#7c2d12;
    classDef continue fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d;
    classDef endNode fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#7f1d1d;

    subgraph Entry[" "]
        PA["parallel_analysis<br/>(RL + GNN + RAG)"]:::entry
    end

    subgraph Process[" "]
        FR["fuse_results"]:::process
        GR["generate_response<br/>(LLM)"]:::process
        US["update_states"]:::process
    end

    subgraph Decision[" "]
        SC{"should_continue"}:::decision
    end

    subgraph Continue[" "]
        AT["advance_turn"]:::continue
    end

    subgraph End[" "]
        E["END"]:::endNode
    end

    PA --> FR --> GR --> US --> SC
    SC -->|next_speaker| AT
    SC -->|next_round| AT
    SC -->|end| E
    AT --> PA
```

---

## 

```python
class DebateState(TypedDict):
    # 
    topic: str                    # 
    current_round: int            # 
    max_rounds: int               # 
    
    # Agent 
    agent_order: List[str]        # Agent 
    current_speaker_index: int    # 
    agent_states: Dict[str, Any]  #  Agent 
    
    #  operator.add 
    history: Annotated[List[Dict], operator.add]
    round_history: List[Dict]
    
    # 
    rl_result: Optional[Dict]
    gnn_result: Optional[Dict]
    rag_result: Optional[Dict]
    fused_result: Optional[Dict]
    
    # 
    debate_ended: bool
    end_reason: Optional[str]
    winner: Optional[str]
```

---

## 

 LangGraph 

|  |  |  |
|------|------|--------|
| `rl_select_strategy` |  | strategy, quality_score, confidence |
| `gnn_analyze_social` |  | influence_score, persuasion_prediction, stance_trend |
| `rag_retrieve_evidence` |  | evidence_pool, best_evidence, evidence_types |
| `evaluate_response_effects` |  | persuasion_score, attack_score, evidence_score |

---

## 

### 

```python
from src.orchestrator import create_langgraph_orchestrator

orchestrator = create_langgraph_orchestrator(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

results = orchestrator.run_debate(
    topic="AI ",
    agent_configs=[
        {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
        {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.7}
    ],
    max_rounds=5
)
```

---

## 

|  |  | LangGraph |
|------|------|-----------|
|  | ~900 | ~400 |
|  |  |  |
|  |  |  |
|  |  |  |

### 

1. ****
2. ****
3. ****
4. ****/
5. **/**

---

## 

```
src/orchestrator/
 __init__.py                 # 
 parallel_orchestrator.py    # 
 langgraph_orchestrator.py   # LangGraph 
 debate_state.py             # 
 debate_tools.py             # LangGraph 
```
