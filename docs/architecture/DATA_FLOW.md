# Data Flow and State Management

## Overview

This document describes how data flows through the Social Debate AI system and how state is managed throughout a debate session.

## State Lifecycle

```

                     Initial State Creation                       
  - Topic                                                         
  - Agent configurations (stance, conviction)                     
  - Max rounds                                                    

                              
                              

                        Debate Loop                               
      
   For each round:                                              
     For each agent:                                            
       1. Analyze (RL + GNN + RAG)                             
       2. Fuse results                                          
       3. Generate response                                     
       4. Evaluate effects                                      
       5. Update all agent states                              
       6. Check end conditions                                 
      

                              
                              

                     Final Summary                                
  - Winner determination                                          
  - Score calculation                                             
  - Debate statistics                                             

```

## State Components

### 1. Agent State

Each agent maintains:

```python
AgentState:
  agent_id: str           # Unique identifier
  current_stance: float   # -1.0 (oppose) to +1.0 (support)
  conviction: float       # 0.0 (easily swayed) to 1.0 (firm)
  social_context: [float] # 128-dim social embedding
  persuasion_history: []  # Last 10 persuasion scores received
  attack_history: []      # Last 10 attack scores received
  has_surrendered: bool   # Whether agent has given up
```

### 2. Debate State

Global debate state includes:

```python
DebateState:
  topic: str
  current_round: int
  max_rounds: int
  agent_order: [str]          # Speaking order
  current_speaker_index: int
  agent_states: {id: AgentState}
  history: [Response]         # All responses (accumulative)
  round_history: [Response]   # Current round only
```

### 3. Analysis Results

Temporary state during each turn:

```python
AnalysisResults:
  rl_result:
    strategy: str
    quality_score: float
    confidence: float
    
  gnn_result:
    social_vector: [float]
    influence_score: float
    stance_trend: float
    persuasion_prediction:
      delta_probability: float
      best_strategy: str
      
  rag_result:
    evidence_pool: [Evidence]
    best_evidence: str
    total_evidence: int
```

## Data Flow Details

### 1. Analysis Phase

```
Context Build
     
      RL Tool  strategy, quality_score
     
      GNN Tool  influence_score, persuasion_prediction
     
      RAG Tool  evidence_pool, best_evidence
```

### 2. Fusion Phase

```
RL Result 
               
GNN Result  Strategy Fusion Logic  final_strategy
               
RAG Result  Evidence Selection  best_evidence
               
                Confidence Calculation  evidence_confidence
```

**Fusion Rules:**
- If GNN's `delta_probability > 0.7`: Use GNN's suggested strategy
- If influence high + strong stance: Prefer aggressive strategy
- If influence low + weak stance: Prefer defensive strategy
- Otherwise: Balance RL and GNN recommendations

### 3. Generation Phase

```
Agent State 
                 
Fused Results  Prompt Builder  LLM  Response
                 
History 
```

### 4. Update Phase

```
Response  Evaluate Effects  persuasion_score
                               attack_score
                               evidence_score
                                      
                                      
                              Update Target Agent States
                                      
                                      
                              Check Surrender Conditions
```

**Surrender Detection:**
1. High persuasion (>0.65) + Low conviction (<0.25)
2. Near-neutral stance (<0.1) + Low conviction (<0.3)
3. 5 consecutive high persuasion rounds (>0.6) + Low conviction (<0.4)

## Response Format

Each response is stored as:

```python
Response:
  agent_id: str
  content: str
  effects:
    persuasion_score: float
    attack_score: float
    evidence_score: float
    length_score: float
  timestamp: float
```

## History Accumulation

The `history` field uses LangGraph's `operator.add` annotation:

```python
history: Annotated[List[Dict], operator.add]
```

This means:
- Each node can return `{"history": [new_response]}`
- LangGraph automatically appends to existing history
- No manual list management needed

## State Persistence

### In-Memory (Default)
- State lives in orchestrator instance
- Lost when process ends

### With Checkpointing (Optional)
```python
from langgraph.checkpoint import MemorySaver

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# State persists across invocations
config = {"configurable": {"thread_id": "debate-1"}}
result = graph.invoke(initial_state, config)
```

## API Data Format

### Request (Start Debate)
```json
{
  "topic": "Should AI be regulated?",
  "max_rounds": 5
}
```

### Response (Debate Results)
```json
{
  "success": true,
  "topic": "Should AI be regulated?",
  "rounds": [
    {
      "round": 1,
      "responses": [
        {
          "agent_id": "Agent_A",
          "content": "...",
          "effects": {...}
        }
      ],
      "agents": {
        "Agent_A": {"stance": 0.78, "conviction": 0.65}
      }
    }
  ],
  "summary": {
    "winner": "Agent_A",
    "scores": {...},
    "verdict": "..."
  }
}
```

