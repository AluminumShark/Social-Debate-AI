# Debate Scoring and Victory Determination System

*English | [](#chinese-version)*

## Table of Contents

1. [System Overview](#system-overview)
2. [Scoring Mechanism](#scoring-mechanism)
3. [State Update Rules](#state-update-rules)
4. [Surrender Mechanism](#surrender-mechanism)
5. [Victory Determination](#victory-determination)
6. [Strategy Recommendations](#strategy-recommendations)
7. [Case Studies](#case-studies)
8. [Technical Implementation](#technical-implementation)

## System Overview

The debate scoring system of Social Debate AI is a multi-dimensional comprehensive evaluation mechanism designed to simulate complex interactions in real debates. The system not only considers the persuasiveness of arguments but also evaluates participants' stance firmness, influence, and resistance to pressure.

### Core Philosophy

- **Dynamic Balance**: Stance and belief adjust dynamically based on debate progress
- **Multi-dimensional Evaluation**: Comprehensive consideration of attack, defense, persuasion, and other capabilities
- **Strategic Depth**: Encourages thoughtful debate strategies rather than simple confrontation

## Scoring Mechanism

### 1. Stance Conviction Score

**Formula**:
```python
stance_score = abs(current_stance) × conviction × 30
```

**Detailed Explanation**:
- `current_stance`: Current stance (-1.0 to 1.0)
  - Positive values indicate support, negative values indicate opposition
  - Higher absolute value means more extreme position
- `conviction`: Belief strength (0.0 to 1.0)
  - Represents firmness in one's position
  - Higher values are less likely to be persuaded
- Base weight: 30 points

**Scoring Logic**:
- Rewards participants with clear and firm stances
- Neutral or wavering positions receive lower scores
- Reflects the importance of "having a position" in debates

**Example Calculation**:
| Agent | Stance | Conviction | Score |
|-------|--------|------------|-------|
| A | 0.8 | 0.7 | 16.8 |
| B | -0.6 | 0.6 | 10.8 |
| C | 0.2 | 0.5 | 3.0 |

### 2. Persuasion Score

**Calculation Method**:
```python
persuasion_score = 0
for other_agent in all_agents:
    if other_agent.has_surrendered:
        persuasion_score += 20  # Surrender bonus
    
    # Influence bonus
    avg_persuasion = mean(other_agent.persuasion_history)
    persuasion_score += avg_persuasion × 10
```

**Scoring Components**:

#### a) Surrender Bonus (20 points/person)
- Successfully persuading opponents to surrender is the highest achievement
- Each surrendering opponent provides 20 point bonus
- Embodies the art of "winning without fighting" in debate

#### b) Influence Bonus (0-10 points/person)
- Based on average persuasion degree on each opponent
- Even without causing surrender, continuous influence has value
- Calculates cumulative effects across all rounds

**Persuasion Degree Assessment Standards**:
- 0.0-0.2: Minimal influence
- 0.2-0.4: Slight influence
- 0.4-0.6: Moderate influence
- 0.6-0.8: Significant influence
- 0.8-1.0: Strong influence

### 3. Resistance Score

**Formula**:
```python
avg_attack = mean(attack_history)
resistance_score = (1 - avg_attack) × conviction × 20
```

**Scoring Logic**:
- Measures ability to maintain position when facing attacks
- Lower attack received, or more successful attack resistance, results in higher scores
- Conviction strength is an important factor in resistance ability

**Resistance Performance Grading**:
| Avg Attack Degree | Conviction | Resistance Rating |
|------------------|-----------|-------------------|
| < 0.3 | > 0.7 | Excellent |
| 0.3-0.5 | 0.5-0.7 | Good |
| 0.5-0.7 | 0.3-0.5 | Average |
| > 0.7 | < 0.3 | Poor |

### 4. Surrender Penalty

**Fixed Penalty**: -50 points

**Penalty Rationale**:
- Surrender represents complete abandonment of one's position
- Loss of ability to continue arguing in the debate
- Severe penalty ensures agents don't surrender easily

## State Update Rules

### 1. Persuasion Effect

When an agent is persuaded (persuasion degree > 0.6):

```python
# Stance moves toward neutral
persuasion_effect = persuasion_score × (1.0 - conviction)
new_stance = current_stance × (1.0 - persuasion_effect × 0.3)

# Conviction weakens
new_conviction = conviction × 0.85
```

**Effect Analysis**:
- Stance gradually trends toward neutral (0)
- Conviction strength decreases, becoming more susceptible to further persuasion
- High conviction individuals have stronger "immunity"

### 2. Attack Effect

When an agent is attacked (attack effect > 0.3):

```python
# Calculate attack resistance
attack_resistance = conviction × 0.8
attack_effect = max(0, attack_score - attack_resistance)

# Stance polarization
new_stance = current_stance × (1.0 + attack_effect × 0.2)

# Conviction strengthening
new_conviction = min(1.0, conviction × 1.1)
```

**Effect Analysis**:
- Attacks may lead to "backlash effect"
- Stance becomes more extreme
- Conviction strengthens due to defensive psychology
- Reflects psychological "confirmation bias"

### 3. History Record Maintenance

```python
# Keep most recent 10 records
if len(history) > 10:
    history.pop(0)
```

- Only considers recent interactions
- Avoids excessive accumulation of early influence
- Maintains timeliness of dynamic assessment

## Surrender Mechanism

### Detailed Surrender Conditions

#### Condition 1: High Persuasion + Low Conviction
```python
if recent_persuasion > 0.6 and conviction < 0.4:
    surrender = True
```

**Trigger Scenarios**:
- Continuous impact from powerful arguments
- Insufficient foundation in one's own arguments
- Beginning to doubt one's position

#### Condition 2: Stance Wavering
```python
if abs(current_stance) < 0.2 and conviction < 0.5:
    surrender = True
```

**Trigger Scenarios**:
- Already essentially persuaded to neutral position
- Lost motivation to continue debating
- Believes both sides have merit

#### Condition 3: Consecutive Persuasion
```python
consecutive_high = all(score > 0.5 for score in persuasion_history[-3:])
if consecutive_high:
    surrender = True
```

**Trigger Scenarios**:
- Effectively persuaded for 3 consecutive rounds
- Unable to provide strong counterarguments
- Psychological defenses gradually collapsing

### Surrender Consequences

1. **Immediate Effects**:
   - 50 point deduction from score
   - Stop participating in subsequent debates
   - Recorded as "persuaded"

2. **Impact on Others**:
   - Persuader receives 20 point bonus
   - May influence bystanders' positions
   - Changes power balance of debate

## Victory Determination

### 1. Total Score Calculation

```python
total_score = (
    stance_score +           # Stance firmness
    persuasion_score +       # Persuading others
    resistance_score -       # Resistance ability
    surrender_penalty        # Surrender penalty
)
```

### 2. Victory Type Determination

#### Overwhelming Victory
```python
if len(surrendered_agents) > 0:
    verdict = f" {winner} achieved overwhelming victory! Successfully persuaded {surrendered} to surrender."
```

**Characteristics**:
- Persuaded at least one opponent to surrender
- Demonstrated exceptional persuasion ability
- Highest level of victory

#### Clear Advantage
```python
if score_difference > 30:
    verdict = f" {winner} won with clear advantage! Demonstrated excellent debate skills."
```

**Characteristics**:
- Leading second place by more than 30 points
- Excellent performance in multiple dimensions
- Convincing victory

#### Narrow Victory
```python
else:
    verdict = f" {winner} won narrowly! This was an evenly matched exciting debate."
```

**Characteristics**:
- Leading advantage less than 30 points
- Similar strength between parties
- Victory possibly decided by details

### 3. Comprehensive Evaluation Dimensions

The system generates detailed evaluation reports:

```json
{
    "total_rounds": 5,
    "winner": "Agent_A",
    "scores": {
        "Agent_A": 67.5,
        "Agent_B": 35.2,
        "Agent_C": -12.3
    },
    "surrendered_agents": ["Agent_C"],
    "final_states": {
        "Agent_A": {
            "stance": 0.75,
            "conviction": 0.65,
            "has_surrendered": false,
            "final_position": "Supportive"
        }
    },
    "verdict": "Overwhelming Victory"
}
```

## Strategy Recommendations

### 1. Aggressive Strategy

**Applicable Situations**:
- Opponent has weak conviction (< 0.5)
- Strong evidence support available
- Need to quickly establish advantage

**Key Points**:
- Focus fire on opponent's argument weaknesses
- Use aggressive strategy
- Combine with high-quality RAG evidence

**Risks**:
- May trigger opponent's defense mechanisms
- Excessive attacks may be seen as irrational

### 2. Defensive Strategy

**Applicable Situations**:
- Strong personal conviction (> 0.7)
- Facing strong opponents
- Need to maintain scoring advantage

**Key Points**:
- Strengthen core arguments
- Use defensive strategy
- Focus on improving resistance ability score

**Advantages**:
- Less likely to surrender under persuasion
- Steadily accumulate stance points

### 3. Balanced Strategy

**Applicable Situations**:
- Multi-party battle situation
- Need flexible response
- Pursuing comprehensive score

**Key Points**:
- Dynamically adjust strategy based on GNN predictions
- Balance attack and defense
- Pay attention to opponent state changes

### 4. Persuasive Strategy

**Applicable Situations**:
- Opponent's position not firm enough
- Personal persuasion advantage
- Pursuing surrender bonus

**Key Points**:
- Use empathetic strategy to build rapport
- Gradually weaken opponent's conviction
- Apply persuasive pressure at key moments

## Case Studies

### Case 1: Dynamic Balance in Three-Way Debate

**Initial State**:
- Agent_A: Stance +0.8 (strongly supportive), Conviction 0.7
- Agent_B: Stance -0.6 (opposed), Conviction 0.6
- Agent_C: Stance 0.0 (neutral), Conviction 0.5

**Round 1**:
- A uses analytical strategy, presents data arguments
- B uses aggressive strategy, attacks A's assumptions
- C uses empathetic strategy, understands both viewpoints

**Effects**:
- A under attack, stance becomes firmer: +0.85
- B's attack resisted, conviction slightly increases: 0.62
- C influenced by both sides, slightly favors A: +0.15

**Round 2**:
- A adjusts to defensive, consolidates arguments
- B continues aggressive, increases attacks
- C switches to analytical, proposes compromise

**Key Turning Point**:
- B's excessive attacks cause resentment
- C's rational analysis gains recognition
- A begins considering C's viewpoint

**Final Result**:
- Agent_A: 45.6 points (firm stance but failed to persuade others)
- Agent_B: 28.3 points (attack strategy had limited effect)
- Agent_C: 52.1 points (successfully influenced both sides, narrow victory)

### Case 2: Victory Through Persuasion

**Scenario**: Debate about "Universal Basic Income"

**Key Strategy**:
1. Agent_A first uses analytical to establish theoretical foundation
2. Identifies Agent_B's weak conviction points
3. Switches to empathetic strategy, understands opponent's concerns
4. Provides specific solutions to eliminate doubts
5. Maintains high persuasion for three consecutive rounds

**Result**:
- Agent_B surrenders in round 4
- Agent_A achieves overwhelming victory
- Final score: A (72.5) vs B (-25.0)

## Technical Implementation

### 1. Core Data Structures

```python
@dataclass
class AgentState:
    agent_id: str
    current_stance: float          # -1.0 to 1.0
    conviction: float              # 0.0 to 1.0
    social_context: List[float]    # 128-dimensional social vector
    persuasion_history: List[float]
    attack_history: List[float]
    has_surrendered: bool = False
```

### 2. Score Calculation Function

```python
def calculate_agent_score(agent_id: str, state: AgentState, 
                         all_states: Dict[str, AgentState]) -> float:
    score = 0
    
    # Stance firmness
    stance_score = abs(state.current_stance) * state.conviction * 30
    score += stance_score
    
    # Persuading others
    persuasion_score = 0
    for other_id, other_state in all_states.items():
        if other_id != agent_id:
            if other_state.has_surrendered:
                persuasion_score += 20
            avg_persuasion = sum(other_state.persuasion_history) / len(other_state.persuasion_history)
            persuasion_score += avg_persuasion * 10
    score += persuasion_score
    
    # Resistance ability
    if len(state.attack_history) > 0:
        avg_attack = sum(state.attack_history) / len(state.attack_history)
        resistance_score = (1 - avg_attack) * state.conviction * 20
        score += resistance_score
    
    # Surrender penalty
    if state.has_surrendered:
        score -= 50
    
    return score
```

### 3. Effect Evaluation Function

```python
def evaluate_response_effects(response: str, target_agents: List[str]) -> Dict:
    # Keyword analysis
    persuasion_indicators = ['however', 'consider', 'but', 'think about', 'understand']
    attack_indicators = ['wrong', 'flawed', 'incorrect', 'mistaken', 'fallacious']
    evidence_indicators = ['research', 'data', 'study', 'evidence', 'statistics']
    
    # Calculate scores
    response_lower = response.lower()
    persuasion_score = min(1.0, sum(ind in response_lower for ind in persuasion_indicators) * 0.3)
    attack_score = min(1.0, sum(ind in response_lower for ind in attack_indicators) * 0.4)
    evidence_score = min(1.0, sum(ind in response_lower for ind in evidence_indicators) * 0.35)
    
    return {
        'persuasion_score': persuasion_score,
        'attack_score': attack_score,
        'evidence_score': evidence_score
    }
```

### 4. Configuration Parameters

System behavior can be adjusted through `configs/debate.yaml`:

```yaml
victory_conditions:
  surrender_threshold: 0.4      # Surrender conviction threshold
  stance_neutral_threshold: 0.2 # Neutral stance threshold
  consecutive_persuasion: 3     # Consecutive persuaded rounds

persuasion_factors:
  base_persuasion: 0.3         # Base persuasion power
  strategy_bonus: 0.2          # Strategy bonus
  evidence_bonus: 0.3          # Evidence bonus
  social_influence: 0.2        # Social influence
```

## Summary

Social Debate AI's scoring system realistically simulates the complexity of debate through multi-dimensional evaluation:

1. **More than just arguments**: Simultaneously evaluates stance, persuasion, and resistance
2. **Dynamic game**: States constantly change with debate progress
3. **Strategic depth**: Different strategies applicable to different scenarios
4. **Psychological realism**: Simulates confirmation bias, defense mechanisms, and other psychological phenomena

This system encourages participants to:
- Maintain rational but firm positions
- Use evidence and logic to persuade opponents
- Find balance between attack and defense
- Demonstrate true debate artistry

Through this design, AI debate is no longer simple viewpoint output, but a comprehensive contest requiring wisdom, strategy, and psychological qualities.

---

## Chinese Version

# 

*[English](#debate-scoring-and-victory-determination-system) | *

## 

1. [](#)
2. [](#)
3. [](#)
4. [](#)
5. [](#)
6. [](#)
7. [](#)
8. [](#)

## 

Social Debate AI 

### 

- ****
- ****
- ****

## 

### 1. Stance Conviction Score

****
```python
stance_score = abs(current_stance) × conviction × 30
```

****
- `current_stance`-1.0  1.0
  - 
  - 
- `conviction`0.0  1.0
  - 
  - 
- 30 

****
- 
- 
- ""

****
| Agent |  |  |  |
|-------|------|------|------|
| A | 0.8 | 0.7 | 16.8 |
| B | -0.6 | 0.6 | 10.8 |
| C | 0.2 | 0.5 | 3.0 |

### 2. Persuasion Score

****
```python
persuasion_score = 0
for other_agent in all_agents:
    if other_agent.has_surrendered:
        persuasion_score += 20  # 
    
    # 
    avg_persuasion = mean(other_agent.persuasion_history)
    persuasion_score += avg_persuasion × 10
```

****

#### a) 20/
- 
-  20 
- ""

#### b) 0-10/
- 
- 
- 

****
- 0.0-0.2
- 0.2-0.4
- 0.4-0.6
- 0.6-0.8
- 0.8-1.0

### 3. Resistance Score

****
```python
avg_attack = mean(attack_history)
resistance_score = (1 - avg_attack) × conviction × 20
```

****
- 
- 
- 

****
|  |  |  |
|-------------|---------|----------|
| < 0.3 | > 0.7 |  |
| 0.3-0.5 | 0.5-0.7 |  |
| 0.5-0.7 | 0.3-0.5 |  |
| > 0.7 | < 0.3 |  |

### 4. Surrender Penalty

****-50 

****
- 
- 
-  Agent 

## 

### 1. Persuasion Effect

 Agent  > 0.6

```python
# 
persuasion_effect = persuasion_score × (1.0 - conviction)
new_stance = current_stance × (1.0 - persuasion_effect × 0.3)

# 
new_conviction = conviction × 0.85
```

****
- 0
- 
- ""

### 2. Attack Effect

 Agent  > 0.3

```python
# 
attack_resistance = conviction × 0.8
attack_effect = max(0, attack_score - attack_resistance)

# 
new_stance = current_stance × (1.0 + attack_effect × 0.2)

# 
new_conviction = min(1.0, conviction × 1.1)
```

****
- ""
- 
- 
- ""

### 3. 

```python
#  10 
if len(history) > 10:
    history.pop(0)
```

- 
- 
- 

## 

### 

####  1 + 
```python
if recent_persuasion > 0.6 and conviction < 0.4:
    surrender = True
```

****
- 
- 
- 

####  2
```python
if abs(current_stance) < 0.2 and conviction < 0.5:
    surrender = True
```

****
- 
- 
- 

####  3
```python
consecutive_high = all(score > 0.5 for score in persuasion_history[-3:])
if consecutive_high:
    surrender = True
```

****
-  3 
- 
- 

### 

1. ****
   -  50 
   - 
   - ""

2. ****
   -  20 
   - 
   - 

## 

### 1. 

```python
total_score = (
    stance_score +           # 
    persuasion_score +       # 
    resistance_score -       # 
    surrender_penalty        # 
)
```

### 2. 

#### Overwhelming Victory
```python
if len(surrendered_agents) > 0:
    verdict = f" {winner}  {surrendered} "
```

****
- 
- 
- 

#### Clear Advantage
```python
if score_difference > 30:
    verdict = f" {winner} "
```

****
-  30 
- 
- 

#### Narrow Victory
```python
else:
    verdict = f" {winner} "
```

****
-  30 
- 
- 

### 3. 



```json
{
    "total_rounds": 5,
    "winner": "Agent_A",
    "scores": {
        "Agent_A": 67.5,
        "Agent_B": 35.2,
        "Agent_C": -12.3
    },
    "surrendered_agents": ["Agent_C"],
    "final_states": {
        "Agent_A": {
            "stance": 0.75,
            "conviction": 0.65,
            "has_surrendered": false,
            "final_position": ""
        }
    },
    "verdict": ""
}
```

## 

### 1. 

****
- < 0.5
- 
- 

****
- 
-  aggressive 
-  RAG 

****
- 
- 

### 2. 

****
- > 0.7
- 
- 

****
- 
-  defensive 
- 

****
- 
- 

### 3. 

****
- 
- 
- 

****
-  GNN 
- 
- 

### 4. 

****
- 
- 
- 

****
-  empathetic 
- 
- 

## 

###  1

****
- Agent_A +0.8 0.7
- Agent_B -0.6 0.6  
- Agent_C 0.0 0.5

****
- A  analytical 
- B  aggressive  A 
- C  empathetic 

****
- A +0.85
- B 0.62
- C  A+0.15

****
- A  defensive
- B  aggressive
- C  analytical

****
- B 
- C 
- A  C 

****
- Agent_A45.6 
- Agent_B28.3 
- Agent_C52.1 

###  2

****""

****
1. Agent_A  analytical 
2.  Agent_B 
3.  empathetic 
4. 
5. 

****
- Agent_B  4 
- Agent_A 
- A (72.5) vs B (-25.0)

## 

### 1. 

```python
@dataclass
class AgentState:
    agent_id: str
    current_stance: float          # -1.0  1.0
    conviction: float              # 0.0  1.0
    social_context: List[float]    # 128 
    persuasion_history: List[float]
    attack_history: List[float]
    has_surrendered: bool = False
```

### 2. 

```python
def calculate_agent_score(agent_id: str, state: AgentState, 
                         all_states: Dict[str, AgentState]) -> float:
    score = 0
    
    # 
    stance_score = abs(state.current_stance) * state.conviction * 30
    score += stance_score
    
    # 
    persuasion_score = 0
    for other_id, other_state in all_states.items():
        if other_id != agent_id:
            if other_state.has_surrendered:
                persuasion_score += 20
            avg_persuasion = sum(other_state.persuasion_history) / len(other_state.persuasion_history)
            persuasion_score += avg_persuasion * 10
    score += persuasion_score
    
    # 
    if len(state.attack_history) > 0:
        avg_attack = sum(state.attack_history) / len(state.attack_history)
        resistance_score = (1 - avg_attack) * state.conviction * 20
        score += resistance_score
    
    # 
    if state.has_surrendered:
        score -= 50
    
    return score
```

### 3. 

```python
def evaluate_response_effects(response: str, target_agents: List[str]) -> Dict:
    # 
    persuasion_indicators = ['however', 'consider', '', '', '']
    attack_indicators = ['wrong', 'flawed', '', '', '']
    evidence_indicators = ['research', 'data', '', '', '']
    
    # 
    response_lower = response.lower()
    persuasion_score = min(1.0, sum(ind in response_lower for ind in persuasion_indicators) * 0.3)
    attack_score = min(1.0, sum(ind in response_lower for ind in attack_indicators) * 0.4)
    evidence_score = min(1.0, sum(ind in response_lower for ind in evidence_indicators) * 0.35)
    
    return {
        'persuasion_score': persuasion_score,
        'attack_score': attack_score,
        'evidence_score': evidence_score
    }
```

### 4. 

 `configs/debate.yaml` 

```yaml
victory_conditions:
  surrender_threshold: 0.4      # 
  stance_neutral_threshold: 0.2 # 
  consecutive_persuasion: 3     # 

persuasion_factors:
  base_persuasion: 0.3         # 
  strategy_bonus: 0.2          # 
  evidence_bonus: 0.3          # 
  social_influence: 0.2        # 
```

## 

Social Debate AI 

1. ****
2. ****
3. ****
4. ****


- 
- 
- 
- 

AI  