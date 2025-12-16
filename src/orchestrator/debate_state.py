"""
Debate State Schema for LangGraph
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator


@dataclass
class AgentState:
    """Individual Agent State"""
    agent_id: str
    current_stance: float  # -1.0 to 1.0, stance intensity
    conviction: float      # 0.0 to 1.0, belief firmness
    social_context: List[float] = field(default_factory=lambda: [0.0] * 128)
    persuasion_history: List[float] = field(default_factory=list)
    attack_history: List[float] = field(default_factory=list)
    has_surrendered: bool = False
    
    def update_stance(self, persuasion_score: float, attack_score: float):
        """Update stance and conviction based on debate effects"""
        # Calculate persuasion effect
        persuasion_effect = persuasion_score * (1.0 - self.conviction)
        
        # Calculate attack resistance
        attack_resistance = self.conviction * 0.8
        attack_effect = max(0, attack_score - attack_resistance)
        
        # Update stance
        if persuasion_score > 0.5:
            self.current_stance *= (1.0 - persuasion_effect * 0.2)
            self.conviction *= 0.9
        
        if attack_effect > 0.3:
            self.current_stance *= (1.0 + attack_effect * 0.2)
            self.conviction = min(1.0, self.conviction * 1.1)
        
        # Record history
        self.persuasion_history.append(persuasion_score)
        self.attack_history.append(attack_score)
        
        # Keep history length
        if len(self.persuasion_history) > 10:
            self.persuasion_history.pop(0)
        if len(self.attack_history) > 10:
            self.attack_history.pop(0)
        
        # Check surrender conditions
        self._check_surrender()
    
    def _check_surrender(self):
        """Check if agent should surrender"""
        if len(self.persuasion_history) >= 4:
            recent_persuasion = sum(self.persuasion_history[-4:]) / 4
            
            if recent_persuasion > 0.65 and self.conviction < 0.25:
                self.has_surrendered = True
            elif abs(self.current_stance) < 0.1 and self.conviction < 0.3:
                self.has_surrendered = True
            elif len(self.persuasion_history) >= 5:
                consecutive_high = all(score > 0.6 for score in self.persuasion_history[-5:])
                if consecutive_high and self.conviction < 0.4:
                    self.has_surrendered = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'current_stance': self.current_stance,
            'conviction': self.conviction,
            'has_surrendered': self.has_surrendered
        }


class DebateState(TypedDict):
    """Main Debate State for LangGraph"""
    # Core debate info
    topic: str
    current_round: int
    max_rounds: int
    
    # Agent management
    agent_order: List[str]
    current_speaker_index: int
    agent_states: Dict[str, Any]  # agent_id -> AgentState dict
    
    # History tracking
    history: Annotated[List[Dict], operator.add]  # All responses
    round_history: List[Dict]  # Current round responses
    
    # Analysis results (from parallel tools)
    rl_result: Optional[Dict]
    gnn_result: Optional[Dict]
    rag_result: Optional[Dict]
    fused_result: Optional[Dict]
    
    # Current turn data
    current_response: Optional[str]
    response_effects: Optional[Dict]
    
    # Control flags
    debate_ended: bool
    end_reason: Optional[str]
    winner: Optional[str]


def create_initial_state(
    topic: str,
    agent_configs: List[Dict],
    max_rounds: int = 5
) -> DebateState:
    """Create initial debate state"""
    
    # Initialize agent states
    agent_states = {}
    agent_order = []
    
    for config in agent_configs:
        agent_id = config['id']
        agent_order.append(agent_id)
        agent_states[agent_id] = {
            'agent_id': agent_id,
            'current_stance': config.get('initial_stance', 0.0),
            'conviction': config.get('initial_conviction', 0.7),
            'social_context': config.get('social_context', [0.0] * 128),
            'persuasion_history': [],
            'attack_history': [],
            'has_surrendered': False
        }
    
    return DebateState(
        topic=topic,
        current_round=1,
        max_rounds=max_rounds,
        agent_order=agent_order,
        current_speaker_index=0,
        agent_states=agent_states,
        history=[],
        round_history=[],
        rl_result=None,
        gnn_result=None,
        rag_result=None,
        fused_result=None,
        current_response=None,
        response_effects=None,
        debate_ended=False,
        end_reason=None,
        winner=None
    )


