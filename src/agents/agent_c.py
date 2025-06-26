"""
Neutral Analytical Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..orchestrator.parallel_orchestrator import ParallelOrchestrator

class AgentC(BaseAgent):
    """Neutral Analytical Agent"""
    
    def __init__(self, name: str = "Agent_C", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.stance = 0.0  # Neutral stance
        self.strategy_preference = "empathetic"
        self.orchestrator = ParallelOrchestrator()
    
    def select_action(self, state: Dict[str, Any]) -> str:
        # Use parallel orchestrator for action selection
        context = state.get('context', '')
        topic = state.get('topic', '')
        history = state.get('history', [])
        
        return self.orchestrator.run_one_round(
            topic=topic,
            context=context,
            history=history,
            current_agent=self.name
        )