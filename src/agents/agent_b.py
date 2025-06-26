"""
Opposition Questioning Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent
from ..orchestrator.parallel_orchestrator import ParallelOrchestrator

class AgentB(BaseAgent):
    """Opposition Questioning Agent"""
    
    def __init__(self, name: str = "Agent_B", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.stance = -0.6  # Opposition stance
        self.strategy_preference = "analytical"
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