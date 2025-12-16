"""
Orchestrator Module
Provides both legacy parallel orchestrator and new LangGraph orchestrator
"""

from .parallel_orchestrator import ParallelOrchestrator, create_parallel_orchestrator
from .langgraph_orchestrator import LangGraphDebateOrchestrator, create_langgraph_orchestrator
from .debate_state import DebateState, AgentState, create_initial_state
from .debate_tools import DEBATE_TOOLS

__all__ = [
    # Legacy
    'ParallelOrchestrator',
    'create_parallel_orchestrator',
    
    # LangGraph
    'LangGraphDebateOrchestrator',
    'create_langgraph_orchestrator',
    
    # State
    'DebateState',
    'AgentState',
    'create_initial_state',
    
    # Tools
    'DEBATE_TOOLS'
]



