"""
Base agent for debate system
"""

from typing import Dict, Any

class BaseAgent:
    """Base Agent Class"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize agent
        
        Args:
            name: Agent name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.history = []

    def select_action(self, state: Dict[str, Any]) -> str:
        """
        Select action (must be implemented by subclass)
        
        Args:
            state: Current state
            
        Returns:
            Action string
            
        Raises:
            NotImplementedError: Subclass must implement this method
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_history(self, message: str):
        """
        Update conversation history
        
        Args:
            message: New message
        """
        self.history.append(message)