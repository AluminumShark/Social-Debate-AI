"""
Unit tests for debate state management
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.debate_state import (
    AgentState,
    create_initial_state
)


class TestAgentState:
    """Tests for AgentState class"""
    
    def test_agent_state_creation(self):
        """Test basic AgentState creation"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.5,
            conviction=0.7
        )
        
        assert state.agent_id == "test_agent"
        assert state.current_stance == 0.5
        assert state.conviction == 0.7
        assert state.has_surrendered is False
        assert len(state.persuasion_history) == 0
    
    def test_update_stance_with_persuasion(self):
        """Test stance update when persuaded"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.8,
            conviction=0.7
        )
        
        # Apply high persuasion
        state.update_stance(persuasion_score=0.6, attack_score=0.1)
        
        assert len(state.persuasion_history) == 1
        assert state.persuasion_history[0] == 0.6
        # Conviction should decrease
        assert state.conviction < 0.7
    
    def test_update_stance_with_attack(self):
        """Test stance update when attacked"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.5,
            conviction=0.5
        )
        
        # Apply strong attack
        state.update_stance(persuasion_score=0.1, attack_score=0.8)
        
        assert len(state.attack_history) == 1
        # Attack should make stance more extreme
    
    def test_surrender_conditions(self):
        """Test surrender detection"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.2,
            conviction=0.2
        )
        
        # Apply multiple high persuasion rounds
        for _ in range(5):
            state.update_stance(persuasion_score=0.7, attack_score=0.1)
        
        # Should eventually surrender
        assert state.has_surrendered is True
    
    def test_history_limit(self):
        """Test that history is limited to 10 items"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.5,
            conviction=0.7
        )
        
        # Apply 15 updates
        for _ in range(15):
            state.update_stance(persuasion_score=0.3, attack_score=0.2)
        
        assert len(state.persuasion_history) <= 10
        assert len(state.attack_history) <= 10
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        state = AgentState(
            agent_id="test_agent",
            current_stance=0.5,
            conviction=0.7
        )
        
        d = state.to_dict()
        
        assert d['agent_id'] == "test_agent"
        assert d['current_stance'] == 0.5
        assert d['conviction'] == 0.7
        assert d['has_surrendered'] is False


class TestCreateInitialState:
    """Tests for create_initial_state function"""
    
    def test_basic_state_creation(self, sample_agent_configs, sample_topic):
        """Test basic initial state creation"""
        state = create_initial_state(
            topic=sample_topic,
            agent_configs=sample_agent_configs,
            max_rounds=5
        )
        
        assert state['topic'] == sample_topic
        assert state['max_rounds'] == 5
        assert state['current_round'] == 1
        assert len(state['agent_order']) == 3
    
    def test_agent_states_initialized(self, sample_agent_configs):
        """Test that agent states are properly initialized"""
        state = create_initial_state(
            topic="Test topic",
            agent_configs=sample_agent_configs,
            max_rounds=3
        )
        
        assert 'Agent_A' in state['agent_states']
        assert 'Agent_B' in state['agent_states']
        assert 'Agent_C' in state['agent_states']
        
        agent_a = state['agent_states']['Agent_A']
        assert agent_a.get('current_stance', agent_a.get('initial_stance')) == 0.8
    
    def test_empty_history(self, sample_agent_configs):
        """Test that history starts empty"""
        state = create_initial_state(
            topic="Test topic",
            agent_configs=sample_agent_configs,
            max_rounds=3
        )
        
        assert state['history'] == []
        assert state['debate_ended'] is False

