"""
Integration tests for LangGraph orchestrator
"""

import pytest
import sys
import os
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from orchestrator.debate_state import create_initial_state
from orchestrator.debate_tools import DEBATE_TOOLS


class TestLangGraphOrchestratorImport:
    """Test LangGraph orchestrator can be imported"""
    
    def test_import_orchestrator(self):
        """Test orchestrator imports successfully"""
        from orchestrator.langgraph_orchestrator import LangGraphDebateOrchestrator
        assert LangGraphDebateOrchestrator is not None
    
    def test_import_factory_function(self):
        """Test factory function imports"""
        from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator
        assert create_langgraph_orchestrator is not None


class TestLangGraphOrchestratorStructure:
    """Test LangGraph orchestrator structure"""
    
    def test_has_required_methods(self):
        """Test orchestrator has all required methods"""
        from orchestrator.langgraph_orchestrator import LangGraphDebateOrchestrator
        
        assert hasattr(LangGraphDebateOrchestrator, 'run_debate')
        assert hasattr(LangGraphDebateOrchestrator, 'run_single_round')
        assert hasattr(LangGraphDebateOrchestrator, 'get_graph_visualization')
    
    def test_graph_visualization(self):
        """Test graph visualization method"""
        from orchestrator.langgraph_orchestrator import LangGraphDebateOrchestrator
        
        # Create instance without full initialization
        orchestrator = object.__new__(LangGraphDebateOrchestrator)
        
        # The method should exist
        assert callable(getattr(LangGraphDebateOrchestrator, 'get_graph_visualization', None))


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)
class TestLangGraphOrchestratorExecution:
    """Integration tests that require OpenAI API key"""
    
    def test_orchestrator_creation(self):
        """Test orchestrator can be created with API key"""
        from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator
        
        orchestrator = create_langgraph_orchestrator(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        assert orchestrator is not None
        assert orchestrator.llm is not None
    
    def test_debate_execution(self, sample_agent_configs, sample_topic):
        """Test full debate execution"""
        from orchestrator.langgraph_orchestrator import create_langgraph_orchestrator
        
        orchestrator = create_langgraph_orchestrator()
        
        results = orchestrator.run_debate(
            topic=sample_topic,
            agent_configs=sample_agent_configs,
            max_rounds=1  # Just one round for testing
        )
        
        assert 'topic' in results
        assert 'history' in results
        assert 'summary' in results


class TestToolsIntegration:
    """Test tools work together"""
    
    def test_all_tools_callable(self):
        """Test all tools are callable"""
        for tool in DEBATE_TOOLS:
            assert callable(tool.invoke)
    
    def test_tools_have_descriptions(self):
        """Test all tools have descriptions"""
        for tool in DEBATE_TOOLS:
            assert tool.description is not None
            assert len(tool.description) > 0


class TestStateAndToolsIntegration:
    """Test state and tools work together"""
    
    def test_create_state_and_analyze(self, sample_agent_configs, sample_topic):
        """Test creating state and running analysis"""
        from orchestrator.debate_tools import gnn_analyze_social
        
        state = create_initial_state(
            topic=sample_topic,
            agent_configs=sample_agent_configs,
            max_rounds=3
        )
        
        # Analyze first agent
        agent_id = state['agent_order'][0]
        agent_state = state['agent_states'][agent_id]
        
        result = gnn_analyze_social.invoke({
            "agent_id": agent_id,
            "current_stance": agent_state.get('current_stance', 0),
            "conviction": agent_state.get('conviction', 0.7),
            "persuasion_history": agent_state.get('persuasion_history', [])
        })
        
        assert 'influence_score' in result
        assert 'social_vector' in result

