"""
Unit tests for debate tools (RAG, GNN, RL)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from orchestrator.debate_tools import (
    rl_select_strategy,
    gnn_analyze_social,
    rag_retrieve_evidence,
    evaluate_response_effects,
    DEBATE_TOOLS
)


class TestRLSelectStrategy:
    """Tests for RL strategy selection tool"""
    
    def test_returns_valid_strategy(self):
        """Test that tool returns a valid strategy"""
        result = rl_select_strategy.invoke({
            "context": "This is a debate about AI regulation",
            "social_context": None
        })
        
        assert 'strategy' in result
        assert result['strategy'] in ['aggressive', 'defensive', 'analytical', 'empathetic']
    
    def test_returns_quality_score(self):
        """Test that tool returns quality score"""
        result = rl_select_strategy.invoke({
            "context": "Testing quality prediction",
            "social_context": None
        })
        
        assert 'quality_score' in result
        assert 0 <= result['quality_score'] <= 1
    
    def test_with_social_context(self):
        """Test with social context vector"""
        social_context = [0.5] * 128
        result = rl_select_strategy.invoke({
            "context": "Test context",
            "social_context": social_context
        })
        
        assert 'strategy' in result


class TestGNNAnalyzeSocial:
    """Tests for GNN social analysis tool"""
    
    def test_returns_influence_score(self):
        """Test that tool returns influence score"""
        result = gnn_analyze_social.invoke({
            "agent_id": "Agent_A",
            "current_stance": 0.8,
            "conviction": 0.7,
            "persuasion_history": []
        })
        
        assert 'influence_score' in result
        assert 0 <= result['influence_score'] <= 1
    
    def test_returns_social_vector(self):
        """Test that tool returns social vector"""
        result = gnn_analyze_social.invoke({
            "agent_id": "Agent_B",
            "current_stance": -0.6,
            "conviction": 0.7,
            "persuasion_history": [0.3, 0.4]
        })
        
        assert 'social_vector' in result
        assert len(result['social_vector']) == 128
    
    def test_returns_persuasion_prediction(self):
        """Test that tool returns persuasion prediction"""
        result = gnn_analyze_social.invoke({
            "agent_id": "Agent_C",
            "current_stance": 0.0,
            "conviction": 0.5,
            "persuasion_history": [0.5, 0.5, 0.5]
        })
        
        assert 'persuasion_prediction' in result
        assert 'delta_probability' in result['persuasion_prediction']


class TestRAGRetrieveEvidence:
    """Tests for RAG evidence retrieval tool"""
    
    def test_returns_evidence_pool(self):
        """Test that tool returns evidence pool"""
        result = rag_retrieve_evidence.invoke({
            "query": "AI regulation benefits",
            "topic": "AI regulation",
            "top_k": 5
        })
        
        assert 'evidence_pool' in result
        assert 'total_evidence' in result
    
    def test_returns_best_evidence(self):
        """Test that tool returns best evidence"""
        result = rag_retrieve_evidence.invoke({
            "query": "Economic impact of technology",
            "topic": "Technology",
            "top_k": 3
        })
        
        assert 'best_evidence' in result
        assert isinstance(result['best_evidence'], str)
    
    def test_returns_evidence_types(self):
        """Test that tool returns evidence types"""
        result = rag_retrieve_evidence.invoke({
            "query": "Climate change debate",
            "topic": "Environment",
            "top_k": 8
        })
        
        assert 'evidence_types' in result
        assert isinstance(result['evidence_types'], dict)


class TestEvaluateResponseEffects:
    """Tests for response evaluation tool"""
    
    def test_returns_persuasion_score(self):
        """Test that tool returns persuasion score"""
        result = evaluate_response_effects.invoke({
            "response": "I understand your perspective, however we should consider the evidence.",
            "target_agents": ["Agent_B"]
        })
        
        assert 'persuasion_score' in result
        assert 0 <= result['persuasion_score'] <= 1
    
    def test_returns_attack_score(self):
        """Test that tool returns attack score"""
        result = evaluate_response_effects.invoke({
            "response": "Your argument is flawed and wrong in several ways.",
            "target_agents": ["Agent_A"]
        })
        
        assert 'attack_score' in result
        assert 0 <= result['attack_score'] <= 1
    
    def test_higher_persuasion_for_persuasive_text(self):
        """Test that persuasive text gets higher persuasion score"""
        persuasive = evaluate_response_effects.invoke({
            "response": "I understand your point, and I agree that we should consider your perspective.",
            "target_agents": ["Agent_B"]
        })
        
        aggressive = evaluate_response_effects.invoke({
            "response": "You are completely wrong about everything.",
            "target_agents": ["Agent_B"]
        })
        
        assert persuasive['persuasion_score'] >= aggressive['persuasion_score']
    
    def test_returns_word_count(self):
        """Test that tool returns word count"""
        result = evaluate_response_effects.invoke({
            "response": "This is a test response with some words.",
            "target_agents": ["Agent_A"]
        })
        
        assert 'word_count' in result
        assert result['word_count'] == 8


class TestDebateTools:
    """Tests for DEBATE_TOOLS list"""
    
    def test_all_tools_present(self):
        """Test that all expected tools are in DEBATE_TOOLS"""
        assert len(DEBATE_TOOLS) == 4
        
        tool_names = [tool.name for tool in DEBATE_TOOLS]
        assert 'rl_select_strategy' in tool_names
        assert 'gnn_analyze_social' in tool_names
        assert 'rag_retrieve_evidence' in tool_names
        assert 'evaluate_response_effects' in tool_names

