"""
LangGraph Tools for Debate System
Wraps RAG, GNN, RL modules as callable tools
"""

from typing import Dict, List, Any, Optional
from langchain_core.tools import tool
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Lazy loading for heavy modules
_rl_module = None
_gnn_module = None
_rag_retriever = None


def _get_rl_module():
    """Lazy load RL module"""
    global _rl_module
    if _rl_module is None:
        try:
            from rl import policy_network
            _rl_module = policy_network
            print("[Tools] RL module loaded")
        except ImportError as e:
            print(f"[Tools] RL module not available: {e}")
            _rl_module = "unavailable"
    return _rl_module if _rl_module != "unavailable" else None


def _get_gnn_module():
    """Lazy load GNN module"""
    global _gnn_module
    if _gnn_module is None:
        try:
            from gnn import social_encoder
            _gnn_module = social_encoder
            print("[Tools] GNN module loaded")
        except ImportError as e:
            print(f"[Tools] GNN module not available: {e}")
            _gnn_module = "unavailable"
    return _gnn_module if _gnn_module != "unavailable" else None


def _get_rag_retriever():
    """Lazy load RAG retriever.

    Prefer the FAISS vector retriever (real semantic search via Ollama
    embeddings). Fall back to the keyword SimpleRetriever if no vector index
    has been built yet.
    """
    global _rag_retriever
    if _rag_retriever is None:
        # 1) Try the FAISS vector retriever
        try:
            from rag.vector_retriever import VectorRetriever
            _rag_retriever = VectorRetriever()
            print("[Tools] RAG: FAISS vector retriever loaded")
            return _rag_retriever
        except Exception as e:
            print(f"[Tools] RAG: vector index unavailable ({e}); trying SimpleRetriever")
        # 2) Fall back to keyword overlap retriever
        try:
            from rag.simple_retriever import SimpleRetriever
            _rag_retriever = SimpleRetriever()
            print("[Tools] RAG: SimpleRetriever (keyword) loaded")
        except ImportError as e:
            print(f"[Tools] RAG retriever not available: {e}")
            _rag_retriever = "unavailable"
    return _rag_retriever if _rag_retriever != "unavailable" else None


@tool
def rl_select_strategy(
    context: str,
    social_context: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Select optimal debate strategy using RL policy network.
    
    Args:
        context: Current debate context including topic and recent history
        social_context: Optional social context vector (128-dim)
    
    Returns:
        Dict with strategy, quality_score, and confidence
    """
    rl = _get_rl_module()
    
    if rl is None:
        # Fallback heuristic
        return {
            'strategy': 'analytical',
            'quality_score': 0.5,
            'confidence': 0.3,
            'source': 'fallback'
        }
    
    try:
        strategy = rl.select_strategy(context, "", social_context)
        
        # Get quality prediction using module-level function
        quality_score = rl.predict_quality(context)
        
        return {
            'strategy': strategy,
            'quality_score': float(quality_score),
            'confidence': 0.8,
            'source': 'rl_model'
        }
    except Exception as e:
        print(f"[Tools] RL analysis error: {e}")
        return {
            'strategy': 'analytical',
            'quality_score': 0.5,
            'confidence': 0.3,
            'source': 'fallback',
            'error': str(e)
        }


def _embed_context(context_text: Optional[str]) -> np.ndarray:
    """Embed the debate context to real 768-d features for the GNN.

    Returns a deterministic zero vector when no text/embeddings are available
    (never random noise — that would make GNN output meaningless).
    """
    if context_text:
        try:
            from llm import embed, resolve_config
            vec = np.asarray(embed([context_text[:2000]], resolve_config())[0], dtype="float32")
            if vec.shape[0] >= 768:
                return vec[:768]
            out = np.zeros(768, dtype="float32")
            out[: vec.shape[0]] = vec
            return out
        except Exception as e:  # noqa: BLE001
            print(f"[Tools] GNN context embed failed ({e}); using zero features")
    return np.zeros(768, dtype="float32")


@tool
def gnn_analyze_social(
    agent_id: str,
    current_stance: float = 0.0,
    conviction: float = 0.7,
    persuasion_history: Optional[List[float]] = None,
    context_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze social influence and persuasion dynamics using GNN.
    
    Args:
        agent_id: The agent's identifier
        current_stance: Current stance value (-1.0 to 1.0)
        conviction: Current conviction level (0.0 to 1.0)
        persuasion_history: Recent persuasion scores
    
    Returns:
        Dict with social_vector, influence_score, stance_trend, and persuasion prediction
    """
    gnn = _get_gnn_module()
    persuasion_history = persuasion_history or []
    
    if gnn is None:
        return {
            'social_vector': [0.0] * 128,
            'influence_score': 0.5,
            'stance_trend': 0.0,
            'current_stance': current_stance,
            'conviction': conviction,
            'persuasion_prediction': {
                'delta_probability': 0.5,
                'best_strategy': 'analytical',
                'strategy_scores': {}
            },
            'source': 'fallback'
        }
    
    try:
        # Get social vector
        social_vector = gnn.social_vec(agent_id)
        
        # Get influence score
        influence_score = gnn.get_social_influence_score(agent_id)
        
        # Predict persuasion from REAL context embeddings (not random noise)
        text_features = _embed_context(context_text)
        persuasion_pred = gnn.predict_persuasion(text_features, agent_id)
        
        # Calculate stance trend
        stance_trend = 0.0
        if len(persuasion_history) >= 2:
            recent_persuasion = sum(persuasion_history[-3:]) / min(3, len(persuasion_history))
            stance_trend = recent_persuasion - 0.5
        
        return {
            'social_vector': social_vector,
            'influence_score': float(influence_score),
            'stance_trend': float(stance_trend),
            'current_stance': current_stance,
            'conviction': conviction,
            'persuasion_prediction': persuasion_pred,
            'source': 'gnn_model'
        }
    except Exception as e:
        print(f"[Tools] GNN analysis error: {e}")
        return {
            'social_vector': [0.0] * 128,
            'influence_score': 0.5,
            'stance_trend': 0.0,
            'current_stance': current_stance,
            'conviction': conviction,
            'persuasion_prediction': {
                'delta_probability': 0.5,
                'best_strategy': 'analytical',
                'strategy_scores': {}
            },
            'source': 'fallback',
            'error': str(e)
        }


@tool
def rag_retrieve_evidence(
    query: str,
    topic: str,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Retrieve relevant evidence for debate arguments using RAG.
    
    Args:
        query: Search query (usually debate context)
        topic: Debate topic for filtering
        top_k: Number of results to retrieve
    
    Returns:
        Dict with evidence_pool, best_evidence, and metadata
    """
    retriever = _get_rag_retriever()
    
    if retriever is None:
        return {
            'evidence_pool': [],
            'best_evidence': "No evidence available",
            'evidence_types': {},
            'total_evidence': 0,
            'source': 'fallback'
        }
    
    try:
        # Retrieve evidence
        results = retriever.retrieve(query=query, top_k=top_k)
        
        # Convert to evidence pool format
        # Handle both dict-style (EnhancedRetriever) and object-style (SimpleRetriever) results
        evidence_pool = []
        for result in results:
            if isinstance(result, dict):
                # Dict-style result from EnhancedRetriever
                evidence_pool.append({
                    'content': result.get('content', ''),
                    'similarity_score': result.get('similarity_score', 0.0),
                    'metadata': result.get('metadata', {})
                })
            else:
                # Object-style result from SimpleRetriever (RetrievalResult dataclass)
                evidence_pool.append({
                    'content': result.text,
                    'similarity_score': result.score,
                    'metadata': result.metadata or {}
                })
        
        # Select best evidence
        best_evidence = evidence_pool[0]['content'] if evidence_pool else "No evidence available"
        
        # Analyze evidence types
        evidence_types = {}
        for item in evidence_pool:
            ev_type = item.get('metadata', {}).get('type', 'unknown')
            evidence_types[ev_type] = evidence_types.get(ev_type, 0) + 1
        
        return {
            'evidence_pool': evidence_pool,
            'best_evidence': best_evidence,
            'evidence_types': evidence_types,
            'total_evidence': len(evidence_pool),
            'source': 'rag_retriever'
        }
    except Exception as e:
        print(f"[Tools] RAG retrieval error: {e}")
        return {
            'evidence_pool': [],
            'best_evidence': "No evidence available",
            'evidence_types': {},
            'total_evidence': 0,
            'source': 'fallback',
            'error': str(e)
        }


def _keyword_effects(response: str, target_agents: List[str]) -> Dict[str, Any]:
    """Fast keyword-based effect estimate (fallback when LLM judge is off)."""
    persuasion_indicators = [
        'however', 'consider', 'understand', 'perspective', 'common',
        'but', 'agree', 'acknowledge', 'point', 'valid'
    ]
    attack_indicators = [
        'wrong', 'flawed', 'mistake', 'ignore', 'fail',
        'error', 'fallacy', 'overlook', 'absurd', 'unreasonable'
    ]
    evidence_indicators = [
        '[CITE]', 'study', 'research', 'data', 'evidence',
        'statistics', 'report', 'survey', 'according'
    ]
    
    response_lower = response.lower()
    
    # Calculate scores
    persuasion_count = sum(1 for ind in persuasion_indicators if ind in response_lower)
    attack_count = sum(1 for ind in attack_indicators if ind in response_lower)
    evidence_count = sum(1 for ind in evidence_indicators if ind in response_lower)
    
    persuasion_score = min(0.7, persuasion_count * 0.1)
    attack_score = min(0.6, attack_count * 0.15)
    evidence_score = min(0.5, evidence_count * 0.15)
    
    # Length score
    word_count = len(response.split())
    length_score = min(1.0, word_count / 80)
    
    return {
        'persuasion_score': persuasion_score,
        'attack_score': attack_score,
        'evidence_score': evidence_score,
        'length_score': length_score,
        'word_count': word_count,
        'target_agents': target_agents,
        'source': 'keyword',
    }


_JUDGE_PROMPT = """You are an impartial debate judge. Rate the following debate \
turn on three axes, each from 0.0 to 1.0:
- persuasion: how likely it is to move a reasonable opponent toward the speaker
- attack: how strongly it challenges/refutes the opponents' arguments
- evidence: how well it uses facts, data, studies, or concrete examples

Respond with ONLY a compact JSON object, e.g.:
{"persuasion": 0.6, "attack": 0.3, "evidence": 0.5}

Debate turn:
\"\"\"%s\"\"\""""


@tool
def evaluate_response_effects(
    response: str,
    target_agents: List[str]
) -> Dict[str, Any]:
    """
    Evaluate the persuasiveness and attack strength of a debate response.

    Uses an LLM-as-a-judge by default (set USE_LLM_JUDGE=false to disable),
    falling back to a fast keyword heuristic on any error.

    Returns dict with persuasion_score, attack_score, evidence_score,
    length_score, word_count, target_agents.
    """
    import os
    import json as _json
    import re as _re

    word_count = len(response.split())
    length_score = min(1.0, word_count / 80)

    use_judge = os.environ.get("USE_LLM_JUDGE", "true").lower() in ("1", "true", "yes")
    if use_judge and response.strip():
        try:
            from llm import chat, resolve_config
            raw = chat(
                [{"role": "user", "content": _JUDGE_PROMPT % response[:2000]}],
                resolve_config(),
            )
            m = _re.search(r"\{.*\}", raw, _re.DOTALL)
            data = _json.loads(m.group(0)) if m else {}
            return {
                "persuasion_score": float(max(0.0, min(1.0, data.get("persuasion", 0.4)))),
                "attack_score": float(max(0.0, min(1.0, data.get("attack", 0.3)))),
                "evidence_score": float(max(0.0, min(1.0, data.get("evidence", 0.3)))),
                "length_score": length_score,
                "word_count": word_count,
                "target_agents": target_agents,
                "source": "llm_judge",
            }
        except Exception as e:  # noqa: BLE001
            print(f"[Tools] LLM judge failed ({e}); using keyword fallback")

    return _keyword_effects(response, target_agents)


# Export all tools
DEBATE_TOOLS = [
    rl_select_strategy,
    gnn_analyze_social,
    rag_retrieve_evidence,
    evaluate_response_effects
]


def get_tools_description() -> str:
    """Get description of all available tools"""
    descriptions = []
    for t in DEBATE_TOOLS:
        descriptions.append(f"- {t.name}: {t.description}")
    return "\n".join(descriptions)


