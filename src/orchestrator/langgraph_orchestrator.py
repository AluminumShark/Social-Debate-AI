"""
LangGraph-based Debate Orchestrator
Replaces the manual parallel_orchestrator with a declarative graph-based approach
"""

import time
from typing import Dict, List, Literal
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .debate_state import DebateState, create_initial_state
from .debate_tools import (
    rl_select_strategy,
    gnn_analyze_social, 
    rag_retrieve_evidence,
    evaluate_response_effects
)

# Strategy guidance templates
STRATEGY_GUIDANCE = {
    'aggressive': "Adopt critical analysis strategy: deeply analyze logical flaws in opponent's arguments, challenge core assumptions with powerful counterexamples and data.",
    'defensive': "Adopt robust argumentation strategy: consolidate core arguments, systematically respond to challenges, strengthen position with more evidence.",
    'analytical': "Adopt rational analysis strategy: use logical reasoning, empirical data and case studies to objectively evaluate pros and cons of various viewpoints.",
    'empathetic': "Adopt constructive dialogue strategy: understand opponent's reasonable concerns, find common ground, propose solutions considering all parties' interests."
}


class LangGraphDebateOrchestrator:
    """
    LangGraph-based Debate Orchestrator
    
    Uses a StateGraph to manage debate flow:
    1. Parallel analysis (RL + GNN + RAG)
    2. Fuse results
    3. Generate response
    4. Update states
    5. Check end conditions
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
        
        print("[LangGraph] Debate orchestrator initialized")
        print(f"[LangGraph] Model: {model_name}")
    
    def _build_graph(self) -> StateGraph:
        """Build the debate workflow graph"""
        
        # Create StateGraph with our state schema
        workflow = StateGraph(DebateState)
        
        # Add nodes
        workflow.add_node("parallel_analysis", self._parallel_analysis_node)
        workflow.add_node("fuse_results", self._fuse_results_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("update_states", self._update_states_node)
        workflow.add_node("advance_turn", self._advance_turn_node)
        
        # Define edges
        workflow.set_entry_point("parallel_analysis")
        workflow.add_edge("parallel_analysis", "fuse_results")
        workflow.add_edge("fuse_results", "generate_response")
        workflow.add_edge("generate_response", "update_states")
        workflow.add_conditional_edges(
            "update_states",
            self._should_continue,
            {
                "next_speaker": "advance_turn",
                "next_round": "advance_turn", 
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "advance_turn",
            self._check_round_complete,
            {
                "continue": "parallel_analysis",
                "end": END
            }
        )
        
        return workflow
    
    def _parallel_analysis_node(self, state: DebateState) -> Dict:
        """Execute RL, GNN, RAG analysis in parallel"""
        
        current_speaker = state["agent_order"][state["current_speaker_index"]]
        agent_state = state["agent_states"][current_speaker]
        topic = state["topic"]
        
        # Build context from history
        recent_history = state["history"][-6:] if state["history"] else []
        context = f"Topic: {topic}\n"
        for h in recent_history:
            context += f"{h['agent_id']}: {h['content'][:200]}...\n"
        
        print(f"[Analysis] Agent {current_speaker} analyzing...")
        
        # Run tools in parallel using ThreadPoolExecutor (synchronous, no asyncio needed)
        try:
            # Submit tasks to thread pool
            rl_future = self.executor.submit(
                rl_select_strategy.invoke,
                {"context": context, "social_context": agent_state.get('social_context')}
            )
            
            gnn_future = self.executor.submit(
                gnn_analyze_social.invoke,
                {
                    "agent_id": current_speaker,
                    "current_stance": agent_state.get('current_stance', 0.0),
                    "conviction": agent_state.get('conviction', 0.7),
                    "persuasion_history": agent_state.get('persuasion_history', [])
                }
            )
            
            rag_future = self.executor.submit(
                rag_retrieve_evidence.invoke,
                {"query": context, "topic": topic, "top_k": 8}
            )
            
            # Wait for results
            rl_result = rl_future.result(timeout=30)
            gnn_result = gnn_future.result(timeout=30)
            rag_result = rag_future.result(timeout=30)
            
            print(f"[Analysis] RL strategy: {rl_result.get('strategy')}")
            print(f"[Analysis] GNN influence: {gnn_result.get('influence_score', 0):.2f}")
            print(f"[Analysis] RAG evidence: {rag_result.get('total_evidence', 0)} items")
            
        except Exception as e:
            print(f"[Analysis] Error: {e}")
            rl_result = {'strategy': 'analytical', 'quality_score': 0.5, 'confidence': 0.3}
            gnn_result = {'influence_score': 0.5, 'stance_trend': 0.0, 'persuasion_prediction': {'best_strategy': 'analytical', 'delta_probability': 0.5}}
            rag_result = {'evidence_pool': [], 'best_evidence': 'No evidence available', 'total_evidence': 0}
        
        return {
            "rl_result": rl_result,
            "gnn_result": gnn_result,
            "rag_result": rag_result
        }
    
    def _fuse_results_node(self, state: DebateState) -> Dict:
        """Fuse analysis results from RL, GNN, RAG"""
        
        rl_result = state["rl_result"] or {}
        gnn_result = state["gnn_result"] or {}
        rag_result = state["rag_result"] or {}
        
        # Get strategies
        base_strategy = rl_result.get('strategy', 'analytical')
        gnn_strategy = gnn_result.get('persuasion_prediction', {}).get('best_strategy', 'analytical')
        influence_score = gnn_result.get('influence_score', 0.5)
        current_stance = gnn_result.get('current_stance', 0.0)
        delta_probability = gnn_result.get('persuasion_prediction', {}).get('delta_probability', 0.5)
        
        # Strategy fusion logic
        if delta_probability > 0.7:
            adjusted_strategy = gnn_strategy
            print(f"[Fuse] Strategy: {base_strategy} → {adjusted_strategy} (high delta prob)")
        elif influence_score > 0.6 and abs(current_stance) > 0.5:
            adjusted_strategy = 'aggressive' if base_strategy == 'analytical' else base_strategy
        elif influence_score < 0.4 and abs(current_stance) < 0.3:
            adjusted_strategy = 'defensive' if base_strategy == 'aggressive' else base_strategy
        else:
            adjusted_strategy = base_strategy
        
        # Evidence confidence
        evidence = rag_result.get('best_evidence', 'No evidence available')
        evidence_confidence = min(1.0, rag_result.get('total_evidence', 0) / 5.0)
        adjusted_confidence = evidence_confidence * (0.5 + 0.5 * delta_probability)
        
        fused_result = {
            'final_strategy': adjusted_strategy,
            'evidence': evidence,
            'evidence_confidence': adjusted_confidence,
            'social_influence': influence_score,
            'stance_strength': abs(current_stance),
            'delta_probability': delta_probability,
            'gnn_suggested_strategy': gnn_strategy
        }
        
        print(f"[Fuse] Final strategy: {adjusted_strategy}, confidence: {adjusted_confidence:.2f}")
        
        return {"fused_result": fused_result}
    
    def _generate_response_node(self, state: DebateState) -> Dict:
        """Generate debate response using LLM"""
        
        current_speaker = state["agent_order"][state["current_speaker_index"]]
        agent_state = state["agent_states"][current_speaker]
        topic = state["topic"]
        fused_result = state["fused_result"] or {}
        history = state["history"]
        
        strategy = fused_result.get('final_strategy', 'analytical')
        evidence = fused_result.get('evidence', '')
        
        # Build prompt
        is_first_round = len(history) == 0 and state["current_speaker_index"] == 0
        
        if is_first_round:
            system_prompt = f"""You are participating in a public issue debate about "{topic}".

Your role settings:
- Position tendency: {agent_state.get('current_stance', 0):.2f} (positive=support, negative=oppose)
- Conviction level: {agent_state.get('conviction', 0.7):.2f}
- Strategy: {strategy}

Requirements:
1. Clearly state your position (support or oppose)
2. Present 3-4 core arguments
3. Use facts and data to support your position
4. Word count: 200-250 words
5. Complete your full argument

Available evidence: {evidence[:500] if evidence else 'None'}"""
        else:
            history_text = "\n".join([
                f"{h['agent_id']}: {h['content'][:150]}..." 
                for h in history[-4:]
            ])
            
            system_prompt = f"""You are participating in a public issue debate about "{topic}".

Your role settings:
- Position tendency: {agent_state.get('current_stance', 0):.2f}
- Conviction level: {agent_state.get('conviction', 0.7):.2f}
- Strategy: {strategy}

Discussion so far:
{history_text}

Strategy guidance: {STRATEGY_GUIDANCE.get(strategy, '')}

Requirements:
1. Respond to opponent's core arguments
2. Provide new perspectives and evidence
3. Use logical transitions
4. Maintain objectivity, avoid personal attacks
5. Word count: 200-250 words

Available evidence: {evidence[:500] if evidence else 'None'}"""

        print(f"[Generate] Agent {current_speaker} generating response...")
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please express your viewpoint:")
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            # Check if truncated
            if not response_text.endswith(('.', '!', '?', '。', '！', '？')):
                response_text += ". In conclusion, I maintain my position based on the above analysis."
            
            print(f"[Generate] Response generated ({len(response_text.split())} words)")
            
        except Exception as e:
            print(f"[Generate] Error: {e}")
            response_text = f"I understand the various perspectives on {topic}. Based on my analysis, I believe we should consider multiple factors before reaching a conclusion."
        
        return {"current_response": response_text}
    
    def _update_states_node(self, state: DebateState) -> Dict:
        """Update agent states based on response effects"""
        
        current_speaker = state["agent_order"][state["current_speaker_index"]]
        response = state["current_response"]
        other_agents = [a for a in state["agent_order"] if a != current_speaker]
        
        # Evaluate response effects
        effects = evaluate_response_effects.invoke({
            "response": response,
            "target_agents": other_agents
        })
        
        print(f"[Update] Effects - Persuasion: {effects['persuasion_score']:.2f}, Attack: {effects['attack_score']:.2f}")
        
        # Create response record
        response_record = {
            'agent_id': current_speaker,
            'content': response,
            'effects': effects,
            'timestamp': time.time()
        }
        
        # Update other agents' states
        updated_states = dict(state["agent_states"])
        for target_id in other_agents:
            if target_id in updated_states:
                target_state = updated_states[target_id]
                
                # Update persuasion/attack history
                persuasion_history = list(target_state.get('persuasion_history', []))
                attack_history = list(target_state.get('attack_history', []))
                
                persuasion_history.append(effects['persuasion_score'])
                attack_history.append(effects['attack_score'])
                
                # Keep last 10
                if len(persuasion_history) > 10:
                    persuasion_history = persuasion_history[-10:]
                if len(attack_history) > 10:
                    attack_history = attack_history[-10:]
                
                # Update stance
                current_stance = target_state.get('current_stance', 0.0)
                conviction = target_state.get('conviction', 0.7)
                
                persuasion_effect = effects['persuasion_score'] * (1.0 - conviction)
                if effects['persuasion_score'] > 0.5:
                    current_stance *= (1.0 - persuasion_effect * 0.2)
                    conviction *= 0.9
                
                attack_resistance = conviction * 0.8
                attack_effect = max(0, effects['attack_score'] - attack_resistance)
                if attack_effect > 0.3:
                    current_stance *= (1.0 + attack_effect * 0.2)
                    conviction = min(1.0, conviction * 1.1)
                
                # Check surrender conditions (matching debate_state.py)
                has_surrendered = target_state.get('has_surrendered', False)
                if len(persuasion_history) >= 4:
                    recent_persuasion = sum(persuasion_history[-4:]) / 4
                    # Condition 1: High persuasion average with very low conviction
                    if recent_persuasion > 0.65 and conviction < 0.25:
                        has_surrendered = True
                    # Condition 2: Very weak stance with low conviction
                    elif abs(current_stance) < 0.1 and conviction < 0.3:
                        has_surrendered = True
                    # Condition 3: 5 consecutive high-persuasion rounds with low conviction
                    elif len(persuasion_history) >= 5:
                        consecutive_high = all(score > 0.6 for score in persuasion_history[-5:])
                        if consecutive_high and conviction < 0.4:
                            has_surrendered = True
                
                updated_states[target_id] = {
                    **target_state,
                    'current_stance': current_stance,
                    'conviction': conviction,
                    'persuasion_history': persuasion_history,
                    'attack_history': attack_history,
                    'has_surrendered': has_surrendered
                }
        
        # Add to round history
        round_history = list(state.get("round_history", []))
        round_history.append(response_record)
        
        return {
            "agent_states": updated_states,
            "response_effects": effects,
            "round_history": round_history,
            "history": [response_record]  # This will be appended via operator.add
        }
    
    def _advance_turn_node(self, state: DebateState) -> Dict:
        """Advance to next speaker or next round"""
        
        next_speaker_index = state["current_speaker_index"] + 1
        current_round = state["current_round"]
        
        # Check if round complete
        if next_speaker_index >= len(state["agent_order"]):
            # Move to next round
            return {
                "current_speaker_index": 0,
                "current_round": current_round + 1,
                "round_history": []  # Reset round history
            }
        else:
            return {
                "current_speaker_index": next_speaker_index
            }
    
    def _should_continue(self, state: DebateState) -> Literal["next_speaker", "next_round", "end"]:
        """Determine next action after updating states"""
        
        # Check for surrender
        for agent_id, agent_state in state["agent_states"].items():
            if agent_state.get('has_surrendered', False):
                print(f"[Control] {agent_id} has surrendered!")
                return "end"
        
        # Check max rounds
        next_speaker = state["current_speaker_index"] + 1
        if next_speaker >= len(state["agent_order"]):
            # Round complete, check if we should continue
            if state["current_round"] >= state["max_rounds"]:
                print(f"[Control] Max rounds ({state['max_rounds']}) reached")
                return "end"
            return "next_round"
        
        return "next_speaker"
    
    def _check_round_complete(self, state: DebateState) -> Literal["continue", "end"]:
        """Check if debate should continue after advancing turn
        
        This method is called after _advance_turn_node to determine if the
        debate should continue with the next speaker or end.
        
        Important: We only end when current_round EXCEEDS max_rounds, not when
        equal. This allows the final round (current_round == max_rounds) to
        complete with all agents speaking.
        """
        
        current_round = state["current_round"]
        max_rounds = state["max_rounds"]
        
        # Only end if we've exceeded max_rounds (safety check)
        # When current_round == max_rounds, we still allow that round to execute
        if current_round > max_rounds:
            return "end"
        
        # Check for surrender
        for agent_id, agent_state in state["agent_states"].items():
            if agent_state.get('has_surrendered', False):
                return "end"
        
        return "continue"
    
    def run_debate(
        self,
        topic: str,
        agent_configs: List[Dict],
        max_rounds: int = 5
    ) -> Dict:
        """
        Run a complete debate session.
        
        Args:
            topic: Debate topic
            agent_configs: List of agent configurations
            max_rounds: Maximum number of rounds
            
        Returns:
            Complete debate results including history, states, and summary
        """
        
        print(f"\n{'='*60}")
        print(f"[Debate] Starting debate on: {topic}")
        print(f"[Debate] Participants: {[c['id'] for c in agent_configs]}")
        print(f"[Debate] Max rounds: {max_rounds}")
        print(f"{'='*60}\n")
        
        # Create initial state
        initial_state = create_initial_state(topic, agent_configs, max_rounds)
        
        # Run the graph
        start_time = time.time()
        final_state = None
        
        for event in self.compiled_graph.stream(initial_state):
            # Log progress
            for node_name, node_state in event.items():
                if 'current_response' in node_state and node_state['current_response']:
                    if isinstance(node_state.get('history'), list) and node_state['history']:
                        latest = node_state['history'][-1] if node_state['history'] else None
                        if latest:
                            print(f"\n[Round {node_state.get('current_round', '?')}] {latest['agent_id']}:")
                            print(f"  {latest['content'][:200]}...")
            
            final_state = node_state
        
        elapsed_time = time.time() - start_time
        print(f"\n[Debate] Completed in {elapsed_time:.2f}s")
        
        # Generate summary
        summary = self._generate_summary(final_state)
        
        return {
            'topic': topic,
            'total_rounds': final_state.get('current_round', 1),
            'history': final_state.get('history', []),
            'agent_states': final_state.get('agent_states', {}),
            'summary': summary,
            'elapsed_time': elapsed_time
        }
    
    async def run_single_round(
        self,
        state: DebateState
    ) -> DebateState:
        """Run a single round of debate (for step-by-step execution)"""
        
        # Run until round changes or debate ends
        initial_round = state["current_round"]
        
        for event in self.compiled_graph.stream(state):
            for node_name, node_state in event.items():
                state = {**state, **node_state}
            
            # Check if round changed or debate ended
            if state.get("current_round", 0) != initial_round:
                break
            if state.get("debate_ended", False):
                break
        
        return state
    
    def _generate_summary(self, final_state: Dict) -> Dict:
        """Generate debate summary"""
        
        if not final_state:
            return {"message": "No debate data available"}
        
        agent_states = final_state.get('agent_states', {})
        
        # Count surrenders
        surrendered_agents = [
            aid for aid, state in agent_states.items()
            if state.get('has_surrendered', False)
        ]
        
        # Calculate scores
        agent_scores = {}
        for agent_id, state in agent_states.items():
            score = 0
            
            # Stance firmness score
            stance_score = abs(state.get('current_stance', 0)) * state.get('conviction', 0.7) * 30
            score += stance_score
            
            # Persuasion score
            for other_id, other_state in agent_states.items():
                if other_id != agent_id:
                    if other_state.get('has_surrendered', False):
                        score += 20
                    history = other_state.get('persuasion_history', [])
                    if history:
                        avg_persuasion = sum(history) / len(history)
                        score += avg_persuasion * 10
            
            # Resistance score
            attack_history = state.get('attack_history', [])
            if attack_history:
                avg_attack = sum(attack_history) / len(attack_history)
                resistance_score = (1 - avg_attack) * state.get('conviction', 0.7) * 20
                score += resistance_score
            
            # Surrender penalty
            if state.get('has_surrendered', False):
                score -= 50
            
            agent_scores[agent_id] = score
        
        # Determine winner
        winner = max(agent_scores.keys(), key=lambda x: agent_scores[x]) if agent_scores else None
        
        # Generate verdict
        if surrendered_agents:
            verdict = f"[WINNER] {winner} achieved overwhelming victory! Successfully persuaded {', '.join(surrendered_agents)} to surrender."
        elif winner:
            scores_sorted = sorted(agent_scores.values(), reverse=True)
            score_diff = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else 0
            if score_diff > 30:
                verdict = f"[WINNER] {winner} won with clear advantage! Demonstrated excellent debate skills."
            else:
                verdict = f"[WINNER] {winner} narrowly won! This was an evenly matched exciting debate."
        else:
            verdict = "Debate ended without a clear winner."
        
        return {
            "winner": winner,
            "scores": agent_scores,
            "surrendered_agents": surrendered_agents,
            "final_states": {
                aid: {
                    "stance": state.get('current_stance', 0),
                    "conviction": state.get('conviction', 0.7),
                    "has_surrendered": state.get('has_surrendered', False)
                }
                for aid, state in agent_states.items()
            },
            "verdict": verdict
        }
    
    def get_graph_visualization(self) -> str:
        """Get ASCII representation of the graph"""
        return """
        ┌─────────────────────┐
        │  parallel_analysis  │
        │  (RL + GNN + RAG)   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │    fuse_results     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  generate_response  │
        │       (LLM)         │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   update_states     │
        └──────────┬──────────┘
                   │
           ┌───────┴───────┐
           │  should_continue  │
           └───────┬───────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
    next_speaker  next_round   end
        │          │          │
        └────┬─────┘          │
             │                │
        ┌────▼────┐           │
        │ advance │           │
        │  turn   │           │
        └────┬────┘           │
             │                │
             └────────────────┘
                    │
                   END
        """


# Convenience function
def create_langgraph_orchestrator(
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7
) -> LangGraphDebateOrchestrator:
    """Create a LangGraph-based debate orchestrator"""
    return LangGraphDebateOrchestrator(model_name=model_name, temperature=temperature)


