"""
Parallel Debate Orchestrator
Integrates parallel processing of RL, GNN, RAG modules
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import random
import numpy as np

# Delayed import to avoid circular dependencies
# PolicyNetwork and Retriever will be dynamically imported when needed

# Import GPT interface
try:
    from gpt_interface.gpt_client import chat
except ImportError:
    # Define dummy function in case GPT is unavailable
    def chat(prompt: str) -> str:
        """Dummy chat function"""
        # This should call the actual GPT interface
        # For demonstration purposes, return a mock response
        responses = [
            "Based on in-depth analysis, I believe this issue needs to be considered from multiple perspectives. First, we must acknowledge its complexity and understand the reasonable concerns behind different positions.",
            "Let me explain this issue from another angle. While the opponent has raised some points, I think they have overlooked several key factors.",
            "I understand the opponent's concerns, but we need to discuss based on facts and data. According to the latest research, the impact of this issue is far more profound than it appears on the surface."
        ]
        return random.choice(responses)

# Try to import required modules
try:
    from rl.policy_network import select_strategy as _select_strategy, choose_snippet as _choose_snippet, PolicyNetwork as _PolicyNetwork
    from gnn.social_encoder import social_vec as _social_vec, get_social_influence_score, predict_persuasion
    from rag.retriever import create_enhanced_retriever as _create_enhanced_retriever
    from utils.config_loader import ConfigLoader as _ConfigLoader
    
    # Create wrapper functions
    def select_strategy(query: str, context: str = "", social_context: List[float] = None) -> str:
        """Wrapper for select_strategy function"""
        return _select_strategy(query, context, social_context)
    
    def choose_snippet(state_text: str, pool: List[Dict]) -> str:
        """Wrapper for choose_snippet function"""
        return _choose_snippet(state_text, pool)
    
    def social_vec(agent_id: str) -> List[float]:
        """Wrapper for social_vec function"""
        return _social_vec(agent_id)
        
except ImportError as e:
    print(f"Module import failed: {e}")
    print("Running with dummy functions")
    
    # Define dummy functions
    def select_strategy(query: str, context: str = "", social_context: List[float] = None) -> str:
        """Dummy select_strategy function"""
        strategies = ['analytical', 'aggressive', 'defensive', 'empathetic']
        return random.choice(strategies)
    
    def choose_snippet(state_text: str, pool: List[Dict]) -> str:
        """Dummy choose_snippet function"""
        if pool:
            return pool[0].get('content', 'No evidence available')
        return "No evidence available"
    
    def social_vec(agent_id: str) -> List[float]:
        """Dummy social_vec function"""
        return [random.random() for _ in range(128)]

@dataclass
class AgentState:
    """Agent State"""
    agent_id: str
    current_stance: float  # -1.0 to 1.0, stance intensity
    conviction: float      # 0.0 to 1.0, belief firmness
    social_context: List[float]  # Social context vector
    persuasion_history: List[float]  # Persuasion history
    attack_history: List[float]     # Attack history
    has_surrendered: bool = False  # Whether surrendered
    
    def update_stance(self, persuasion_score: float, attack_score: float):
        """Update stance and conviction"""
        # Calculate persuasion effect
        persuasion_effect = persuasion_score * (1.0 - self.conviction)
        
        # Calculate attack resistance
        attack_resistance = self.conviction * 0.8
        attack_effect = max(0, attack_score - attack_resistance)
        
        # Update stance (persuasion moves towards neutral, attack polarizes)
        if persuasion_score > 0.5:  # Persuaded (降低閾值)
            self.current_stance *= (1.0 - persuasion_effect * 0.2)  # 減少立場改變幅度
            self.conviction *= 0.9  # 信念減弱更慢
        
        if attack_effect > 0.3:  # Attacked
            self.current_stance *= (1.0 + attack_effect * 0.2)  # Stance becomes more extreme
            self.conviction = min(1.0, self.conviction * 1.1)  # Conviction strengthens
        
        # Record history
        self.persuasion_history.append(persuasion_score)
        self.attack_history.append(attack_score)
        
        # Keep history length
        if len(self.persuasion_history) > 10:
            self.persuasion_history.pop(0)
        if len(self.attack_history) > 10:
            self.attack_history.pop(0)
        
        # Check if should surrender (much stricter conditions)
        if len(self.persuasion_history) >= 4:  # 需要至少4輪歷史
            recent_persuasion = sum(self.persuasion_history[-4:]) / 4
            # Condition 1: Extremely high persuasion + Extremely low conviction
            if recent_persuasion > 0.65 and self.conviction < 0.25:
                self.has_surrendered = True
                print(f"{self.agent_id} surrendered (highly persuaded)")
            # Condition 2: Stance extremely close to neutral + very low conviction
            elif abs(self.current_stance) < 0.1 and self.conviction < 0.3:
                self.has_surrendered = True
                print(f"{self.agent_id} surrendered (stance neutralized)")
            # Condition 3: Consistently highly persuaded for many rounds
            elif len(self.persuasion_history) >= 5:
                consecutive_high = all(score > 0.6 for score in self.persuasion_history[-5:])
                if consecutive_high and self.conviction < 0.4:
                    self.has_surrendered = True
                    print(f"{self.agent_id} surrendered (overwhelmingly persuaded)")

@dataclass
class DebateRound:
    """Debate Round"""
    round_number: int
    topic: str
    agent_states: Dict[str, AgentState]
    history: List[Dict]
    
class ParallelOrchestrator:
    """Parallel Processing Debate Orchestrator"""
    
    def __init__(self):
        self.policy_network = None  # Lazy loading
        self.retriever = None  # Lazy loading
        self.agent_states = {}
        self.debate_history = []
        
        # Executor pool
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        print("Parallel orchestrator initialized")
        print("Executor pool: 4 threads")
        print("Models will load on first use")
    
    def _get_policy_network(self):
        """Lazy load PolicyNetwork"""
        if self.policy_network is None:
            print("Loading RL policy network...")
            start_time = time.time()
            from rl.policy_network import PolicyNetwork
            self.policy_network = PolicyNetwork()
            load_time = time.time() - start_time
            print(f"RL policy loaded ({load_time:.2f}s)")
        return self.policy_network
    
    def _get_retriever(self):
        """Lazy load Retriever"""
        if self.retriever is None:
            print("Loading RAG retriever...")
            start_time = time.time()
            try:
                from rag.retriever import create_enhanced_retriever
                self.retriever = create_enhanced_retriever()
                load_time = time.time() - start_time
                print(f"RAG retriever loaded ({load_time:.2f}s)")
            except Exception as e:
                print(f"Enhanced retriever failed: {e}")
                from rag.simple_retriever import SimpleRetriever
                self.retriever = SimpleRetriever()
                load_time = time.time() - start_time
                print(f"Simple retriever loaded ({load_time:.2f}s)")
                stats = self.retriever.get_stats()
                print(f"Index: {stats.get('total_documents', 0):,} documents")
        return self.retriever
    
    def initialize_agents(self, agent_configs: List[Dict]) -> Dict[str, AgentState]:
        """Initialize Agent states"""
        agents = {}
        
        for config in agent_configs:
            agent_id = config['id']
            agents[agent_id] = AgentState(
                agent_id=agent_id,
                current_stance=config.get('initial_stance', 0.0),
                conviction=config.get('initial_conviction', 0.7),
                social_context=config.get('social_context', [0.0] * 128),
                persuasion_history=[],
                attack_history=[]
            )
        
        self.agent_states = agents
        print(f"Initialized {len(agents)} agents")
        return agents
    
    async def parallel_analysis(self, agent_id: str, topic: str, 
                              history: List[str]) -> Dict:
        """Parallel execution RL + GNN + RAG analysis"""
        
        # Build query context
        recent_turns = history[-3:] if history else []
        context = f"Topic: {topic}\nRecent: {' '.join(recent_turns)}"
        agent_state = self.agent_states[agent_id]
        
        # Create asynchronous tasks
        loop = asyncio.get_event_loop()
        
        # 1. RL strategy selection
        rl_task = loop.run_in_executor(
            self.executor,
            self._rl_analysis,
            context, agent_state.social_context
        )
        
        # 2. GNN social analysis
        gnn_task = loop.run_in_executor(
            self.executor,
            self._gnn_analysis,
            agent_id, agent_state
        )
        
        # 3. RAG evidence retrieval
        rag_task = loop.run_in_executor(
            self.executor,
            self._rag_analysis,
            context, topic
        )
        
        # Wait for all tasks to complete
        try:
            rl_result, gnn_result, rag_result = await asyncio.gather(
                rl_task, gnn_task, rag_task
            )
            
            return {
                'rl': rl_result,
                'gnn': gnn_result,
                'rag': rag_result,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"Parallel analysis failed: {e}")
            return self._fallback_analysis(context, agent_id)
    
    def _rl_analysis(self, context: str, social_context: List[float]) -> Dict:
        """RL strategy analysis"""
        try:
            strategy = select_strategy(context, "", social_context)
            print(f"RL strategy: {strategy}")
            
            # Predict quality score
            quality_score = self._get_policy_network().predict_quality(context)
            print(f"Quality score: {quality_score:.2f}")
            
            return {
                'strategy': strategy,
                'quality_score': quality_score,
                'confidence': 0.8  # Can be obtained from model
            }
        except Exception as e:
            print(f"RL analysis failed: {e}")
            return {'strategy': 'analytical', 'quality_score': 0.5, 'confidence': 0.3}
    
    def _gnn_analysis(self, agent_id: str, agent_state: AgentState) -> Dict:
        """GNN social relationship analysis"""
        try:
            # Get social vector
            social_vector = social_vec(agent_id)
            # Use new influence calculation method
            influence_score = get_social_influence_score(agent_id)
            
            # If there are recent responses, predict persuasion strategy
            if hasattr(agent_state, 'last_response') and agent_state.last_response:
                # Simplified text feature extraction (should use BERT in practice)
                text_features = np.random.randn(768)  # Temporarily use random features
                persuasion_pred = predict_persuasion(text_features, agent_id)
            else:
                persuasion_pred = {
                    'delta_probability': 0.5,
                    'best_strategy': 'analytical',
                    'strategy_scores': {}
                }
            
            # Analyze stance change trend
            stance_trend = 0.0
            if len(agent_state.persuasion_history) >= 2:
                recent_persuasion = sum(agent_state.persuasion_history[-3:]) / 3
                stance_trend = recent_persuasion - 0.5
            
            return {
                'social_vector': social_vector,
                'influence_score': influence_score,
                'stance_trend': stance_trend,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction,
                'persuasion_prediction': persuasion_pred
            }
        except Exception as e:
            print(f"GNN analysis failed: {e}")
            return {
                'social_vector': [0.0] * 128,
                'influence_score': 0.5,
                'stance_trend': 0.0,
                'current_stance': agent_state.current_stance,
                'conviction': agent_state.conviction,
                'persuasion_prediction': {
                    'delta_probability': 0.5,
                    'best_strategy': 'analytical',
                    'strategy_scores': {}
                }
            }
    
    def _rag_analysis(self, context: str, topic: str) -> Dict:
        """RAG evidence retrieval analysis"""
        try:
            # Retrieve evidence pool
            retrieval_results = self._get_retriever().retrieve(
                query=context,
                top_k=8
            )
            print(f"Retrieved {len(retrieval_results)} evidence pieces")
            
            # Convert to dictionary format for choose_snippet usage
            evidence_pool = []
            for result in retrieval_results:
                # Check if result is already a dictionary (from SimpleRetrieverAdapter)
                if isinstance(result, dict):
                    evidence_dict = {
                        'content': result.get('content', ''),
                        'similarity_score': result.get('score', 0.0),
                        'metadata': result.get('metadata', {}),
                        'doc_id': result.get('doc_id', '')
                    }
                else:
                    # If it's an object (from EnhancedRetriever)
                    evidence_dict = {
                        'content': getattr(result, 'content', ''),
                        'similarity_score': getattr(result, 'score', 0.0),
                        'metadata': getattr(result, 'metadata', {}),
                        'doc_id': getattr(result, 'doc_id', '')
                    }
                evidence_pool.append(evidence_dict)
            
            # Select best evidence
            best_evidence = choose_snippet(context, evidence_pool)
            
            # Analyze evidence type distribution
            evidence_types = {}
            for result in retrieval_results:
                # Safely get metadata
                if isinstance(result, dict):
                    metadata = result.get('metadata', {})
                else:
                    metadata = getattr(result, 'metadata', {})
                
                ev_type = metadata.get('type', 'unknown')
                evidence_types[ev_type] = evidence_types.get(ev_type, 0) + 1
            
            print(f"  [INFO] RAG: Evidence type distribution = {evidence_types}")
            
            return {
                'evidence_pool': evidence_pool,
                'best_evidence': best_evidence,
                'evidence_types': evidence_types,
                'total_evidence': len(evidence_pool)
            }
        except Exception as e:
            print(f"[WARN] RAG analysis failed: {e}")
            return {
                'evidence_pool': [],
                'best_evidence': "No evidence available",
                'evidence_types': {},
                'total_evidence': 0
            }
    
    def _fallback_analysis(self, context: str, agent_id: str) -> Dict:
        """Fallback analysis"""
        return {
            'rl': {'strategy': 'analytical', 'quality_score': 0.5, 'confidence': 0.3},
            'gnn': {'social_vector': [0.0] * 128, 'influence_score': 0.5, 
                   'stance_trend': 0.0, 'current_stance': 0.0, 'conviction': 0.7},
            'rag': {'evidence_pool': [], 'best_evidence': "No evidence available",
                   'evidence_types': {}, 'total_evidence': 0},
            'timestamp': time.time()
        }
    
    def fuse_analysis_results(self, analysis_results: Dict, agent_id: str) -> Dict:
        """Fuse analysis results"""
        rl_result = analysis_results['rl']
        gnn_result = analysis_results['gnn']
        rag_result = analysis_results['rag']
        
        # Strategy adjustment: combine RL and GNN recommendations
        base_strategy = rl_result['strategy']
        gnn_strategy = gnn_result['persuasion_prediction']['best_strategy']
        influence_score = gnn_result['influence_score']
        current_stance = gnn_result['current_stance']
        delta_probability = gnn_result['persuasion_prediction']['delta_probability']
        
        # Strategy fusion logic
        if delta_probability > 0.7:
            # High persuasion success rate, prioritize GNN suggested strategy
            adjusted_strategy = gnn_strategy
            print(f"  [ADJUST] Strategy adjustment: {base_strategy} → {adjusted_strategy} (based on high Delta probability)")
        elif influence_score > 0.6 and abs(current_stance) > 0.5:
            # High influence + strong stance = more aggressive
            if base_strategy == 'analytical' and gnn_strategy == 'aggressive':
                adjusted_strategy = 'aggressive'
                print(f"  [ADJUST] Strategy adjustment: {base_strategy} → {adjusted_strategy} (high influence + strong stance)")
            else:
                adjusted_strategy = base_strategy
        elif influence_score < 0.4 and abs(current_stance) < 0.3:
            # Low influence + weak stance = more cautious
            if base_strategy == 'aggressive':
                adjusted_strategy = 'defensive'
                print(f"  [ADJUST] Strategy adjustment: {base_strategy} → {adjusted_strategy} (low influence + weak stance)")
            else:
                adjusted_strategy = base_strategy
        else:
            # Balance RL and GNN recommendations
            strategy_scores = gnn_result['persuasion_prediction']['strategy_scores']
            if strategy_scores.get(base_strategy, 0) < 0.2:
                # If RL-selected strategy scores poorly in GNN, consider switching
                adjusted_strategy = gnn_strategy
                print(f"  [ADJUST] Strategy adjustment: {base_strategy} → {adjusted_strategy} (GNN recommendation)")
            else:
                adjusted_strategy = base_strategy
        
        # Evidence selection: adjust based on strategy and predicted persuasiveness
        evidence = rag_result['best_evidence']
        evidence_confidence = min(1.0, rag_result['total_evidence'] / 5.0)
        
        # Adjust evidence confidence based on predicted Delta probability
        adjusted_confidence = evidence_confidence * (0.5 + 0.5 * delta_probability)
        
        print(f"  [RESULT] Fusion result: Final strategy={adjusted_strategy}, Evidence confidence={adjusted_confidence:.2f}, Delta probability={delta_probability:.2f}")
        
        return {
            'final_strategy': adjusted_strategy,
            'evidence': evidence,
            'evidence_confidence': adjusted_confidence,
            'social_influence': influence_score,
            'stance_strength': abs(current_stance),
            'conviction': gnn_result['conviction'],
            'delta_probability': delta_probability,
            'gnn_suggested_strategy': gnn_strategy,
            'fusion_timestamp': time.time()
        }
    
    async def generate_response(self, agent_id: str, topic: str, 
                              history: List[str], target_agents: List[str]) -> str:
        """Generate debate response"""
        
        # 1. Parallel analysis
        print(f"\n[ANALYSIS] Agent {agent_id} starting parallel analysis...")
        print(f"   [TOPIC] Topic: {topic}")
        print(f"   [INFO] History round count: {len(history)}")
        analysis_start = time.time()
        
        analysis_results = await self.parallel_analysis(agent_id, topic, history)
        
        analysis_time = time.time() - analysis_start
        print(f"[COMPLETE] Parallel analysis completed ({analysis_time:.2f}s)")
        
        # 2. Fuse results
        print(f"[FUSE] Starting result fusion...")
        fused_results = self.fuse_analysis_results(analysis_results, agent_id)
        print(f"[OK] Fusion completed")
        
        # 3. Build prompt
        agent_state = self.agent_states[agent_id]
        recent_history = history[-4:] if history else []
        
        # Analyze target agent weaknesses
        target_analysis = self._analyze_targets(target_agents, history)
        
        prompt = self._build_enhanced_prompt(
            agent_id, topic, recent_history, fused_results, target_analysis
        )
        
        # 4. Generate response
        print(f"[GENERATE] Agent {agent_id} using {fused_results['final_strategy']} strategy to generate response...")
        generation_start = time.time()
        response = chat(prompt)
        generation_time = time.time() - generation_start
        print(f"[OK] Response generation completed ({generation_time:.2f}s)")
        
        # Check if response is truncated (check if it ends with period, question mark, or exclamation mark)
        if response and not response.rstrip().endswith(('。', '！', '？', '.', '!', '?')):
            print(f"[WARN] Detected response may be truncated, attempting to complete...")
            # If response is truncated, add ending
            response += ". In conclusion, based on the above analysis, I maintain my position."
        
        # 5. Evaluate response effectiveness
        response_effects = self._evaluate_response(response, target_agents)
        
        total_time = time.time() - analysis_start
        print(f"[TIME] Agent {agent_id} total processing time: {total_time:.2f}s")
        
        return response
    
    def _analyze_targets(self, target_agents: List[str], history: List[str]) -> Dict:
        """Analyze target agent weaknesses and opportunities"""
        target_analysis = {}
        
        for target_id in target_agents:
            if target_id in self.agent_states:
                target_state = self.agent_states[target_id]
                
                # Analyze persuasion opportunities
                persuasion_opportunity = 1.0 - target_state.conviction
                
                # Analyze attack opportunities
                attack_opportunity = abs(target_state.current_stance)
                
                # Analyze historical trends
                recent_persuasion = 0.0
                if target_state.persuasion_history:
                    recent_persuasion = sum(target_state.persuasion_history[-2:]) / 2
                
                target_analysis[target_id] = {
                    'persuasion_opportunity': persuasion_opportunity,
                    'attack_opportunity': attack_opportunity,
                    'recent_persuasion': recent_persuasion,
                    'stance': target_state.current_stance,
                    'conviction': target_state.conviction
                }
        
        return target_analysis
    
    def _build_enhanced_prompt(self, agent_id: str, topic: str, history: List[str],
                              fused_results: Dict, target_analysis: Dict) -> str:
        """Build enhanced prompt"""
        
        agent_state = self.agent_states[agent_id]
        strategy = fused_results['final_strategy']
        evidence = fused_results['evidence']
        
        # Determine if this is the first round
        is_first_round = len(history) == 0
        
        # Historical dialogue
        history_text = '\n'.join(f"Turn {i+1}: {turn}" for i, turn in enumerate(history))
        
        # Analyze used arguments (to avoid repetition)
        used_arguments = self._extract_used_arguments(history, agent_id)
        
        # Target analysis (not needed for first round)
        target_info = ""
        if not is_first_round:
            for target_id, analysis in target_analysis.items():
                target_info += f"\n- {target_id}: Stance {analysis['stance']:.2f}, Conviction {analysis['conviction']:.2f}, Persuasion opportunity {analysis['persuasion_opportunity']:.2f}"
        
        # Enhanced strategy guidance
        strategy_guidance = {
            'aggressive': "Adopt critical analysis strategy: deeply analyze logical flaws in opponent's arguments, challenge core assumptions with powerful counterexamples and data.",
            'defensive': "Adopt robust argumentation strategy: consolidate core arguments, systematically respond to challenges, strengthen position with more evidence.",
            'analytical': "Adopt rational analysis strategy: use logical reasoning, empirical data and case studies to objectively evaluate pros and cons of various viewpoints.",
            'empathetic': "Adopt constructive dialogue strategy: understand opponent's reasonable concerns, find common ground, propose solutions considering all parties' interests."
        }
        
        # Debate style prompt
        if is_first_round:
            debate_style = """
This is a rational debate about public issues. You need to:
- Clearly state your position on the issue (support or oppose)
- Present your core arguments and reasoning
- Use facts and logic to support your viewpoints
- Demonstrate critical thinking skills
- Do not mention other participants (since you haven't heard their statements yet)
"""
            round_instruction = """
First round requirements:
1. Clearly state whether you support or oppose the issue
2. Present 3-4 core arguments
3. Use specific facts and data to support your position
4. Analyze from perspectives of social impact, economic benefits, long-term development, etc.
5. Do not mention other participants
6. Word count requirement: 200-250 words
7. Must complete full argumentation, do not cut off midway
"""
        else:
            # Adjust strategy based on round number
            round_num = len([h for h in history if agent_id in h]) + 1
            if round_num >= 3 and agent_state.conviction < 0.4:
                debate_style = """
After in-depth discussion, you begin to reconsider this issue:
- Acknowledge that some of the opponent's arguments make sense
- Reflect on the limitations of your own position
- Try to find middle ground or compromise solutions
- Show openness and rational attitude
"""
            elif round_num >= 4:
                debate_style = """
The debate enters the summary stage. You need to:
- Integrate viewpoints from all parties and provide comprehensive analysis
- Point out core disagreements and consensus
- Propose feasible solutions or suggestions
- Make constructive contributions to discussion of this public issue
"""
            else:
                debate_style = """
This is an in-depth public issue debate. You need to:
- Respond to specific arguments from opponents
- Provide more evidence and analysis
- Deepen your arguments from different perspectives
- Maintain rational and professional discussion atmosphere
"""
            
            # Guidance to avoid repetition
            avoid_repetition = ""
            if used_arguments:
                avoid_repetition = f"""
Note: You have already discussed the following aspects, please explore new angles:
{chr(10).join(f"- {arg}" for arg in used_arguments[:3])}

Try to analyze from other dimensions such as: social equity, sustainable development, international comparison, historical experience, etc.
"""
            
            round_instruction = f"""
Debate requirements:
1. Respond to opponent's core arguments, point out logical flaws or provide counterevidence
2. Deepen your arguments, provide new perspectives and evidence
3. Use connecting words like "however", "nevertheless", "on the other hand"
4. Use [CITE] tags when referencing specific cases or data
5. Maintain objectivity and rationality, avoid personal attacks
6. If persuaded, appropriately adjust your position
7. Avoid repeating already discussed content
8. Word count requirement: 200-250 words
9. Must complete full argumentation, ensure arguments have clear beginning and end

{avoid_repetition}
"""
        
        # Build prompt
        if is_first_round:
            return f"""You are participating in a public issue debate about "{topic}".

{debate_style}

Your role settings:
- Position tendency: {agent_state.current_stance:.2f} (positive values lean support, negative values lean oppose)
- Conviction level: {agent_state.conviction:.2f} (higher values are harder to change position)
- Argumentation style: {strategy}

Available evidence:
{evidence}

{round_instruction}

Please express your viewpoint:"""
        else:
            return f"""You are participating in a public issue debate about "{topic}".

{debate_style}

Your current status:
- Position tendency: {agent_state.current_stance:.2f} (positive values lean support, negative values lean oppose)
- Conviction level: {agent_state.conviction:.2f} (higher values are harder to change position)
- Argumentation style: {strategy}

Discussion record:
{history_text}

Available evidence:
{evidence}

Other participants' status:{target_info}

Argumentation strategy: {strategy_guidance.get(strategy, '')}

{round_instruction}

Please express your viewpoint:"""
    
    def _extract_used_arguments(self, history: List[str], agent_id: str) -> List[str]:
        """Extract used argument keywords"""
        used_arguments = []
        
        # Simple keyword extraction
        keywords = ['economic', 'unemployment', 'diplomacy', 'trade', 'environment', 'climate', 'security', 'policy', 
                   'fiscal', 'deficit', 'international', 'leadership', 'social', 'division']
        
        for turn in history:
            if agent_id in turn:
                for keyword in keywords:
                    if keyword in turn and keyword not in used_arguments:
                        used_arguments.append(keyword)
        
        return used_arguments
    
    def _evaluate_response(self, response: str, target_agents: List[str]) -> Dict:
        """Evaluate response persuasiveness and aggressiveness"""
        
        # English and Chinese keyword evaluation
        persuasion_indicators = [
            'however', 'consider', 'understand', 'perspective', 'common',
            'but', 'consider', 'understand', 'viewpoint', 'common', 'agree', 'acknowledge'
        ]
        attack_indicators = [
            'wrong', 'flawed', 'mistake', 'ignore', 'fail',
            'error', 'flaw', 'fallacy', 'overlook', 'failure', 'absurd', 'unreasonable'
        ]
        evidence_indicators = [
            '[CITE]', 'study', 'research', 'data', 'evidence',
            'research', 'data', 'evidence', 'fact', 'statistics', 'report', 'survey'
        ]
        
        # Calculate scores (using more sensitive calculation)
        response_lower = response.lower()
        persuasion_count = sum(1 for indicator in persuasion_indicators if indicator in response_lower)
        attack_count = sum(1 for indicator in attack_indicators if indicator in response_lower)
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in response_lower)
        
        # Adjust score calculation to be more balanced
        persuasion_score = min(0.7, persuasion_count * 0.1)  # 進一步降低說服力係數，最高70%
        attack_score = min(0.6, attack_count * 0.15)  # 進一步降低攻擊力係數，最高60%
        evidence_score = min(0.5, evidence_count * 0.15)  # 進一步降低證據係數，最高50%
        
        # Length score
        word_count = len(response.split())
        length_score = min(1.0, word_count / 80)
        
        return {
            'persuasion_score': persuasion_score,
            'attack_score': attack_score,
            'evidence_score': evidence_score,
            'length_score': length_score
        }
    
    async def run_debate_round(self, round_number: int, topic: str, 
                             agent_order: List[str]) -> DebateRound:
        """Execute one round of debate"""
        
        print(f"\n[ROUND] Starting round {round_number} debate")
        print(f"Topic: {topic}")
        print(f"Speaking order: {' → '.join(agent_order)}")
        
        # Get all previous historical records
        all_history = []
        for past_round in self.debate_history:
            # Handle both DebateRound objects and dict formats
            if hasattr(past_round, 'history'):
                responses = past_round.history
            else:
                responses = past_round.get('responses', [])
            
            for response in responses:
                all_history.append(f"{response['agent_id']}: {response['content']}")
        
        # Current round responses
        round_responses = []
        
        for i, agent_id in enumerate(agent_order):
            print(f"\n--- Agent {agent_id} speaking ---")
            
            # Determine target opponents
            other_agents = [aid for aid in agent_order if aid != agent_id]
            
            # Combine history: all previous rounds + current round already spoken
            current_history = all_history + [f"{r['agent_id']}: {r['content']}" for r in round_responses]
            
            # Generate response
            response = await self.generate_response(
                agent_id, topic, 
                current_history, 
                other_agents
            )
            
            # Evaluate effects
            effects = self._evaluate_response(response, other_agents)
            
            # Record response
            round_responses.append({
                'agent_id': agent_id,
                'content': response,
                'effects': effects,
                'timestamp': time.time()
            })
            
            print(f"Agent {agent_id}: {response[:100]}...")
            print(f"Effect evaluation: Persuasion {effects['persuasion_score']:.2f}, Attack {effects['attack_score']:.2f}")
            
            # Update other agents' states
            for target_id in other_agents:
                if target_id in self.agent_states:
                    self.agent_states[target_id].update_stance(
                        effects['persuasion_score'],
                        effects['attack_score']
                    )
        
        # Create round record
        debate_round = DebateRound(
            round_number=round_number,
            topic=topic,
            agent_states=dict(self.agent_states),
            history=round_responses
        )
        
        self.debate_history.append(debate_round)
        
        # Display round results
        self._display_round_summary(debate_round)
        
        return debate_round
    
    def _display_round_summary(self, debate_round: DebateRound):
        """Display round summary"""
        print(f"\n[SUMMARY] Round {debate_round.round_number} summary")
        print("=" * 50)
        
        for agent_id, state in debate_round.agent_states.items():
            print(f"Agent {agent_id}:")
            print(f"  Stance: {state.current_stance:+.2f} | Conviction: {state.conviction:.2f}")
            if state.persuasion_history:
                avg_persuasion = sum(state.persuasion_history[-3:]) / min(3, len(state.persuasion_history))
                print(f"  Recent persuasion level: {avg_persuasion:.2f}")
    
    def get_debate_summary(self) -> Dict:
        """Get debate summary and victory determination"""
        if not self.debate_history:
            return {"message": "Debate has not started yet"}
        
        # Count surrender situations
        surrendered_agents = [aid for aid, state in self.agent_states.items() 
                            if state.has_surrendered]
        
        # Calculate comprehensive score for each agent
        agent_scores = {}
        for agent_id, state in self.agent_states.items():
            # Base score
            score = 0
            
            # Stance firmness score (more extreme stance + stronger conviction = higher score)
            stance_score = abs(state.current_stance) * state.conviction * 30
            score += stance_score
            
            # Score for persuading others (based on others' surrender and stance changes)
            persuasion_score = 0
            for other_id, other_state in self.agent_states.items():
                if other_id != agent_id:
                    if other_state.has_surrendered:
                        persuasion_score += 20
                    # Calculate impact on others
                    if len(other_state.persuasion_history) > 0:
                        avg_persuasion = sum(other_state.persuasion_history) / len(other_state.persuasion_history)
                        persuasion_score += avg_persuasion * 10
            score += persuasion_score
            
            # Stress resistance score (attacked but still maintaining stance)
            if len(state.attack_history) > 0:
                avg_attack = sum(state.attack_history) / len(state.attack_history)
                resistance_score = (1 - avg_attack) * state.conviction * 20
                score += resistance_score
            
            # Surrender penalty
            if state.has_surrendered:
                score -= 50
            
            agent_scores[agent_id] = score
        
        # Determine winner
        winner = max(agent_scores.keys(), key=lambda x: agent_scores[x])
        
        # Generate summary report
        summary = {
            "total_rounds": len(self.debate_history),
            "winner": winner,
            "scores": agent_scores,
            "surrendered_agents": surrendered_agents,
            "final_states": {},
            "verdict": ""
        }
        
        # Add final states
        for aid, state in self.agent_states.items():
            summary["final_states"][aid] = {
                "stance": state.current_stance,
                "conviction": state.conviction,
                "has_surrendered": state.has_surrendered,
                "final_position": "support" if state.current_stance > 0 else "oppose"
            }
        
        # Generate verdict
        if len(surrendered_agents) > 0:
            summary["verdict"] = f"[WINNER] {winner} achieved overwhelming victory! Successfully persuaded {', '.join(surrendered_agents)} to surrender."
        else:
            score_diff = agent_scores[winner] - sorted(agent_scores.values())[-2]
            if score_diff > 30:
                summary["verdict"] = f"[WINNER] {winner} won with clear advantage! Demonstrated excellent debate skills."
            else:
                summary["verdict"] = f"[WINNER] {winner} narrowly won! This was an evenly matched exciting debate."
        
        return summary

# Convenience function
def create_parallel_orchestrator():
    """Create parallel orchestrator"""
    return ParallelOrchestrator() 