"""
Flask web application for Social Debate AI
Supports both legacy ParallelOrchestrator and new LangGraph orchestrator
"""

import json
import time
import asyncio
import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from pathlib import Path

# Import core modules
try:
    from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
    from src.orchestrator.langgraph_orchestrator import LangGraphDebateOrchestrator, create_langgraph_orchestrator
    from src.utils.config_loader import ConfigLoader
    print("Core modules loaded successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")

app = Flask(__name__)

# Global variables
orchestrator = None
langgraph_orchestrator = None
config = None
USE_LANGGRAPH = os.environ.get('USE_LANGGRAPH', 'true').lower() == 'true'

def initialize_system():
    global orchestrator, langgraph_orchestrator, config
    
    try:
        # Load configuration
        config = ConfigLoader.load('debate')
        config_path = Path('configs/debate.yaml')
        
        if config_path.exists():
            print(f"Loaded configuration: {config_path}")
            print(f"Max rounds: {config.get('debate', {}).get('max_rounds', 5)}")
            print(f"Participants: {', '.join(config.get('debate', {}).get('agents', []))}")
        else:
            print("Using default configuration")
        
        # Initialize orchestrators
        if USE_LANGGRAPH:
            print("Initializing LangGraph orchestrator...")
            try:
                langgraph_orchestrator = create_langgraph_orchestrator()
                print("LangGraph orchestrator initialized successfully")
            except Exception as e:
                print(f"LangGraph initialization failed: {e}")
                print("Falling back to parallel orchestrator")
                USE_LANGGRAPH = False
        
        # Always initialize legacy orchestrator as fallback
        print("Initializing parallel orchestrator...")
        orchestrator = ParallelOrchestrator()
        
        # Initialize agents for legacy orchestrator
        agent_configs = []
        for agent_name in ['Agent_A', 'Agent_B', 'Agent_C']:
            agent_configs.append({
                'id': agent_name,
                'initial_stance': 0.8 if agent_name == 'Agent_A' else (-0.6 if agent_name == 'Agent_B' else 0.0),
                'initial_conviction': 0.7
            })
        
        orchestrator.initialize_agents(agent_configs)
        print("System initialization complete")
        print(f"Using {'LangGraph' if USE_LANGGRAPH and langgraph_orchestrator else 'Legacy'} orchestrator")
        
        return True
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         title="Social Debate AI",
                         description="AI-powered debate simulation system",
                         use_langgraph=USE_LANGGRAPH and langgraph_orchestrator is not None)

@app.route('/api/debate', methods=['POST'])
def start_debate():
    """Start debate - full automated run"""
    global orchestrator, langgraph_orchestrator
    
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({'error': 'Topic cannot be empty'}), 400
    
    # Limit topic length
    if len(topic) > 200:
        topic = topic[:200] + "..."
    
    try:
        print(f"Starting debate on topic: {topic}")
        max_rounds = config.get('debate', {}).get('max_rounds', 5)
        
        # Use LangGraph if available
        if USE_LANGGRAPH and langgraph_orchestrator:
            return _run_langgraph_debate(topic, max_rounds)
        else:
            return _run_legacy_debate(topic, max_rounds)
        
    except Exception as e:
        print(f"Debate execution failed: {str(e)}")
        return jsonify({'error': f'Debate failed: {str(e)}'}), 500


def _run_langgraph_debate(topic: str, max_rounds: int):
    """Run debate using LangGraph orchestrator"""
    global langgraph_orchestrator
    
    agent_configs = [
        {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
        {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.7}
    ]
    
    # Run the debate
    results = langgraph_orchestrator.run_debate(
        topic=topic,
        agent_configs=agent_configs,
        max_rounds=max_rounds
    )
    
    # Format for API response
    debate_results = []
    current_round = 1
    round_responses = []
    
    for response in results.get('history', []):
        round_responses.append(response)
        
        # Check if round complete (all 3 agents spoke)
        if len(round_responses) >= 3:
            round_data = {
                'round': current_round,
                'topic': topic,
                'responses': round_responses,
                'agents': {}
            }
            
            for agent_id, state in results.get('agent_states', {}).items():
                round_data['agents'][agent_id] = {
                    'stance': round(state.get('current_stance', 0), 2),
                    'conviction': round(state.get('conviction', 0.7), 2),
                    'has_surrendered': state.get('has_surrendered', False)
                }
            
            debate_results.append(round_data)
            round_responses = []
            current_round += 1
    
    return jsonify({
        'success': True,
        'topic': topic,
        'rounds': debate_results,
        'summary': results.get('summary', {}),
        'orchestrator': 'langgraph',
        'elapsed_time': results.get('elapsed_time', 0)
    })


def _run_legacy_debate(topic: str, max_rounds: int):
    """Run debate using legacy parallel orchestrator"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({'error': 'System not initialized'}), 500
    
    agent_order = ['Agent_A', 'Agent_B', 'Agent_C']
    debate_results = []
    
    # Execute debate rounds
    for round_num in range(1, max_rounds + 1):
        print(f"Round {round_num}")
        
        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            debate_round = loop.run_until_complete(
                orchestrator.run_debate_round(round_num, topic, agent_order)
            )
            
            # Convert to serializable format
            round_data = {
                'round': round_num,
                'topic': topic,
                'agents': {}
            }
            
            for agent_id, state in debate_round.agent_states.items():
                round_data['agents'][agent_id] = {
                    'stance': round(state.current_stance, 2),
                    'conviction': round(state.conviction, 2),
                    'has_surrendered': state.has_surrendered
                }
            
            # Add responses from history
            if debate_round.history:
                round_data['responses'] = debate_round.history
            
            debate_results.append(round_data)
            
            # Check if any agent surrendered
            if any(state.has_surrendered for state in debate_round.agent_states.values()):
                print(f"Debate ended early due to surrender in round {round_num}")
                break
                
        finally:
            loop.close()
    
    # Generate summary
    summary = orchestrator.get_debate_summary()
    
    return jsonify({
        'success': True,
        'topic': topic,
        'rounds': debate_results,
        'summary': summary,
        'orchestrator': 'legacy'
    })


@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'ok',
        'system_ready': orchestrator is not None or langgraph_orchestrator is not None,
        'orchestrator_type': 'langgraph' if (USE_LANGGRAPH and langgraph_orchestrator) else 'legacy',
        'timestamp': time.time()
    })

@app.route('/api/init', methods=['POST'])
def init_system():
    """Initialize system"""
    global orchestrator, langgraph_orchestrator
    
    if orchestrator or langgraph_orchestrator:
        return jsonify({
            'success': True,
            'message': 'System already initialized',
            'orchestrator_type': 'langgraph' if (USE_LANGGRAPH and langgraph_orchestrator) else 'legacy'
        })
    
    try:
        init_result = initialize_system()
        
        if init_result:
            return jsonify({
                'success': True,
                'message': 'System initialized successfully',
                'orchestrator_type': 'langgraph' if (USE_LANGGRAPH and langgraph_orchestrator) else 'legacy'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'System initialization failed'
            }), 500
            
    except Exception as e:
        print(f"Init failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Initialization error: {str(e)}'
        }), 500

@app.route('/api/set_topic', methods=['POST'])
def set_topic():
    """Set debate topic"""
    global orchestrator, langgraph_orchestrator
    
    if not orchestrator and not langgraph_orchestrator:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 500
    
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({
            'success': False,
            'message': 'Topic cannot be empty'
        }), 400
    
    try:
        # Reset legacy orchestrator state
        if orchestrator:
            orchestrator.agent_states = {}
            orchestrator.debate_history = []
            
            agent_configs = []
            for agent_name in ['Agent_A', 'Agent_B', 'Agent_C']:
                agent_configs.append({
                    'id': agent_name,
                    'initial_stance': 0.8 if agent_name == 'Agent_A' else (-0.6 if agent_name == 'Agent_B' else 0.0),
                    'initial_conviction': 0.7
                })
            
            orchestrator.initialize_agents(agent_configs)
        
        return jsonify({
            'success': True,
            'message': 'Topic set successfully',
            'topic': topic
        })
        
    except Exception as e:
        print(f"Set topic failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Failed to set topic: {str(e)}'
        }), 500

@app.route('/api/debate_round', methods=['POST'])
def debate_round():
    """Execute single debate round"""
    global orchestrator, langgraph_orchestrator
    
    if not orchestrator and not langgraph_orchestrator:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 500
    
    try:
        data = request.get_json() or {}
        topic = data.get('topic', 'Default debate topic')
        
        # For step-by-step, use legacy orchestrator
        if not orchestrator:
            return jsonify({
                'success': False,
                'message': 'Step-by-step mode requires legacy orchestrator'
            }), 500
        
        if not hasattr(orchestrator, 'debate_history'):
            orchestrator.debate_history = []
        current_round = len(orchestrator.debate_history) + 1
        max_rounds = config.get('debate', {}).get('max_rounds', 5)
        
        if current_round > max_rounds:
            return jsonify({
                'success': False,
                'message': 'Maximum rounds reached'
            }), 400
        
        agent_order = ['Agent_A', 'Agent_B', 'Agent_C']
        
        print(f"Executing round {current_round}")
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            debate_round_result = loop.run_until_complete(
                orchestrator.run_debate_round(current_round, topic, agent_order)
            )
            
            round_data = {
                'round': current_round,
                'topic': topic,
                'agents': {}
            }
            
            for agent_id, state in debate_round_result.agent_states.items():
                round_data['agents'][agent_id] = {
                    'stance': round(state.current_stance, 2),
                    'conviction': round(state.conviction, 2),
                    'has_surrendered': state.has_surrendered
                }
            
            if debate_round_result.history:
                round_data['responses'] = debate_round_result.history
            
            has_surrender = any(state.has_surrendered for state in debate_round_result.agent_states.values())
            
            summary = None
            if has_surrender or current_round >= max_rounds:
                try:
                    summary = orchestrator.get_debate_summary()
                except Exception as e:
                    print(f"Failed to generate summary: {e}")
                    summary = {"message": "Could not generate summary", "error": str(e)}
            
            return jsonify({
                'success': True,
                'round': current_round,
                'topic': topic,
                'responses': round_data.get('responses', []),
                'agent_states': round_data.get('agents', {}),
                'max_rounds': max_rounds,
                'has_surrender': has_surrender,
                'debate_ended': has_surrender or current_round >= max_rounds,
                'summary': summary
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        print(f"Debate round failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Debate round failed: {str(e)}'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_debate():
    """Reset debate"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 500
    
    try:
        orchestrator.debate_history = []
        orchestrator.agent_states = {}
        
        agent_configs = []
        for agent_name in ['Agent_A', 'Agent_B', 'Agent_C']:
            agent_configs.append({
                'id': agent_name,
                'initial_stance': 0.8 if agent_name == 'Agent_A' else (-0.6 if agent_name == 'Agent_B' else 0.0),
                'initial_conviction': 0.7
            })
        
        orchestrator.initialize_agents(agent_configs)
        
        return jsonify({
            'success': True,
            'message': 'Debate reset successfully'
        })
        
    except Exception as e:
        print(f"Reset failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Reset failed: {str(e)}'
        }), 500

@app.route('/api/export', methods=['GET'])
def export_debate():
    """Export debate records"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': 'System not initialized'
        }), 500
    
    try:
        export_data = {
            'debate_history': orchestrator.debate_history,
            'agent_states': {},
            'summary': orchestrator.get_debate_summary(),
            'export_time': time.time()
        }
        
        for agent_id, state in orchestrator.agent_states.items():
            export_data['agent_states'][agent_id] = {
                'stance': state.current_stance,
                'conviction': state.conviction,
                'has_surrendered': state.has_surrendered
            }
        
        return jsonify({
            'success': True,
            'data': export_data
        })
        
    except Exception as e:
        print(f"Export failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Export failed: {str(e)}'
        }), 500

@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get LangGraph visualization"""
    global langgraph_orchestrator
    
    if langgraph_orchestrator:
        return jsonify({
            'success': True,
            'graph': langgraph_orchestrator.get_graph_visualization(),
            'orchestrator': 'langgraph'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'LangGraph orchestrator not available'
        }), 404

@app.route('/debug')
def debug_info():
    """Debug information page"""
    system_info = {
        'orchestrator_loaded': orchestrator is not None,
        'langgraph_orchestrator_loaded': langgraph_orchestrator is not None,
        'use_langgraph': USE_LANGGRAPH,
        'config_loaded': config is not None,
        'config_path': str(Path('configs/debate.yaml').absolute()),
        'agents_initialized': len(orchestrator.agent_states) if orchestrator else 0
    }
    
    return jsonify(system_info)

# Initialize system on startup
if __name__ == '__main__':
    print("Starting Social Debate AI Web Application")
    print("=" * 50)
    
    if initialize_system():
        print("Server ready at http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize system")
        exit(1)
