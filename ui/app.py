"""
Flask web application for Social Debate AI
"""

import json
import time
import asyncio
from flask import Flask, render_template, request, jsonify, redirect, url_for
from pathlib import Path

# Import core modules
try:
    from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
    from src.utils.config_loader import ConfigLoader
    print("Core modules loaded successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the project root directory")

app = Flask(__name__)

# Global variables
orchestrator = None
config = None

def initialize_system():
    global orchestrator, config
    
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
        
        # Initialize orchestrator
        print("Initializing parallel orchestrator...")
        orchestrator = ParallelOrchestrator()
        
        # Initialize agents
        agent_configs = []
        for agent_name in ['Agent_A', 'Agent_B', 'Agent_C']:
            agent_configs.append({
                'id': agent_name,
                'initial_stance': 0.8 if agent_name == 'Agent_A' else (-0.6 if agent_name == 'Agent_B' else 0.0),
                'initial_conviction': 0.7
            })
        
        orchestrator.initialize_agents(agent_configs)
        print("System initialization complete")
        
        return True
        
    except Exception as e:
        print(f"System initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', 
                         title="Social Debate AI",
                         description="AI-powered debate simulation system")

@app.route('/api/debate', methods=['POST'])
def start_debate():
    """Start debate"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({'error': 'System not initialized'}), 500
    
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({'error': 'Topic cannot be empty'}), 400
    
    # Limit topic length
    if len(topic) > 200:
        topic = topic[:200] + "..."
    
    try:
        print(f"Starting debate on topic: {topic}")
        
        # Run debate simulation
        max_rounds = config.get('debate', {}).get('max_rounds', 5)
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
            'summary': summary
        })
        
    except Exception as e:
        print(f"Debate execution failed: {str(e)}")
        return jsonify({'error': f'Debate failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """System health check"""
    return jsonify({
        'status': 'ok',
        'system_ready': orchestrator is not None,
        'timestamp': time.time()
    })

@app.route('/api/init', methods=['POST'])
def init_system():
    """初始化系統"""
    global orchestrator
    
    if orchestrator:
        return jsonify({
            'success': True,
            'message': '系統已初始化'
        })
    
    try:
        # 重新初始化系統
        init_result = initialize_system()
        
        if init_result:
            return jsonify({
                'success': True,
                'message': '系統初始化成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '系統初始化失敗'
            }), 500
            
    except Exception as e:
        print(f"Init failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'初始化錯誤: {str(e)}'
        }), 500

@app.route('/api/set_topic', methods=['POST'])
def set_topic():
    """設置辯論主題"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統尚未初始化'
        }), 500
    
    data = request.get_json()
    topic = data.get('topic', '').strip()
    
    if not topic:
        return jsonify({
            'success': False,
            'message': '主題不能為空'
        }), 400
    
    try:
        # 重置辯論狀態
        orchestrator.agent_states = {}
        orchestrator.debate_history = []
        
        # 重新初始化代理
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
            'message': '主題設置成功',
            'topic': topic
        })
        
    except Exception as e:
        print(f"Set topic failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'設置主題失敗: {str(e)}'
        }), 500

@app.route('/api/debate_round', methods=['POST'])
def debate_round():
    """執行單回合辯論"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統尚未初始化'
        }), 500
    
    try:
        # 獲取當前回合數 - orchestrator 會自動管理 debate_history
        if not hasattr(orchestrator, 'debate_history'):
            orchestrator.debate_history = []
        current_round = len(orchestrator.debate_history) + 1
        max_rounds = config.get('debate', {}).get('max_rounds', 5)
        
        if current_round > max_rounds:
            return jsonify({
                'success': False,
                'message': '已達到最大回合數'
            }), 400
        
        try:
            data = request.get_json() or {}
        except Exception as e:
            print(f"JSON parse error: {e}")
            data = {}
        
        topic = data.get('topic', '預設辯論主題')
        agent_order = ['Agent_A', 'Agent_B', 'Agent_C']
        
        print(f"執行第 {current_round} 回合辯論")
        
        # 創建事件循環執行異步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            debate_round_result = loop.run_until_complete(
                orchestrator.run_debate_round(current_round, topic, agent_order)
            )
            
            # 轉換為可序列化格式
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
            
            # 添加回應內容
            if debate_round_result.history:
                round_data['responses'] = debate_round_result.history
            
            # 注意：不要手動添加到 debate_history，orchestrator 會自己管理
            # orchestrator.debate_history.append(round_data)
            
            # 檢查是否有代理投降
            has_surrender = any(state.has_surrendered for state in debate_round_result.agent_states.values())
            
            # 如果辯論結束，生成摘要
            summary = None
            if has_surrender or current_round >= max_rounds:
                try:
                    summary = orchestrator.get_debate_summary()
                    print(f"Generated summary: {summary}")
                except Exception as e:
                    print(f"Failed to generate summary: {e}")
                    summary = {
                        "message": "無法生成摘要",
                        "error": str(e)
                    }
            
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
            'message': f'辯論回合執行失敗: {str(e)}'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_debate():
    """重置辯論"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統尚未初始化'
        }), 500
    
    try:
        # 清空辯論歷史
        orchestrator.debate_history = []
        orchestrator.agent_states = {}
        
        # 重新初始化代理
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
            'message': '辯論已重置'
        })
        
    except Exception as e:
        print(f"Reset failed: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'重置失敗: {str(e)}'
        }), 500

@app.route('/api/export', methods=['GET'])
def export_debate():
    """導出辯論記錄"""
    global orchestrator
    
    if not orchestrator:
        return jsonify({
            'success': False,
            'message': '系統尚未初始化'
        }), 500
    
    try:
        export_data = {
            'debate_history': orchestrator.debate_history,
            'agent_states': {},
            'summary': orchestrator.get_debate_summary(),
            'export_time': time.time()
        }
        
        # 轉換代理狀態
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
            'message': f'導出失敗: {str(e)}'
        }), 500

@app.route('/debug')
def debug_info():
    """Debug information page"""
    system_info = {
        'orchestrator_loaded': orchestrator is not None,
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