"""
System integrity tests for Social Debate AI
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, description):
    try:
        importlib.import_module(module_name)
        return f"✓ {description}"
    except ImportError as e:
        return f"✗ {description}: {e}"
    except Exception as e:
        return f"⚠ {description}: {e}"

def test_file_exists(file_path, description):
    if Path(file_path).exists():
        return f"✓ {description}"
    else:
        return f"✗ {description}: File not found"

def test_directory_exists(dir_path, description):
    if Path(dir_path).exists() and Path(dir_path).is_dir():
        return f"✓ {description}"
    else:
        return f"✗ {description}: Directory not found"

def run_tests():
    tests = [
        # Core modules
        ("src.agents.base_agent", "Base agent class"),
        ("src.agents.agent_a", "Agent A"),
        ("src.agents.agent_b", "Agent B"),
        ("src.agents.agent_c", "Agent C"),
        
        # System components
        ("src.orchestrator.parallel_orchestrator", "Parallel orchestrator"),
        ("src.gpt_interface.gpt_client", "GPT client"),
        ("src.dialogue.dialogue_manager", "Dialogue manager"),
        
        # RAG system
        ("src.rag.retriever", "RAG retriever"),
        ("src.rag.simple_retriever", "Simple retriever"),
        ("src.rag.build_index", "Index builder"),
        
        # GNN system
        ("src.gnn.social_encoder", "Social encoder"),
        ("src.gnn.train_supervised", "GNN trainer"),
        
        # RL system
        ("src.rl.policy_network", "Policy network"),
        ("src.rl.train_ppo", "PPO trainer"),
        ("src.rl.ppo_trainer", "PPO trainer class"),
        
        # Utilities
        ("src.utils.config_loader", "Config loader"),
        
        # Web interface
        ("ui.app", "Flask app"),
    ]
    
    file_tests = [
        ("configs/debate.yaml", "Main config file"),
        ("run_flask.py", "Flask runner"),
        ("train_all.py", "Training script"),
    ]
    
    dir_tests = [
        ("src/", "Source directory"),
        ("ui/", "UI directory"),
        ("configs/", "Config directory"),
        ("data/", "Data directory"),
    ]
    
    results = []
    
    # Test imports
    for module, desc in tests:
        results.append(test_import(module, desc))
    
    # Test files
    for file_path, desc in file_tests:
        results.append(test_file_exists(file_path, desc))
    
    # Test directories
    for dir_path, desc in dir_tests:
        results.append(test_directory_exists(dir_path, desc))
    
    return results

def main():
    print("Social Debate AI - System Integrity Check")
    print("=" * 50)
    
    results = run_tests()
    
    success_count = sum(1 for r in results if r.startswith("✓"))
    warning_count = sum(1 for r in results if r.startswith("⚠"))
    error_count = sum(1 for r in results if r.startswith("✗"))
    
    print("\nCore Modules")
    print("-" * 30)
    for result in results[:14]:  # Core modules
        print(result)
    
    print("\nSystem Files")
    print("-" * 30)
    for result in results[14:]:  # Files and directories
        print(result)
    
    print("\nSummary")
    print("-" * 30)
    print(f"Success: {success_count}")
    print(f"Warning: {warning_count}")
    print(f"Error: {error_count}")
    print(f"Total: {len(results)} tests")
    
    if error_count > 0:
        print(f"\n{error_count} critical issues found!")
        sys.exit(1)
    elif warning_count > 0:
        print(f"\n{warning_count} warnings found.")
        sys.exit(0)
    else:
        print("\nAll tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main() 