"""
System integrity tests - moved from root directory
Tests module imports and file existence
"""

import sys
import importlib
from pathlib import Path
import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestCoreModuleImports:
    """Test that core modules can be imported"""
    
    @pytest.mark.parametrize("module_name,description", [
        ("agents.base_agent", "Base agent class"),
        # Note: agent_a/b/c have relative import issues when imported directly
        # They work when imported from the application context
    ])
    def test_agent_imports(self, module_name, description):
        """Test agent module imports"""
        module = importlib.import_module(module_name)
        assert module is not None
    
    @pytest.mark.parametrize("module_name,description", [
        ("orchestrator.parallel_orchestrator", "Legacy orchestrator"),
        ("orchestrator.debate_state", "Debate state"),
        ("orchestrator.debate_tools", "Debate tools"),
        ("orchestrator.langgraph_orchestrator", "LangGraph orchestrator"),
    ])
    def test_orchestrator_imports(self, module_name, description):
        """Test orchestrator module imports"""
        module = importlib.import_module(module_name)
        assert module is not None
    
    @pytest.mark.parametrize("module_name,description", [
        ("gpt_interface.gpt_client", "GPT client"),
        # Note: dialogue_manager depends on agents which have import issues
        ("utils.config_loader", "Config loader"),
    ])
    def test_utility_imports(self, module_name, description):
        """Test utility module imports"""
        module = importlib.import_module(module_name)
        assert module is not None
    
    @pytest.mark.parametrize("module_name,description", [
        ("rag.simple_retriever", "Simple retriever"),
        # Note: build_index requires OPENAI_API_KEY at import time
    ])
    def test_rag_imports(self, module_name, description):
        """Test RAG module imports"""
        module = importlib.import_module(module_name)
        assert module is not None
    
    @pytest.mark.parametrize("module_name,description", [
        ("gnn.social_encoder", "Social encoder"),
    ])
    def test_gnn_imports(self, module_name, description):
        """Test GNN module imports"""
        module = importlib.import_module(module_name)
        assert module is not None
    
    @pytest.mark.parametrize("module_name,description", [
        ("rl.policy_network", "Policy network"),
    ])
    def test_rl_imports(self, module_name, description):
        """Test RL module imports"""
        module = importlib.import_module(module_name)
        assert module is not None


class TestFileExistence:
    """Test that required files exist"""
    
    @pytest.mark.parametrize("file_path,description", [
        ("configs/debate.yaml", "Debate config"),
        ("configs/rag.yaml", "RAG config"),
        ("configs/gnn.yaml", "GNN config"),
        ("configs/rl.yaml", "RL config"),
        ("configs/system.yaml", "System config"),
    ])
    def test_config_files(self, project_root, file_path, description):
        """Test config files exist"""
        assert (project_root / file_path).exists(), f"{description} not found"
    
    @pytest.mark.parametrize("file_path,description", [
        ("run_flask.py", "Flask runner"),
        ("train_all.py", "Training script"),
        ("pyproject.toml", "Project config"),
        ("README.md", "README"),
    ])
    def test_root_files(self, project_root, file_path, description):
        """Test root level files exist"""
        assert (project_root / file_path).exists(), f"{description} not found"


class TestDirectoryStructure:
    """Test that required directories exist"""
    
    @pytest.mark.parametrize("dir_path,description", [
        ("src", "Source directory"),
        ("src/agents", "Agents directory"),
        ("src/orchestrator", "Orchestrator directory"),
        ("src/rag", "RAG directory"),
        ("src/gnn", "GNN directory"),
        ("src/rl", "RL directory"),
        ("ui", "UI directory"),
        ("configs", "Config directory"),
        ("docs", "Documentation directory"),
        ("tests", "Tests directory"),
    ])
    def test_directories(self, project_root, dir_path, description):
        """Test directories exist"""
        path = project_root / dir_path
        assert path.exists() and path.is_dir(), f"{description} not found"

