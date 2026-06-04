"""
System integrity tests: core modules import and required files/dirs exist.
"""

import sys
import importlib
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestCoreModuleImports:
    @pytest.mark.parametrize("module_name,description", [
        ("llm.provider", "LLM provider seam"),
        ("orchestrator.debate_state", "Debate state"),
        ("orchestrator.debate_tools", "Debate tools"),
        ("orchestrator.langgraph_orchestrator", "LangGraph orchestrator"),
        ("rag.vector_retriever", "FAISS retriever"),
        ("rag.simple_retriever", "Keyword retriever (fallback)"),
        ("gnn.social_encoder", "GNN social encoder"),
        ("rl.policy_network", "RL policy"),
        ("storage.debate_store", "Debate persistence"),
        ("utils.config_loader", "Config loader"),
    ])
    def test_module_imports(self, module_name, description):
        module = importlib.import_module(module_name)
        assert module is not None


class TestFileExistence:
    @pytest.mark.parametrize("file_path,description", [
        ("configs/debate.yaml", "Debate config"),
        ("configs/rag.yaml", "RAG config"),
        ("configs/gnn.yaml", "GNN config"),
        ("configs/rl.yaml", "RL config"),
        ("configs/system.yaml", "System config"),
        ("wsgi.py", "WSGI entrypoint"),
        ("train_all.py", "Training script"),
        ("pyproject.toml", "Project config"),
        ("README.md", "README"),
        ("env.example", "Env template"),
    ])
    def test_files(self, project_root, file_path, description):
        assert (project_root / file_path).exists(), f"{description} not found"


class TestDirectoryStructure:
    @pytest.mark.parametrize("dir_path,description", [
        ("src/llm", "LLM seam"),
        ("src/orchestrator", "Orchestrator"),
        ("src/rag", "RAG"),
        ("src/gnn", "GNN"),
        ("src/rl", "RL"),
        ("src/storage", "Storage"),
        ("ui", "UI"),
        ("configs", "Config"),
        ("docs", "Docs"),
        ("tests", "Tests"),
        ("docker", "Docker"),
        ("scripts", "Scripts"),
    ])
    def test_directories(self, project_root, dir_path, description):
        path = project_root / dir_path
        assert path.exists() and path.is_dir(), f"{description} not found"
