"""
Pytest configuration and fixtures for Social Debate AI tests
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Fix Windows encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


@pytest.fixture
def project_root():
    """Return project root path"""
    return PROJECT_ROOT


@pytest.fixture
def sample_agent_configs():
    """Return sample agent configurations for testing"""
    return [
        {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
        {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.7},
        {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.7}
    ]


@pytest.fixture
def sample_topic():
    """Return sample debate topic"""
    return "Should artificial intelligence be regulated by government?"


@pytest.fixture
def sample_debate_context():
    """Return sample debate context for testing"""
    return {
        'topic': "Should AI be regulated?",
        'history': [
            {'agent_id': 'Agent_A', 'content': 'I believe AI needs oversight...'},
            {'agent_id': 'Agent_B', 'content': 'Regulation would stifle innovation...'}
        ]
    }

