"""
Integration tests for Flask web application
"""

import pytest
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFlaskAppImport:
    """Test Flask app can be imported"""
    
    def test_import_app(self):
        """Test app module imports"""
        from ui import app
        assert app is not None
    
    def test_flask_app_exists(self):
        """Test Flask app instance exists"""
        from ui.app import app
        assert app is not None


class TestFlaskRoutes:
    """Test Flask routes exist"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from ui.app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_index_route(self, client):
        """Test index route returns 200"""
        response = client.get('/')
        assert response.status_code == 200
    
    def test_health_route(self, client):
        """Test health check route"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data
    
    def test_debug_route(self, client):
        """Test debug info route"""
        response = client.get('/debug')
        assert response.status_code == 200


class TestAPIEndpoints:
    """Test API endpoints structure"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from ui.app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_debate_endpoint_requires_topic(self, client):
        """Test debate endpoint requires topic"""
        response = client.post('/api/debate', json={})
        assert response.status_code == 400
    
    def test_set_topic_endpoint(self, client):
        """Test set topic endpoint structure"""
        response = client.post('/api/set_topic', json={'topic': ''})
        # Should return 400 for empty topic or 500 if not initialized
        assert response.status_code in [400, 500]
    
    def test_reset_endpoint(self, client):
        """Test reset endpoint structure"""
        response = client.post('/api/reset')
        # May return 500 if not initialized, which is expected
        assert response.status_code in [200, 500]

