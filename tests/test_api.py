"""
Tests for the Spine Fracture Detection API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.utils import apply_window, preprocess_single_slice
from app.model import SpineModelSingle, SpineModel3Channel
import numpy as np
import torch


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a sample image array."""
    return np.random.randint(0, 1000, size=(512, 512)).astype(np.float32)


# ============================================================
# API Tests
# ============================================================
class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Spine Fracture Detection API"
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "device" in data
    
    def test_predict_invalid_file_type(self, client):
        """Test predict endpoint rejects non-DICOM files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        assert response.status_code == 400
        assert "dcm" in response.json()["detail"].lower()


# ============================================================
# Utility Function Tests
# ============================================================
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_apply_window(self, sample_image):
        """Test CT windowing function."""
        result = apply_window(sample_image)
        
        # Check output is normalized to 0-1
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.dtype == np.float32
    
    def test_apply_window_custom_params(self):
        """Test windowing with custom parameters."""
        image = np.array([[100, 400, 700], [200, 500, 800]], dtype=np.float32)
        result = apply_window(image, center=400, width=400)
        
        # Values outside window should be clipped
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_preprocess_single_slice(self, sample_image):
        """Test single slice preprocessing."""
        result = preprocess_single_slice(sample_image, img_size=256)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1, 256, 256)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ============================================================
# Model Tests
# ============================================================
class TestModels:
    """Test model classes."""
    
    def test_single_model_forward(self):
        """Test SpineModelSingle forward pass."""
        model = SpineModelSingle(num_classes=7)
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 1, 512, 512)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 7)
    
    def test_3channel_model_forward(self):
        """Test SpineModel3Channel forward pass."""
        model = SpineModel3Channel(num_classes=7)
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 512, 512)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 7)
    
    def test_model_output_range(self):
        """Test that sigmoid of model output is in valid range."""
        model = SpineModel3Channel(num_classes=7)
        model.eval()
        
        x = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.sigmoid(output)
        
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


# ============================================================
# Integration Tests
# ============================================================
class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline_single(self, sample_image):
        """Test full pipeline with single slice model."""
        # Preprocess
        image = preprocess_single_slice(sample_image)
        
        # Model forward
        model = SpineModelSingle()
        model.eval()
        
        with torch.no_grad():
            output = model(image)
            probs = torch.sigmoid(output).numpy()[0]
        
        # Check output format
        assert len(probs) == 7
        assert all(0 <= p <= 1 for p in probs)


# ============================================================
# Run tests
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
