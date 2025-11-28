"""
tests/test_threshold_logic.py
-----------------------------
Tests threshold logic and Flask routes for PneumoDetect.
"""

import pytest
from app.utils import apply_threshold
from app.app import app

# -------------------------------
# Threshold Logic Unit Tests
# -------------------------------
def test_apply_threshold_high():
    assert apply_threshold(0.86, 0.8) == "High Risk"

def test_apply_threshold_low():
    assert apply_threshold(0.3, 0.8) == "Low Risk"


# -------------------------------
# Flask Route Tests
# -------------------------------
@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Ensure home page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"PneumoDetect" in response.data


def test_predict_route_redirects_without_file(client):
    """POST without file should redirect to index."""
    response = client.post("/predict", data={})
    assert response.status_code in (302, 308)
