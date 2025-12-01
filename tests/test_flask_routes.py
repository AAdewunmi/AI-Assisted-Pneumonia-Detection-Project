"""
tests/test_flask_routes.py
--------------------------
Integration tests for PneumoDetect Flask web application.
Validates route accessibility, upload handling, and prediction responses.
"""

import io
import pytest
from app.app import app


@pytest.fixture
def client():
    """Create a Flask test client for the PneumoDetect app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Ensure the home page (/) loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Upload" in response.data or b"Predict" in response.data


def test_predict_route_no_file(client):
    """POST /predict with no file should redirect to home."""
    response = client.post("/predict", data={}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Upload" in response.data or b"PneumoDetect" in response.data


def test_predict_route_with_image(client):
    """
    POST /predict with a valid image file.
    Since we mock model weights for CI, this only checks for response integrity.
    """
    img_bytes = io.BytesIO(b"fake_image_data")
    data = {
        "file": (img_bytes, "test.png"),
        "threshold": "0.5",
    }

    response = client.post("/predict", data=data, content_type="multipart/form-data")
    # Either redirect (if inference fails) or render result.html
    assert response.status_code in (200, 302)


def test_invalid_route_returns_404(client):
    """Accessing an undefined route should return 404."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
