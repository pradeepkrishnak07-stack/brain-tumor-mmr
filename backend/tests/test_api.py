"""
API Unit and Integration Tests for FastAPI backend
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "service" in data
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_list_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 4


def test_analyze_sample_valid():
    response = client.post("/api/analyze-sample/sample_1")
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "results_summary" in data
    assert "images" in data
    assert "original" in data["images"]
    assert "overlay" in data["images"]
    assert data["results_summary"]["candidate_regions_count"] >= 1


def test_analyze_sample_invalid_id():
    response = client.post("/api/analyze-sample/non_existent_id")
    assert response.status_code == 404


def test_analyze_upload_valid_image():
    # Create synthetic test image with a small hyperintense nodule (radius ~ 4 pixels)
    img = Image.new("L", (128, 128), color=50)
    # Draw a bright spot of size 8x8
    for r in range(50, 58):
        for c in range(50, 58):
            img.putpixel((c, r), 230)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        files={"file": ("test_slice.png", buf.getvalue(), "image/png")},
        data={
            "enable_denoise": "false",
            "enable_clahe": "false",
            "recon_method": "opening_by_reconstruction",
            "selem_shape": "disk",
            "selem_size": "10",  # SE larger than 8x8 spot
            "thresh_method": "otsu",
            "min_area": "20"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_slice.png"
    assert "dimensions" in data
    assert data["dimensions"]["width"] == 128
    assert data["dimensions"]["height"] == 128
    assert "images" in data
    assert "overlay" in data["images"]
    assert len(data["regions"]) >= 1


def test_analyze_empty_file():
    response = client.post(
        "/api/analyze",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_analyze_unsupported_file_type():
    response = client.post(
        "/api/analyze",
        files={"file": ("malicious.exe", b"MZDummyPayload", "application/x-msdownload")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
