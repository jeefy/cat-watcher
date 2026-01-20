"""Tests for inference API."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

# Skip all tests if FastAPI is not available
fastapi = pytest.importorskip("fastapi")
PIL = pytest.importorskip("PIL")

from fastapi.testclient import TestClient

from cat_watcher.inference.api import router, set_pipeline
from cat_watcher.inference.pipeline import InferencePipeline


@pytest.fixture
def client() -> TestClient:
    """Create test client with mock pipeline."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Create a mock pipeline
    mock_pipeline = MagicMock(spec=InferencePipeline)
    mock_pipeline.is_loaded = True
    mock_pipeline.has_detector = True
    mock_pipeline.has_identifier = True
    set_pipeline(mock_pipeline)

    return TestClient(app)


@pytest.fixture
def client_no_pipeline() -> TestClient:
    """Create test client without pipeline."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    set_pipeline(None)  # type: ignore[arg-type]

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check_healthy(self, client: TestClient) -> None:
        """Test healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline_loaded"] is True

    def test_health_check_no_pipeline(self, client_no_pipeline: TestClient) -> None:
        """Test unhealthy status when no pipeline."""
        response = client_no_pipeline.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["pipeline_loaded"] is False


class TestBehaviorsEndpoint:
    """Tests for behaviors endpoint."""

    def test_list_behaviors(self, client: TestClient) -> None:
        """Test listing behaviors."""
        response = client.get("/api/v1/behaviors")

        assert response.status_code == 200
        behaviors = response.json()
        assert "cat_eating" in behaviors
        assert "cat_drinking" in behaviors
        assert "cat_vomiting" in behaviors
        assert "cat_litterbox" in behaviors
        assert "cat_yowling" in behaviors


class TestCatsEndpoint:
    """Tests for cats endpoint."""

    def test_list_cats(self, client: TestClient) -> None:
        """Test listing cats."""
        response = client.get("/api/v1/cats")

        assert response.status_code == 200
        cats = response.json()
        assert "starbuck" in cats
        assert "apollo" in cats
        assert "mia" in cats
        assert "unknown" in cats


class TestInferEndpoint:
    """Tests for inference endpoint."""

    def test_infer_no_pipeline(self, client_no_pipeline: TestClient) -> None:
        """Test inference without pipeline returns 503."""
        from PIL import Image

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client_no_pipeline.post(
            "/api/v1/infer",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

        assert response.status_code == 503

    def test_infer_invalid_file_type(self, client: TestClient) -> None:
        """Test inference with non-image file."""
        response = client.post(
            "/api/v1/infer",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )

        assert response.status_code == 400
        assert "image" in response.json()["detail"].lower()


class TestDetectEndpoint:
    """Tests for detect endpoint."""

    def test_detect_no_detector(self, client: TestClient) -> None:
        """Test detect when detector not configured."""
        from PIL import Image

        # Mock pipeline without detector
        mock_pipeline = MagicMock(spec=InferencePipeline)
        mock_pipeline.has_detector = False
        set_pipeline(mock_pipeline)

        img = Image.new("RGB", (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post(
            "/api/v1/detect",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

        assert response.status_code == 503


class TestIdentifyEndpoint:
    """Tests for identify endpoint."""

    def test_identify_no_identifier(self, client: TestClient) -> None:
        """Test identify when identifier not configured."""
        from PIL import Image

        # Mock pipeline without identifier
        mock_pipeline = MagicMock(spec=InferencePipeline)
        mock_pipeline.has_identifier = False
        set_pipeline(mock_pipeline)

        img = Image.new("RGB", (100, 100))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = client.post(
            "/api/v1/identify",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )

        assert response.status_code == 503
