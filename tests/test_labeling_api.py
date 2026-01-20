"""Tests for labeling API."""

import tempfile
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cat_watcher.collection.storage import FrameStorage
from cat_watcher.config import Settings
from cat_watcher.labeling.app import create_app
from cat_watcher.schemas import BehaviorType, BoundingBox, CatName


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_settings(temp_data_dir: Path) -> Settings:
    """Create test settings."""
    return Settings(data_dir=temp_data_dir)


@pytest.fixture
def storage(temp_data_dir: Path) -> FrameStorage:
    """Create storage instance."""
    return FrameStorage(temp_data_dir)


@pytest.fixture
def app(test_settings: Settings, storage: FrameStorage) -> FastAPI:
    """Create test FastAPI app with storage."""
    app = create_app(test_settings)
    # Manually set storage since lifespan doesn't run in tests
    app.state.storage = storage
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_frame_data() -> bytes:
    """Create minimal valid JPEG data."""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Create sample bounding box."""
    return BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.7)


class TestLabelingAPI:
    """Tests for labeling API endpoints."""

    async def test_health_check(self, client: AsyncClient) -> None:
        """Test health endpoint."""
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_get_behaviors(self, client: AsyncClient) -> None:
        """Test getting behavior types."""
        response = await client.get("/api/behaviors")
        assert response.status_code == 200
        behaviors = response.json()

        assert len(behaviors) > 0
        behavior_values = [b["value"] for b in behaviors]
        assert "cat_eating" in behavior_values
        assert "cat_litterbox" in behavior_values
        assert "cat_yowling" in behavior_values

    async def test_get_cats(self, client: AsyncClient) -> None:
        """Test getting cat identities."""
        response = await client.get("/api/cats")
        assert response.status_code == 200
        cats = response.json()

        assert len(cats) > 0
        cat_values = [c["value"] for c in cats]
        assert "starbuck" in cat_values
        assert "apollo" in cat_values
        assert "mia" in cat_values

    async def test_get_stats_empty(self, client: AsyncClient) -> None:
        """Test stats with no samples."""
        response = await client.get("/api/stats")
        assert response.status_code == 200
        stats = response.json()

        assert stats["total_samples"] == 0
        assert stats["labeled_samples"] == 0
        assert stats["unlabeled_samples"] == 0

    async def test_get_unlabeled_empty(self, client: AsyncClient) -> None:
        """Test getting unlabeled samples when empty."""
        response = await client.get("/api/samples/unlabeled")
        assert response.status_code == 200
        data = response.json()

        assert data["items"] == []
        assert data["total"] == 0
        assert not data["has_more"]

    async def test_get_sample_not_found(self, client: AsyncClient) -> None:
        """Test getting non-existent sample."""
        response = await client.get("/api/samples/nonexistent-id")
        assert response.status_code == 404

    async def test_label_sample_not_found(self, client: AsyncClient) -> None:
        """Test labeling non-existent sample."""
        response = await client.post(
            "/api/samples/nonexistent-id/label",
            json={"behavior": "cat_eating"},
        )
        assert response.status_code == 404


class TestLabelingAPIWithData:
    """Tests for labeling API with sample data."""

    @pytest.fixture(autouse=True)
    def setup_samples(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Set up sample data before tests."""
        # Create test samples
        for i in range(5):
            storage.save_sample(
                sample_id=f"test-sample-{i}",
                timestamp=datetime(2024, 1, 15, 10 + i, 0, 0),
                camera="apollo-dish",
                frame_data=sample_frame_data,
                bounding_box=sample_bbox,
                frigate_score=0.8 + i * 0.02,
            )

        # Label some samples
        storage.update_labels(
            "test-sample-0",
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
        )
        storage.update_labels(
            "test-sample-1",
            behavior=BehaviorType.DRINKING,
            cat=CatName.APOLLO,
        )

    async def test_get_stats_with_data(self, client: AsyncClient) -> None:
        """Test stats with samples."""
        response = await client.get("/api/stats")
        assert response.status_code == 200
        stats = response.json()

        assert stats["total_samples"] == 5
        assert stats["labeled_samples"] == 2
        assert stats["unlabeled_samples"] == 3

    async def test_get_unlabeled_samples(self, client: AsyncClient) -> None:
        """Test getting unlabeled samples."""
        response = await client.get("/api/samples/unlabeled")
        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 3
        assert data["total"] == 3

    async def test_get_labeled_samples(self, client: AsyncClient) -> None:
        """Test getting labeled samples."""
        response = await client.get("/api/samples/labeled")
        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 2

    async def test_get_labeled_samples_filtered(self, client: AsyncClient) -> None:
        """Test filtering labeled samples."""
        # Filter by behavior
        response = await client.get("/api/samples/labeled?behavior=cat_eating")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["behavior_label"] == "cat_eating"

        # Filter by cat
        response = await client.get("/api/samples/labeled?cat=apollo")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["cat_label"] == "apollo"

    async def test_get_sample(self, client: AsyncClient) -> None:
        """Test getting a specific sample."""
        response = await client.get("/api/samples/test-sample-2")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "test-sample-2"
        assert data["camera"] == "apollo-dish"
        assert not data["is_labeled"]

    async def test_label_sample(self, client: AsyncClient) -> None:
        """Test labeling a sample."""
        response = await client.post(
            "/api/samples/test-sample-2/label",
            json={
                "behavior": "cat_litterbox",
                "cat": "mia",
                "notes": "Test label",
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["behavior_label"] == "cat_litterbox"
        assert data["cat_label"] == "mia"
        assert data["is_labeled"]
        assert data["notes"] == "Test label"

    async def test_skip_sample(self, client: AsyncClient) -> None:
        """Test skipping a sample."""
        response = await client.post("/api/samples/test-sample-3/skip")
        assert response.status_code == 200
        data = response.json()

        assert data["skip_training"]

    async def test_label_with_quality_flags(self, client: AsyncClient) -> None:
        """Test labeling with quality flags."""
        response = await client.post(
            "/api/samples/test-sample-4/label",
            json={
                "behavior": "cat_vomiting",
                "is_blurry": True,
                "is_ambiguous": True,
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert data["behavior_label"] == "cat_vomiting"
        assert data["is_blurry"]
        assert data["is_ambiguous"]

    async def test_get_next_unlabeled(self, client: AsyncClient) -> None:
        """Test getting next unlabeled sample."""
        response = await client.get("/api/next-unlabeled")
        assert response.status_code == 200
        data = response.json()

        assert data is not None
        assert not data["is_labeled"]

    async def test_pagination(self, client: AsyncClient) -> None:
        """Test pagination of samples."""
        response = await client.get("/api/samples/unlabeled?page_size=2")
        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 2
        assert data["page_size"] == 2
        assert data["has_more"]

        # Get next page
        response = await client.get("/api/samples/unlabeled?page=2&page_size=2")
        assert response.status_code == 200
        data = response.json()

        assert len(data["items"]) == 1
        assert not data["has_more"]

    async def test_export_coco(self, client: AsyncClient) -> None:
        """Test COCO format export."""
        response = await client.post(
            "/api/export",
            json={"format": "coco", "output_dir": "test_export"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["format"] == "coco"
        assert data["samples"] == 2  # Only labeled samples

    async def test_export_csv(self, client: AsyncClient) -> None:
        """Test CSV format export."""
        response = await client.post(
            "/api/export",
            json={"format": "csv", "output_dir": "test_csv_export"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["format"] == "csv"
        assert data["samples"] == 2
