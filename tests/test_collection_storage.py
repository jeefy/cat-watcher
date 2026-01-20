"""Tests for data collection storage."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from cat_watcher.collection.storage import CollectedSample, FrameStorage
from cat_watcher.schemas import BehaviorType, BoundingBox, CatName


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage(temp_data_dir: Path) -> FrameStorage:
    """Create FrameStorage instance."""
    return FrameStorage(temp_data_dir)


@pytest.fixture
def sample_frame_data() -> bytes:
    """Create minimal valid JPEG data."""
    # Create a simple 1x1 pixel JPEG
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Create sample bounding box."""
    return BoundingBox(x_min=0.2, y_min=0.3, x_max=0.6, y_max=0.8)


class TestFrameStorage:
    """Tests for FrameStorage class."""

    def test_init_creates_directories(self, temp_data_dir: Path) -> None:
        """Test that initialization creates required directories."""
        storage = FrameStorage(temp_data_dir)

        assert storage.frames_dir.exists()
        assert storage.crops_dir.exists()
        assert storage.db_path.exists()

    def test_save_sample(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test saving a new sample."""
        sample = storage.save_sample(
            sample_id="test-event-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            camera="apollo-dish",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.85,
        )

        assert sample.id == "test-event-123"
        assert sample.camera == "apollo-dish"
        assert sample.frigate_score == 0.85
        assert sample.frame_path.exists()
        assert sample.crop_path is not None
        assert sample.crop_path.exists()
        assert not sample.is_labeled

    def test_sample_exists(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test checking if sample exists."""
        assert not storage.sample_exists("nonexistent")

        storage.save_sample(
            sample_id="exists-123",
            timestamp=datetime.now(),
            camera="test",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.9,
        )

        assert storage.sample_exists("exists-123")

    def test_get_sample(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test retrieving a sample by ID."""
        storage.save_sample(
            sample_id="get-test-123",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            camera="test-cam",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.75,
        )

        sample = storage.get_sample("get-test-123")
        assert sample is not None
        assert sample.id == "get-test-123"
        assert sample.camera == "test-cam"
        assert sample.frigate_score == 0.75

        # Non-existent sample
        assert storage.get_sample("nonexistent") is None

    def test_update_labels(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test updating sample labels."""
        storage.save_sample(
            sample_id="label-test-123",
            timestamp=datetime.now(),
            camera="test",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.9,
        )

        # Update labels
        updated = storage.update_labels(
            sample_id="label-test-123",
            behavior=BehaviorType.EATING,
            cat=CatName.STARBUCK,
            notes="Test annotation",
        )

        assert updated is not None
        assert updated.behavior_label == BehaviorType.EATING
        assert updated.cat_label == CatName.STARBUCK
        assert updated.is_labeled
        assert updated.labeled_at is not None
        assert updated.notes == "Test annotation"

    def test_get_unlabeled_samples(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test getting unlabeled samples."""
        # Create some samples
        for i in range(5):
            storage.save_sample(
                sample_id=f"unlabeled-{i}",
                timestamp=datetime.now(),
                camera="test",
                frame_data=sample_frame_data,
                bounding_box=sample_bbox,
                frigate_score=0.9,
            )

        # Label one
        storage.update_labels("unlabeled-2", behavior=BehaviorType.DRINKING)

        unlabeled = storage.get_unlabeled_samples()
        assert len(unlabeled) == 4
        assert all(not s.is_labeled for s in unlabeled)

    def test_get_labeled_samples_with_filter(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test getting labeled samples with filters."""
        # Create and label samples
        for i, (behavior, cat) in enumerate([
            (BehaviorType.EATING, CatName.STARBUCK),
            (BehaviorType.EATING, CatName.APOLLO),
            (BehaviorType.DRINKING, CatName.STARBUCK),
            (BehaviorType.LITTERBOX, CatName.MIA),
        ]):
            storage.save_sample(
                sample_id=f"labeled-{i}",
                timestamp=datetime.now(),
                camera="test",
                frame_data=sample_frame_data,
                bounding_box=sample_bbox,
                frigate_score=0.9,
            )
            storage.update_labels(f"labeled-{i}", behavior=behavior, cat=cat)

        # Filter by behavior
        eating = storage.get_labeled_samples(behavior=BehaviorType.EATING)
        assert len(eating) == 2

        # Filter by cat
        starbuck = storage.get_labeled_samples(cat=CatName.STARBUCK)
        assert len(starbuck) == 2

        # Filter by both
        eating_starbuck = storage.get_labeled_samples(
            behavior=BehaviorType.EATING, cat=CatName.STARBUCK
        )
        assert len(eating_starbuck) == 1

    def test_get_stats(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test getting collection statistics."""
        # Create samples
        for i in range(3):
            storage.save_sample(
                sample_id=f"stats-{i}",
                timestamp=datetime.now(),
                camera="test",
                frame_data=sample_frame_data,
                bounding_box=sample_bbox,
                frigate_score=0.9,
            )

        # Label some
        storage.update_labels("stats-0", behavior=BehaviorType.EATING, cat=CatName.STARBUCK)
        storage.update_labels("stats-1", behavior=BehaviorType.EATING, cat=CatName.APOLLO)

        stats = storage.get_stats()
        assert stats["total_samples"] == 3
        assert stats["labeled_samples"] == 2
        assert stats["unlabeled_samples"] == 1
        assert stats["behavior_distribution"]["cat_eating"] == 2

    def test_export_csv(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
        temp_data_dir: Path,
    ) -> None:
        """Test CSV export."""
        storage.save_sample(
            sample_id="export-1",
            timestamp=datetime.now(),
            camera="test",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.9,
        )
        storage.update_labels("export-1", behavior=BehaviorType.VOMITING, cat=CatName.MIA)

        output_dir = temp_data_dir / "exports"
        result = storage.export_for_training(output_dir, format="csv")

        assert result["format"] == "csv"
        assert result["samples"] == 1
        assert Path(result["path"]).exists()

    def test_skip_training_flag(
        self,
        storage: FrameStorage,
        sample_frame_data: bytes,
        sample_bbox: BoundingBox,
    ) -> None:
        """Test skip_training flag excludes samples."""
        storage.save_sample(
            sample_id="skip-test",
            timestamp=datetime.now(),
            camera="test",
            frame_data=sample_frame_data,
            bounding_box=sample_bbox,
            frigate_score=0.9,
        )
        storage.update_labels("skip-test", skip_training=True)

        unlabeled = storage.get_unlabeled_samples()
        assert len(unlabeled) == 0  # Skipped samples excluded


class TestCollectedSample:
    """Tests for CollectedSample model."""

    def test_model_creation(self, sample_bbox: BoundingBox, temp_data_dir: Path) -> None:
        """Test creating a CollectedSample."""
        sample = CollectedSample(
            id="test-123",
            timestamp=datetime.now(),
            camera="test-cam",
            frame_path=temp_data_dir / "frame.jpg",
            bounding_box=sample_bbox,
            frigate_score=0.88,
            frame_width=1920,
            frame_height=1080,
        )

        assert sample.id == "test-123"
        assert sample.frigate_score == 0.88
        assert not sample.is_labeled
        assert sample.behavior_label is None
