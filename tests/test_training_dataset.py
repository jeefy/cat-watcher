"""Tests for training dataset classes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from cat_watcher.training.dataset import (
    CatBehaviorDataset,
    CatIDDataset,
    create_data_yaml,
    split_cat_id_dataset,
    split_dataset,
)


class TestCatBehaviorDataset:
    """Tests for CatBehaviorDataset."""

    def test_class_names(self) -> None:
        """Test class names are correct."""
        names = CatBehaviorDataset.class_names()
        assert "cat_eating" in names
        assert "cat_drinking" in names
        assert "cat_vomiting" in names
        assert "cat_waiting" in names
        assert "cat_litterbox" in names
        assert "cat_yowling" in names
        assert "cat_sleeping" in names
        assert "cat_present" in names

    def test_num_classes(self) -> None:
        """Test number of classes."""
        assert CatBehaviorDataset.num_classes() == 8

    def test_empty_dataset(self) -> None:
        """Test behavior with non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CatBehaviorDataset(tmpdir, split="train")
            assert len(dataset) == 0

    def test_loads_images(self) -> None:
        """Test loading images from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure
            images_dir = tmpdir / "images" / "train"
            labels_dir = tmpdir / "labels" / "train"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            # Create a simple image
            from PIL import Image

            img = Image.new("RGB", (640, 480), color=(255, 0, 0))
            img.save(images_dir / "test.jpg")

            # Create label file
            (labels_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.2\n")

            # Load dataset
            dataset = CatBehaviorDataset(tmpdir, split="train", augment=False)
            assert len(dataset) == 1

            # Get sample
            img_tensor, target = dataset[0]
            assert img_tensor.shape == (3, 640, 640)
            assert len(target["boxes"]) == 1
            assert len(target["labels"]) == 1

    def test_augmentation_flag(self) -> None:
        """Test augmentation only applies to train split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_ds = CatBehaviorDataset(tmpdir, split="train", augment=True)
            val_ds = CatBehaviorDataset(tmpdir, split="val", augment=True)

            assert train_ds.augment is True
            assert val_ds.augment is False


class TestCatIDDataset:
    """Tests for CatIDDataset."""

    def test_class_names(self) -> None:
        """Test class names are correct."""
        names = CatIDDataset.class_names()
        assert "starbuck" in names
        assert "apollo" in names
        assert "mia" in names
        assert "unknown" in names

    def test_num_classes(self) -> None:
        """Test number of classes."""
        assert CatIDDataset.num_classes() == 4

    def test_empty_dataset(self) -> None:
        """Test behavior with non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CatIDDataset(tmpdir, split="train")
            assert len(dataset) == 0

    def test_loads_images_by_class(self) -> None:
        """Test loading images organized by class."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create directory structure for classification
            for split in ["train", "val"]:
                for cat in ["starbuck", "apollo", "mia"]:
                    (tmpdir / split / cat).mkdir(parents=True)

            # Create test images
            from PIL import Image

            img = Image.new("RGB", (224, 224), color=(0, 255, 0))
            img.save(tmpdir / "train" / "starbuck" / "img1.jpg")
            img.save(tmpdir / "train" / "apollo" / "img2.jpg")
            img.save(tmpdir / "val" / "mia" / "img3.jpg")

            # Load datasets
            train_ds = CatIDDataset(tmpdir, split="train", augment=False)
            val_ds = CatIDDataset(tmpdir, split="val", augment=False)

            assert len(train_ds) == 2
            assert len(val_ds) == 1

            # Check sample format
            img_tensor, label = train_ds[0]
            assert img_tensor.shape == (3, 224, 224)
            assert isinstance(label, int)
            assert 0 <= label < 4

    def test_normalization(self) -> None:
        """Test ImageNet normalization is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "train" / "starbuck").mkdir(parents=True)

            from PIL import Image

            img = Image.new("RGB", (224, 224), color=(127, 127, 127))
            img.save(tmpdir / "train" / "starbuck" / "test.jpg")

            dataset = CatIDDataset(tmpdir, split="train", augment=False)
            img_tensor, _ = dataset[0]

            # Values should be normalized (not in [0, 1] range)
            # Check that mean is close to 0 (normalized)
            assert img_tensor.mean().abs() < 1.0


class TestDataYaml:
    """Tests for data.yaml creation."""

    def test_create_data_yaml(self) -> None:
        """Test creating data.yaml file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            yaml_path = create_data_yaml(tmpdir)

            assert yaml_path.exists()
            content = yaml_path.read_text()

            assert "train: images/train" in content
            assert "val: images/val" in content
            assert "nc: 8" in content
            assert "cat_eating" in content
            assert "cat_sleeping" in content

    def test_custom_output_path(self) -> None:
        """Test custom output path for data.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            custom_path = tmpdir / "custom" / "my_data.yaml"

            yaml_path = create_data_yaml(tmpdir, custom_path)

            assert yaml_path == custom_path
            assert custom_path.exists()


class TestSplitDataset:
    """Tests for dataset splitting functions."""

    def test_split_empty_db(self) -> None:
        """Test splitting with empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"

            # Create empty database
            import sqlite3

            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE samples (
                    id INTEGER PRIMARY KEY,
                    image_path TEXT,
                    behavior_label TEXT,
                    cat_label TEXT,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL
                )
            """)
            conn.close()

            output_dir = tmpdir / "output"
            stats = split_dataset(db_path, output_dir)

            assert stats["train"] == 0
            assert stats["val"] == 0
            assert stats["total"] == 0

    def test_split_cat_id_empty_db(self) -> None:
        """Test cat ID splitting with empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            db_path = tmpdir / "test.db"

            import sqlite3

            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE samples (
                    id INTEGER PRIMARY KEY,
                    image_path TEXT,
                    behavior_label TEXT,
                    cat_label TEXT,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL
                )
            """)
            conn.close()

            output_dir = tmpdir / "output"
            stats = split_cat_id_dataset(db_path, output_dir)

            assert stats["train"] == 0
            assert stats["val"] == 0
            assert stats["total"] == 0


class TestBehaviorDatasetAugmentation:
    """Tests for augmentation in behavior dataset."""

    def test_horizontal_flip_boxes(self) -> None:
        """Test horizontal flip transforms boxes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            from PIL import Image

            img = Image.new("RGB", (100, 100))
            boxes = [[10, 20, 30, 40]]  # x1, y1, x2, y2

            dataset = CatBehaviorDataset(tmpdir, split="train", augment=True)

            # Mock random to always flip
            with patch("random.random", return_value=0.0):
                flipped_img, flipped_boxes = dataset._augment(img, boxes)

            # After horizontal flip, x coordinates should be mirrored
            # x1' = width - x2 = 100 - 30 = 70
            # x2' = width - x1 = 100 - 10 = 90
            assert flipped_boxes[0][0] == 70  # new x1
            assert flipped_boxes[0][2] == 90  # new x2
            # y coordinates unchanged
            assert flipped_boxes[0][1] == 20
            assert flipped_boxes[0][3] == 40


class TestCatIDDatasetPreprocessing:
    """Tests for preprocessing in cat ID dataset."""

    def test_resize_and_crop(self) -> None:
        """Test resize and center crop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            from PIL import Image

            # Create a rectangular image
            img = Image.new("RGB", (300, 200), color=(100, 150, 200))

            dataset = CatIDDataset(tmpdir, split="train", img_size=224)
            result = dataset._resize_and_crop(img)

            assert result.size == (224, 224)

    def test_resize_tall_image(self) -> None:
        """Test resize with tall image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            from PIL import Image

            img = Image.new("RGB", (200, 300))

            dataset = CatIDDataset(tmpdir, split="train", img_size=224)
            result = dataset._resize_and_crop(img)

            assert result.size == (224, 224)
