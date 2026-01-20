"""Tests for behavior model trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if torch is not available
pytest.importorskip("torch")

from cat_watcher.training.behavior import (
    BehaviorTrainer,
    BehaviorTrainerConfig,
    train_behavior_model,
)


class TestBehaviorTrainerConfig:
    """Tests for BehaviorTrainerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BehaviorTrainerConfig()

        assert config.model_name == "yolov8n.pt"
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.img_size == 640
        assert config.patience == 20
        assert config.amp is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BehaviorTrainerConfig(
            model_name="yolov8s.pt",
            epochs=50,
            batch_size=8,
            device="cpu",
        )

        assert config.model_name == "yolov8s.pt"
        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.device == "cpu"

    def test_augmentation_defaults(self) -> None:
        """Test augmentation default values."""
        config = BehaviorTrainerConfig()

        assert config.hsv_h == 0.015
        assert config.fliplr == 0.5
        assert config.mosaic == 1.0
        assert config.flipud == 0.0

    def test_export_formats_default(self) -> None:
        """Test export formats default to ONNX."""
        config = BehaviorTrainerConfig()
        assert "onnx" in config.export_formats


class TestBehaviorTrainer:
    """Tests for BehaviorTrainer."""

    def test_init(self) -> None:
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BehaviorTrainer(tmpdir)

            assert trainer.data_dir == Path(tmpdir)
            assert trainer.config is not None
            assert trainer.model is None

    def test_init_with_config(self) -> None:
        """Test trainer initialization with custom config."""
        config = BehaviorTrainerConfig(epochs=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BehaviorTrainer(tmpdir, config)

            assert trainer.config.epochs == 10

    def test_prepare_data_creates_yaml(self) -> None:
        """Test prepare_data creates data.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BehaviorTrainer(tmpdir)
            yaml_path = trainer.prepare_data()

            assert yaml_path.exists()
            assert yaml_path.name == "data.yaml"

    @patch("cat_watcher.training.behavior.YOLO")
    def test_train_calls_yolo(self, mock_yolo_class: MagicMock) -> None:
        """Test that train initializes and trains YOLO model."""
        mock_model = MagicMock()
        mock_model.train.return_value = MagicMock(
            results_dict={"metrics/mAP50(B)": 0.85}
        )
        mock_yolo_class.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BehaviorTrainer(tmpdir)
            results = trainer.train()

            mock_yolo_class.assert_called_once()
            mock_model.train.assert_called_once()
            assert "best_model" in results
            assert "last_model" in results

    @patch("cat_watcher.training.behavior.YOLO")
    def test_validate(self, mock_yolo_class: MagicMock) -> None:
        """Test validation method."""
        mock_model = MagicMock()
        mock_model.val.return_value = MagicMock(
            results_dict={
                "metrics/mAP50(B)": 0.80,
                "metrics/mAP50-95(B)": 0.65,
                "metrics/precision(B)": 0.85,
                "metrics/recall(B)": 0.75,
            }
        )
        mock_yolo_class.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create fake model file
            model_dir = tmpdir / "runs" / "detect" / "behavior" / "weights"
            model_dir.mkdir(parents=True)
            (model_dir / "best.pt").touch()

            trainer = BehaviorTrainer(tmpdir)
            results = trainer.validate(model_dir / "best.pt")

            assert "mAP50" in results
            assert "mAP50-95" in results
            assert "precision" in results
            assert "recall" in results

    @patch("cat_watcher.training.behavior.YOLO")
    def test_export(self, mock_yolo_class: MagicMock) -> None:
        """Test model export."""
        mock_model = MagicMock()
        mock_model.export.return_value = "/tmp/model.onnx"
        mock_yolo_class.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            model_path = tmpdir / "best.pt"
            model_path.touch()

            trainer = BehaviorTrainer(tmpdir)
            exported = trainer.export(model_path, formats=["onnx"])

            mock_model.export.assert_called_once()
            assert "onnx" in exported

    @patch("cat_watcher.training.behavior.YOLO")
    def test_predict(self, mock_yolo_class: MagicMock) -> None:
        """Test prediction method."""
        # Mock detection result
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(item=lambda: 0)]
        mock_box.conf = [MagicMock(item=lambda: 0.95)]
        mock_box.xyxy = [MagicMock(tolist=lambda: [10, 20, 100, 150])]

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__iter__ = lambda s: iter([mock_box])
        mock_result.boxes.__len__ = lambda s: 1
        mock_result.path = "/tmp/test.jpg"

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            model_path = tmpdir / "best.pt"
            model_path.touch()

            # Create test image
            from PIL import Image

            img_path = tmpdir / "test.jpg"
            Image.new("RGB", (640, 480)).save(img_path)

            trainer = BehaviorTrainer(tmpdir)
            results = trainer.predict(img_path, model_path)

            assert len(results) == 1
            assert "detections" in results[0]

    def test_train_without_ultralytics_raises(self) -> None:
        """Test that missing ultralytics raises ImportError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BehaviorTrainer(tmpdir)

            with patch.dict("sys.modules", {"ultralytics": None}), patch(
                "cat_watcher.training.behavior.YOLO",
                side_effect=ImportError("No module named 'ultralytics'"),
            ), pytest.raises(ImportError, match="ultralytics"):
                trainer.train()


class TestTrainBehaviorModelFunction:
    """Tests for train_behavior_model convenience function."""

    @patch("cat_watcher.training.behavior.BehaviorTrainer")
    def test_convenience_function(self, mock_trainer_class: MagicMock) -> None:
        """Test convenience function creates trainer and trains."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_model": "/tmp/best.pt",
            "last_model": "/tmp/last.pt",
        }
        mock_trainer.export.return_value = {"onnx": Path("/tmp/model.onnx")}
        mock_trainer_class.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_behavior_model(
                data_dir=tmpdir,
                epochs=10,
                model_size="s",
            )

            mock_trainer_class.assert_called_once()
            mock_trainer.train.assert_called_once()
            mock_trainer.export.assert_called_once()
            assert "exported" in results

    @patch("cat_watcher.training.behavior.BehaviorTrainer")
    def test_no_export_option(self, mock_trainer_class: MagicMock) -> None:
        """Test skipping ONNX export."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"best_model": "/tmp/best.pt"}
        mock_trainer_class.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_behavior_model(
                data_dir=tmpdir,
                export_onnx=False,
            )

            mock_trainer.export.assert_not_called()
            assert "exported" not in results
