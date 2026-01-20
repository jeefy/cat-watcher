"""Tests for cat ID model trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from cat_watcher.training.cat_id import (
    CatIDModel,
    CatIDTrainer,
    CatIDTrainerConfig,
    train_cat_id_model,
)


class TestCatIDTrainerConfig:
    """Tests for CatIDTrainerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CatIDTrainerConfig()

        assert config.model_name == "efficientnet_b0"
        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.img_size == 224
        assert config.pretrained is True
        assert config.amp is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CatIDTrainerConfig(
            model_name="efficientnet_b2",
            epochs=25,
            batch_size=16,
            device="cpu",
        )

        assert config.model_name == "efficientnet_b2"
        assert config.epochs == 25
        assert config.batch_size == 16
        assert config.device == "cpu"

    def test_learning_rate_config(self) -> None:
        """Test learning rate configuration."""
        config = CatIDTrainerConfig(
            lr=0.0001,
            lr_scheduler="step",
            lr_step_size=5,
            lr_gamma=0.5,
        )

        assert config.lr == 0.0001
        assert config.lr_scheduler == "step"
        assert config.lr_step_size == 5
        assert config.lr_gamma == 0.5


class TestCatIDModel:
    """Tests for CatIDModel."""

    def test_model_creation(self) -> None:
        """Test model can be created."""
        model = CatIDModel(num_classes=4, pretrained=False)

        assert model.num_classes == 4
        assert model.classifier is not None

    def test_forward_pass(self) -> None:
        """Test forward pass produces correct output shape."""
        model = CatIDModel(num_classes=4, pretrained=False)
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 4)

    def test_predict_proba(self) -> None:
        """Test predict_proba returns valid probabilities."""
        model = CatIDModel(num_classes=4, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        probs = model.predict_proba(x)

        assert probs.shape == (1, 4)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)
        # All probabilities should be >= 0
        assert (probs >= 0).all()

    def test_different_model_sizes(self) -> None:
        """Test creating different model variants."""
        for model_name in ["efficientnet_b0", "efficientnet_b1"]:
            model = CatIDModel(model_name=model_name, pretrained=False)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, 4)


class TestCatIDTrainer:
    """Tests for CatIDTrainer."""

    def test_init(self) -> None:
        """Test trainer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CatIDTrainer(tmpdir)

            assert trainer.data_dir == Path(tmpdir)
            assert trainer.config is not None
            assert trainer.model is None
            assert trainer.best_accuracy == 0.0

    def test_init_with_config(self) -> None:
        """Test trainer initialization with custom config."""
        config = CatIDTrainerConfig(epochs=10, device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CatIDTrainer(tmpdir, config)

            assert trainer.config.epochs == 10
            assert str(trainer.device) == "cpu"

    def test_device_auto_selection(self) -> None:
        """Test automatic device selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CatIDTrainer(tmpdir)

            # Should be either cuda or cpu
            assert trainer.device.type in ["cuda", "cpu"]

    def test_history_initialization(self) -> None:
        """Test training history is initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CatIDTrainer(tmpdir)

            assert "train_loss" in trainer.history
            assert "train_acc" in trainer.history
            assert "val_loss" in trainer.history
            assert "val_acc" in trainer.history

    def test_save_and_load_checkpoint(self) -> None:
        """Test saving and loading model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config = CatIDTrainerConfig(device="cpu")

            trainer = CatIDTrainer(tmpdir, config)
            trainer.model = CatIDModel(num_classes=4, pretrained=False)
            trainer.best_accuracy = 0.85

            # Save checkpoint
            checkpoint_path = tmpdir / "test_checkpoint.pt"
            optimizer = torch.optim.Adam(trainer.model.parameters())
            trainer._save_checkpoint(checkpoint_path, epoch=5, optimizer=optimizer)

            assert checkpoint_path.exists()

            # Load checkpoint
            trainer2 = CatIDTrainer(tmpdir, config)
            trainer2.load_checkpoint(checkpoint_path)

            assert trainer2.model is not None
            assert trainer2.best_accuracy == 0.85

    def test_export_onnx(self) -> None:
        """Test ONNX export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            config = CatIDTrainerConfig(device="cpu")

            trainer = CatIDTrainer(tmpdir, config)
            trainer.model = CatIDModel(num_classes=4, pretrained=False)

            # Save a checkpoint first
            checkpoint_path = tmpdir / "model.pt"
            optimizer = torch.optim.Adam(trainer.model.parameters())
            trainer._save_checkpoint(checkpoint_path, epoch=1, optimizer=optimizer)

            # Export to ONNX
            onnx_path = trainer.export_onnx(checkpoint_path)

            assert onnx_path.exists()
            assert onnx_path.suffix == ".onnx"

    def test_train_empty_dataset_raises(self) -> None:
        """Test training with empty dataset raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CatIDTrainerConfig(device="cpu", epochs=1)
            trainer = CatIDTrainer(tmpdir, config)

            with pytest.raises(ValueError, match="empty"):
                trainer.train()


class TestTrainerTrainEpoch:
    """Tests for training epoch methods."""

    def test_train_epoch_updates_model(self) -> None:
        """Test that train epoch updates model parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal dataset
            (tmpdir / "train" / "starbuck").mkdir(parents=True)
            (tmpdir / "val" / "starbuck").mkdir(parents=True)

            from PIL import Image

            for i in range(4):
                img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
                img.save(tmpdir / "train" / "starbuck" / f"img{i}.jpg")
                img.save(tmpdir / "val" / "starbuck" / f"img{i}.jpg")

            config = CatIDTrainerConfig(
                device="cpu",
                epochs=1,
                batch_size=2,
                workers=0,
                pretrained=False,
            )

            trainer = CatIDTrainer(tmpdir, config)

            # Get initial parameters
            trainer.model = CatIDModel(num_classes=4, pretrained=False)
            initial_params = {
                name: param.clone()
                for name, param in trainer.model.named_parameters()
            }

            # Create dataloader and train
            train_loader, _ = trainer._create_dataloaders()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(trainer.model.parameters())

            loss, acc = trainer._train_epoch(train_loader, criterion, optimizer, None)

            # Check some parameters changed
            params_changed = False
            for name, param in trainer.model.named_parameters():
                if not torch.allclose(param, initial_params[name]):
                    params_changed = True
                    break

            assert params_changed
            assert isinstance(loss, float)
            assert isinstance(acc, float)


class TestTrainCatIDModelFunction:
    """Tests for train_cat_id_model convenience function."""

    @patch("cat_watcher.training.cat_id.CatIDTrainer")
    def test_convenience_function(self, mock_trainer_class: MagicMock) -> None:
        """Test convenience function creates trainer and trains."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "best_model": "/tmp/best.pt",
            "best_accuracy": 0.92,
        }
        mock_trainer.export_onnx.return_value = Path("/tmp/model.onnx")
        mock_trainer_class.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_cat_id_model(
                data_dir=tmpdir,
                epochs=10,
                model_name="efficientnet_b0",
            )

            mock_trainer_class.assert_called_once()
            mock_trainer.train.assert_called_once()
            mock_trainer.export_onnx.assert_called_once()
            assert "onnx_model" in results

    @patch("cat_watcher.training.cat_id.CatIDTrainer")
    def test_no_export_option(self, mock_trainer_class: MagicMock) -> None:
        """Test skipping ONNX export."""
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"best_model": "/tmp/best.pt"}
        mock_trainer_class.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_cat_id_model(
                data_dir=tmpdir,
                export_onnx=False,
            )

            mock_trainer.export_onnx.assert_not_called()
            assert "onnx_model" not in results
