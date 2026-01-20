"""EfficientNet cat identification model trainer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cat_watcher.training.dataset import CatIDDataset


@dataclass
class CatIDTrainerConfig:
    """Configuration for cat ID model training."""

    # Model
    model_name: str = "efficientnet_b0"  # efficientnet_b0 to b7
    pretrained: bool = True
    dropout: float = 0.2

    # Training
    epochs: int = 50
    batch_size: int = 32
    img_size: int = 224
    workers: int = 4

    # Optimization
    lr: float = 0.001
    weight_decay: float = 0.0001
    lr_scheduler: str = "cosine"  # 'cosine' or 'step'
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    warmup_epochs: int = 5

    # Hardware
    device: str = ""  # 'cpu', 'cuda', 'cuda:0', etc. Empty for auto
    amp: bool = True  # Automatic mixed precision

    # Output
    output_dir: str = "runs/catid"
    save_every: int = 5  # Save checkpoint every N epochs

    # Early stopping
    patience: int = 15
    min_delta: float = 0.001


class CatIDModel(nn.Module):
    """EfficientNet-based cat identification model."""

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Initialize model.

        Args:
            num_classes: Number of output classes
            model_name: EfficientNet variant name
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name

        # Try to load from torchvision or timm
        try:
            import timm

            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                drop_rate=dropout,
            )
            self.feature_dim = self.backbone.num_features
        except ImportError:
            # Fallback to torchvision
            import torchvision.models as models

            if model_name == "efficientnet_b0":
                weights = "IMAGENET1K_V1" if pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)
                self.feature_dim = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            elif model_name == "efficientnet_b1":
                weights = "IMAGENET1K_V1" if pretrained else None
                self.backbone = models.efficientnet_b1(weights=weights)
                self.feature_dim = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported model without timm: {model_name}") from None

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input tensor

        Returns:
            Probability tensor
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class CatIDTrainer:
    """Trainer for cat identification model."""

    def __init__(
        self,
        data_dir: Path | str,
        config: CatIDTrainerConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            data_dir: Directory containing training data
            config: Training configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or CatIDTrainerConfig()

        # Set device
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: CatIDModel | None = None
        self.best_accuracy = 0.0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _create_dataloaders(self) -> tuple[DataLoader[Any], DataLoader[Any]]:
        """Create train and validation dataloaders.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = CatIDDataset(
            self.data_dir,
            split="train",
            img_size=self.config.img_size,
            augment=True,
        )
        val_dataset = CatIDDataset(
            self.data_dir,
            split="val",
            img_size=self.config.img_size,
            augment=False,
        )

        train_loader: DataLoader[Any] = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True,
        )
        val_loader: DataLoader[Any] = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train(self) -> dict[str, Any]:
        """Train the cat ID model.

        Returns:
            Training results dictionary
        """
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create model
        self.model = CatIDModel(
            num_classes=CatIDDataset.num_classes(),
            model_name=self.config.model_name,
            pretrained=self.config.pretrained,
            dropout=self.config.dropout,
        )
        self.model = self.model.to(self.device)

        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()

        if len(train_loader) == 0:
            raise ValueError("Training dataset is empty")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs - self.config.warmup_epochs,
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )

        # Mixed precision scaler
        scaler = torch.amp.GradScaler("cuda") if self.config.amp and self.device.type == "cuda" else None

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # Training loop
        for epoch in range(self.config.epochs):
            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                warmup_lr = self.config.lr * (epoch + 1) / self.config.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr

            # Train epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, scaler
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Step scheduler after warmup
            if epoch >= self.config.warmup_epochs:
                scheduler.step()

            # Print progress
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                f"LR: {current_lr:.6f}"
            )

            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(output_dir / "best.pt", epoch, optimizer)

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(
                    output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    epoch,
                    optimizer,
                )

            # Early stopping
            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final model
        self._save_checkpoint(output_dir / "last.pt", self.config.epochs - 1, optimizer)

        # Save training history
        with open(output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return {
            "best_model": str(output_dir / "best.pt"),
            "last_model": str(output_dir / "last.pt"),
            "best_accuracy": self.best_accuracy,
            "history": self.history,
        }

    def _train_epoch(
        self,
        loader: DataLoader[Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
    ) -> tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (loss, accuracy)
        """
        assert self.model is not None
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / total, correct / total

    def _validate(
        self,
        loader: DataLoader[Any],
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Validate model.

        Returns:
            Tuple of (loss, accuracy)
        """
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Save model checkpoint."""
        assert self.model is not None
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": self.best_accuracy,
                "config": {
                    "model_name": self.config.model_name,
                    "num_classes": CatIDDataset.num_classes(),
                    "class_names": CatIDDataset.class_names(),
                    "img_size": self.config.img_size,
                },
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> None:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        config = checkpoint.get("config", {})
        self.model = CatIDModel(
            num_classes=config.get("num_classes", CatIDDataset.num_classes()),
            model_name=config.get("model_name", self.config.model_name),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

    def export_onnx(
        self,
        model_path: Path | str,
        output_path: Path | str | None = None,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            model_path: Path to PyTorch checkpoint
            output_path: Output ONNX file path

        Returns:
            Path to exported ONNX file
        """
        self.load_checkpoint(model_path)
        assert self.model is not None

        if output_path is None:
            output_path = Path(model_path).with_suffix(".onnx")
        output_path = Path(output_path)

        self.model.eval()

        # Dummy input
        dummy_input = torch.randn(1, 3, self.config.img_size, self.config.img_size).to(
            self.device
        )

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        return output_path

    def predict(
        self,
        image_path: Path | str,
        model_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Predict cat identity from image.

        Args:
            image_path: Path to image file
            model_path: Path to model checkpoint

        Returns:
            Prediction result with class name and probabilities
        """
        from PIL import Image

        if model_path:
            self.load_checkpoint(model_path)
        elif self.model is None:
            raise ValueError("Model not loaded. Provide model_path or call load_checkpoint first.")

        assert self.model is not None  # Type narrowing for mypy
        self.model.eval()

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Create a temporary dataset just for preprocessing
        temp_dataset = CatIDDataset(
            self.data_dir,
            split="val",
            img_size=self.config.img_size,
            augment=False,
        )

        # Manually preprocess
        image = temp_dataset._resize_and_crop(image)
        img_tensor = torch.from_numpy(
            __import__("numpy").array(image).transpose(2, 0, 1)
        ).float() / 255.0
        img_tensor = (img_tensor - temp_dataset.mean) / temp_dataset.std
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        class_names = CatIDDataset.class_names()
        pred_idx = int(probs.argmax().item())

        return {
            "predicted_class": class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                name: float(probs[i]) for i, name in enumerate(class_names)
            },
        }


def train_cat_id_model(
    data_dir: Path | str,
    output_dir: Path | str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    model_name: str = "efficientnet_b0",
    device: str = "",
    export_onnx: bool = True,
) -> dict[str, Any]:
    """Convenience function to train cat ID model.

    Args:
        data_dir: Directory with classification training data
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Batch size
        model_name: EfficientNet variant
        device: Device to train on
        export_onnx: Whether to export ONNX model

    Returns:
        Training results
    """
    config = CatIDTrainerConfig(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        output_dir=str(output_dir) if output_dir else "runs/catid",
    )

    trainer = CatIDTrainer(data_dir, config)
    results = trainer.train()

    # Export to ONNX if requested
    if export_onnx:
        onnx_path = trainer.export_onnx(results["best_model"])
        results["onnx_model"] = str(onnx_path)

    return results
