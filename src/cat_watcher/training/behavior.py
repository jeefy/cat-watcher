"""YOLOv8 behavior detection model trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cat_watcher.training.dataset import CatBehaviorDataset, create_data_yaml


@dataclass
class BehaviorTrainerConfig:
    """Configuration for behavior model training."""

    # Model
    model_name: str = "yolov8n.pt"  # nano model for speed, can use s/m/l/x
    img_size: int = 640

    # Training
    epochs: int = 100
    batch_size: int = 16
    patience: int = 20  # Early stopping patience
    workers: int = 0  # 0 for GPU training to avoid multiprocessing issues

    # Optimization
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8

    # Augmentation
    hsv_h: float = 0.015  # HSV-Hue augmentation
    hsv_s: float = 0.7  # HSV-Saturation augmentation
    hsv_v: float = 0.4  # HSV-Value augmentation
    degrees: float = 0.0  # Rotation degrees
    translate: float = 0.1  # Translation
    scale: float = 0.5  # Scale
    shear: float = 0.0  # Shear degrees
    perspective: float = 0.0  # Perspective
    flipud: float = 0.0  # Flip up-down probability
    fliplr: float = 0.5  # Flip left-right probability
    mosaic: float = 1.0  # Mosaic augmentation
    mixup: float = 0.0  # Mixup augmentation

    # Hardware
    device: str = ""  # 'cpu', '0', '0,1', etc. Empty for auto
    amp: bool = False  # Disable AMP - causes issues on older GPUs like GTX 1660

    # Output
    project: str = "runs/detect"
    name: str = "behavior"
    exist_ok: bool = True  # Overwrite existing experiment
    pretrained: bool = True  # Use pretrained weights
    verbose: bool = True

    # Export
    export_formats: list[str] = field(default_factory=lambda: ["onnx"])


class BehaviorTrainer:
    """Trainer for YOLOv8-based cat behavior detection model."""

    def __init__(
        self,
        data_dir: Path | str,
        config: BehaviorTrainerConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            data_dir: Directory containing training data in YOLO format
            config: Training configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or BehaviorTrainerConfig()
        self.model: Any = None
        self.results: Any = None

    def prepare_data(self) -> Path:
        """Prepare data.yaml file for training.

        Returns:
            Path to data.yaml file
        """
        return create_data_yaml(self.data_dir)

    def train(self, on_epoch_end: callable = None) -> dict[str, Any]:
        """Train the behavior detection model.

        Args:
            on_epoch_end: Optional callback called at end of each epoch with trainer instance

        Returns:
            Training results dictionary
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        # Prepare data config
        data_yaml = self.prepare_data()

        # Load model
        self.model = YOLO(self.config.model_name)

        # Add callback if provided
        if on_epoch_end:
            self.model.add_callback("on_train_epoch_end", on_epoch_end)

        # Train
        self.results = self.model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            batch=self.config.batch_size,
            imgsz=self.config.img_size,
            patience=self.config.patience,
            workers=self.config.workers,
            device=self.config.device if self.config.device else None,
            project=self.config.project,
            name=self.config.name,
            exist_ok=self.config.exist_ok,
            pretrained=self.config.pretrained,
            verbose=self.config.verbose,
            amp=self.config.amp,
            # Learning rate
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,
            warmup_momentum=self.config.warmup_momentum,
            # Augmentation
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            degrees=self.config.degrees,
            translate=self.config.translate,
            scale=self.config.scale,
            shear=self.config.shear,
            perspective=self.config.perspective,
            flipud=self.config.flipud,
            fliplr=self.config.fliplr,
            mosaic=self.config.mosaic,
            mixup=self.config.mixup,
        )

        return {
            "best_model": str(Path(self.config.project) / self.config.name / "weights" / "best.pt"),
            "last_model": str(Path(self.config.project) / self.config.name / "weights" / "last.pt"),
            "metrics": self._extract_metrics(),
        }

    def _extract_metrics(self) -> dict[str, float]:
        """Extract training metrics from results.

        Returns:
            Dictionary of metric names to values
        """
        if self.results is None:
            return {}

        try:
            return {
                "mAP50": float(self.results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(self.results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(self.results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(self.results.results_dict.get("metrics/recall(B)", 0)),
            }
        except (AttributeError, KeyError):
            return {}

    def validate(self, model_path: Path | str | None = None) -> dict[str, Any]:
        """Validate model on validation set.

        Args:
            model_path: Path to model weights (uses best.pt if not specified)

        Returns:
            Validation metrics
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        if model_path is None:
            model_path = Path(self.config.project) / self.config.name / "weights" / "best.pt"

        model = YOLO(str(model_path))
        data_yaml = self.data_dir / "data.yaml"

        results = model.val(
            data=str(data_yaml),
            imgsz=self.config.img_size,
            device=self.config.device if self.config.device else None,
            verbose=self.config.verbose,
        )

        return {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            "classes": CatBehaviorDataset.class_names(),
        }

    def export(
        self,
        model_path: Path | str | None = None,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export model to various formats.

        Args:
            model_path: Path to model weights
            formats: List of export formats (onnx, torchscript, etc.)

        Returns:
            Dictionary mapping format to exported file path
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        if model_path is None:
            model_path = Path(self.config.project) / self.config.name / "weights" / "best.pt"

        if formats is None:
            formats = self.config.export_formats

        model = YOLO(str(model_path))
        exported: dict[str, Path] = {}

        for fmt in formats:
            result = model.export(
                format=fmt,
                imgsz=self.config.img_size,
                device=self.config.device if self.config.device else None,
            )
            if result:
                exported[fmt] = Path(result)

        return exported

    def predict(
        self,
        source: Path | str,
        model_path: Path | str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = False,
    ) -> list[dict[str, Any]]:
        """Run inference on images/video.

        Args:
            source: Image, video, or directory path
            model_path: Path to model weights
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save annotated results

        Returns:
            List of detection results
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        if model_path is None:
            model_path = Path(self.config.project) / self.config.name / "weights" / "best.pt"

        model = YOLO(str(model_path))
        results = model.predict(
            source=str(source),
            conf=conf,
            iou=iou,
            imgsz=self.config.img_size,
            device=self.config.device if self.config.device else None,
            save=save,
            verbose=self.config.verbose,
        )

        detections: list[dict[str, Any]] = []
        class_names = CatBehaviorDataset.class_names()

        for result in results:
            frame_detections: list[dict[str, Any]] = []

            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    box = boxes[i]
                    cls_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    frame_detections.append({
                        "class_id": cls_id,
                        "class_name": class_names[cls_id] if cls_id < len(class_names) else "unknown",
                        "confidence": conf_score,
                        "bbox": {
                            "x1": xyxy[0],
                            "y1": xyxy[1],
                            "x2": xyxy[2],
                            "y2": xyxy[3],
                        },
                    })

            detections.append({
                "source": str(result.path) if hasattr(result, "path") else "",
                "detections": frame_detections,
            })

        return detections


def train_behavior_model(
    data_dir: Path | str,
    output_dir: Path | str | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    model_size: str = "n",
    device: str = "",
    export_onnx: bool = True,
) -> dict[str, Any]:
    """Convenience function to train behavior detection model.

    Args:
        data_dir: Directory with YOLO-format training data
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Batch size
        model_size: YOLOv8 model size (n, s, m, l, x)
        device: Device to train on
        export_onnx: Whether to export ONNX model

    Returns:
        Training results
    """
    config = BehaviorTrainerConfig(
        model_name=f"yolov8{model_size}.pt",
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        project=str(output_dir) if output_dir else "runs/detect",
        name="behavior",
        export_formats=["onnx"] if export_onnx else [],
    )

    trainer = BehaviorTrainer(data_dir, config)
    results = trainer.train()

    # Export to ONNX if requested
    if export_onnx:
        exported = trainer.export()
        results["exported"] = {fmt: str(path) for fmt, path in exported.items()}

    return results
