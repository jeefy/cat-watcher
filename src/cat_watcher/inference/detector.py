"""Behavior detection using YOLOv8 or ONNX models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cat_watcher.schemas import BehaviorType, BoundingBox


@dataclass
class Detection:
    """A single behavior detection result."""

    behavior: BehaviorType
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "behavior": self.behavior.value,
            "confidence": self.confidence,
            "bbox": {
                "x_min": self.bbox.x_min,
                "y_min": self.bbox.y_min,
                "x_max": self.bbox.x_max,
                "y_max": self.bbox.y_max,
            },
        }


class BehaviorDetector:
    """YOLOv8-based behavior detection.

    Supports both PyTorch (.pt) and ONNX (.onnx) models.
    """

    BEHAVIOR_CLASSES = [
        BehaviorType.EATING,
        BehaviorType.DRINKING,
        BehaviorType.VOMITING,
        BehaviorType.WAITING,
        BehaviorType.LITTERBOX,
        BehaviorType.YOWLING,
        BehaviorType.PRESENT,
    ]

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "",
        use_onnx: bool | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            model_path: Path to model weights (.pt or .onnx)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cpu', 'cuda', etc.)
            use_onnx: Force ONNX runtime. Auto-detect if None.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Auto-detect model type
        if use_onnx is None:
            use_onnx = self.model_path.suffix.lower() == ".onnx"
        self.use_onnx = use_onnx

        self._model: Any = None
        self._onnx_session: Any = None
        self._img_size = 640

    def load(self) -> None:
        """Load the model."""
        if self.use_onnx:
            self._load_onnx()
        else:
            self._load_pytorch()

    def _load_onnx(self) -> None:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime required for ONNX inference. "
                "Install with: pip install onnxruntime-gpu"
            ) from e

        # Select execution provider
        providers = ["CPUExecutionProvider"]
        if self.device != "cpu":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._onnx_session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

    def _load_pytorch(self) -> None:
        """Load PyTorch/Ultralytics model."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics required for PyTorch inference. "
                "Install with: pip install ultralytics"
            ) from e

        self._model = YOLO(str(self.model_path))

    def detect(
        self,
        image: Image.Image | np.ndarray | Path | str,
    ) -> list[Detection]:
        """Run detection on an image.

        Args:
            image: PIL Image, numpy array, or path to image

        Returns:
            List of Detection objects
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.use_onnx:
            return self._detect_onnx(image)
        else:
            return self._detect_pytorch(image)

    def _detect_pytorch(self, image: Image.Image) -> list[Detection]:
        """Run detection with PyTorch model."""
        if self._model is None:
            self.load()

        # Get image dimensions for normalization
        img_w, img_h = image.size

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device if self.device else None,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()

                if cls_id < len(self.BEHAVIOR_CLASSES):
                    # Convert to normalized coordinates (0-1)
                    bbox = BoundingBox(
                        x_min=float(xyxy[0]) / img_w,
                        y_min=float(xyxy[1]) / img_h,
                        x_max=float(xyxy[2]) / img_w,
                        y_max=float(xyxy[3]) / img_h,
                    )
                    detections.append(Detection(
                        behavior=self.BEHAVIOR_CLASSES[cls_id],
                        confidence=conf,
                        bbox=bbox,
                    ))

        return detections

    def _detect_onnx(self, image: Image.Image) -> list[Detection]:
        """Run detection with ONNX model."""
        if self._onnx_session is None:
            self.load()

        # Preprocess image
        orig_w, orig_h = image.size
        img_resized = image.resize((self._img_size, self._img_size))

        # Convert to numpy and normalize
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

        # Run inference
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: img_array})

        # Parse YOLO output
        # YOLOv8 output shape: [1, num_classes + 4, num_predictions]
        # or [1, num_predictions, num_classes + 4] depending on export
        output = outputs[0]

        detections = self._parse_yolo_output(output)

        return detections

    def _parse_yolo_output(
        self,
        output: np.ndarray,
    ) -> list[Detection]:
        """Parse YOLOv8 ONNX output.

        Args:
            output: Raw model output

        Returns:
            List of detections (with normalized coordinates)
        """
        # Handle different output formats
        if output.ndim == 3:
            output = output[0]  # Remove batch dim

        # YOLOv8 outputs [4 + num_classes, num_predictions] or transposed
        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to [num_predictions, 4 + num_classes]

        num_classes = len(self.BEHAVIOR_CLASSES)

        detections: list[Detection] = []

        for pred in output:
            # First 4 values are bbox (x_center, y_center, width, height) in img_size coords
            x_center, y_center, w, h = pred[:4]
            class_scores = pred[4:4 + num_classes]

            # Get best class
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.confidence_threshold:
                continue

            # Convert to normalized coordinates (0-1)
            # First convert center format to corner format in img_size space
            x1 = (x_center - w / 2) / self._img_size
            y1 = (y_center - h / 2) / self._img_size
            x2 = (x_center + w / 2) / self._img_size
            y2 = (y_center + h / 2) / self._img_size

            bbox = BoundingBox(
                x_min=max(0.0, min(1.0, float(x1))),
                y_min=max(0.0, min(1.0, float(y1))),
                x_max=max(0.0, min(1.0, float(x2))),
                y_max=max(0.0, min(1.0, float(y2))),
            )

            detections.append(Detection(
                behavior=self.BEHAVIOR_CLASSES[cls_id],
                confidence=conf,
                bbox=bbox,
            ))

        # Apply NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression.

        Args:
            detections: List of detections

        Returns:
            Filtered detections
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep: list[Detection] = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Filter out overlapping detections
            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self.iou_threshold
            ]

        return keep

    @staticmethod
    def _iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU between two boxes (using normalized coordinates)."""
        x1 = max(box1.x_min, box2.x_min)
        y1 = max(box1.y_min, box2.y_min)
        x2 = min(box1.x_max, box2.x_max)
        y2 = min(box1.y_max, box2.y_max)

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
        area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None or self._onnx_session is not None
