"""Cat detector using YOLOv8 pre-trained model.

Provides Stage 1 detection: finding cats in frames using COCO-pretrained YOLOv8.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from PIL import Image

from cat_watcher.schemas import BoundingBox

logger = structlog.get_logger(__name__)


@dataclass
class CatDetection:
    """A detected cat in a frame."""

    bbox: BoundingBox
    confidence: float
    bbox_pixels: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

    def __repr__(self) -> str:
        return (
            f"CatDetection(conf={self.confidence:.2f}, "
            f"bbox=[{self.bbox.x_min:.3f}, {self.bbox.y_min:.3f}, "
            f"{self.bbox.x_max:.3f}, {self.bbox.y_max:.3f}])"
        )


class CatDetector:
    """Detects cats using YOLOv8 pre-trained model.

    This is Stage 1 of the detection pipeline. It uses a COCO-pretrained
    YOLOv8 model to detect cats (class 15) in frames.

    Example:
        ```python
        detector = CatDetector(confidence_threshold=0.5)
        detector.load()

        # Detect cats in a frame (BGR numpy array from OpenCV)
        detections = detector.detect(frame)
        for det in detections:
            print(f"Cat found at {det.bbox} with confidence {det.confidence}")
        ```
    """

    # COCO class ID for "cat"
    COCO_CAT_CLASS_ID = 15

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        iou_threshold: float = 0.45,
    ):
        """Initialize cat detector.

        Args:
            model_path: Path to YOLO model or model name (e.g., "yolov8n.pt").
                       Will auto-download if not found locally.
            confidence_threshold: Minimum confidence for detections (0-1).
            device: Inference device ("auto", "cuda", or "cpu").
            iou_threshold: IoU threshold for NMS.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.iou_threshold = iou_threshold

        self._model = None
        self._resolved_device: str | None = None

    def load(self) -> None:
        """Load the YOLO model.

        This is called automatically on first detection, but can be called
        explicitly for eager loading.
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package required. Install with: pip install ultralytics"
            ) from e

        logger.info("Loading cat detector model", model=self.model_path)

        self._model = YOLO(self.model_path)

        # Resolve device
        if self.device == "auto":
            import torch

            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.device

        logger.info(
            "Cat detector loaded",
            model=self.model_path,
            device=self._resolved_device,
        )

    def detect(
        self,
        frame: np.ndarray | Image.Image | Path | str,
    ) -> list[CatDetection]:
        """Detect cats in a frame.

        Args:
            frame: Input image as:
                   - BGR numpy array (from OpenCV)
                   - PIL Image
                   - Path to image file
                   - String path to image file

        Returns:
            List of CatDetection objects for detected cats.
        """
        if self._model is None:
            self.load()

        # Get image dimensions
        if isinstance(frame, np.ndarray):
            # OpenCV format: (height, width, channels)
            img_h, img_w = frame.shape[:2]
        elif isinstance(frame, Image.Image):
            img_w, img_h = frame.size
        elif isinstance(frame, (Path, str)):
            # Load image to get dimensions
            img = Image.open(frame)
            img_w, img_h = img.size
            frame = img
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")

        # Run inference
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.COCO_CAT_CLASS_ID],  # Only detect cats
            device=self._resolved_device,
            verbose=False,
        )

        detections: list[CatDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()

                # Should only be cats due to class filter, but double-check
                if cls_id != self.COCO_CAT_CLASS_ID:
                    continue

                # Pixel coordinates
                x_min_px = int(xyxy[0])
                y_min_px = int(xyxy[1])
                x_max_px = int(xyxy[2])
                y_max_px = int(xyxy[3])

                # Normalize to 0-1 range
                bbox = BoundingBox(
                    x_min=xyxy[0] / img_w,
                    y_min=xyxy[1] / img_h,
                    x_max=xyxy[2] / img_w,
                    y_max=xyxy[3] / img_h,
                )

                detections.append(
                    CatDetection(
                        bbox=bbox,
                        confidence=conf,
                        bbox_pixels=(x_min_px, y_min_px, x_max_px, y_max_px),
                    )
                )

        logger.debug(
            "Cat detection complete",
            num_detections=len(detections),
            image_size=f"{img_w}x{img_h}",
        )

        return detections

    def detect_and_annotate(
        self,
        frame: np.ndarray | Image.Image | Path | str,
        output_path: Path | str | None = None,
        box_color: tuple[int, int, int] = (0, 255, 0),  # Green in BGR
        box_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> tuple[list[CatDetection], np.ndarray]:
        """Detect cats and draw bounding boxes on the frame.

        Args:
            frame: Input image (same formats as detect())
            output_path: Optional path to save annotated image
            box_color: BGR color for bounding boxes
            box_thickness: Line thickness for boxes
            font_scale: Font scale for labels

        Returns:
            Tuple of (detections, annotated_frame as BGR numpy array)
        """
        import cv2

        # Ensure we have a numpy array
        if isinstance(frame, (Path, str)):
            frame = cv2.imread(str(frame))
        elif isinstance(frame, Image.Image):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Make a copy to annotate
        annotated = frame.copy()

        # Detect
        detections = self.detect(frame)

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = det.bbox_pixels

            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

            # Draw label
            label = f"cat {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Background for label
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                box_color,
                -1,
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                1,
            )

        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), annotated)
            logger.info("Saved annotated image", path=str(output_path))

        return detections, annotated

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def resolved_device(self) -> str | None:
        """Get the resolved device (after loading)."""
        return self._resolved_device
