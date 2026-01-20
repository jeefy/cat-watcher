"""Cat identification using EfficientNet or ONNX models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cat_watcher.schemas import BoundingBox, CatName


@dataclass
class Identification:
    """A single cat identification result."""

    cat: CatName
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cat": self.cat.value,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class CatIdentifier:
    """EfficientNet-based cat identification.

    Supports both PyTorch (.pt) and ONNX (.onnx) models.
    """

    CAT_CLASSES = [
        CatName.STARBUCK,
        CatName.APOLLO,
        CatName.MIA,
        CatName.UNKNOWN,
    ]

    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.5,
        device: str = "",
        use_onnx: bool | None = None,
        img_size: int = 224,
    ) -> None:
        """Initialize identifier.

        Args:
            model_path: Path to model weights (.pt or .onnx)
            confidence_threshold: Minimum confidence for identification
            device: Device to run on ('cpu', 'cuda', etc.)
            use_onnx: Force ONNX runtime. Auto-detect if None.
            img_size: Input image size
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.img_size = img_size

        # Auto-detect model type
        if use_onnx is None:
            use_onnx = self.model_path.suffix.lower() == ".onnx"
        self.use_onnx = use_onnx

        self._model: Any = None
        self._onnx_session: Any = None
        self._torch_device: Any = None

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
        """Load PyTorch model from checkpoint."""
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "torch required for PyTorch inference. "
                "Install with: pip install torch"
            ) from e

        # Determine device
        if self.device:
            self._torch_device = torch.device(self.device)
        else:
            self._torch_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self._torch_device,
        )

        # Get config from checkpoint
        config = checkpoint.get("config", {})
        model_name = config.get("model_name", "efficientnet_b0")
        num_classes = config.get("num_classes", len(self.CAT_CLASSES))

        # Create model
        from cat_watcher.training.cat_id import CatIDModel

        self._model = CatIDModel(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=False,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model = self._model.to(self._torch_device)
        self._model.eval()

    def identify(
        self,
        image: Image.Image | np.ndarray | Path | str,
        bbox: BoundingBox | None = None,
    ) -> Identification:
        """Identify the cat in an image.

        Args:
            image: PIL Image, numpy array, or path to image
            bbox: Optional bounding box to crop before identification (normalized coords)

        Returns:
            Identification result
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Crop to bounding box if provided (coords are normalized 0-1)
        if bbox is not None:
            # Convert normalized coords to pixel coords with padding
            w, h = image.size
            pad = 20  # pixels
            x1 = max(0, int(bbox.x_min * w) - pad)
            y1 = max(0, int(bbox.y_min * h) - pad)
            x2 = min(w, int(bbox.x_max * w) + pad)
            y2 = min(h, int(bbox.y_max * h) + pad)
            image = image.crop((x1, y1, x2, y2))

        if self.use_onnx:
            return self._identify_onnx(image)
        else:
            return self._identify_pytorch(image)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed numpy array
        """
        # Resize and center crop
        image = self._resize_and_crop(image)

        # Convert to numpy and normalize
        img_array = np.array(image).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        img_array = (img_array - self.MEAN) / self.STD

        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)

        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        """Resize and center crop image.

        Args:
            image: PIL Image

        Returns:
            Processed image
        """
        w, h = image.size

        # Resize so shorter side is img_size
        if w < h:
            new_w = self.img_size
            new_h = int(h * self.img_size / w)
        else:
            new_h = self.img_size
            new_w = int(w * self.img_size / h)

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop
        left = (new_w - self.img_size) // 2
        top = (new_h - self.img_size) // 2
        right = left + self.img_size
        bottom = top + self.img_size

        return image.crop((left, top, right, bottom))

    def _identify_pytorch(self, image: Image.Image) -> Identification:
        """Run identification with PyTorch model."""
        import torch

        if self._model is None:
            self.load()

        # Preprocess
        img_array = self._preprocess(image)
        img_tensor = torch.from_numpy(img_array).to(self._torch_device)

        # Inference
        with torch.no_grad():
            logits = self._model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        probs_np = probs.cpu().numpy()
        return self._create_identification(probs_np)

    def _identify_onnx(self, image: Image.Image) -> Identification:
        """Run identification with ONNX model."""
        if self._onnx_session is None:
            self.load()

        # Preprocess
        img_array = self._preprocess(image)

        # Run inference
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: img_array})

        # Apply softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return self._create_identification(probs)

    def _create_identification(self, probs: np.ndarray) -> Identification:
        """Create Identification from probabilities.

        Args:
            probs: Class probabilities array

        Returns:
            Identification result
        """
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # If confidence is below threshold, return unknown
        if confidence < self.confidence_threshold:
            cat = CatName.UNKNOWN
        else:
            cat = self.CAT_CLASSES[pred_idx]

        return Identification(
            cat=cat,
            confidence=confidence,
            probabilities={
                self.CAT_CLASSES[i].value: float(probs[i])
                for i in range(len(self.CAT_CLASSES))
            },
        )

    def identify_multiple(
        self,
        images: list[Image.Image | np.ndarray],
    ) -> list[Identification]:
        """Identify cats in multiple images (batched).

        Args:
            images: List of images

        Returns:
            List of Identification results
        """
        if not images:
            return []

        # Convert all to PIL
        pil_images: list[Image.Image] = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            pil_images.append(img)

        # Preprocess all
        batch = np.concatenate([
            self._preprocess(img) for img in pil_images
        ], axis=0)

        if self.use_onnx:
            return self._identify_batch_onnx(batch)
        else:
            return self._identify_batch_pytorch(batch)

    def _identify_batch_pytorch(self, batch: np.ndarray) -> list[Identification]:
        """Run batched identification with PyTorch."""
        import torch

        if self._model is None:
            self.load()

        batch_tensor = torch.from_numpy(batch).to(self._torch_device)

        with torch.no_grad():
            logits = self._model(batch_tensor)
            probs = torch.softmax(logits, dim=1)

        results: list[Identification] = []
        for i in range(probs.shape[0]):
            probs_np = probs[i].cpu().numpy()
            results.append(self._create_identification(probs_np))

        return results

    def _identify_batch_onnx(self, batch: np.ndarray) -> list[Identification]:
        """Run batched identification with ONNX."""
        if self._onnx_session is None:
            self.load()

        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: batch})

        results: list[Identification] = []
        for logits in outputs[0]:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            results.append(self._create_identification(probs))

        return results

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None or self._onnx_session is not None
