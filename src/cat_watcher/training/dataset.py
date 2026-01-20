"""Dataset classes for training behavior and cat ID models."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from cat_watcher.schemas import BehaviorType, CatName


class CatBehaviorDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    """YOLO-format dataset for behavior detection.

    Expects directory structure:
        data_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
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
        data_dir: Path | str,
        split: str = "train",
        img_size: int = 640,
        augment: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Root directory containing images/ and labels/
            split: Either 'train' or 'val'
            img_size: Target image size
            augment: Whether to apply augmentations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"

        self.images_dir = self.data_dir / "images" / split
        self.labels_dir = self.data_dir / "labels" / split

        # Find all images
        self.image_files: list[Path] = []
        if self.images_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                self.image_files.extend(self.images_dir.glob(ext))
        self.image_files.sort()

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, target_dict) where target_dict contains
            boxes, labels, and original image size
        """
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load YOLO labels (class x_center y_center width height)
        boxes: list[list[float]] = []
        labels: list[int] = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # Convert from YOLO format (normalized center) to pixel coords
                        x1 = (x_center - width / 2) * orig_w
                        y1 = (y_center - height / 2) * orig_h
                        x2 = (x_center + width / 2) * orig_w
                        y2 = (y_center + height / 2) * orig_h

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)

        # Apply augmentations
        if self.augment:
            image, boxes = self._augment(image, boxes)

        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        # Scale boxes to new size
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        scaled_boxes = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            for b in boxes
        ]

        # Convert to tensors
        img_tensor = torch.from_numpy(
            __import__("numpy").array(image).transpose(2, 0, 1)
        ).float() / 255.0

        target = {
            "boxes": torch.tensor(scaled_boxes, dtype=torch.float32)
            if scaled_boxes
            else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros(0, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }

        return img_tensor, target

    def _augment(
        self, image: Image.Image, boxes: list[list[float]]
    ) -> tuple[Image.Image, list[list[float]]]:
        """Apply random augmentations.

        Args:
            image: PIL Image
            boxes: List of [x1, y1, x2, y2] boxes

        Returns:
            Augmented image and boxes
        """
        orig_w, orig_h = image.size

        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[orig_w - b[2], b[1], orig_w - b[0], b[3]] for b in boxes]

        # Random brightness/contrast (simple version)
        if random.random() > 0.5:
            from PIL import ImageEnhance

            # Brightness
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(random.uniform(0.8, 1.2))

            # Contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(random.uniform(0.8, 1.2))

        return image, boxes

    @classmethod
    def class_names(cls) -> list[str]:
        """Return list of class names."""
        return [b.value for b in cls.BEHAVIOR_CLASSES]

    @classmethod
    def num_classes(cls) -> int:
        """Return number of classes."""
        return len(cls.BEHAVIOR_CLASSES)


class CatIDDataset(Dataset[tuple[torch.Tensor, int]]):
    """Classification dataset for cat identification.

    Expects directory structure:
        data_dir/
            train/
                starbuck/
                apollo/
                mia/
                unknown/
            val/
                starbuck/
                ...
    """

    CAT_CLASSES = [
        CatName.STARBUCK,
        CatName.APOLLO,
        CatName.MIA,
        CatName.UNKNOWN,
    ]

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            data_dir: Root directory containing train/ and val/ subdirs
            split: Either 'train' or 'val'
            img_size: Target image size
            augment: Whether to apply augmentations
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"

        self.split_dir = self.data_dir / split

        # Build class to index mapping
        self.class_to_idx = {cat.value: i for i, cat in enumerate(self.CAT_CLASSES)}

        # Find all images with their labels
        self.samples: list[tuple[Path, int]] = []
        if self.split_dir.exists():
            for cat in self.CAT_CLASSES:
                cat_dir = self.split_dir / cat.value
                if cat_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png"]:
                        for img_path in cat_dir.glob(ext):
                            self.samples.append(
                                (img_path, self.class_to_idx[cat.value])
                            )
        self.samples.sort(key=lambda x: x[0])

        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, class_label)
        """
        img_path, label = self.samples[idx]

        # Load and convert image
        image = Image.open(img_path).convert("RGB")

        # Apply augmentations
        if self.augment:
            image = self._augment(image)

        # Resize and center crop
        image = self._resize_and_crop(image)

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(
            __import__("numpy").array(image).transpose(2, 0, 1)
        ).float() / 255.0

        # Normalize with ImageNet stats
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor, label

    def _augment(self, image: Image.Image) -> Image.Image:
        """Apply random augmentations for classification.

        Args:
            image: PIL Image

        Returns:
            Augmented image
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

        # Color jitter
        if random.random() > 0.5:
            from PIL import ImageEnhance

            # Brightness
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(random.uniform(0.8, 1.2))

            # Contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(random.uniform(0.8, 1.2))

            # Saturation
            saturation_enhancer = ImageEnhance.Color(image)
            image = saturation_enhancer.enhance(random.uniform(0.8, 1.2))

        return image

    def _resize_and_crop(self, image: Image.Image) -> Image.Image:
        """Resize and center crop image.

        Args:
            image: PIL Image

        Returns:
            Resized and cropped image
        """
        # Resize so shorter side is img_size
        w, h = image.size
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

    @classmethod
    def class_names(cls) -> list[str]:
        """Return list of class names."""
        return [c.value for c in cls.CAT_CLASSES]

    @classmethod
    def num_classes(cls) -> int:
        """Return number of classes."""
        return len(cls.CAT_CLASSES)


def create_data_yaml(data_dir: Path, output_path: Path | None = None) -> Path:
    """Create YOLO data.yaml file for training.

    Args:
        data_dir: Root data directory
        output_path: Optional output path for yaml file

    Returns:
        Path to created yaml file
    """
    if output_path is None:
        output_path = data_dir / "data.yaml"

    yaml_content = f"""# Cat Behavior Detection Dataset
path: {data_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: {CatBehaviorDataset.num_classes()}
names: {CatBehaviorDataset.class_names()}
"""

    output_path.write_text(yaml_content)
    return output_path


def split_dataset(
    storage_db: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    _format_type: str = "yolo",
) -> dict[str, int]:
    """Split labeled data into train/val sets.

    Args:
        storage_db: Path to SQLite database from labeling
        output_dir: Output directory for split data
        val_ratio: Fraction of data for validation
        format_type: Either 'yolo' or 'coco'

    Returns:
        Dictionary with split statistics
    """
    import sqlite3

    conn = sqlite3.connect(storage_db)
    cursor = conn.cursor()

    # Get all labeled samples - use actual schema columns
    cursor.execute("""
        SELECT id, frame_path, behavior_label, cat_label, 
               bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
               frame_width, frame_height
        FROM samples
        WHERE (behavior_label IS NOT NULL OR cat_label IS NOT NULL)
          AND is_labeled = 1
          AND skip_training = 0
    """)
    samples = cursor.fetchall()
    conn.close()

    if not samples:
        return {"train": 0, "val": 0, "total": 0}

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Create directories
    output_dir = Path(output_dir)
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Get behavior label to class index mapping
    behavior_to_idx = {b.value: i for i, b in enumerate(CatBehaviorDataset.BEHAVIOR_CLASSES)}

    skipped_samples = []

    def process_samples(samples_list: list[Any], split: str) -> int:
        """Process and save samples."""
        count = 0
        for sample in samples_list:
            (sample_id, img_path, behavior, cat, 
             bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
             frame_width, frame_height) = sample
            img_path = Path(img_path)

            if not img_path.exists():
                skipped_samples.append((sample_id, "image not found"))
                continue

            # Validate and fix bounding box coordinates
            # Handle swapped min/max (common Frigate issue)
            if bbox_x_min > bbox_x_max:
                bbox_x_min, bbox_x_max = bbox_x_max, bbox_x_min
            if bbox_y_min > bbox_y_max:
                bbox_y_min, bbox_y_max = bbox_y_max, bbox_y_min
            
            # Clamp to valid 0-1 range
            bbox_x_min = max(0.0, min(1.0, bbox_x_min))
            bbox_y_min = max(0.0, min(1.0, bbox_y_min))
            bbox_x_max = max(0.0, min(1.0, bbox_x_max))
            bbox_y_max = max(0.0, min(1.0, bbox_y_max))
            
            # Calculate YOLO format values
            norm_w = bbox_x_max - bbox_x_min
            norm_h = bbox_y_max - bbox_y_min
            x_center = bbox_x_min + norm_w / 2
            y_center = bbox_y_min + norm_h / 2
            
            # Skip if box is too small or invalid after fixing
            if norm_w < 0.01 or norm_h < 0.01:
                skipped_samples.append((sample_id, f"bbox too small: {norm_w:.3f}x{norm_h:.3f}"))
                continue
            
            # Final validation - all values must be in [0, 1]
            if not all(0 <= v <= 1 for v in [x_center, y_center, norm_w, norm_h]):
                skipped_samples.append((sample_id, f"invalid coords after fix: center=({x_center:.3f},{y_center:.3f}) size=({norm_w:.3f}x{norm_h:.3f})"))
                continue

            # Copy image
            import shutil
            dest_img = output_dir / "images" / split / img_path.name
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)

            # Create YOLO label file
            if behavior:
                label_path = output_dir / "labels" / split / f"{img_path.stem}.txt"

                class_idx = behavior_to_idx.get(behavior, 0)

                # Append to label file (supports multiple objects)
                with open(label_path, "a") as f:
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            count += 1
        return count

    train_count = process_samples(train_samples, "train")
    val_count = process_samples(val_samples, "val")

    # Log skipped samples
    if skipped_samples:
        print(f"\nWarning: Skipped {len(skipped_samples)} samples with invalid bounding boxes:")
        for sample_id, reason in skipped_samples[:10]:  # Show first 10
            print(f"  - {sample_id}: {reason}")
        if len(skipped_samples) > 10:
            print(f"  ... and {len(skipped_samples) - 10} more")

    # Create data.yaml
    create_data_yaml(output_dir)

    return {
        "train": train_count,
        "val": val_count,
        "total": train_count + val_count,
        "skipped": len(skipped_samples),
    }


def split_cat_id_dataset(
    storage_db: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
) -> dict[str, int]:
    """Split labeled data into train/val for cat ID classification.

    Args:
        storage_db: Path to SQLite database from labeling
        output_dir: Output directory for split data
        val_ratio: Fraction of data for validation

    Returns:
        Dictionary with split statistics
    """
    import shutil
    import sqlite3

    conn = sqlite3.connect(storage_db)
    cursor = conn.cursor()

    # Get samples with cat labels - use actual schema
    cursor.execute("""
        SELECT id, crop_path, cat_label
        FROM samples
        WHERE cat_label IS NOT NULL
          AND is_labeled = 1
          AND skip_training = 0
          AND crop_path IS NOT NULL
    """)
    samples = cursor.fetchall()
    conn.close()

    if not samples:
        return {"train": 0, "val": 0, "total": 0}

    # Group by cat label
    by_cat: dict[str, list[Any]] = {}
    for sample in samples:
        cat = sample[2]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(sample)

    # Create directories
    output_dir = Path(output_dir)
    for split in ["train", "val"]:
        for cat in CatIDDataset.class_names():
            (output_dir / split / cat).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "total": 0}

    # Split each cat class separately for balanced splits
    for _cat, cat_samples in by_cat.items():
        random.shuffle(cat_samples)
        split_idx = int(len(cat_samples) * (1 - val_ratio))

        for i, sample in enumerate(cat_samples):
            sample_id, crop_path, cat_label = sample
            crop_path = Path(crop_path)

            if not crop_path.exists():
                continue

            split = "train" if i < split_idx else "val"

            # Copy the pre-cropped image (already extracted around the cat)
            dest = output_dir / split / cat_label / f"{sample_id}.jpg"
            shutil.copy(crop_path, dest)

            stats[split] += 1
            stats["total"] += 1

    return stats
