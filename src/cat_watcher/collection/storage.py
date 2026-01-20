"""Storage management for collected frames and metadata."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import structlog
from PIL import Image
from pydantic import BaseModel, Field

from cat_watcher.schemas import BehaviorType, BoundingBox, CatName

logger = structlog.get_logger(__name__)


class CollectedSample(BaseModel):
    """A collected training sample."""

    id: str = Field(description="Unique sample ID (event ID)")
    timestamp: datetime = Field(description="When the sample was collected")
    camera: str = Field(description="Camera name")
    frame_path: Path = Field(description="Path to full frame image")
    crop_path: Path | None = Field(default=None, description="Path to cropped cat image")
    bounding_box: BoundingBox = Field(description="Cat bounding box in frame")
    frigate_score: float = Field(description="Detection confidence (Frigate or pipeline)")
    frame_width: int = Field(description="Frame width in pixels")
    frame_height: int = Field(description="Frame height in pixels")

    # Source tracking
    source: str = Field(default="frigate", description="Data source: 'frigate' or 'detection'")
    detection_confidence: float | None = Field(default=None, description="Pipeline detection confidence")
    track_id: int | None = Field(default=None, description="Tracker ID for linking related samples")
    event_id: str | None = Field(default=None, description="Detection event ID")

    # Labels (filled in during annotation)
    behavior_label: BehaviorType | None = Field(default=None, description="Labeled behavior")
    cat_label: CatName | None = Field(default=None, description="Labeled cat identity")
    is_labeled: bool = Field(default=False, description="Whether sample has been labeled")
    labeled_at: datetime | None = Field(default=None, description="When labels were added")
    notes: str | None = Field(default=None, description="Annotator notes")

    # Quality flags
    is_blurry: bool = Field(default=False, description="Image is too blurry")
    is_occluded: bool = Field(default=False, description="Cat is partially occluded")
    is_ambiguous: bool = Field(default=False, description="Behavior is ambiguous")
    skip_training: bool = Field(default=False, description="Exclude from training")

    model_config = {"from_attributes": True}


class FrameStorage:
    """Manages storage of frames and metadata in SQLite + filesystem."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize storage.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        self.crops_dir = self.data_dir / "crops"
        self.db_path = self.data_dir / "samples.db"

        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(
            "Initialized frame storage",
            data_dir=str(self.data_dir),
            db_path=str(self.db_path),
        )

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    camera TEXT NOT NULL,
                    frame_path TEXT NOT NULL,
                    crop_path TEXT,
                    bbox_x_min REAL NOT NULL,
                    bbox_y_min REAL NOT NULL,
                    bbox_x_max REAL NOT NULL,
                    bbox_y_max REAL NOT NULL,
                    frigate_score REAL NOT NULL,
                    frame_width INTEGER NOT NULL,
                    frame_height INTEGER NOT NULL,
                    behavior_label TEXT,
                    cat_label TEXT,
                    is_labeled INTEGER DEFAULT 0,
                    labeled_at TEXT,
                    notes TEXT,
                    is_blurry INTEGER DEFAULT 0,
                    is_occluded INTEGER DEFAULT 0,
                    is_ambiguous INTEGER DEFAULT 0,
                    skip_training INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'frigate',
                    detection_confidence REAL,
                    track_id INTEGER,
                    event_id TEXT
                )
            """)
            
            # Cats table for user-defined cats
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Settings table for configurable options
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Run migrations for existing databases
            self._migrate_db(conn)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_labeled
                ON samples(is_labeled)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_timestamp
                ON samples(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_behavior
                ON samples(behavior_label)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_cat
                ON samples(cat_label)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_samples_source
                ON samples(source)
            """)

    def _migrate_db(self, conn: sqlite3.Connection) -> None:
        """Run database migrations for schema updates."""
        # Get existing columns
        cursor = conn.execute("PRAGMA table_info(samples)")
        columns = {row[1] for row in cursor.fetchall()}

        # Add new columns if they don't exist
        migrations = [
            ("source", "TEXT DEFAULT 'frigate'"),
            ("detection_confidence", "REAL"),
            ("track_id", "INTEGER"),
            ("event_id", "TEXT"),
        ]

        for col_name, col_def in migrations:
            if col_name not in columns:
                conn.execute(f"ALTER TABLE samples ADD COLUMN {col_name} {col_def}")
                logger.info(f"Added column {col_name} to samples table")

    @contextmanager
    def _get_db(self) -> Iterator[sqlite3.Connection]:
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _get_frame_path(self, sample_id: str, timestamp: datetime) -> Path:
        """Generate frame storage path organized by date."""
        date_dir = self.frames_dir / timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{sample_id}.jpg"

    def _get_crop_path(self, sample_id: str, timestamp: datetime) -> Path:
        """Generate crop storage path organized by date."""
        date_dir = self.crops_dir / timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{sample_id}_crop.jpg"

    def sample_exists(self, sample_id: str) -> bool:
        """Check if a sample already exists.

        Args:
            sample_id: Frigate event ID

        Returns:
            True if sample exists
        """
        with self._get_db() as conn:
            result = conn.execute(
                "SELECT 1 FROM samples WHERE id = ?", (sample_id,)
            ).fetchone()
            return result is not None

    def save_sample(
        self,
        sample_id: str,
        timestamp: datetime,
        camera: str,
        frame_data: bytes,
        bounding_box: BoundingBox,
        frigate_score: float,
        save_crop: bool = True,
        crop_padding: float = 0.1,
    ) -> CollectedSample:
        """Save a new sample.

        Args:
            sample_id: Frigate event ID
            timestamp: Event timestamp
            camera: Camera name
            frame_data: JPEG frame bytes
            bounding_box: Normalized bounding box
            frigate_score: Frigate detection confidence
            save_crop: Whether to save cropped image
            crop_padding: Padding around crop (fraction of box size)

        Returns:
            Saved CollectedSample
        """
        # Load image to get dimensions
        image = Image.open(BytesIO(frame_data))
        width, height = image.size

        # Save full frame
        frame_path = self._get_frame_path(sample_id, timestamp)
        frame_path.write_bytes(frame_data)

        # Save crop if requested
        crop_path: Path | None = None
        if save_crop:
            crop_path = self._get_crop_path(sample_id, timestamp)
            crop_image = self._extract_crop(image, bounding_box, crop_padding)
            crop_image.save(crop_path, "JPEG", quality=95)

        # Create sample object
        sample = CollectedSample(
            id=sample_id,
            timestamp=timestamp,
            camera=camera,
            frame_path=frame_path,
            crop_path=crop_path,
            bounding_box=bounding_box,
            frigate_score=frigate_score,
            frame_width=width,
            frame_height=height,
        )

        # Save to database
        self._save_to_db(sample)

        logger.debug(
            "Saved sample",
            sample_id=sample_id,
            frame_path=str(frame_path),
            crop_path=str(crop_path) if crop_path else None,
        )

        return sample

    def save_detection_sample(
        self,
        event_id: str,
        camera: str,
        timestamp: float,
        frame: "np.ndarray",
        bbox: BoundingBox,
        confidence: float,
        track_id: int,
        save_crop: bool = True,
        crop_padding: float = 0.1,
    ) -> CollectedSample:
        """Save a sample from the detection pipeline.

        Unlike Frigate samples, detection samples have:
        - The actual frame (BGR numpy array, not a URL fetch)
        - Accurate bbox at the moment of detection
        - Track ID for linking related samples

        Args:
            event_id: Detection event ID
            camera: Camera name
            timestamp: Unix timestamp of detection
            frame: BGR numpy array from OpenCV
            bbox: Normalized bounding box
            confidence: Detection confidence
            track_id: Tracker ID for linking related samples
            save_crop: Whether to save cropped image
            crop_padding: Padding around crop (fraction of box size)

        Returns:
            Saved CollectedSample
        """
        import cv2

        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Encode frame as JPEG
        _, jpeg_data = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg_data.tobytes()

        # Save full frame
        frame_path = self._get_frame_path(event_id, dt)
        frame_path.write_bytes(frame_bytes)

        # Save crop if requested
        crop_path: Path | None = None
        if save_crop:
            crop_path = self._get_crop_path(event_id, dt)
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            crop_image = self._extract_crop(image, bbox, crop_padding)
            crop_image.save(crop_path, "JPEG", quality=95)

        # Create sample object
        sample = CollectedSample(
            id=event_id,
            timestamp=dt,
            camera=camera,
            frame_path=frame_path,
            crop_path=crop_path,
            bounding_box=bbox,
            frigate_score=confidence,  # Use same field for compatibility
            frame_width=width,
            frame_height=height,
            source="detection",
            detection_confidence=confidence,
            track_id=track_id,
            event_id=event_id,
        )

        # Save to database
        self._save_to_db(sample)

        logger.debug(
            "Saved detection sample",
            event_id=event_id,
            camera=camera,
            confidence=confidence,
            track_id=track_id,
        )

        return sample

    def _extract_crop(
        self, image: Image.Image, bbox: BoundingBox, padding: float
    ) -> Image.Image:
        """Extract padded crop from image.

        Args:
            image: PIL Image
            bbox: Normalized bounding box
            padding: Padding fraction

        Returns:
            Cropped PIL Image
        """
        width, height = image.size

        # Convert to pixels
        x_min, y_min, x_max, y_max = bbox.to_pixel_coords(width, height)

        # Handle invalid bounding boxes from Frigate (sometimes coords are swapped)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min

        # Ensure minimum crop size (at least 10x10 pixels)
        if x_max - x_min < 10:
            center_x = (x_min + x_max) // 2
            x_min = max(0, center_x - 50)
            x_max = min(width, center_x + 50)
        if y_max - y_min < 10:
            center_y = (y_min + y_max) // 2
            y_min = max(0, center_y - 50)
            y_max = min(height, center_y + 50)

        # Add padding
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)

        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(width, x_max + pad_x)
        y_max = min(height, y_max + pad_y)

        return image.crop((x_min, y_min, x_max, y_max))

    def _save_to_db(self, sample: CollectedSample) -> None:
        """Save sample metadata to database."""
        with self._get_db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO samples (
                    id, timestamp, camera, frame_path, crop_path,
                    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max,
                    frigate_score, frame_width, frame_height,
                    behavior_label, cat_label, is_labeled, labeled_at,
                    notes, is_blurry, is_occluded, is_ambiguous, skip_training,
                    source, detection_confidence, track_id, event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    sample.id,
                    sample.timestamp.isoformat(),
                    sample.camera,
                    str(sample.frame_path),
                    str(sample.crop_path) if sample.crop_path else None,
                    sample.bounding_box.x_min,
                    sample.bounding_box.y_min,
                    sample.bounding_box.x_max,
                    sample.bounding_box.y_max,
                    sample.frigate_score,
                    sample.frame_width,
                    sample.frame_height,
                    sample.behavior_label.value if sample.behavior_label else None,
                    sample.cat_label.value if sample.cat_label else None,
                    1 if sample.is_labeled else 0,
                    sample.labeled_at.isoformat() if sample.labeled_at else None,
                    sample.notes,
                    1 if sample.is_blurry else 0,
                    1 if sample.is_occluded else 0,
                    1 if sample.is_ambiguous else 0,
                    1 if sample.skip_training else 0,
                    sample.source,
                    sample.detection_confidence,
                    sample.track_id,
                    sample.event_id,
                ),
            )

    def _row_to_sample(self, row: sqlite3.Row) -> CollectedSample:
        """Convert database row to CollectedSample."""
        return CollectedSample(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            camera=row["camera"],
            frame_path=Path(row["frame_path"]),
            crop_path=Path(row["crop_path"]) if row["crop_path"] else None,
            bounding_box=BoundingBox(
                x_min=row["bbox_x_min"],
                y_min=row["bbox_y_min"],
                x_max=row["bbox_x_max"],
                y_max=row["bbox_y_max"],
            ),
            frigate_score=row["frigate_score"],
            frame_width=row["frame_width"],
            frame_height=row["frame_height"],
            behavior_label=BehaviorType(row["behavior_label"])
            if row["behavior_label"]
            else None,
            cat_label=CatName(row["cat_label"]) if row["cat_label"] else None,
            is_labeled=bool(row["is_labeled"]),
            labeled_at=datetime.fromisoformat(row["labeled_at"])
            if row["labeled_at"]
            else None,
            notes=row["notes"],
            is_blurry=bool(row["is_blurry"]),
            is_occluded=bool(row["is_occluded"]),
            is_ambiguous=bool(row["is_ambiguous"]),
            skip_training=bool(row["skip_training"]),
            source=row["source"] if "source" in row.keys() else "frigate",
            detection_confidence=row["detection_confidence"] if "detection_confidence" in row.keys() else None,
            track_id=row["track_id"] if "track_id" in row.keys() else None,
            event_id=row["event_id"] if "event_id" in row.keys() else None,
        )

    def get_sample(self, sample_id: str) -> CollectedSample | None:
        """Get a sample by ID.

        Args:
            sample_id: Frigate event ID

        Returns:
            CollectedSample or None if not found
        """
        with self._get_db() as conn:
            row = conn.execute(
                "SELECT * FROM samples WHERE id = ?", (sample_id,)
            ).fetchone()
            if row:
                return self._row_to_sample(row)
            return None

    def get_unlabeled_samples(
        self, limit: int = 100, offset: int = 0, source: str | None = None
    ) -> list[CollectedSample]:
        """Get unlabeled samples for annotation.

        Args:
            limit: Maximum number of samples
            offset: Pagination offset
            source: Filter by source ("frigate" or "detection")

        Returns:
            List of unlabeled samples
        """
        query = "SELECT * FROM samples WHERE is_labeled = 0 AND skip_training = 0"
        params: list[Any] = []

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_db() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_sample(row) for row in rows]

    def get_labeled_samples(
        self,
        behavior: BehaviorType | None = None,
        cat: CatName | None = None,
        limit: int = 100,
        offset: int = 0,
        source: str | None = None,
    ) -> list[CollectedSample]:
        """Get labeled samples, optionally filtered.

        Args:
            behavior: Filter by behavior type
            cat: Filter by cat identity
            limit: Maximum number of samples
            offset: Pagination offset
            source: Filter by source ("frigate" or "detection")

        Returns:
            List of labeled samples
        """
        query = "SELECT * FROM samples WHERE is_labeled = 1 AND skip_training = 0"
        params: list[Any] = []

        if behavior:
            query += " AND behavior_label = ?"
            params.append(behavior.value)

        if cat:
            query += " AND cat_label = ?"
            params.append(cat.value)

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_db() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_sample(row) for row in rows]

    def update_labels(
        self,
        sample_id: str,
        behavior: BehaviorType | None = None,
        cat: CatName | None = None,
        notes: str | None = None,
        is_blurry: bool = False,
        is_occluded: bool = False,
        is_ambiguous: bool = False,
        skip_training: bool = False,
    ) -> CollectedSample | None:
        """Update labels for a sample.

        Args:
            sample_id: Sample ID
            behavior: Behavior label
            cat: Cat identity label
            notes: Annotator notes
            is_blurry: Mark as blurry
            is_occluded: Mark as occluded
            is_ambiguous: Mark as ambiguous
            skip_training: Exclude from training

        Returns:
            Updated sample or None if not found
        """
        is_labeled = behavior is not None or cat is not None
        labeled_at = datetime.now() if is_labeled else None

        with self._get_db() as conn:
            conn.execute(
                """
                UPDATE samples SET
                    behavior_label = ?,
                    cat_label = ?,
                    is_labeled = ?,
                    labeled_at = ?,
                    notes = ?,
                    is_blurry = ?,
                    is_occluded = ?,
                    is_ambiguous = ?,
                    skip_training = ?
                WHERE id = ?
            """,
                (
                    behavior.value if behavior else None,
                    cat.value if cat else None,
                    1 if is_labeled else 0,
                    labeled_at.isoformat() if labeled_at else None,
                    notes,
                    1 if is_blurry else 0,
                    1 if is_occluded else 0,
                    1 if is_ambiguous else 0,
                    1 if skip_training else 0,
                    sample_id,
                ),
            )

        return self.get_sample(sample_id)

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_db() as conn:
            total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE is_labeled = 1"
            ).fetchone()[0]
            skipped = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE skip_training = 1"
            ).fetchone()[0]
            # Unlabeled samples that are available for labeling (not skipped)
            unlabeled = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE is_labeled = 0 AND skip_training = 0"
            ).fetchone()[0]

            # Behavior distribution
            behavior_counts = {}
            rows = conn.execute(
                """
                SELECT behavior_label, COUNT(*) as count
                FROM samples
                WHERE behavior_label IS NOT NULL
                GROUP BY behavior_label
            """
            ).fetchall()
            for row in rows:
                behavior_counts[row["behavior_label"]] = row["count"]

            # Cat distribution
            cat_counts = {}
            rows = conn.execute(
                """
                SELECT cat_label, COUNT(*) as count
                FROM samples
                WHERE cat_label IS NOT NULL
                GROUP BY cat_label
            """
            ).fetchall()
            for row in rows:
                cat_counts[row["cat_label"]] = row["count"]

            return {
                "total_samples": total,
                "labeled_samples": labeled,
                "unlabeled_samples": unlabeled,
                "skipped_samples": skipped,
                "trainable_samples": labeled - skipped,
                "behavior_distribution": behavior_counts,
                "cat_distribution": cat_counts,
            }

    def export_for_training(
        self, output_path: Path, format: str = "coco"
    ) -> dict[str, Any]:
        """Export labeled samples for training.

        Args:
            output_path: Output directory
            format: Export format ('coco', 'yolo', or 'csv')

        Returns:
            Export summary
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = self.get_labeled_samples(limit=100000)
        trainable = [s for s in samples if not s.skip_training]

        if format == "csv":
            return self._export_csv(trainable, output_path)
        elif format == "yolo":
            return self._export_yolo(trainable, output_path)
        else:
            return self._export_coco(trainable, output_path)

    def _export_csv(
        self, samples: list[CollectedSample], output_path: Path
    ) -> dict[str, Any]:
        """Export as CSV."""
        import csv

        csv_path = output_path / "labels.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "frame_path",
                    "crop_path",
                    "behavior",
                    "cat",
                    "x_min",
                    "y_min",
                    "x_max",
                    "y_max",
                ]
            )
            for s in samples:
                writer.writerow(
                    [
                        s.id,
                        str(s.frame_path),
                        str(s.crop_path) if s.crop_path else "",
                        s.behavior_label.value if s.behavior_label else "",
                        s.cat_label.value if s.cat_label else "",
                        s.bounding_box.x_min,
                        s.bounding_box.y_min,
                        s.bounding_box.x_max,
                        s.bounding_box.y_max,
                    ]
                )

        return {"format": "csv", "path": str(csv_path), "samples": len(samples)}

    def _export_coco(
        self, samples: list[CollectedSample], output_path: Path
    ) -> dict[str, Any]:
        """Export in COCO format."""
        # Build category mappings
        behavior_categories = [
            {"id": i + 1, "name": b.value}
            for i, b in enumerate(BehaviorType)
        ]

        # Build COCO structure
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []

        annotation_id = 1
        for i, sample in enumerate(samples):
            image_id = i + 1

            # Add image entry
            images.append(
                {
                    "id": image_id,
                    "file_name": str(sample.frame_path),
                    "width": sample.frame_width,
                    "height": sample.frame_height,
                }
            )

            # Add annotation
            if sample.behavior_label:
                bbox = sample.bounding_box
                x, y, w, h = (
                    bbox.x_min * sample.frame_width,
                    bbox.y_min * sample.frame_height,
                    (bbox.x_max - bbox.x_min) * sample.frame_width,
                    (bbox.y_max - bbox.y_min) * sample.frame_height,
                )

                category_id = [
                    c["id"]
                    for c in behavior_categories
                    if c["name"] == sample.behavior_label.value
                ][0]

                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "attributes": {
                            "cat_identity": sample.cat_label.value
                            if sample.cat_label
                            else None
                        },
                    }
                )
                annotation_id += 1

        coco_data = {
            "info": {
                "description": "Cat Watcher Training Data",
                "date_created": datetime.now().isoformat(),
            },
            "categories": behavior_categories,
            "images": images,
            "annotations": annotations,
        }

        # Save JSON
        json_path = output_path / "annotations.json"
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=2)

        return {
            "format": "coco",
            "path": str(json_path),
            "samples": len(samples),
            "annotations": annotation_id - 1,
        }

    def _export_yolo(
        self, samples: list[CollectedSample], output_path: Path
    ) -> dict[str, Any]:
        """Export in YOLO format."""
        labels_dir = output_path / "labels"
        labels_dir.mkdir(exist_ok=True)

        # Create class mapping
        classes = [b.value for b in BehaviorType]
        classes_path = output_path / "classes.txt"
        classes_path.write_text("\n".join(classes))

        exported = 0
        for sample in samples:
            if not sample.behavior_label:
                continue

            class_id = classes.index(sample.behavior_label.value)
            bbox = sample.bounding_box

            # YOLO format: class x_center y_center width height (normalized)
            x_center = (bbox.x_min + bbox.x_max) / 2
            y_center = (bbox.y_min + bbox.y_max) / 2
            width = bbox.x_max - bbox.x_min
            height = bbox.y_max - bbox.y_min

            label_path = labels_dir / f"{sample.id}.txt"
            label_path.write_text(f"{class_id} {x_center} {y_center} {width} {height}")
            exported += 1

        return {
            "format": "yolo",
            "labels_dir": str(labels_dir),
            "classes_file": str(classes_path),
            "samples": exported,
        }

    # Cat management methods
    
    def get_cats(self) -> list[dict[str, Any]]:
        """Get all cats from the database.
        
        Returns:
            List of cat dictionaries with id, name, age, notes, created_at, updated_at
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT id, name, age, notes, created_at, updated_at
                FROM cats
                ORDER BY name
            """)
            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "age": row[2],
                    "notes": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                }
                for row in cursor.fetchall()
            ]
    
    def get_cat(self, cat_id: int) -> dict[str, Any] | None:
        """Get a single cat by ID.
        
        Args:
            cat_id: Cat ID
            
        Returns:
            Cat dictionary or None if not found
        """
        with self._get_db() as conn:
            cursor = conn.execute("""
                SELECT id, name, age, notes, created_at, updated_at
                FROM cats
                WHERE id = ?
            """, (cat_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "age": row[2],
                    "notes": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                }
            return None
    
    def add_cat(self, name: str, age: int | None = None, notes: str | None = None) -> dict[str, Any]:
        """Add a new cat to the database.
        
        Args:
            name: Cat name (must be unique)
            age: Cat age in years (optional)
            notes: Additional notes (optional)
            
        Returns:
            Created cat dictionary
            
        Raises:
            ValueError: If cat name already exists
        """
        with self._get_db() as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO cats (name, age, notes)
                    VALUES (?, ?, ?)
                """, (name.lower().strip(), age, notes))
                cat_id = cursor.lastrowid
                logger.info("Added cat", name=name, id=cat_id)
            except sqlite3.IntegrityError:
                raise ValueError(f"Cat '{name}' already exists")
        
        # Fetch the cat after commit (outside the with block)
        return self.get_cat(cat_id)  # type: ignore
    
    def update_cat(self, cat_id: int, name: str | None = None, age: int | None = None, notes: str | None = None) -> dict[str, Any] | None:
        """Update an existing cat.
        
        Args:
            cat_id: Cat ID to update
            name: New name (optional)
            age: New age (optional)
            notes: New notes (optional)
            
        Returns:
            Updated cat dictionary or None if not found
        """
        # Get current cat
        cat = self.get_cat(cat_id)
        if not cat:
            return None
        
        # Build update
        updates = []
        params = []
        
        if name is not None:
            updates.append("name = ?")
            params.append(name.lower().strip())
        if age is not None:
            updates.append("age = ?")
            params.append(age)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if not updates:
            return cat
        
        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(cat_id)
        
        with self._get_db() as conn:
            try:
                conn.execute(f"""
                    UPDATE cats
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                logger.info("Updated cat", id=cat_id)
            except sqlite3.IntegrityError:
                raise ValueError(f"Cat name '{name}' already exists")
        
        # Fetch the updated cat after commit (outside the with block)
        return self.get_cat(cat_id)
    
    def delete_cat(self, cat_id: int) -> bool:
        """Delete a cat from the database.
        
        Args:
            cat_id: Cat ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_db() as conn:
            cursor = conn.execute("DELETE FROM cats WHERE id = ?", (cat_id,))
            if cursor.rowcount > 0:
                logger.info("Deleted cat", id=cat_id)
                return True
            return False

    # Settings management methods
    
    # Default settings with descriptions
    DEFAULT_SETTINGS: dict[str, dict[str, Any]] = {
        # Frigate settings
        "frigate.url": {
            "value": "http://192.168.50.36:5000",
            "category": "frigate",
            "description": "Frigate API base URL",
        },
        "frigate.cameras": {
            "value": "[]",
            "category": "frigate",
            "description": "List of cameras to monitor (JSON array, empty = all)",
        },
        # Detection settings
        "detection.frame_rate": {
            "value": "5.0",
            "category": "detection",
            "description": "Target frames per second for processing",
        },
        "detection.cat_confidence": {
            "value": "0.2",
            "category": "detection",
            "description": "Minimum confidence for cat detection (0.0-1.0)",
        },
        "detection.behavior_confidence": {
            "value": "0.6",
            "category": "detection",
            "description": "Minimum confidence for behavior classification (0.0-1.0)",
        },
        "detection.save_frames": {
            "value": "true",
            "category": "detection",
            "description": "Save event frames to disk",
        },
        "detection.min_event_duration": {
            "value": "0.5",
            "category": "detection",
            "description": "Minimum event duration in seconds",
        },
        "detection.max_event_duration": {
            "value": "300.0",
            "category": "detection",
            "description": "Maximum event duration in seconds",
        },
        # MQTT settings
        "mqtt.broker": {
            "value": "192.168.1.82",
            "category": "mqtt",
            "description": "MQTT broker hostname or IP",
        },
        "mqtt.port": {
            "value": "1883",
            "category": "mqtt",
            "description": "MQTT broker port",
        },
        "mqtt.enabled": {
            "value": "true",
            "category": "mqtt",
            "description": "Enable MQTT event publishing",
        },
        # Notifications
        "notifications.enabled": {
            "value": "false",
            "category": "notifications",
            "description": "Enable event notifications",
        },
    }
    
    def get_setting(self, key: str) -> str | None:
        """Get a single setting value.
        
        Args:
            key: Setting key (e.g., 'frigate.url')
            
        Returns:
            Setting value as string, or default if not set
        """
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]
            # Return default if exists
            if key in self.DEFAULT_SETTINGS:
                return self.DEFAULT_SETTINGS[key]["value"]
            return None
    
    def get_settings_by_category(self, category: str) -> dict[str, Any]:
        """Get all settings for a category.
        
        Args:
            category: Settings category (e.g., 'frigate', 'detection', 'mqtt')
            
        Returns:
            Dictionary of key -> {value, description}
        """
        result = {}
        
        # Start with defaults for this category
        for key, info in self.DEFAULT_SETTINGS.items():
            if info["category"] == category:
                result[key] = {
                    "value": info["value"],
                    "description": info["description"],
                }
        
        # Override with database values
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT key, value, description FROM settings WHERE category = ?",
                (category,)
            )
            for row in cursor.fetchall():
                result[row[0]] = {
                    "value": row[1],
                    "description": row[2] or result.get(row[0], {}).get("description", ""),
                }
        
        return result
    
    def get_all_settings(self) -> dict[str, dict[str, Any]]:
        """Get all settings grouped by category.
        
        Returns:
            Dictionary of category -> {key -> {value, description}}
        """
        result: dict[str, dict[str, Any]] = {}
        
        # Start with defaults
        for key, info in self.DEFAULT_SETTINGS.items():
            category = info["category"]
            if category not in result:
                result[category] = {}
            result[category][key] = {
                "value": info["value"],
                "description": info["description"],
            }
        
        # Override with database values
        with self._get_db() as conn:
            cursor = conn.execute(
                "SELECT key, value, category, description FROM settings"
            )
            for row in cursor.fetchall():
                key, value, category, description = row
                if category not in result:
                    result[category] = {}
                result[category][key] = {
                    "value": value,
                    "description": description or result.get(category, {}).get(key, {}).get("description", ""),
                }
        
        return result
    
    def set_setting(self, key: str, value: str, category: str | None = None, description: str | None = None) -> None:
        """Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value (as string)
            category: Setting category (inferred from key if not provided)
            description: Setting description (optional)
        """
        # Infer category from key if not provided
        if category is None:
            category = key.split(".")[0] if "." in key else "general"
        
        # Get description from defaults if not provided
        if description is None and key in self.DEFAULT_SETTINGS:
            description = self.DEFAULT_SETTINGS[key].get("description", "")
        
        with self._get_db() as conn:
            conn.execute("""
                INSERT INTO settings (key, value, category, description, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, value, category, description))
            logger.info("Updated setting", key=key, value=value)
    
    def set_settings_bulk(self, settings: dict[str, str]) -> None:
        """Set multiple settings at once.
        
        Args:
            settings: Dictionary of key -> value
        """
        with self._get_db() as conn:
            for key, value in settings.items():
                category = key.split(".")[0] if "." in key else "general"
                description = self.DEFAULT_SETTINGS.get(key, {}).get("description", "")
                conn.execute("""
                    INSERT INTO settings (key, value, category, description, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                """, (key, str(value), category, description))
            logger.info("Updated settings bulk", count=len(settings))
    
    def delete_setting(self, key: str) -> bool:
        """Delete a setting (revert to default).
        
        Args:
            key: Setting key
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_db() as conn:
            cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            if cursor.rowcount > 0:
                logger.info("Deleted setting", key=key)
                return True
            return False
