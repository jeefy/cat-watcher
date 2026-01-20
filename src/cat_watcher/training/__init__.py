"""Training module for cat behavior and identification models."""

from cat_watcher.training.behavior import BehaviorTrainer
from cat_watcher.training.cat_id import CatIDTrainer
from cat_watcher.training.dataset import CatBehaviorDataset, CatIDDataset

__all__ = [
    "BehaviorTrainer",
    "CatIDTrainer",
    "CatBehaviorDataset",
    "CatIDDataset",
]
