"""Labeling service module for annotating training data."""

from cat_watcher.labeling.api import router
from cat_watcher.labeling.app import create_app

__all__ = ["create_app", "router"]
