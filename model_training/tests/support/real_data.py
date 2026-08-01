"""Own the explicit flag and data-root boundary for real-package tests."""

from __future__ import annotations

import os
from pathlib import Path

from src import common, datasets

REAL_DATA_FLAG = "RUN_REAL_DATA_TESTS"
MODEL_TRAINING_DATA_ROOT = "MODEL_TRAINING_DATA_ROOT"


def real_data_tests_enabled() -> bool:
    """Return whether strict mounted-package acceptance was explicitly enabled."""
    return os.environ.get(REAL_DATA_FLAG) == "1"


def require_real_data_root() -> Path:
    """Return the explicit model-training root without a repository fallback."""
    if not real_data_tests_enabled():
        message = f"{REAL_DATA_FLAG}=1 is required for real-data acceptance"
        raise RuntimeError(message)
    configured = os.environ.get(MODEL_TRAINING_DATA_ROOT, "").strip()
    if not configured:
        message = f"{MODEL_TRAINING_DATA_ROOT} is required for real-data acceptance"
        raise RuntimeError(message)
    root = Path(configured).expanduser()
    if common.paths.get_model_training_data_root() != root:
        message = "Maintained model-training path ownership disagrees with the explicit real-data root"
        raise RuntimeError(message)
    return root


def require_real_metadata_package(dataset_id: str) -> Path:
    """Return one mandatory real metadata package or fail without fallback."""
    require_real_data_root()
    metadata_dir = common.paths.resolve_dataset_metadata_dir(dataset_id)
    required = (
        metadata_dir / datasets.metadata.METADATA_FILENAME,
        metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME,
    )
    if not all(path.is_file() for path in required):
        message = f"Required real production package {dataset_id!r} is missing"
        raise FileNotFoundError(message)
    return metadata_dir
