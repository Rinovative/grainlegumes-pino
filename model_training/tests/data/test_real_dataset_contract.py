# ruff: noqa: S101
"""Validate mounted production packages only under explicit real-data acceptance."""

from __future__ import annotations

import json

import pytest
from src import datasets
from support import real_data

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(
        not real_data.real_data_tests_enabled(),
        reason="set RUN_REAL_DATA_TESTS=1 for strict mounted-package acceptance",
    ),
]


@pytest.mark.parametrize("dataset_id", ["lhs_var80_seed3001", "lhs_var120_seed4001"])
def test_current_production_metadata_package_passes_read_only_cross_binding(
    dataset_id: str,
) -> None:
    """Require and fully validate each mounted production metadata package."""
    metadata_dir = real_data.require_real_metadata_package(dataset_id)
    metadata_path = metadata_dir / datasets.metadata.METADATA_FILENAME
    manifest_path = metadata_dir / datasets.metadata.SOURCE_MANIFEST_FILENAME

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scientific = metadata["scientific_identity"]
    identity = datasets.identity.DatasetIdentity(
        dataset_id=dataset_id,
        task=scientific["task_id"],
        task_contract_digest=scientific["task_contract_digest"],
        fingerprint=scientific["dataset_fingerprint"],
        sample_ids=tuple(manifest["intended_case_ids"]),
        sample_count=scientific["sample_count"],
        spatial_shape=tuple(scientific["spatial_shape"]),
        generated_batch_identity_sha256=scientific["generated_batch_identity_sha256"],
    )

    package = datasets.metadata.validate_dataset_metadata_directory(
        metadata_dir,
        dataset_identity=identity,
    )

    assert package.source_manifest["intended_case_ids"] == manifest["intended_case_ids"]
    assert package.metadata["scientific_identity"]["generated_batch_identity_sha256"] == scientific["generated_batch_identity_sha256"]
