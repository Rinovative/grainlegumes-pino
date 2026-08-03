# ruff: noqa: BLE001, S101, SLF001
"""
Protect current artifact reader identity, cache admission, and contained rebuilds.

The tests cover current table parsing, metadata collision rejection, ordered
membership, symlink/path containment, concurrent publication, and provenance
completion races. Numerical artifact generation is covered in
``test_artifact_provenance``; plot usability is covered separately.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import threading
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from src import analysis, common, experiments, learning


def _current_generic_frame(*, metadata: object, index: int = 0, source_index: int = 2) -> pd.DataFrame:
    """
    Build one compact current-schema artifact table.

    ``metadata`` is intentionally left unvalidated so callers can exercise the
    reader's JSON, collision, and alignment checks against one canonical row.
    """
    return pd.DataFrame(
        [
            {
                "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
                "task_id": "synthetic",
                "output_fields": ["target"],
                "output_units": ["1"],
                "case_index": source_index + 1,
                "source_index": source_index,
                "split_local_index": 0,
                "npz_path": "case_0003.npz",
                "meta": metadata,
                "inference_time_ms": None,
                "rel_l2": 0.0,
                "rel_h1": 0.0,
                "rmse_target": 0.0,
                "normalized_sse_target": 0.0,
                "normalized_count_target": 1,
                "normalized_rmse_target": 0.0,
            }
        ],
        index=[index],
    )


def test_dataframe_reader_exposes_authoritative_normalized_aggregate() -> None:
    """
    Reconstruct the authoritative aggregate from normalized SSE and count.

    A current one-row table must expose schema metadata and a zero macro RMSE;
    this protects analysis consumers from treating per-case means as primary.
    """
    enriched = analysis.evaluation.dataframe.build_eval_df(_current_generic_frame(metadata='{"label": "valid"}'))

    assert enriched.attrs["artifact_schema_version"] == analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION
    assert enriched.attrs["output_fields"] == ("target",)
    assert enriched.attrs["normalized_macro_rmse"]["value"] == 0.0


def test_dataframe_reader_rejects_every_unexpected_column() -> None:
    """Reject an undeclared column through the generic exact-schema boundary."""
    frame = _current_generic_frame(metadata="{}")
    frame["unexpected_metric"] = 0.0

    with pytest.raises(ValueError, match="schema mismatch"):
        analysis.evaluation.dataframe.build_eval_df(frame)


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_dataframe_reader_requires_integer_artifact_version_one(schema_version: bool | float) -> None:
    """Reject alternate representations and unsupported Parquet schema versions."""
    frame = _current_generic_frame(metadata="{}")
    frame["artifact_schema_version"] = schema_version  # pyright: ignore[reportCallIssue, reportArgumentType]

    with pytest.raises((TypeError, ValueError), match=r"artifact_schema_version|schema version"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_dataframe_reader_rejects_missing_and_duplicate_columns() -> None:
    """Reject missing evidence and duplicate columns through exact schema admission."""
    missing = _current_generic_frame(metadata="{}").drop(columns="normalized_sse_target")
    with pytest.raises(ValueError, match="schema mismatch"):
        analysis.evaluation.dataframe.build_eval_df(missing)

    base = _current_generic_frame(metadata="{}")
    duplicate = pd.concat([base, base[["rmse_target"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        analysis.evaluation.dataframe.build_eval_df(duplicate)


def test_metadata_cannot_duplicate_authoritative_identity_columns() -> None:
    """
    Reject metadata that duplicates an authoritative identity column.

    A forged ``source_index`` must fail before flattening so metadata cannot
    override the row identity used for provenance and case navigation.
    """
    frame = _current_generic_frame(metadata='{"source_index": 999}')

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_cannot_reuse_the_raw_meta_column() -> None:
    """
    Reject metadata that reuses the raw ``meta`` column name.

    The reader must preserve the original payload boundary rather than allow a
    flattened key to erase or ambiguously replace its own source column.
    """
    frame = _current_generic_frame(metadata='{"meta": "ambiguous"}')

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_expansion_preserves_noncontiguous_row_indices() -> None:
    """
    Preserve metadata alignment for a noncontiguous DataFrame index.

    A row indexed at seven must retain its case identity and label after
    expansion; positional concatenation would attach metadata to the wrong row.
    """
    row_index = 7
    case_index = 8
    frame = _current_generic_frame(
        metadata='{"label": "case-eight"}',
        index=row_index,
        source_index=case_index - 1,
    )

    enriched = analysis.evaluation.dataframe.build_eval_df(frame)

    assert enriched.index.tolist() == [row_index]
    assert enriched.loc[row_index, "case_index"] == case_index
    assert enriched.loc[row_index, "label"] == "case-eight"


@pytest.mark.parametrize(
    ("metadata", "error_type", "match"),
    [
        ("{'source': 1}", ValueError, "valid JSON"),
        ("[1, 2]", TypeError, "decode to an object"),
        (42, TypeError, "JSON object or mapping"),
    ],
    ids=("python-literal", "json-array", "non-string"),
)
def test_artifact_metadata_requires_a_json_object(
    metadata: object,
    error_type: type[Exception],
    match: str,
) -> None:
    """
    Reject malformed JSON, JSON arrays, and unsupported metadata values.

    The parameter matrix covers syntax, decoded shape, and input type so the
    reader never turns an invalid provenance payload into an empty mapping.
    """
    with pytest.raises(error_type, match=match):
        analysis.evaluation.dataframe.build_eval_df(_current_generic_frame(metadata=metadata))


def test_scientific_cache_identity_tracks_science_but_not_runtime_device() -> None:
    """
    Separate scientific cache identity from operational runtime facts.

    Mutating every required scientific component must change the projected
    provenance, while device, batch size, outputs, and aggregates must not.
    """
    provenance = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "task_contract_digest": "task-a",
            "best_checkpoint_sha256": "checkpoint-a",
            "normalizer_sha256": "normalizer-a",
        },
        "dataset": {"fingerprint": "dataset-a"},
        "selection": {"effective_ordered_source_indices_sha256": "membership-a"},
        "normalizer": {"sha256": "normalizer-a"},
        "evaluator": {
            "objective": {"id": "normalized_macro_rmse", "reduction": "field_macro_element_mean"},
            "output_fields": ["first", "second"],
        },
        "physics": {
            "residual_schema_version": 1,
            "selected_training_continuity": "div_eps_velocity",
            "derivatives": {"kind": "spectral", "extension": "reflect"},
            "interior_crop": 2,
            "constants": {"dynamic_viscosity_pa_s": 1.8139e-5},
            "scalar_definitions": {"div_velocity_mse": "mean(div(u)**2)"},
        },
        "generation": {"effective_case_limit": None},
        "runtime": {"requested_policy": "cpu", "resolved_device": "cpu", "batch_size": 1},
    }
    baseline = analysis.artifacts.service._scientific_provenance(provenance)
    mutations = [
        (("artifact_schema_version",), 999),
        (("run", "task_contract_digest"), "task-b"),
        (("run", "best_checkpoint_sha256"), "checkpoint-b"),
        (("normalizer", "sha256"), "normalizer-b"),
        (("dataset", "fingerprint"), "dataset-b"),
        (("selection", "effective_ordered_source_indices_sha256"), "membership-b"),
        (("evaluator", "objective", "reduction"), "wrong-reduction"),
        (("evaluator", "output_fields"), ["second", "first"]),
        (("physics", "residual_schema_version"), 2),
        (("physics", "selected_training_continuity"), "div_velocity"),
        (("physics", "derivatives", "kind"), "unsupported"),
        (("physics", "derivatives", "extension"), "none"),
        (("physics", "interior_crop"), 3),
        (("physics", "constants", "dynamic_viscosity_pa_s"), 2.0e-5),
        (("physics", "scalar_definitions", "div_velocity_mse"), "different"),
        (("generation", "effective_case_limit"), 4),
    ]
    for keys, replacement in mutations:
        changed = copy.deepcopy(provenance)
        target: object = changed
        for key in keys[:-1]:
            assert isinstance(target, dict)
            target = target[key]
        assert isinstance(target, dict)
        target[keys[-1]] = replacement
        assert analysis.artifacts.service._scientific_provenance(changed) != baseline

    operational = copy.deepcopy(provenance)
    operational["runtime"] = {
        "requested_policy": "auto",
        "resolved_device": "cuda:7",
        "batch_size": 19,
    }
    operational["outputs"] = {"generated": "digest"}
    operational["aggregate"] = {"value": 123.0}
    assert analysis.artifacts.service._scientific_provenance(operational) == baseline


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_artifact_provenance_requires_integer_version_one(schema_version: object) -> None:
    """Reject alternate representations in both artifact provenance version fields."""
    current: dict[str, object] = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
    }
    for field in tuple(current):
        invalid = dict(current)
        invalid[field] = schema_version
        with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match=field):
            analysis.artifacts.service._require_current_provenance_schema(invalid)


def test_rebuild_removes_only_one_exact_target(tmp_path: Path) -> None:
    """
    Confine rebuild deletion to one exact ID or named OOD target.

    Removing one OOD fixture must preserve the ID and sibling OOD markers, and
    broad or out-of-run paths must be rejected before filesystem mutation.
    """
    run_dir = tmp_path / "run"
    id_target = run_dir / "analysis" / "id"
    first_ood = run_dir / "analysis" / "ood" / "first"
    second_ood = run_dir / "analysis" / "ood" / "second"
    for target in (id_target, first_ood, second_ood):
        target.mkdir(parents=True)
        (target / "marker").write_text(target.name, encoding="utf-8")

    analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=first_ood)

    assert not first_ood.exists()
    assert (id_target / "marker").is_file()
    assert (second_ood / "marker").is_file()
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "ood")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=tmp_path / "outside")


def test_upload_gate_requires_an_explicit_complete_current_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Admit only an explicit, complete, current artifact for upload.

    The valid fixture must pass without changing its bytes; after the manifest
    is made stale, validation must fail before any tracking-side effect.
    """
    run_dir = tmp_path / "run"
    artifact_root = common.paths.resolve_id_analysis_dir(run_dir)
    artifact_root.mkdir(parents=True)
    marker = artifact_root / "aggregate.parquet"
    marker.write_bytes(b"complete-local-artifact")
    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(artifact_root)
    provenance_path.write_text("{}\n", encoding="utf-8")
    outputs = {"aggregate.parquet": {"sha256": "a" * 64, "size_bytes": 23}}
    provenance = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "outputs": outputs,
        "run": {
            "name": "run-name",
            "task": "steady_flow",
            "task_contract_digest": "t" * 64,
            "effective_config_digest": "c" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "normalizer_sha256": "n" * 64,
        },
    }
    completed = {
        "summary": {
            "effective_config_digest": "c" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "normalizer_sha256": "n" * 64,
        },
        "config": {"run": {"name": "run-name"}},
    }
    monkeypatch.setattr(
        analysis.artifacts.service,
        "_read_artifact_provenance",
        lambda _path: provenance,
    )
    monkeypatch.setattr(
        analysis.artifacts.service,
        "_require_current_provenance_schema",
        lambda _provenance: None,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.contracts,
        "artifact_output_manifest",
        lambda _root: outputs,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.run,
        "validate_completed_run",
        lambda _run_dir: completed,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: SimpleNamespace(
            id="steady_flow",
            contract_digest="t" * 64,
        ),
    )

    validated = analysis.artifacts.service.validate_artifact_upload_source(
        run_dir=run_dir,
        artifact_root=artifact_root,
    )

    assert validated == provenance
    assert marker.read_bytes() == b"complete-local-artifact"
    monkeypatch.setattr(
        analysis.artifacts.service.contracts,
        "artifact_output_manifest",
        lambda _root: {"aggregate.parquet": {"sha256": "stale"}},
    )
    with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match="manifest mismatch"):
        analysis.artifacts.service.validate_artifact_upload_source(
            run_dir=run_dir,
            artifact_root=artifact_root,
        )
    assert marker.read_bytes() == b"complete-local-artifact"


def test_unrelated_artifact_targets_use_independent_locks(tmp_path: Path) -> None:
    """
    Allow unrelated artifact targets to acquire independent locks.

    Holding the ID lock must not prevent a worker from taking a named OOD lock;
    target-scoped locking preserves concurrency without weakening serialization.
    """
    run_dir = tmp_path / "run"
    id_target = common.paths.resolve_id_analysis_dir(run_dir)
    ood_target = common.paths.resolve_ood_analysis_dir(run_dir, "other")
    id_lock = analysis.artifacts.service._artifact_lock_path(run_dir=run_dir, save_root=id_target)
    ood_lock = analysis.artifacts.service._artifact_lock_path(run_dir=run_dir, save_root=ood_target)
    assert id_lock.parent == common.paths.get_run_locks_root()
    assert ood_lock.parent == common.paths.get_run_locks_root()
    assert id_lock != ood_lock
    assert not (common.paths.resolve_analysis_root(run_dir) / ".locks").exists()
    acquired = threading.Event()

    def acquire_ood() -> None:
        """Acquire the unrelated OOD lock and signal successful entry."""
        with common.locking.exclusive_file_lock(ood_lock, blocking=True):
            acquired.set()

    with common.locking.exclusive_file_lock(id_lock, blocking=True):
        worker = threading.Thread(target=acquire_ood)
        worker.start()
        assert acquired.wait(timeout=1)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert id_lock.is_file()
    assert ood_lock.is_file()
    assert id_lock.stat().st_size == 0
    assert ood_lock.stat().st_size == 0


def test_rebuild_rejects_symlink_escape(tmp_path: Path) -> None:
    """
    Reject analysis-root and target symlinks that escape containment.

    Each attempted rebuild points through a different escape shape, and every
    outside or sibling marker must remain intact after rejection.
    """
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside_target = outside / "id"
    outside_target.mkdir(parents=True)
    marker = outside_target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_dir.mkdir()
    (run_dir / "analysis").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "id")

    assert marker.read_text(encoding="utf-8") == "keep"

    target_symlink_run = tmp_path / "target-symlink-run"
    (target_symlink_run / "analysis").mkdir(parents=True)
    (target_symlink_run / "analysis" / "id").symlink_to(outside_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(
            run_dir=target_symlink_run,
            save_root=target_symlink_run / "analysis" / "id",
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    sibling_run = tmp_path / "sibling-symlink-run"
    sibling_target = sibling_run / "analysis" / "ood" / "keep"
    sibling_target.mkdir(parents=True)
    sibling_marker = sibling_target / "keep.txt"
    sibling_marker.write_text("keep", encoding="utf-8")
    (sibling_run / "analysis" / "id").symlink_to(sibling_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(
            run_dir=sibling_run,
            save_root=sibling_run / "analysis" / "id",
        )
    assert sibling_marker.read_text(encoding="utf-8") == "keep"


def test_atomic_publication_rejects_incomplete_stage_without_touching_target(
    tmp_path: Path,
) -> None:
    """
    Reject an incomplete stage without touching the published target.

    Publication must require the provenance completion marker, preserving both
    the existing target and the failed stage for deterministic recovery.
    """
    run_dir = tmp_path / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    target.mkdir(parents=True)
    marker = target / "complete.txt"
    marker.write_text("old", encoding="utf-8")
    stage = analysis.artifacts.service._create_artifact_staging_root(target)
    (stage / "partial.txt").write_text("partial", encoding="utf-8")

    with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match="completion marker"):
        analysis.artifacts.service._publish_staged_artifact(
            run_dir=run_dir,
            save_root=target,
            staging_root=stage,
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert (stage / "partial.txt").is_file()


def test_atomic_publication_rolls_back_when_replacement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Restore the previous target when the staged replacement rename fails.

    An injected filesystem error must leave old content published, retain the
    stage for diagnosis, and clean any temporary backup directory.
    """
    run_dir = tmp_path / "run"
    target = common.paths.resolve_id_analysis_dir(run_dir)
    target.mkdir(parents=True)
    marker = target / "complete.txt"
    marker.write_text("old", encoding="utf-8")
    stage = analysis.artifacts.service._create_artifact_staging_root(target)
    analysis.artifacts.contracts.artifact_provenance_path(stage).write_text("{}\n", encoding="utf-8")
    (stage / "new.txt").write_text("new", encoding="utf-8")
    original_replace = Path.replace

    def fail_stage_replace(source: Path, destination: Path) -> Path:
        """Inject failure only when the new stage would become public."""
        if source == stage:
            message = "injected publication failure"
            raise OSError(message)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", fail_stage_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        analysis.artifacts.service._publish_staged_artifact(
            run_dir=run_dir,
            save_root=target,
            staging_root=stage,
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()
    assert stage.is_dir()
    assert not list(target.parent.glob(".id.backup.*"))


def test_failed_rebuild_generation_preserves_previous_complete_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve the complete cache when rebuild generation fails.

    The injected generator error occurs after request setup; only staging may be
    cleaned, while the prior marker remains the externally visible artifact.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = common.paths.resolve_id_analysis_dir(run_dir)
    target.mkdir(parents=True)
    marker = target / "complete.txt"
    marker.write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        analysis.artifacts.service,
        "_build_artifact_request",
        lambda **_kwargs: analysis.artifacts.service.ArtifactRequest(
            provenance={"request": "replacement"},
            source_indices=(0,),
        ),
    )
    monkeypatch.setattr(analysis.artifacts.service, "_load_run_config", lambda _run_dir: {})
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: object(),
    )
    monkeypatch.setattr(
        learning.inference.context,
        "load_inference_context_with_resolution",
        lambda **_kwargs: (object(), object(), object(), None),
    )

    def fail_generation(**_kwargs: Any) -> None:
        """Model a generator failure after the prior target is established."""
        message = "injected generation failure"
        raise RuntimeError(message)

    monkeypatch.setattr(analysis.artifacts.service.generation, "generate_artifacts", fail_generation)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)

    with pytest.raises(RuntimeError, match="injected generation failure"):
        analysis.artifacts.service._run_or_load_artifacts_locked(
            run_dir=run_dir,
            dataset_name="dataset",
            split="eval",
            max_cases=1,
            batch_size=2,
            device_resolution=learning.device.resolve_device("cpu"),
            dataset_root=tmp_path / "datasets",
            metadata_root=tmp_path / "meta",
            rebuild=True,
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert not list(target.parent.glob(".id.staging.*"))


def test_requested_run_names_cannot_escape_discovery_root(tmp_path: Path) -> None:
    """
    Reject requested run names that escape the discovery root.

    Treating a traversal string as a logical run name must fail before directory
    discovery, preventing explicit selection from bypassing containment.
    """
    with pytest.raises(ValueError, match="single non-empty path component"):
        list(analysis.artifacts.service.iter_run_dirs(tmp_path, run_names=["../outside"]))


def _require_generation(value: object) -> int:
    """
    Validate and normalize one process-worker generation marker.

    Non-integral values signal a malformed worker result and raise rather than
    being silently coerced into a successful concurrency outcome.
    """
    if not isinstance(value, Integral):
        msg = f"Unexpected generation marker: {value!r}"
        raise TypeError(msg)
    return int(value)


def _run_artifact_worker(arguments: dict[str, Any], outcomes: Any) -> None:
    """
    Execute one artifact request and publish a process-safe outcome.

    Success sends the normalized generation marker; failures are reduced to a
    stable type-and-message tuple that the parent process can assert on.
    """
    try:
        frame = analysis.artifacts.service.run_or_load_artifacts(**arguments)
        outcomes.put(("ok", _require_generation(frame.loc[0, "generation"])))
    except Exception as error:
        outcomes.put(("error", f"{type(error).__name__}: {error}"))


@pytest.mark.parametrize("marker_name", common.paths.CURRENT_RUN_REQUIRED_FILES)
def test_discovery_rejects_every_malformed_child_run_marker(
    tmp_path: Path,
    marker_name: str,
) -> None:
    """
    Reject a child containing any lone required run marker.

    The parameter covers every current lifecycle marker; discovery must surface
    each partial run as corruption instead of silently omitting it.
    """
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / marker_name).touch()

    with pytest.raises(experiments.run.RunLifecycleError, match="incomplete and not loadable"):
        list(analysis.artifacts.service.iter_run_dirs(tmp_path))


def test_concurrent_rebuilds_coalesce_to_one_generation_and_one_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Coalesce concurrent rebuilds into one generation and one cache reuse.

    Two forked workers cross the same pre-lock barrier. The target lock must span
    invalidation through validation so both succeed while generation runs once.
    """
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("POSIX fork context is required for artifact lock coverage")
    context = mp.get_context("fork")
    request_barrier = context.Barrier(2)
    generation_started = context.Event()
    release_generation = context.Event()
    generation_count = context.Value("i", 0)
    outcomes = context.Queue()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_root = common.paths.resolve_id_analysis_dir(run_dir)
    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(save_root)

    original_completion_identity = analysis.artifacts.service._completion_marker_identity
    observed_initial_completion = False

    def completion_identity(path: Path) -> tuple[int, int, int, int] | None:
        """Synchronize both workers at their first completion observation."""
        nonlocal observed_initial_completion
        if not observed_initial_completion:
            observed_initial_completion = True
            request_barrier.wait(timeout=10)
        return original_completion_identity(path)

    def build_request(**_kwargs: Any) -> analysis.artifacts.service.ArtifactRequest:
        """Return the shared scientific request used by both workers."""
        return analysis.artifacts.service.ArtifactRequest(
            provenance={"request": "shared"},
            source_indices=(0,),
        )

    def cache_has_outputs(**_kwargs: Any) -> bool:
        """Treat the published completion marker as the cache-output signal."""
        return provenance_path.is_file()

    def load_validated_cache(**kwargs: Any) -> pd.DataFrame:
        """Load generation one only after its completion marker is public."""
        completion = analysis.artifacts.contracts.artifact_provenance_path(Path(kwargs["save_root"]))
        if not completion.is_file():
            msg = "completion marker missing"
            raise RuntimeError(msg)
        return pd.DataFrame([{"generation": 1}])

    def load_context(**_kwargs: Any) -> tuple[object, object, object, None]:
        """Return inert inference collaborators for the lock-focused test."""
        return object(), object(), object(), None

    def generate(**kwargs: Any) -> None:
        """Count, pause, and complete the sole staged artifact generation."""
        with generation_count.get_lock():
            generation_count.value += 1
        generation_started.set()
        if not release_generation.wait(timeout=10):
            msg = "Parent did not release artifact generation"
            raise TimeoutError(msg)
        stage_root = Path(kwargs["save_root"])
        stage_root.mkdir(parents=True, exist_ok=True)
        analysis.artifacts.contracts.artifact_provenance_path(stage_root).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(analysis.artifacts.service, "_completion_marker_identity", completion_identity)
    monkeypatch.setattr(analysis.artifacts.service, "_build_artifact_request", build_request)
    monkeypatch.setattr(analysis.artifacts.service, "_cache_has_outputs", cache_has_outputs)
    monkeypatch.setattr(analysis.artifacts.service, "_load_validated_artifact_cache", load_validated_cache)
    monkeypatch.setattr(analysis.artifacts.service, "_load_run_config", lambda _run_dir: {})
    monkeypatch.setattr(analysis.artifacts.service.experiments.config.loader, "validate_resolved_task_contract", lambda _config: object())
    monkeypatch.setattr(learning.inference.context, "load_inference_context_with_resolution", load_context)
    monkeypatch.setattr(analysis.artifacts.service.generation, "generate_artifacts", generate)
    monkeypatch.setattr(analysis.artifacts.service, "cleanup_runtime", lambda _device: None)

    arguments = {
        "run_dir": run_dir,
        "dataset_name": "dataset",
        "split": "eval",
        "max_cases": 1,
        "batch_size": 1,
        "device_resolution": learning.device.resolve_device("cpu"),
        "dataset_root": tmp_path / "datasets",
        "metadata_root": tmp_path / "meta",
        "rebuild": True,
    }
    workers = [context.Process(target=_run_artifact_worker, args=(arguments, outcomes)) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert generation_started.wait(timeout=10)
        release_generation.set()
        for worker in workers:
            worker.join(timeout=15)
    finally:
        release_generation.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

    assert all(worker.exitcode == 0 for worker in workers)
    assert sorted(outcomes.get(timeout=5) for _ in workers) == [("ok", 1), ("ok", 1)]
    assert generation_count.value == 1
    assert provenance_path.is_file()


def test_rebuild_waiter_recovers_after_prior_generator_removed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep rebuild intent when the preceding owner removed completion.

    The waiter observes a changed completion identity and must enter the locked
    path with ``rebuild=True`` rather than accepting a partial predecessor state.
    """
    observations = iter(((1, 2, 3, 4), None))
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        analysis.artifacts.service,
        "_completion_marker_identity",
        lambda _path: next(observations),
    )

    def run_locked(**kwargs: Any) -> pd.DataFrame:
        """Capture the effective rebuild flag at the serialized boundary."""
        captured["rebuild"] = bool(kwargs["rebuild"])
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(analysis.artifacts.service, "_run_or_load_artifacts_locked", run_locked)
    analysis.artifacts.service.run_or_load_artifacts(
        run_dir=tmp_path / "run",
        dataset_name="dataset",
        split="eval",
        max_cases=1,
        batch_size=1,
        device_resolution=learning.device.resolve_device("cpu"),
        dataset_root=tmp_path / "datasets",
        metadata_root=tmp_path / "meta",
        rebuild=True,
    )

    assert captured == {"rebuild": True}


def test_artifact_operation_waits_for_the_active_run_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Serialize artifact work behind the active run-writer lease.

    A worker may observe request identity before blocking, but its locked body
    must execute only after the resume/training writer releases ownership.
    """
    monkeypatch.setenv("MODEL_TRAINING_DATA_ROOT", str(tmp_path / "training"))
    run_dir = tmp_path / "run"
    observed = threading.Event()
    executed = threading.Event()
    errors: list[Exception] = []

    def completion_identity(_path: Path) -> None:
        """Signal that the artifact worker reached its pre-lock observation."""
        observed.set()

    def run_locked(**_kwargs: Any) -> pd.DataFrame:
        """Signal entry into the serialized artifact implementation."""
        executed.set()
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(analysis.artifacts.service, "_completion_marker_identity", completion_identity)
    monkeypatch.setattr(analysis.artifacts.service, "_run_or_load_artifacts_locked", run_locked)

    def build() -> None:
        """Run the artifact request in a thread and retain unexpected errors."""
        try:
            analysis.artifacts.service.run_or_load_artifacts(
                run_dir=run_dir,
                dataset_name="dataset",
                split="eval",
                max_cases=1,
                batch_size=1,
                device_resolution=learning.device.resolve_device("cpu"),
                dataset_root=tmp_path / "datasets",
                metadata_root=tmp_path / "meta",
                rebuild=True,
            )
        except Exception as error:
            errors.append(error)

    with experiments.run.run_writer_lease(run_dir):
        worker = threading.Thread(target=build)
        worker.start()
        assert observed.wait(timeout=5)
        assert not executed.wait(timeout=0.1)
        assert worker.is_alive()

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors
    assert executed.is_set()
