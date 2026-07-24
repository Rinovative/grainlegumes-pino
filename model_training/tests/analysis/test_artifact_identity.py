# ruff: noqa: BLE001, S101, SLF001, TC003
"""Verify artifact metadata integrity and exact, contained rebuild behavior."""

from __future__ import annotations

import multiprocessing as mp
import threading
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from src import analysis, common, experiments


def test_metadata_cannot_duplicate_authoritative_identity_columns() -> None:
    """Flattening rejects metadata that would create ambiguous columns."""
    frame = pd.DataFrame(
        [{"case_index": 3, "source_index": 2, "meta": '{"source_index": 999}'}],
    )

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_cannot_reuse_the_raw_meta_column() -> None:
    """Flattened metadata cannot erase itself through a duplicate raw column name."""
    frame = pd.DataFrame([{"case_index": 3, "meta": '{"meta": "ambiguous"}'}])

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_expansion_preserves_noncontiguous_row_indices() -> None:
    """Metadata rows remain aligned with the authoritative Parquet row index."""
    row_index = 7
    case_index = 8
    frame = pd.DataFrame(
        [{"case_index": case_index, "meta": '{"label": "case-eight"}'}],
        index=[row_index],
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
    """Malformed or non-object metadata payloads fail loudly."""
    with pytest.raises(error_type, match=match):
        analysis.evaluation.dataframe.build_eval_df(pd.DataFrame([{"meta": metadata}]))


def test_rebuild_removes_only_one_exact_target(tmp_path: Path) -> None:
    """ID rebuild cannot remove OOD siblings or the shared analysis root."""
    run_dir = tmp_path / "run"
    id_target = run_dir / "analysis" / "id"
    first_ood = run_dir / "analysis" / "ood" / "first"
    second_ood = run_dir / "analysis" / "ood" / "second"
    for target in (id_target, first_ood, second_ood):
        target.mkdir(parents=True)
        (target / "marker").write_text(target.name, encoding="utf-8")

    analysis.artifact_service.rebuild_artifact_target(run_dir=run_dir, save_root=first_ood)

    assert not first_ood.exists()
    assert (id_target / "marker").is_file()
    assert (second_ood / "marker").is_file()
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifact_service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifact_service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "ood")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifact_service.rebuild_artifact_target(run_dir=run_dir, save_root=tmp_path / "outside")


def test_rebuild_rejects_symlink_escape(tmp_path: Path) -> None:
    """An analysis or target symlink cannot authorize deletion outside its run."""
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside_target = outside / "id"
    outside_target.mkdir(parents=True)
    marker = outside_target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_dir.mkdir()
    (run_dir / "analysis").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifact_service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "id")

    assert marker.read_text(encoding="utf-8") == "keep"

    target_symlink_run = tmp_path / "target-symlink-run"
    (target_symlink_run / "analysis").mkdir(parents=True)
    (target_symlink_run / "analysis" / "id").symlink_to(outside_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifact_service.rebuild_artifact_target(
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
        analysis.artifact_service.rebuild_artifact_target(
            run_dir=sibling_run,
            save_root=sibling_run / "analysis" / "id",
        )
    assert sibling_marker.read_text(encoding="utf-8") == "keep"


def test_requested_run_names_cannot_escape_discovery_root(tmp_path: Path) -> None:
    """Explicit run selection accepts one logical child name only."""
    with pytest.raises(ValueError, match="single non-empty path component"):
        list(analysis.artifact_service.iter_run_dirs(tmp_path, run_names=["../outside"]))


def _require_generation(value: object) -> int:
    """Return one integral worker generation marker."""
    if not isinstance(value, Integral):
        msg = f"Unexpected generation marker: {value!r}"
        raise TypeError(msg)
    return int(value)


def _run_artifact_worker(arguments: dict[str, Any], outcomes: Any) -> None:
    """Run one artifact request and return a compact process-safe outcome."""
    try:
        frame = analysis.artifact_service.run_or_load_artifacts(**arguments)
        outcomes.put(("ok", _require_generation(frame.loc[0, "generation"])))
    except Exception as error:
        outcomes.put(("error", f"{type(error).__name__}: {error}"))


@pytest.mark.parametrize("marker_name", common.paths.CURRENT_RUN_REQUIRED_FILES)
def test_discovery_rejects_every_malformed_child_run_marker(
    tmp_path: Path,
    marker_name: str,
) -> None:
    """A partial child run is never silently omitted from container discovery."""
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / marker_name).touch()

    with pytest.raises(experiments.run.RunLifecycleError, match="incomplete and not loadable"):
        list(analysis.artifact_service.iter_run_dirs(tmp_path))


def test_concurrent_rebuilds_coalesce_to_one_generation_and_one_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target lock spans rebuild, cache check, generation, and validation."""
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
    provenance_path = analysis.artifacts.artifact_provenance_path(save_root)

    original_completion_identity = analysis.artifact_service._completion_marker_identity
    observed_initial_completion = False

    def completion_identity(path: Path) -> tuple[int, int, int, int] | None:
        nonlocal observed_initial_completion
        if not observed_initial_completion:
            observed_initial_completion = True
            request_barrier.wait(timeout=10)
        return original_completion_identity(path)

    def build_request(**_kwargs: Any) -> analysis.artifact_service.ArtifactRequest:
        return analysis.artifact_service.ArtifactRequest(
            provenance={"request": "shared"},
            source_indices=(0,),
        )

    def cache_has_outputs(**_kwargs: Any) -> bool:
        return provenance_path.is_file()

    def load_validated_cache(**_kwargs: Any) -> pd.DataFrame:
        if not provenance_path.is_file():
            msg = "completion marker missing"
            raise RuntimeError(msg)
        return pd.DataFrame([{"generation": 1}])

    def load_context(**_kwargs: Any) -> tuple[object, object, object, None]:
        return object(), object(), object(), None

    def generate(**_kwargs: Any) -> None:
        with generation_count.get_lock():
            generation_count.value += 1
        generation_started.set()
        if not release_generation.wait(timeout=10):
            msg = "Parent did not release artifact generation"
            raise TimeoutError(msg)
        save_root.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(analysis.artifact_service, "_completion_marker_identity", completion_identity)
    monkeypatch.setattr(analysis.artifact_service, "_build_artifact_request", build_request)
    monkeypatch.setattr(analysis.artifact_service, "_cache_has_outputs", cache_has_outputs)
    monkeypatch.setattr(analysis.artifact_service, "_load_validated_artifact_cache", load_validated_cache)
    monkeypatch.setattr(analysis.artifact_service, "_load_run_config", lambda _run_dir: {})
    monkeypatch.setattr(analysis.artifact_service.experiments.config.loader, "validate_resolved_task_contract", lambda _config: object())
    monkeypatch.setattr(analysis.artifact_service.learning.inference.context, "load_inference_context", load_context)
    monkeypatch.setattr(analysis.artifact_service.artifacts, "generate_artifacts", generate)
    monkeypatch.setattr(analysis.artifact_service, "cleanup_gpu", lambda: None)

    arguments = {
        "run_dir": run_dir,
        "dataset_name": "dataset",
        "split": "eval",
        "max_cases": 1,
        "batch_size": 1,
        "prefer_cuda": False,
        "dataset_root": tmp_path / "datasets",
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
    """A waiter still rebuilds when the preceding rebuild owner failed partially."""
    observations = iter(((1, 2, 3, 4), None))
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        analysis.artifact_service,
        "_completion_marker_identity",
        lambda _path: next(observations),
    )

    def run_locked(**kwargs: Any) -> pd.DataFrame:
        captured["rebuild"] = bool(kwargs["rebuild"])
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(analysis.artifact_service, "_run_or_load_artifacts_locked", run_locked)
    analysis.artifact_service.run_or_load_artifacts(
        run_dir=tmp_path / "run",
        dataset_name="dataset",
        split="eval",
        max_cases=1,
        batch_size=1,
        prefer_cuda=False,
        dataset_root=tmp_path / "datasets",
        rebuild=True,
    )

    assert captured == {"rebuild": True}


def test_artifact_operation_waits_for_the_active_run_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact reads and writes cannot race an active resume/training writer."""
    run_dir = tmp_path / "run"
    observed = threading.Event()
    executed = threading.Event()
    errors: list[Exception] = []

    def completion_identity(_path: Path) -> None:
        observed.set()

    def run_locked(**_kwargs: Any) -> pd.DataFrame:
        executed.set()
        return pd.DataFrame([{"ok": 1}])

    monkeypatch.setattr(analysis.artifact_service, "_completion_marker_identity", completion_identity)
    monkeypatch.setattr(analysis.artifact_service, "_run_or_load_artifacts_locked", run_locked)

    def build() -> None:
        try:
            analysis.artifact_service.run_or_load_artifacts(
                run_dir=run_dir,
                dataset_name="dataset",
                split="eval",
                max_cases=1,
                batch_size=1,
                prefer_cuda=False,
                dataset_root=tmp_path / "datasets",
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
