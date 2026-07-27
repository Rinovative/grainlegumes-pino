"""
===============================================================================
experiments_tracking.py
===============================================================================
Mirror authoritative local experiment state to optional W&B observability.

Responsibilities:
  - Keep disabled tracking free of SDK imports and tracking side effects
  - Persist opaque fresh identities and exact-resume identities locally first
  - Publish bounded semantic config, epoch history, summaries, files and media
  - Degrade online transport failures without invalidating local correctness
  - Surface requested offline-record I/O failures as local operation failures

Design principles:
  - Local config, split, normalizer, checkpoints, summaries and artifacts win
  - W&B observes training and artifacts but never chooses or reconstructs them
  - Fresh and exact-resume sessions have distinct fail-closed identity semantics
  - Secrets, arbitrary environment state and incidental absolute paths are absent

This module does NOT:
  - Own training, checkpoint, scheduler, pruning, or local lifecycle decisions
  - Upload raw datasets, resume-only checkpoints, arbitrary files, or cache internals
  - Make remote availability or W&B state a prerequisite for local correctness
===============================================================================
"""

from __future__ import annotations

import copy
import importlib
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, Unpack, cast

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer

EpochEndCallback = Callable[[int, dict[str, float]], None]
TrackingStateUpdater = Callable[[Mapping[str, Any]], None]
MonitorEvaluator = Callable[[], Mapping[str, float]]

POST_ARTIFACT_MEDIA_KEYS = frozenset(
    {
        "run_summary_table",
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    }
)
_POST_ARTIFACT_FILE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".html", ".pdf"})
_SECRET_KEY_PATTERN = re.compile(r"(?i)(WANDB_API_KEY|api[_-]?key|password|secret|token)(\s*[:=]\s*)([^\s,;]+)")
_MAX_SAFE_ERROR_LENGTH = 600


class TrackingError(RuntimeError):
    """
    Base class for failures owned by the optional tracking boundary.

    These errors describe observer initialization, local offline persistence, or
    upload-admission failure; they never represent local scientific-run validity.
    """


class TrackingInitializationError(TrackingError):
    """
    Represent failure to initialize an explicitly enabled session before epoch one.

    Raised after sanitized failure facts are persisted locally; no training
    telemetry has been accepted by the observer at this boundary.
    """


class TrackingIOError(TrackingError):
    """
    Represent loss of required local records in requested offline mode.

    Unlike an online transport failure, this cannot degrade silently because
    local W&B persistence is the explicitly requested observer destination.
    """


class TrackingUploadError(TrackingError):
    """
    Represent rejection by the run-file or curated-media upload allowlist.

    The error is raised before an SDK mutation for unsupported kinds, paths,
    formats, cache ownership, or disabled checkpoint publication.
    """


class _WandbRun(Protocol):
    """Describe the SDK run surface used by the lifecycle adapter."""

    summary: MutableMapping[str, Any]

    def log(self, data: Mapping[str, Any], *, step: int) -> None:
        """Log one completed epoch."""

    def finish(self, exit_code: int = 0) -> None:
        """Finalize the tracking run."""


class _WandbArtifact(Protocol):
    """Describe the bounded W&B artifact bundle surface."""

    def add_file(self, path: str, *, name: str) -> None:
        """Add one explicit rendered media file."""

    def add(self, value: Any, name: str) -> None:
        """Add one explicit prebuilt table object."""


class _WandbInitKwargs(TypedDict):
    """Type the exact W&B initialization keywords used by this adapter."""

    project: str
    entity: str | None
    group: str | None
    tags: list[str]
    mode: str
    name: str
    id: str
    resume: str | None
    job_type: str
    dir: str
    config: Mapping[str, Any]
    save_code: bool
    settings: Mapping[str, Any]


class _WandbModule(Protocol):
    """Describe the lazily imported W&B module surface."""

    def init(self, **kwargs: Unpack[_WandbInitKwargs]) -> _WandbRun | None:
        """Initialize one W&B run."""

    def Table(self, *, columns: Sequence[str], data: Sequence[Sequence[object]]) -> Any:  # noqa: N802
        """Build one W&B table from a neutral curated payload."""


def _require_initialized_run(run: _WandbRun | None) -> _WandbRun:
    """Return the SDK run or fail through the initialization wrapper."""
    if run is None:
        msg = "wandb.init() did not return a run."
        raise RuntimeError(msg)
    return run


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _safe_error(error: BaseException) -> dict[str, str]:
    """
    Return bounded exception context safe for local or remote tracking records.

    Active W&B keys, key-like assignments, and the current home path are
    redacted before truncation. Only the exception class and sanitized message
    are returned; traceback, environment, and arbitrary object state are absent.
    """
    message = str(error)
    secret = os.environ.get("WANDB_API_KEY")
    if secret:
        message = message.replace(secret, "<redacted>")
    message = _SECRET_KEY_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}<redacted>", message)
    with suppress(RuntimeError):
        home = str(Path.home())
        if home and home != "/":
            message = message.replace(home, "<home>")
    return {
        "error_class": type(error).__name__,
        "error_message": message[:_MAX_SAFE_ERROR_LENGTH],
    }


def _sanitize_semantic_value(value: Any, *, key: str = "") -> Any:
    """
    Recursively copy a semantic value while excluding secrets and host paths.

    Secret-like mapping keys are removed, absolute paths retain only a basename,
    and unsupported objects fail instead of being stringified. The result is a
    bounded JSON-like structure suitable for W&B config, never scientific state.
    """
    lowered = key.lower()
    if any(marker in lowered for marker in ("api_key", "password", "secret", "token")):
        return None
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            name = str(child_key)
            if any(marker in name.lower() for marker in ("api_key", "password", "secret", "token")):
                continue
            sanitized[name] = _sanitize_semantic_value(child_value, key=name)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_semantic_value(item, key=key) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if Path(value).is_absolute():
            return Path(value).name or "<absolute-path-omitted>"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    msg = f"Semantic tracking payload contains unsupported value at {key or '<root>'}: {type(value).__name__}."
    raise TypeError(msg)


def _git_metadata() -> dict[str, Any]:
    """
    Return read-only commit and dirty-state facts without repository identity.

    Fixed local Git commands run with a short timeout. Missing Git, command
    errors, or timeouts return unknown values; no remote URL, branch, author,
    path, or environment state is collected.
    """
    project_root = Path(__file__).resolve().parents[3]
    git_path = shutil.which("git")
    if git_path is None:
        return {"commit": None, "dirty": None}
    try:
        revision = subprocess.run(  # noqa: S603 -- absolute git resolved from trusted PATH
            [git_path, "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(  # noqa: S603 -- fixed read-only git arguments
            [git_path, "status", "--porcelain", "--untracked-files=normal"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    commit = revision.stdout.strip() if revision.returncode == 0 else None
    return {
        "commit": commit or None,
        "dirty": None if status.returncode != 0 else bool(status.stdout),
    }


def build_semantic_config(
    config: Mapping[str, Any],
    *,
    split_indices: Mapping[str, Any],
    normalizer_sha256: str,
    checkpoint_identity: Mapping[str, Any],
    model: Any,
    device_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build the bounded semantic W&B configuration after local admission.

    The payload contains scientific identities and an absolute-path-free
    effective config. Dataset tensors, split tensors, normalizer tensors,
    arbitrary environment variables and secrets are never included.

    Parameters
    ----------
    config : Mapping[str, Any]
        Locally admitted resolved experiment config.
    split_indices : Mapping[str, Any]
        Saved split identity and membership metadata.
    normalizer_sha256 : str
        Digest of the authoritative local normalizer artifact.
    checkpoint_identity : Mapping[str, Any]
        Immutable run identity shared by best and last checkpoints.
    model : Any
        Constructed model used only for bounded parameter counts.
    device_metadata : Mapping[str, Any]
        Serialization-safe requested and resolved runtime facts.

    Returns
    -------
    dict[str, Any]
        Sanitized JSON-like semantic observer configuration.

    Raises
    ------
    TypeError
        If required identity mappings or payload values violate the bounded schema.

    """
    task_contract = config.get("task_contract")
    split_metadata = split_indices.get("metadata")
    if not isinstance(task_contract, Mapping) or not isinstance(split_metadata, Mapping):
        msg = "Semantic tracking config requires resolved task and split metadata."
        raise TypeError(msg)
    datasets = split_metadata.get("datasets")
    membership = split_metadata.get("membership_digests")
    if not isinstance(datasets, Mapping) or not isinstance(membership, Mapping):
        msg = "Semantic tracking config requires dataset identities and membership digests."
        raise TypeError(msg)

    dataset_payload: dict[str, Any] = {}
    for role, raw_identity in datasets.items():
        if not isinstance(raw_identity, Mapping):
            msg = f"split_indices metadata dataset {role!r} must be a mapping."
            raise TypeError(msg)
        dataset_payload[str(role)] = {
            key: copy.deepcopy(raw_identity.get(key))
            for key in (
                "dataset_id",
                "fingerprint",
                "sample_count",
                "spatial_shape",
                "task_contract_digest",
            )
        }

    effective_config = copy.deepcopy(dict(config))
    effective_config.pop("paths", None)
    parameter_counts = {
        "total": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable": sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    }
    physics = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["loss"])["physics"])
    payload = {
        "task": {
            "id": config["task"],
            "schema_version": task_contract.get("schema_version"),
            "contract_digest": task_contract.get("digest"),
            "inputs": task_contract.get("inputs"),
            "outputs": task_contract.get("outputs"),
        },
        "effective_config": effective_config,
        "effective_config_digest": checkpoint_identity.get("effective_config_digest"),
        "model": {
            "kind": cast("Mapping[str, Any]", config["model"]).get("kind"),
            "architecture": cast("Mapping[str, Any]", config["model"]).get("params"),
            "parameter_counts": parameter_counts,
        },
        "datasets": dataset_payload,
        "split_membership_digests": copy.deepcopy(dict(membership)),
        "normalizer": {
            "identity": "saved_run_normalizer.pt",
            "sha256": normalizer_sha256,
            "fit_split": cast("Mapping[str, Any]", task_contract.get("preprocessing", {})).get("fit_split"),
        },
        "objective": copy.deepcopy(cast("Mapping[str, Any]", config["evaluation"])["objective"]),
        "training_loss": copy.deepcopy(cast("Mapping[str, Any]", config["loss"])["data"]),
        "physics": {
            "enabled": physics.get("enabled"),
            "continuity": physics.get("continuity"),
            "derivatives": physics.get("derivatives"),
            "interior_crop": physics.get("interior_crop"),
            "residual_weight": physics.get("residual_weight"),
            "boundary_weight": physics.get("boundary_weight"),
        },
        "reproducibility": {
            "seed": cast("Mapping[str, Any]", config["run"]).get("seed"),
            "deterministic": cast("Mapping[str, Any]", config["run"]).get("deterministic"),
        },
        "device": copy.deepcopy(dict(device_metadata)),
        "source": _git_metadata(),
    }
    return cast("dict[str, Any]", _sanitize_semantic_value(payload))


def build_monitor_membership(
    config: Mapping[str, Any],
    split_indices: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    Build the exact saved-evaluation prefix identity used by physics monitors.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved experiment config with validated W&B monitor settings.
    split_indices : Mapping[str, Any]
        Persisted split artifact including ordered indices and dataset identity.

    Returns
    -------
    dict[str, Any] | None
        Source indices, sample IDs, membership digests, and configured bound when
        monitoring is enabled; otherwise ``None``.

    Notes
    -----
    The membership is fixed before training telemetry and never resampled by epoch.

    """
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    monitor = cast("Mapping[str, Any]", settings["monitor"])
    if not bool(settings["enabled"]) or not bool(monitor["enabled"]):
        return None
    raw_indices = split_indices.get("eval_indices")
    metadata = split_indices.get("metadata")
    if not hasattr(raw_indices, "tolist") or not isinstance(metadata, Mapping):
        msg = "Physics monitor membership requires saved evaluation indices and metadata."
        raise TypeError(msg)
    datasets = metadata.get("datasets")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get("train"), Mapping):
        msg = "Physics monitor membership requires the saved training dataset identity."
        raise TypeError(msg)
    dataset_identity = cast("Mapping[str, Any]", datasets["train"])
    sample_ids = dataset_identity.get("sample_ids")
    fingerprint = dataset_identity.get("fingerprint")
    if not isinstance(sample_ids, list) or not isinstance(fingerprint, str):
        msg = "Physics monitor membership requires ordered sample IDs and fingerprint."
        raise TypeError(msg)
    selected = [int(value) for value in cast("Any", raw_indices).tolist()[: int(monitor["max_cases"])]]
    from src import datasets as dataset_package  # noqa: PLC0415

    digest = dataset_package.identity.membership_digest(
        role="wandb_physics_monitor",
        dataset_fingerprint=fingerprint,
        sample_ids=sample_ids,
        indices=selected,
    )
    membership_digests = metadata.get("membership_digests")
    saved_eval_digest = membership_digests.get("eval") if isinstance(membership_digests, Mapping) else None
    return {
        "source_indices": selected,
        "sample_ids": [sample_ids[index] for index in selected],
        "membership_digest": digest,
        "saved_eval_membership_digest": saved_eval_digest,
        "max_cases": int(monitor["max_cases"]),
    }


def persisted_wandb_identity(summary: Mapping[str, Any]) -> tuple[str, int | None]:
    """
    Recover the sole W&B identity and latest successful epoch from local sessions.

    Parameters
    ----------
    summary : Mapping[str, Any]
        Authoritative run summary containing append-only runtime session records.

    Returns
    -------
    tuple[str, int | None]
        Persisted W&B run ID and maximum locally recorded logged epoch.

    Raises
    ------
    TrackingInitializationError
        If runtime sessions are malformed or contain zero/multiple run IDs.

    """
    raw_sessions = summary.get("runtime_sessions", [])
    if not isinstance(raw_sessions, list):
        msg = "Run summary runtime_sessions must be a list."
        raise TrackingInitializationError(msg)
    identities: set[str] = set()
    last_epoch: int | None = None
    for raw_session in raw_sessions:
        if not isinstance(raw_session, Mapping):
            continue
        state = raw_session.get("tracking")
        if not isinstance(state, Mapping):
            continue
        run_id = state.get("wandb_run_id")
        if isinstance(run_id, str) and run_id:
            identities.add(run_id)
        raw_epoch = state.get("last_logged_epoch")
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool):
            last_epoch = raw_epoch if last_epoch is None else max(last_epoch, raw_epoch)
    if len(identities) != 1:
        msg = f"Exact W&B resume requires one persisted run ID, found {len(identities)}."
        raise TrackingInitializationError(msg)
    return next(iter(identities)), last_epoch


@dataclass(slots=True)
class WandbSession:
    """
    Own one optional W&B mirror while keeping local run state authoritative.

    The session accepts completed-epoch telemetry, a fixed physics monitor, an
    allowlisted file set, and the exact curated post-artifact bundle. Disabled
    sessions are inert. Online write failures transition once to ``degraded`` and
    stop remote mutation; offline persistence failures raise ``TrackingIOError``.

    Attributes
    ----------
    objective_id, objective_direction : str
        Resolved metric identity and model-selection direction mirrored remotely.
    evaluation_metric_ids : frozenset[str]
        Exact evaluation keys admitted into epoch history.
    mode : str
        ``online``, ``offline``, or inert ``disabled`` observer mode.
    run_id : str | None
        Opaque W&B identity persisted locally for exact resume.
    run_dir : pathlib.Path
        Authoritative local run leaf constraining every file upload.
    state_updater : TrackingStateUpdater | None
        Callback that atomically persists observer-only session facts.

    Notes
    -----
    Construction is owned by :func:`initialize_wandb`. ``state_updater`` never
    changes checkpoint selection, scheduler input, Optuna input, saved split
    membership, or the run's scientific result. The session is mutable and
    single-lifecycle; it is not thread-safe and ``finish`` is idempotent.

    """

    _run: _WandbRun | None
    _wandb: Any | None
    objective_id: str
    objective_direction: str
    evaluation_metric_ids: frozenset[str]
    mode: str
    run_id: str | None
    run_dir: Path
    run_name: str
    task_id: str
    upload_settings: Mapping[str, Any]
    semantic_config: Mapping[str, Any] = field(default_factory=dict)
    state_updater: TrackingStateUpdater | None = None
    monitor_evaluator: MonitorEvaluator | None = None
    monitor_interval: int = 1
    terminal_epoch: int = 1
    _last_logged_epoch: int | None = None
    _degraded: bool = False
    _finished: bool = False
    _uploaded_media: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Return whether this session owns an initialized SDK run."""
        return self._run is not None

    @property
    def degraded(self) -> bool:
        """Return whether online remote writes have been disabled."""
        return self._degraded

    def _persist(self, updates: Mapping[str, Any]) -> None:
        """Mirror observer-only facts into the authoritative local runtime session."""
        if self.state_updater is not None:
            self.state_updater(updates)

    def _degrade(self, error: BaseException, *, operation: str) -> None:
        """
        Enter online degraded mode once and persist sanitized failure context.

        Subsequent remote mutations become no-ops. Local training correctness
        and authoritative files remain untouched.
        """
        if self._degraded:
            return
        self._degraded = True
        self._persist(
            {
                "status": "degraded",
                "degraded_operation": operation,
                **_safe_error(error),
            }
        )

    def _offline_failure(self, error: BaseException, *, operation: str) -> TrackingIOError:
        """
        Convert an offline persistence loss into a recorded ``TrackingIOError``.

        Sanitized context is written through ``state_updater`` before the error
        is returned for exception chaining by the caller.
        """
        context = _safe_error(error)
        self._persist({"status": "failed", "failed_operation": operation, **context})
        msg = f"Requested offline W&B {operation} failed: {context['error_class']}: {context['error_message']}"
        return TrackingIOError(msg)

    def _run_operation(self, operation: str, action: Callable[[], None]) -> bool:
        """
        Apply online-degrade or offline-fail policy around one SDK mutation.

        Returns whether the action completed. Disabled, finished, or already
        degraded sessions return false without invoking ``action``.
        """
        if self._run is None or self._finished or self._degraded:
            return False
        try:
            action()
        except Exception as error:
            if self.mode == "online":
                self._degrade(error, operation=operation)
                return False
            raise self._offline_failure(error, operation=operation) from error
        return True

    def log_epoch(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """
        Log one strictly increasing completed epoch with stable semantic keys.

        Parameters
        ----------
        epoch : int
            Actual one-based completed epoch used as the W&B step.
        metrics : Mapping[str, float]
            Sample-weighted training values and declared evaluation metric IDs.

        Raises
        ------
        TrackingError
            If an epoch would rewrite or move behind the last successful step.

        Notes
        -----
        The fixed physics monitor runs only on its interval and terminal epoch.
        Unsupported monitor keys degrade the online observer rather than changing
        local training state.

        """
        if self._run is None or self._finished or self._degraded:
            return
        if self._last_logged_epoch is not None and epoch <= self._last_logged_epoch:
            msg = f"W&B completed-epoch history cannot rewrite step {epoch}; last successful epoch is {self._last_logged_epoch}."
            raise TrackingError(msg)

        payload: dict[str, float | int] = {"epoch": epoch}
        global_step = metrics.get("global_step")
        if global_step is not None:
            payload["global_step"] = int(global_step)
        for key, value in metrics.items():
            if key.startswith(("train/", "monitor/")):
                payload[key] = float(value)
            elif key in self.evaluation_metric_ids:
                payload[f"eval/{key}"] = float(value)
        if self.objective_id in metrics:
            objective_value = float(metrics[self.objective_id])
            payload["eval/objective_value"] = objective_value
            payload["objective/value"] = objective_value

        should_monitor = self.monitor_evaluator is not None and (epoch % self.monitor_interval == 0 or epoch == self.terminal_epoch)
        if should_monitor:
            monitor_evaluator = cast("MonitorEvaluator", self.monitor_evaluator)
            try:
                monitor_values = monitor_evaluator()
            except Exception as error:  # noqa: BLE001 -- observer failures must never escape online
                self._degrade(error, operation="physics_monitor")
                return
            for key, value in monitor_values.items():
                if key not in {
                    "monitor/momentum_residual_mse",
                    "monitor/div_velocity_mse",
                    "monitor/div_eps_velocity_mse",
                    "monitor/pressure_boundary_mse",
                }:
                    msg = f"Physics monitor produced unsupported key {key!r}."
                    self._degrade(ValueError(msg), operation="physics_monitor")
                    return
                payload[key] = float(value)

        if self._run_operation(
            "history",
            lambda: cast("_WandbRun", self._run).log(payload, step=epoch),
        ):
            self._last_logged_epoch = epoch
            self._persist({"last_logged_epoch": epoch})

    def _validate_run_file(self, kind: str, path: Path) -> None:
        """
        Admit one complete run-owned file against the explicit upload allowlist.

        Core config/summary/checkpoint paths must be exact; provenance must be
        the named file below this run's analysis root. No glob, directory, or
        cross-run path is accepted.
        """
        resolved = path.resolve()
        expected = {
            "config": self.run_dir / "config.yaml",
            "summary": self.run_dir / "summary.json",
            "best_checkpoint": self.run_dir / "best_checkpoint.pt",
        }
        if kind in expected and resolved != expected[kind].resolve():
            msg = f"Upload kind {kind!r} requires exactly {expected[kind].name}."
            raise TrackingUploadError(msg)
        if kind == "artifact_provenance":
            try:
                resolved.relative_to((self.run_dir / "analysis").resolve())
            except ValueError as error:
                msg = "Artifact provenance must be inside the current run analysis root."
                raise TrackingUploadError(msg) from error
            if resolved.name != "artifact_provenance.json":
                msg = "Artifact provenance upload requires artifact_provenance.json."
                raise TrackingUploadError(msg)
        if kind not in {*expected, "artifact_provenance"}:
            msg = f"Unsupported tracked file kind {kind!r}."
            raise TrackingUploadError(msg)
        if kind == "best_checkpoint" and not bool(self.upload_settings["best_checkpoint"]):
            msg = "best_checkpoint.pt upload is disabled by configuration."
            raise TrackingUploadError(msg)
        if not resolved.is_file():
            msg = f"Tracked file is not complete: {resolved}"
            raise FileNotFoundError(msg)

    def upload_files(self, files: Mapping[str, Path]) -> None:
        """
        Upload only explicit, validated run-owned files without directory globs.

        Parameters
        ----------
        files : Mapping[str, pathlib.Path]
            Allowlisted semantic kind to exact complete file. Supported kinds are
            config, summary, optional best checkpoint, and artifact provenance.

        Raises
        ------
        TrackingUploadError
            If a kind/path is not allowlisted or checkpoint upload is disabled.
        FileNotFoundError
            If an admitted path is not a complete file.

        """
        if self._run is None or self._finished or self._degraded:
            return
        raw_save = getattr(self._run, "save", None)
        if not callable(raw_save):
            error = TrackingUploadError("The active W&B run does not expose bounded file upload.")
            if self.mode == "online":
                self._degrade(error, operation="file_upload")
                return
            raise self._offline_failure(error, operation="file_upload") from error
        save = cast("Callable[..., Any]", raw_save)
        for kind, path in files.items():
            candidate = Path(path)
            self._validate_run_file(kind, candidate)

            def upload_candidate(candidate: Path = candidate) -> None:
                """Upload one prevalidated file relative to the current run only."""
                save(
                    str(candidate),
                    base_path=str(self.run_dir),
                    policy="now",
                )

            self._run_operation(f"{kind}_upload", upload_candidate)
            if self._degraded:
                return
        if files:
            self._persist({"uploaded_file_kinds": sorted(files)})

    def upload_post_artifact(
        self,
        *,
        artifact_root: Path,
        media_files: Mapping[str, Path] | None = None,
        tables: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Upload one explicit curated post-artifact bundle without plot rendering.

        The artifact service must validate ``artifact_root`` immediately before
        this call. Only the fixed curated scientific inventory is accepted, and
        no directory is scanned by this adapter.

        Parameters
        ----------
        artifact_root : pathlib.Path
            Immutable artifact-cache root; rendered upload files must be outside it.
        media_files : Mapping[str, pathlib.Path] | None, optional
            Curated semantic keys mapped to complete rendered files.
        tables : Mapping[str, Any] | None, optional
            Optional prebuilt or neutral ``run_summary_table`` payload.

        Raises
        ------
        TrackingUploadError
            If inventory, ownership, suffix, table schema, or SDK capability is invalid.
        TrackingIOError
            If requested offline artifact persistence fails.

        Notes
        -----
        Online SDK failures degrade the observer and return without invalidating
        the locally validated artifact cache.

        """
        files = dict(media_files or {})
        table_values = dict(tables or {})
        names = set(files).union(table_values)
        unsupported = sorted(names.difference(POST_ARTIFACT_MEDIA_KEYS))
        if unsupported:
            msg = f"Unsupported post-artifact media key(s): {unsupported}."
            raise TrackingUploadError(msg)
        if set(files).intersection(table_values):
            msg = "Post-artifact keys cannot identify both a file and a table."
            raise TrackingUploadError(msg)
        if "run_summary_table" in files:
            msg = "run_summary_table must be supplied as an already-built table object."
            raise TrackingUploadError(msg)
        if any(name != "run_summary_table" for name in table_values):
            msg = "Only run_summary_table accepts a table object."
            raise TrackingUploadError(msg)
        if not names:
            return
        if self._run is None or self._finished or self._degraded:
            return

        root = Path(artifact_root).resolve()
        for name, raw_path in files.items():
            candidate = Path(raw_path).resolve()
            if candidate.is_relative_to(root):
                msg = f"Curated media {name!r} must be rendered outside the immutable artifact cache."
                raise TrackingUploadError(msg)
            if candidate.suffix.lower() not in _POST_ARTIFACT_FILE_SUFFIXES or not candidate.is_file():
                msg = f"Curated media {name!r} is not an allowed complete rendered file."
                raise TrackingUploadError(msg)

        raw_artifact_factory = getattr(self._wandb, "Artifact", None)
        raw_log_artifact = getattr(self._run, "log_artifact", None)
        if not callable(raw_artifact_factory) or not callable(raw_log_artifact):
            capability_error = TrackingUploadError("The active W&B SDK does not expose artifact media upload.")
            if self.mode == "online":
                self._degrade(capability_error, operation="post_artifact_media")
                return
            raise self._offline_failure(capability_error, operation="post_artifact_media") from capability_error
        artifact_factory = cast("Callable[..., _WandbArtifact]", raw_artifact_factory)
        log_artifact = cast("Callable[..., None]", raw_log_artifact)
        bundle = artifact_factory(
            name=f"{self.task_id}-{self.run_name}-curated-media",
            type="evaluation",
            metadata={"wandb_run_id": self.run_id, "inventory": sorted(names)},
        )
        for name, path in files.items():
            bundle.add_file(str(path), name=f"{name}{path.suffix.lower()}")
        for name, table in table_values.items():
            value = table
            if isinstance(table, Mapping) and set(table) == {"columns", "data"}:
                columns = table["columns"]
                data = table["data"]
                if (
                    not isinstance(columns, Sequence)
                    or isinstance(columns, (str, bytes))
                    or any(not isinstance(column, str) or not column for column in columns)
                    or not isinstance(data, Sequence)
                    or isinstance(data, (str, bytes))
                ):
                    msg = "Neutral run_summary_table payload must contain string columns and row data."
                    raise TrackingUploadError(msg)
                table_factory = getattr(self._wandb, "Table", None)
                if not callable(table_factory):
                    msg = "The active W&B SDK cannot serialize the neutral run summary table."
                    raise TrackingUploadError(msg)
                value = table_factory(columns=list(columns), data=list(data))
            bundle.add(value, name)
        if self._run_operation(
            "post_artifact_media",
            lambda: log_artifact(bundle, aliases=["latest"]),
        ):
            self._uploaded_media = sorted(names)
            self._persist({"uploaded_media": self._uploaded_media})

    def _terminal_summary(
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None,
        local_summary: Mapping[str, Any] | None,
        error: BaseException | str | None,
    ) -> dict[str, Any]:
        """
        Build remote terminal facts exclusively from admitted local identities.

        Scientific identity comes from the sanitized semantic config and the
        authoritative local summary. Training results contribute only bounded
        progress/objective fields, device facts stay operational, and any error
        is redacted through :func:`_safe_error` before SDK publication.
        """
        summary: dict[str, Any] = {
            "run/status": status,
            "objective/id": self.objective_id,
            "objective/direction": self.objective_direction,
            "tracking/status": "finished",
            "tracking/mode": self.mode,
            "tracking/run_id": self.run_id,
            "run/local_role": f"{self.task_id}/runs/{self.run_name}",
        }
        task = self.semantic_config.get("task")
        if isinstance(task, Mapping):
            summary["task/id"] = task.get("id")
            summary["task/contract_digest"] = task.get("contract_digest")
        datasets = self.semantic_config.get("datasets")
        if isinstance(datasets, Mapping):
            summary["data/identities"] = copy.deepcopy(dict(datasets))
        membership = self.semantic_config.get("split_membership_digests")
        if isinstance(membership, Mapping):
            summary["split/membership_digests"] = copy.deepcopy(dict(membership))
        normalizer = self.semantic_config.get("normalizer")
        if isinstance(normalizer, Mapping):
            summary["normalizer/identity"] = normalizer.get("identity")
            summary["normalizer/sha256"] = normalizer.get("sha256")
        physics = self.semantic_config.get("physics")
        if isinstance(physics, Mapping):
            summary["physics/continuity"] = physics.get("continuity")
        if result is not None:
            mapping = {
                "best_metric": "objective/best_value",
                "best_epoch": "objective/best_epoch",
                "completed_epoch": "training/completed_epoch",
                "global_step": "training/global_step",
            }
            for source, target in mapping.items():
                if source in result:
                    summary[target] = result[source]
        if local_summary is not None:
            for source, target in (
                ("task", "task/id"),
                ("effective_config_digest", "config/digest"),
                ("normalizer_sha256", "normalizer/sha256"),
                ("split_indices_sha256", "split/artifact_sha256"),
            ):
                if source in local_summary:
                    summary[target] = local_summary[source]
            runtime_device = local_summary.get("runtime_device")
            if isinstance(runtime_device, Mapping):
                summary["device/requested"] = runtime_device.get("requested_policy")
                summary["device/resolved"] = runtime_device.get("resolved_device")
        if error is not None:
            context = _safe_error(error if isinstance(error, BaseException) else RuntimeError(error))
            summary["run/error_class"] = context["error_class"]
            summary["run/error_message"] = context["error_message"]
        return summary

    def finish(
        self,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        local_summary: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
    ) -> None:
        """
        Mirror terminal local facts and finish the SDK session exactly once.

        Parameters
        ----------
        status : str
            Authoritative local terminal status.
        result : Mapping[str, Any] | None, optional
            Completed training objective/epoch/global-step facts.
        local_summary : Mapping[str, Any] | None, optional
            Validated local digests and device resolution.
        error : BaseException | str | None, optional
            Sanitized terminal failure context.

        Notes
        -----
        Online finish/upload errors degrade and close best-effort. Requested
        offline failures raise after being recorded locally.

        """
        if self._run is None or self._finished:
            return
        exit_code = 0 if status == "completed" else 1
        if self._degraded:
            with suppress(Exception):
                self._run.finish(exit_code=exit_code)
            self._finished = True
            return

        try:
            terminal = self._terminal_summary(
                status=status,
                result=result,
                local_summary=local_summary,
                error=error,
            )
            for key, value in terminal.items():
                self._run.summary[key] = value

            files: dict[str, Path] = {}
            if bool(self.upload_settings["config"]):
                files["config"] = self.run_dir / "config.yaml"
            if bool(self.upload_settings["summary"]):
                files["summary"] = self.run_dir / "summary.json"
            if bool(self.upload_settings["best_checkpoint"]) and (self.run_dir / "best_checkpoint.pt").is_file():
                files["best_checkpoint"] = self.run_dir / "best_checkpoint.pt"
            self.upload_files(files)
            if self._degraded:
                with suppress(Exception):
                    self._run.finish(exit_code=exit_code)
                self._finished = True
                return
            self._run.finish(exit_code=exit_code)
        except Exception as failure:
            if self.mode == "online":
                self._degrade(failure, operation="finish")
                with suppress(Exception):
                    self._run.finish(exit_code=exit_code)
                self._finished = True
                return
            self._finished = True
            raise self._offline_failure(failure, operation="finish") from failure

        self._finished = True
        self._persist({"status": "finished", "finished_at": _utc_now()})


def initialize_wandb(
    config: Mapping[str, Any],
    *,
    run_dir: Path | str,
    semantic_config: Mapping[str, Any] | None = None,
    resume: bool = False,
    persisted_run_id: str | None = None,
    previous_last_logged_epoch: int | None = None,
    state_updater: TrackingStateUpdater | None = None,
    monitor_evaluator: MonitorEvaluator | None = None,
    job_type: str = "training",
) -> WandbSession:
    """
    Initialize W&B only for an explicitly enabled, locally admitted session.

    Fresh online sessions use strict no-resume semantics. Exact online resume
    uses strict must-resume semantics. W&B 0.26 ignores resume in offline mode,
    so offline resume uses the same persisted ID with resume omitted and records
    that narrow same-ID-segment fallback locally.

    Parameters
    ----------
    config : Mapping[str, Any]
        Resolved experiment and validated tracking policy.
    run_dir : Path | str
        Locally admitted run leaf used as the SDK working directory.
    semantic_config : Mapping[str, Any] | None, optional
        Bounded payload built after local task/data/checkpoint admission.
    resume : bool, optional
        Require continuation of ``persisted_run_id`` when true.
    persisted_run_id : str | None, optional
        Sole locally recovered W&B identity for exact resume.
    previous_last_logged_epoch : int | None, optional
        Latest successful local observer step, preventing history rewrites.
    state_updater : TrackingStateUpdater | None, optional
        Atomic local observer-state publisher.
    monitor_evaluator : MonitorEvaluator | None, optional
        Fixed-membership physics monitor invoked at configured epochs.
    job_type : str, optional
        Bounded SDK job classification.

    Returns
    -------
    WandbSession
        Inert disabled session or initialized mutable observer lifecycle.

    Raises
    ------
    TrackingInitializationError
        If enabled initialization, credentials, local storage, or exact identity fails.

    Notes
    -----
    Disabled configuration returns before importing W&B. Enabled failures are
    sanitized and persisted as ``failed_before_start`` before propagation.

    """
    objective = cast("Mapping[str, Any]", config["evaluation"])["objective"]
    objective_mapping = cast("Mapping[str, Any]", objective)
    objective_id = str(objective_mapping["id"])
    objective_direction = str(objective_mapping["direction"])
    evaluation_metrics = cast("Sequence[Mapping[str, Any]]", cast("Mapping[str, Any]", config["evaluation"])["metrics"])
    metric_ids = frozenset(str(metric["id"]) for metric in evaluation_metrics)
    settings = cast("Mapping[str, Any]", cast("Mapping[str, Any]", config["tracking"])["wandb"])
    path = Path(run_dir)
    run_name = str(cast("Mapping[str, Any]", config["run"])["name"])
    task_id = str(config["task"])
    upload_settings = cast("Mapping[str, Any]", settings["upload"])
    monitor_settings = cast("Mapping[str, Any]", settings["monitor"])

    if not bool(settings["enabled"]):
        return WandbSession(
            None,
            None,
            objective_id,
            objective_direction,
            metric_ids,
            "disabled",
            None,
            path,
            run_name,
            task_id,
            upload_settings,
        )

    mode = str(settings["mode"])
    run_id = persisted_run_id if resume else uuid.uuid4().hex
    if not isinstance(run_id, str) or not run_id:
        msg = "Exact W&B resume requires the previously persisted non-empty run ID."
        raise TrackingInitializationError(msg)
    resume_policy: str | None
    offline_fallback: str | None = None
    if mode == "offline":
        resume_policy = None
        if resume:
            offline_fallback = "wandb_0_26_ignores_resume_same_persisted_id_segment"
    else:
        resume_policy = "must" if resume else "never"

    base_state = {
        "enabled": True,
        "requested_mode": mode,
        "wandb_run_id": run_id,
        "project": settings["project"],
        "entity": settings["entity"],
        "group": settings["group"],
        "tags": list(cast("list[str]", settings["tags"])),
        "session_started_at": _utc_now(),
        "session_kind": "resume" if resume else "fresh",
        "status": "offline" if mode == "offline" else "active",
    }
    if offline_fallback is not None:
        base_state["offline_resume_fallback"] = offline_fallback
    if state_updater is not None:
        state_updater(base_state)

    try:
        wandb = cast("_WandbModule", importlib.import_module("wandb"))
        sdk_run = wandb.init(
            project=str(settings["project"]),
            entity=cast("str | None", settings["entity"]),
            group=cast("str | None", settings["group"]),
            tags=list(cast("list[str]", settings["tags"])),
            mode=mode,
            name=run_name,
            id=run_id,
            resume=resume_policy,
            job_type=job_type,
            dir=str(path),
            config=copy.deepcopy(dict(semantic_config or {})),
            save_code=False,
            settings={
                "disable_git": True,
                "disable_code": True,
                "_disable_stats": True,
            },
        )
        sdk_run = _require_initialized_run(sdk_run)
    except Exception as error:
        context = _safe_error(error)
        if state_updater is not None:
            state_updater(
                {
                    "status": "failed_before_start",
                    "failed_operation": "initialization",
                    **context,
                }
            )
        message = (
            f"tracking.wandb.mode={mode!r} initialization failed before epoch 1: "
            f"{context['error_class']}: {context['error_message']}. "
            "Verify W&B installation, local write access, and online WANDB_API_KEY authentication when applicable."
        )
        raise TrackingInitializationError(message) from error

    return WandbSession(
        sdk_run,
        wandb,
        objective_id,
        objective_direction,
        metric_ids,
        mode,
        run_id,
        path,
        run_name,
        task_id,
        upload_settings,
        semantic_config=copy.deepcopy(dict(semantic_config or {})),
        state_updater=state_updater,
        monitor_evaluator=monitor_evaluator,
        monitor_interval=int(monitor_settings["interval"]),
        terminal_epoch=int(cast("Mapping[str, Any]", config["training"])["epochs"]),
        _last_logged_epoch=previous_last_logged_epoch,
    )


def epoch_callback(
    session: WandbSession,
    optimizer: Optimizer | None = None,
) -> EpochEndCallback | None:
    """
    Bind an enabled W&B session to completed-epoch telemetry.

    Parameters
    ----------
    session : WandbSession
        Initialized or disabled observer session.
    optimizer : torch.optim.Optimizer | None, optional
        Optimizer whose first parameter-group learning rate is added when absent.

    Returns
    -------
    EpochEndCallback | None
        Callback for the training loop, or ``None`` for a disabled session.

    """
    if not session.enabled:
        return None

    def callback(epoch: int, metrics: dict[str, float]) -> None:
        """Add the current optimizer rate and forward one completed-epoch payload."""
        values = dict(metrics)
        if "train/learning_rate" not in values and optimizer is not None:
            parameter_groups = optimizer.param_groups
            if not parameter_groups:
                msg = "Cannot log W&B learning rate: optimizer has no parameter groups."
                raise RuntimeError(msg)
            values["train/learning_rate"] = float(parameter_groups[0]["lr"])
        session.log_epoch(epoch, values)

    return callback


def combine_epoch_callbacks(
    *callbacks: EpochEndCallback | None,
) -> EpochEndCallback | None:
    """
    Combine non-null completed-epoch consumers without changing their order.

    Parameters
    ----------
    *callbacks : EpochEndCallback | None
        Scheduler-independent lifecycle callbacks such as Optuna reporting and W&B.

    Returns
    -------
    EpochEndCallback | None
        Ordered composite, or ``None`` when every input is disabled.

    Notes
    -----
    Exceptions propagate immediately, so later consumers do not observe an epoch
    rejected by an earlier authoritative consumer.

    """
    active = tuple(callback for callback in callbacks if callback is not None)
    if not active:
        return None

    def combined(epoch: int, metrics: dict[str, float]) -> None:
        """Invoke lifecycle consumers in caller-specified order for one epoch."""
        for callback in active:
            callback(epoch, metrics)

    return combined
