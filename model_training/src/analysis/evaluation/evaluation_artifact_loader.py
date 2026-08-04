"""
===============================================================================
evaluation_artifact_loader.py
===============================================================================
Load strict path-selected evaluation artifacts without generation or inference.

Responsibilities:
  - Admit one explicit current run directory independently of its storage alias
  - Derive exact ID and OOD roots from authoritative saved split identity
  - Validate artifact schemas, payload digests, provenance, and membership
  - Return comparison-ready DataFrames with stable canonical identity digests
  - Report actionable host commands for missing or incompatible artifacts

Design principles:
  - Scientific run and saved-split evidence remain the source of artifact identity
  - ID and OOD roles are explicit and cannot substitute for one another
  - Artifact admission is read-only, exact, and fail-closed
  - Presentation labels and operational runtime facts remain outside scientific identity

This module does NOT:
  - Import artifact generation, reconstruct models, or execute inference
  - Create, repair, rebuild, publish, upload, or remove artifact payloads
  - Render figures or manage evaluation-session caches
===============================================================================
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src import common, domain, experiments
from src.analysis.artifacts import contracts

from . import evaluation_dataframe as dataframe

if TYPE_CHECKING:
    import pandas as pd

    from src.domain.tasks.domain_task_spec import TaskSpec

ArtifactRole = Literal["eval", "ood"]
ArtifactSelectionRole = Literal["id", "ood"]

__all__ = [
    "ArtifactRole",
    "ArtifactSelectionRole",
    "EvaluationArtifactLoadError",
    "IncompatibleEvaluationArtifactsError",
    "LoadedEvaluationArtifact",
    "LoadedRunArtifacts",
    "MissingEvaluationArtifactsError",
    "load_completed_run_artifacts",
    "load_run_artifacts",
]


class EvaluationArtifactLoadError(RuntimeError):
    """
    Represent failure to admit run-owned evaluation artifacts.

    Parameters
    ----------
    message : str
        Human-readable admission failure and recovery guidance.
    role : {"eval", "ood"} | None
        Rejected artifact role when the failure is role-local.
    run_dir : pathlib.Path | None
        Exact current run directory when it is known.

    Notes
    -----
    Missing and incompatible targets use distinct subclasses so notebook callers
    can present the appropriate host-side recovery command without inspecting
    exception text.

    """

    def __init__(self, message: str, *, role: ArtifactRole | None = None, run_dir: Path | None = None) -> None:
        """Initialize the message and structured recovery context."""
        super().__init__(message)
        self.role = role
        self.run_dir = run_dir


class MissingEvaluationArtifactsError(EvaluationArtifactLoadError):
    """
    Represent absence of a required ID or OOD artifact target.

    Notes
    -----
    This error is reserved for an absent or empty target. Partial targets are
    incompatible because normal artifact generation must not trust them.

    """


class IncompatibleEvaluationArtifactsError(EvaluationArtifactLoadError):
    """
    Represent a stale, partial, changed, or contradictory artifact target.

    Notes
    -----
    The associated message identifies the rejected role and supplies the exact
    host-side rebuild command for the selected task and canonical run name.

    """


@dataclass(frozen=True, slots=True)
class LoadedEvaluationArtifact:
    """
    Store one validated role-local artifact owned by an evaluable run.

    Parameters
    ----------
    split_role : {"eval", "ood"}
        Exact saved split role represented by the artifact.
    dataset_name : str
        Canonical saved dataset identity for the role.
    root : pathlib.Path
        Exact resolved run-owned artifact root.
    frame : pandas.DataFrame
        Comparison-ready table with complete validated provenance attributes.
    identity_sha256 : str
        Canonical SHA-256 digest of the complete validated provenance document.

    """

    split_role: ArtifactRole
    dataset_name: str
    root: Path
    frame: pd.DataFrame
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class LoadedRunArtifacts:
    """
    Store validated ID and OOD artifacts for one scientifically identified run.

    Parameters
    ----------
    task : str
        Canonical registered task identifier.
    run_name : str
        Immutable scientific run name used for persisted identity.
    run_dir : pathlib.Path
        Exact resolved current run directory.
    id_artifact : LoadedEvaluationArtifact
        Artifact bound to saved evaluation membership and ``analysis/id``.
    ood_artifact : LoadedEvaluationArtifact
        Artifact bound to saved OOD membership and its named OOD root.

    """

    task: str
    run_name: str
    run_dir: Path
    id_artifact: LoadedEvaluationArtifact | None
    ood_artifact: LoadedEvaluationArtifact | None
    lifecycle_status: str = "completed"
    is_completed: bool = True
    is_provisional: bool = False
    selected_checkpoint_role: str = "best"
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_sha256: str = ""
    summary: Mapping[str, Any] | None = None

    @property
    def scientific_run_name(self) -> str:
        """Return the immutable scientific identity saved in config.yaml."""
        return self.run_name

    @property
    def storage_alias(self) -> str:
        """Return the mutable current directory leaf used for storage."""
        return self.run_dir.name


@dataclass(frozen=True, slots=True)
class _SavedRole:
    """
    Store authoritative saved-run evidence for one artifact role.

    Parameters
    ----------
    split_role : {"eval", "ood"}
        Exact persisted membership role.
    dataset_name : str
        Canonical saved dataset name.
    dataset_identity : collections.abc.Mapping
        Validated compact dataset identity from ``split_indices.pt``.
    source_indices : tuple[int, ...]
        Exact ordered saved source membership.
    saved_membership_digest : str
        Saved role-local membership digest validated by the evaluable-run gate.
    root : pathlib.Path
        Exact resolved run-owned artifact root for the role.

    """

    split_role: ArtifactRole
    dataset_name: str
    dataset_identity: Mapping[str, Any]
    source_indices: tuple[int, ...]
    saved_membership_digest: str
    root: Path


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """
    Return one mapping with a path-qualified error.

    Parameters
    ----------
    value : Any
        Candidate mapping value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    collections.abc.Mapping
        The admitted mapping without copying it.

    Raises
    ------
    TypeError
        If ``value`` is not a mapping.

    """
    if not isinstance(value, Mapping):
        msg = f"{label} must be a mapping."
        raise TypeError(msg)
    return value


def _positive_int(value: Any, *, label: str) -> int:
    """
    Return one positive integer while rejecting boolean aliases.

    Parameters
    ----------
    value : Any
        Candidate integral value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    int
        Exact positive integer value.

    Raises
    ------
    TypeError
        If the value is boolean, non-integral, zero, or negative.

    """
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        msg = f"{label} must be a positive integer."
        raise TypeError(msg)
    return int(value)


def _nonempty_string(value: Any, *, label: str) -> str:
    """
    Return one non-empty string.

    Parameters
    ----------
    value : Any
        Candidate string value.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    str
        The admitted non-empty string.

    Raises
    ------
    TypeError
        If the value is not non-empty text.

    """
    if not isinstance(value, str) or not value:
        msg = f"{label} must be a non-empty string."
        raise TypeError(msg)
    return value


def _indices(value: Any, *, label: str) -> tuple[int, ...]:
    """
    Return exact non-negative ordered source indices from saved split state.

    Parameters
    ----------
    value : Any
        Tensor-like or sequence-like saved membership.
    label : str
        Semantic path used in validation failures.

    Returns
    -------
    tuple[int, ...]
        Exact ordered source indices.

    Raises
    ------
    TypeError
        If the membership is empty or contains non-integral values.
    ValueError
        If the membership contains duplicate source indices.

    """
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)) or not raw:
        msg = f"{label} must contain a non-empty ordered index sequence."
        raise TypeError(msg)
    if any(isinstance(item, bool) or not isinstance(item, Integral) or int(item) < 0 for item in raw):
        msg = f"{label} must contain only non-negative integers."
        raise TypeError(msg)
    result = tuple(int(item) for item in raw)
    if len(result) != len(set(result)):
        msg = f"{label} contains duplicate source indices."
        raise ValueError(msg)
    return result


def _artifact_command(*, run_dir: Path, rebuild: bool) -> str:
    """Return the host wrapper command for one exact current run directory."""
    suffix = " --rebuild" if rebuild else ""
    return f"./scripts/docker_job.sh --queue-gpu auto artifacts --run-dir {shlex.quote(str(run_dir))}{suffix}"


def _missing(*, task: str, run_name: str, run_dir: Path, role: ArtifactRole, root: Path) -> MissingEvaluationArtifactsError:
    """
    Build an actionable missing-artifact failure.

    Parameters
    ----------
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory used by the recovery command.
    role : {"eval", "ood"}
        Missing saved split role.
    root : pathlib.Path
        Exact expected artifact root.

    Returns
    -------
    MissingEvaluationArtifactsError
        Failure carrying the generic host-side build command.

    """
    command = _artifact_command(run_dir=run_dir, rebuild=False)
    return MissingEvaluationArtifactsError(
        f"Evaluation artifacts are missing for run {task}/{run_name} ({role} expected at {root}).\nBuild them locally:\n  {command}",
        role=role,
        run_dir=run_dir,
    )


def _incompatible(
    *,
    task: str,
    run_name: str,
    run_dir: Path,
    role: ArtifactRole,
    root: Path,
    reason: str,
) -> IncompatibleEvaluationArtifactsError:
    """
    Build an actionable incompatible-artifact failure.

    Parameters
    ----------
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory used by the recovery command.
    role : {"eval", "ood"}
        Rejected saved split role.
    root : pathlib.Path
        Exact observed artifact root.
    reason : str
        Path-qualified admission failure.

    Returns
    -------
    IncompatibleEvaluationArtifactsError
        Failure carrying the generic host-side rebuild command.

    """
    command = _artifact_command(run_dir=run_dir, rebuild=True)
    return IncompatibleEvaluationArtifactsError(
        f"Evaluation artifacts are incompatible with run {task}/{run_name} "
        f"({role} at {root}): {reason}\n"
        f"Rebuild only the exact run-owned target explicitly:\n  {command}",
        role=role,
        run_dir=run_dir,
    )


def _saved_dataset_identity(
    metadata: Mapping[str, Any],
    *,
    identity_key: Literal["train", "ood"],
    task: str,
    data_contract_digest: str,
) -> Mapping[str, Any]:
    """
    Admit one compact dataset identity from validated saved split evidence.

    Parameters
    ----------
    metadata : collections.abc.Mapping
        Saved split metadata containing dataset identities.
    identity_key : {"train", "ood"}
        Dataset identity supplying the requested role.
    task : str
        Canonical completed-run task.
    data_contract_digest : str
        Exact validated dataset data-contract digest.

    Returns
    -------
    collections.abc.Mapping
        Admitted saved dataset identity.

    Raises
    ------
    TypeError, ValueError
        If names, counts, fingerprints, or task identity are malformed or
        contradictory.

    """
    datasets = _mapping(metadata.get("datasets"), label="split_indices.pt metadata.datasets")
    identity = _mapping(
        datasets.get(identity_key),
        label=f"split_indices.pt metadata.datasets.{identity_key}",
    )
    dataset_name = common.paths.validate_logical_name(
        identity.get("dataset_id"),
        label=f"split_indices.pt metadata.datasets.{identity_key}.dataset_id",
    )
    if identity.get("task") != task or identity.get("data_contract_digest") != data_contract_digest:
        msg = f"Saved {identity_key} dataset identity contradicts the evaluable run data contract."
        raise ValueError(msg)
    _nonempty_string(identity.get("fingerprint"), label=f"saved {identity_key} dataset fingerprint")
    _positive_int(identity.get("sample_count"), label=f"saved {identity_key} dataset sample_count")
    if identity.get("dataset_id") != dataset_name:
        msg = f"Saved {identity_key} dataset name is not canonical."
        raise ValueError(msg)
    return identity


def _saved_roles(
    *,
    completed: Mapping[str, Any],
    run_dir: Path,
    task: str,
    task_spec: TaskSpec,
) -> tuple[_SavedRole, _SavedRole]:
    """
    Resolve exact ID and OOD roots and ordered memberships from one run.

    Parameters
    ----------
    completed : collections.abc.Mapping
        Result of strict completed-run validation.
    run_dir : pathlib.Path
        Exact canonical completed-run directory.
    task : str
        Canonical registered task identifier.
    task_spec : TaskSpec
        Validated persisted task contract.

    Returns
    -------
    tuple[_SavedRole, _SavedRole]
        ID evaluation role followed by the sole named OOD role.

    Raises
    ------
    TypeError, ValueError
        If config dataset names, compact identities, counts, or membership
        digests disagree.

    """
    config = _mapping(completed.get("config"), label="completed config")
    split = _mapping(completed.get("split_indices"), label="completed split_indices")
    metadata = _mapping(split.get("metadata"), label="split_indices.pt metadata")
    data_config = _mapping(config.get("data"), label="config.yaml data")
    membership = _mapping(metadata.get("membership_digests"), label="split_indices.pt metadata.membership_digests")

    id_identity = _saved_dataset_identity(
        metadata,
        identity_key="train",
        task=task,
        data_contract_digest=task_spec.data_contract_digest,
    )
    ood_identity = _saved_dataset_identity(
        metadata,
        identity_key="ood",
        task=task,
        data_contract_digest=task_spec.data_contract_digest,
    )
    id_name = str(id_identity["dataset_id"])
    ood_name = str(ood_identity["dataset_id"])
    if data_config.get("train_dataset") != id_name:
        msg = "config.yaml data.train_dataset contradicts saved ID dataset identity."
        raise ValueError(msg)
    configured_ood = data_config.get("ood_datasets")
    if not isinstance(configured_ood, list) or configured_ood != [ood_name]:
        msg = "config.yaml data.ood_datasets must exactly match the sole saved OOD dataset identity."
        raise ValueError(msg)

    role_specs: tuple[tuple[ArtifactRole, str, Mapping[str, Any], str, str, str, str, Path], ...] = (
        (
            "eval",
            id_name,
            id_identity,
            "eval_indices",
            "eval",
            "n_eval",
            "n_train_full",
            common.paths.resolve_id_analysis_dir(run_dir),
        ),
        (
            "ood",
            ood_name,
            ood_identity,
            "ood_indices",
            "ood",
            "n_ood",
            "n_ood_full",
            common.paths.resolve_ood_analysis_dir(run_dir, ood_name),
        ),
    )
    roles: list[_SavedRole] = []
    for split_role, dataset_name, identity, index_key, membership_key, count_key, full_count_key, root in role_specs:
        source_indices = _indices(split.get(index_key), label=f"split_indices.pt {index_key}")
        if _positive_int(metadata.get(count_key), label=f"split_indices.pt metadata.{count_key}") != len(source_indices):
            msg = f"Saved {split_role} membership count contradicts its ordered indices."
            raise ValueError(msg)
        full_count = _positive_int(metadata.get(full_count_key), label=f"split_indices.pt metadata.{full_count_key}")
        if full_count != _positive_int(identity.get("sample_count"), label=f"saved {split_role} sample_count"):
            msg = f"Saved {split_role} full count contradicts its dataset identity."
            raise ValueError(msg)
        saved_digest = _nonempty_string(
            membership.get(membership_key),
            label=f"split_indices.pt metadata.membership_digests.{membership_key}",
        )
        roles.append(
            _SavedRole(
                split_role=split_role,
                dataset_name=dataset_name,
                dataset_identity=identity,
                source_indices=source_indices,
                saved_membership_digest=saved_digest,
                root=Path(root).absolute(),
            )
        )
    id_role, ood_role = roles
    return id_role, ood_role


def _expected_run_provenance(
    *,
    summary: Mapping[str, Any],
    task_spec: TaskSpec,
    run_name: str,
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project immutable scientific and optional lifecycle evaluation evidence."""
    result = {
        "name": run_name,
        "task": task_spec.id,
        "task_contract_digest": task_spec.contract_digest,
        "effective_config_digest": summary.get("effective_config_digest"),
        "best_checkpoint_sha256": summary.get("best_checkpoint_sha256"),
        "normalizer_sha256": summary.get("normalizer_sha256"),
    }
    if evaluation is not None:
        result.update(
            {
                "best_checkpoint_epoch": evaluation.get("selected_checkpoint_epoch"),
                "lifecycle_status": evaluation.get("lifecycle_status"),
                "is_completed": evaluation.get("is_completed"),
                "is_provisional": evaluation.get("is_provisional"),
                "selected_checkpoint_role": evaluation.get("selected_checkpoint_role"),
            }
        )
    return result


def _expected_model_provenance(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Project architecture, exact capacity, and physics selection.

    Parameters
    ----------
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary carrying parameter counts.

    Returns
    -------
    dict[str, Any]
        Exact model provenance required from the artifact.

    Raises
    ------
    TypeError, ValueError
        If model, physics, or parameter-count evidence is malformed.

    """
    model = _mapping(config.get("model"), label="config.yaml model")
    architecture = _mapping(model.get("params"), label="config.yaml model.params")
    loss = _mapping(config.get("loss"), label="config.yaml loss")
    physics = _mapping(loss.get("physics"), label="config.yaml loss.physics")
    result = {
        "kind": model.get("kind"),
        "architecture": dict(architecture),
        "physics_enabled": physics.get("enabled") is True,
    }
    raw_counts = summary.get("model_parameter_counts")
    if raw_counts is not None:
        counts = _mapping(raw_counts, label="summary model_parameter_counts")
        exact_counts = {
            "total": _positive_int(counts.get("total"), label="summary model_parameter_counts.total"),
            "trainable": _positive_int(counts.get("trainable"), label="summary model_parameter_counts.trainable"),
        }
        if exact_counts["trainable"] > exact_counts["total"]:
            msg = "Run trainable parameter count exceeds total parameter count."
            raise ValueError(msg)
        result["parameter_counts"] = exact_counts
    return result


def _expected_normalizer_provenance(
    *,
    task_spec: TaskSpec,
    normalizer_sha256: Any,
    normalizer_state: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Project current saved-normalizer semantics.

    Parameters
    ----------
    task_spec : TaskSpec
        Validated persisted task contract.
    normalizer_sha256 : Any
        Completed-run digest of ``normalizer.pt``.
    normalizer_state : collections.abc.Mapping[str, Any]
        Validated saved normalizer tensors from completed-run admission.

    Returns
    -------
    dict[str, Any]
        Exact normalizer provenance required from the artifact.

    """
    return {
        "sha256": normalizer_sha256,
        "identity": "saved_run_normalizer.pt",
        "fit_split": task_spec.preprocessing.fit_split,
        "output_normalization": task_spec.preprocessing.output_normalization,
        "denominator_floor": 1e-7,
        "output_standard_deviations": contracts.output_standard_deviations_from_state(
            normalizer_state,
            output_fields=task_spec.output_names,
        ),
    }


def _expected_evaluator_provenance(
    *,
    config: Mapping[str, Any],
    task_spec: TaskSpec,
) -> dict[str, Any]:
    """
    Project the task-resolved evaluator contract without runtime workloads.

    Parameters
    ----------
    config : collections.abc.Mapping
        Validated resolved run configuration.
    task_spec : TaskSpec
        Validated persisted task contract.

    Returns
    -------
    dict[str, Any]
        Exact evaluator fields, units, metrics, objective, and formulas.

    Raises
    ------
    TypeError, ValueError
        If evaluation configuration is malformed or inconsistent.

    """
    evaluation = _mapping(config.get("evaluation"), label="config.yaml evaluation")
    metrics = evaluation.get("metrics")
    if not isinstance(metrics, list):
        msg = "config.yaml evaluation.metrics must be a list."
        raise TypeError(msg)
    return {
        "metrics": metrics,
        "objective": experiments.config.loader.get_resolved_objective(config),
        "input_fields": list(task_spec.input_names),
        "input_units": {field.name: field.unit for field in task_spec.inputs},
        "output_fields": list(task_spec.output_names),
        "output_units": {field.name: field.unit for field in task_spec.outputs},
        "output_groups": contracts.output_group_payload(task_spec.output_groups),
        "physics_kind": task_spec.physics.kind,
        "group_objective_evidence": {
            "squared_error_accumulation_dtype": "float64",
            "per_case_physical_columns": {field: list(contracts.physical_statistic_columns(field)) for field in task_spec.output_names},
            "per_case_normalized_columns": {field: list(contracts.normalized_statistic_columns(field)) for field in task_spec.output_names},
            "train_scale_source": "saved_run_normalizer.output_standard_deviations",
            "dataset_reduction": ("sum physical field SSE/count, finalize shared-scale group RMSE, equal macro mean over output groups"),
        },
        "predictive_metrics": {
            "rel_l2": "per-case arithmetic mean of physical per-field relative L2 ratios",
            "rel_h1": ("per-case arithmetic mean of physical per-field relative H1 ratios on the declared artifact region where available"),
            "physical_rmse_columns": {field: contracts.physical_statistic_columns(field)[2] for field in task_spec.output_names},
        },
    }


def _expected_physics_provenance(
    *,
    config: Mapping[str, Any],
    task_spec: TaskSpec,
) -> dict[str, Any] | None:
    """
    Return the exact current steady-Brinkman diagnostic contract.

    Parameters
    ----------
    config : collections.abc.Mapping
        Validated resolved run configuration.
    task_spec : TaskSpec
        Validated persisted task contract.

    Returns
    -------
    dict[str, Any] | None
        Exact physics provenance for steady Brinkman tasks, otherwise ``None``.

    Raises
    ------
    TypeError, ValueError
        If the selected continuity is malformed or not allowed by the task.

    """
    if task_spec.physics.kind != domain.physics.contracts.STEADY_BRINKMAN_KIND:
        return None
    loss = _mapping(config.get("loss"), label="config.yaml loss")
    physics_config = _mapping(loss.get("physics"), label="config.yaml loss.physics")
    continuity = _nonempty_string(
        physics_config.get("continuity"),
        label="config.yaml loss.physics.continuity",
    )
    if continuity not in task_spec.physics.allowed_continuities:
        msg = f"Selected training continuity {continuity!r} is not allowed by task {task_spec.id!r}."
        raise ValueError(msg)
    return {
        "residual_schema_version": contracts.RESIDUAL_SCHEMA_VERSION,
        "task_id": task_spec.id,
        "task_contract_digest": task_spec.contract_digest,
        "equation_kind": task_spec.physics.kind,
        "equation_set": task_spec.physics.equation_set,
        "boundary_condition_kind": task_spec.physics.boundary,
        "selected_training_continuity": continuity,
        "evaluated_continuity_formulations": list(domain.physics.contracts.available_continuity_kinds()),
        "constants": {
            "dynamic_viscosity_pa_s": domain.physics.brinkman.AIR_DYNAMIC_VISCOSITY,
            "porosity_floor": domain.physics.brinkman.POROSITY_FLOOR,
            "permeability_scale_floor_m2": domain.physics.brinkman.PERMEABILITY_SCALE_FLOOR,
            "permeability_determinant_floor": domain.physics.brinkman.PERMEABILITY_DETERMINANT_FLOOR,
            "permeability_cross_ratio_clip": domain.physics.brinkman.PERMEABILITY_CROSS_RATIO_CLIP,
        },
        "permeability_representation": {
            "kxx": "10**stored_log10_ratio_to_1_m2",
            "kxy": "stored_dimensionless_ratio_times_sqrt(kxx*kyy)",
            "kyy": "10**stored_log10_ratio_to_1_m2",
            "inverse": "normalized_symmetric_2x2_inverse_with_declared_floors",
        },
        "derivatives": {
            "kind": contracts.ARTIFACT_DERIVATIVE_KIND,
            "extension": contracts.ARTIFACT_DERIVATIVE_EXTENSION,
            "operator_axes": list(task_spec.operator_axes),
            "grid_axes": ["y", "x"],
        },
        "interior_crop": contracts.EVAL_PAD,
        "residual_evaluation_region": {
            "momentum_residual_mse": "interior grid after symmetric cell crop",
            "div_velocity_mse": "interior grid after symmetric cell crop",
            "div_eps_velocity_mse": "interior grid after symmetric cell crop",
            "pressure_boundary_mse": "full-grid inlet and outlet masks",
            "pressure_inlet_mse": "full-grid y-min inlet mask",
            "pressure_outlet_mean_square": "square of the sample mean on the full-grid y-max outlet mask",
            "residual_arrays": "full grid",
        },
        "scalar_definitions": {
            "momentum_residual_mse": {"formula": "mean(Rx**2 + Ry**2)", "unit": "(Pa/m)^2"},
            "div_velocity_mse": {"formula": "mean(div(u)**2)", "unit": "1/s^2"},
            "div_eps_velocity_mse": {"formula": "mean(div(eps*u)**2)", "unit": "1/s^2"},
            "pressure_boundary_mse": {
                "formula": "pressure_inlet_mse + pressure_outlet_mean_square",
                "unit": "Pa^2",
            },
            "pressure_inlet_mse": {"formula": "mean_inlet((p-p_bc)**2)", "unit": "Pa^2"},
            "pressure_outlet_mean_square": {"formula": "mean_outlet(p)**2", "unit": "Pa^2"},
        },
        "array_definitions": {
            "Rx": {"formula": "-dp/dx + div(tau)_x - mu*(K^-1*u)_x", "unit": "Pa/m"},
            "Ry": {"formula": "-dp/dy + div(tau)_y - mu*(K^-1*u)_y", "unit": "Pa/m"},
            "div_u": {"formula": "du/dx + dv/dy", "unit": "1/s"},
            "div_eps_u": {"formula": "d(eps*u)/dx + d(eps*v)/dy", "unit": "1/s"},
        },
    }


def _validate_runtime_provenance(value: Any) -> int:
    """
    Validate operational runtime facts without comparing device choices.

    Parameters
    ----------
    value : Any
        Persisted artifact runtime provenance.

    Raises
    ------
    TypeError, ValueError
        If required runtime facts are missing, unexpected, or malformed.

    Notes
    -----
    Runtime facts remain part of complete provenance and its canonical digest.
    They do not determine whether two scientific requests are equivalent.

    """
    runtime = _mapping(value, label="artifact provenance runtime")
    required = {
        "requested_policy",
        "resolved_device",
        "device_type",
        "pytorch_version",
        "batch_size",
    }
    optional = {"cuda_index", "cuda_device_name", "cuda_runtime_version"}
    missing = sorted(required.difference(runtime))
    unexpected = sorted(set(runtime).difference(required | optional))
    if missing or unexpected:
        msg = f"Artifact runtime provenance schema mismatch: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)
    for key in ("requested_policy", "resolved_device", "device_type", "pytorch_version"):
        _nonempty_string(runtime.get(key), label=f"artifact runtime {key}")
    batch_size = _positive_int(runtime.get("batch_size"), label="artifact runtime batch_size")
    if "cuda_index" in runtime:
        cuda_index = runtime["cuda_index"]
        if isinstance(cuda_index, bool) or not isinstance(cuda_index, Integral) or int(cuda_index) < 0:
            msg = "Artifact runtime cuda_index must be a non-negative integer."
            raise TypeError(msg)
    for key in ("cuda_device_name", "cuda_runtime_version"):
        if key in runtime:
            _nonempty_string(runtime[key], label=f"artifact runtime {key}")
    return batch_size


def _effective_selection(
    role: _SavedRole,
    provenance: Mapping[str, Any],
    *,
    evaluation_batch_size: int,
) -> tuple[int, tuple[int, ...]]:
    """Validate and return the complete saved membership and batching contract."""
    selection = _mapping(provenance.get("selection"), label="artifact provenance selection")
    membership_digest = contracts.ordered_indices_sha256(role.source_indices)
    expected = {
        "index_key": "eval_indices" if role.split_role == "eval" else "ood_indices",
        "full_selected_case_count": len(role.source_indices),
        "effective_case_count": len(role.source_indices),
        "generation_limit": None,
        "full_ordered_source_indices_sha256": membership_digest,
        "effective_ordered_source_indices_sha256": membership_digest,
    }
    if dict(selection) != expected:
        msg = "Artifact selection must match the complete saved membership without a generation limit."
        raise ValueError(msg)
    generation = _mapping(provenance.get("generation"), label="artifact provenance generation")
    expected_generation = {
        "effective_case_limit": None,
        "inference_batch_size": evaluation_batch_size,
        "compression": "numpy savez_compressed",
    }
    if dict(generation) != expected_generation:
        msg = "Artifact generation provenance must use complete membership and the saved evaluation batch size."
        raise ValueError(msg)
    return len(role.source_indices), role.source_indices


def _validate_payload_membership(
    *,
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
    role: _SavedRole,
    effective_indices: tuple[int, ...],
) -> None:
    """
    Bind table rows and manifest NPZ paths to exact saved membership.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current-schema evaluation table.
    provenance : collections.abc.Mapping
        Complete admitted artifact provenance.
    role : _SavedRole
        Authoritative role identity and root.
    effective_indices : tuple[int, ...]
        Exact saved-order membership represented by the artifact.

    Raises
    ------
    TypeError, ValueError
        If row identities, case paths, or output-manifest membership disagree.

    Notes
    -----
    Validation uses persisted scalar identities and file digests. It never opens
    or decompresses the NPZ arrays.

    """
    source_indices = tuple(int(value) for value in frame["source_index"].tolist())
    split_local_indices = tuple(int(value) for value in frame["split_local_index"].tolist())
    case_indices = tuple(int(value) for value in frame["case_index"].tolist())
    if source_indices != effective_indices:
        msg = "Artifact Parquet ordered source_index values do not match saved membership."
        raise ValueError(msg)
    if split_local_indices != tuple(range(len(effective_indices))):
        msg = "Artifact Parquet split_local_index values do not preserve saved membership order."
        raise ValueError(msg)
    expected_case_indices = tuple(source_index + 1 for source_index in effective_indices)
    if case_indices != expected_case_indices:
        msg = "Artifact Parquet case_index values do not match saved source identities."
        raise ValueError(msg)

    expected_paths = tuple((role.root / "npz" / f"case_{case_index:04d}.npz").absolute() for case_index in expected_case_indices)
    raw_paths = frame["npz_path"].tolist()
    if any(not isinstance(value, str) or not value for value in raw_paths):
        msg = "Artifact Parquet npz_path values must be non-empty strings."
        raise TypeError(msg)
    actual_paths = tuple(Path(value).expanduser().absolute() for value in raw_paths)
    if actual_paths != expected_paths:
        msg = "Artifact Parquet NPZ paths do not identify exact run-owned case payloads."
        raise ValueError(msg)
    for expected_path in expected_paths:
        if not expected_path.is_file() or expected_path.is_symlink() or expected_path.resolve() != expected_path:
            msg = f"Artifact NPZ payload is missing or uses a symbolic-link alias: {expected_path}"
            raise ValueError(msg)

    outputs = _mapping(provenance.get("outputs"), label="artifact provenance outputs")
    parquet_output = _mapping(outputs.get("parquet"), label="artifact provenance outputs.parquet")
    if parquet_output.get("path") != f"{role.dataset_name}.parquet":
        msg = "Artifact manifest Parquet path does not match the saved dataset identity."
        raise ValueError(msg)
    npz_outputs = outputs.get("npz")
    if not isinstance(npz_outputs, list):
        msg = "Artifact provenance outputs.npz must be a list."
        raise TypeError(msg)
    manifest_paths = tuple(_mapping(item, label=f"artifact provenance outputs.npz[{index}]").get("path") for index, item in enumerate(npz_outputs))
    expected_manifest_paths = tuple(sorted(f"npz/case_{case_index:04d}.npz" for case_index in expected_case_indices))
    if manifest_paths != expected_manifest_paths:
        msg = "Artifact manifest NPZ membership does not match saved source membership."
        raise ValueError(msg)


def _bind_artifact_to_run(
    *,
    frame: pd.DataFrame,
    role: _SavedRole,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    task_spec: TaskSpec,
    run_name: str,
    evaluation: Mapping[str, Any],
) -> str:
    """
    Bind one admitted artifact to its authoritative evaluable run.

    Parameters
    ----------
    frame : pandas.DataFrame
        Evaluation table with complete admitted provenance.
    role : _SavedRole
        Authoritative saved role and artifact root.
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary.
    normalizer_state : collections.abc.Mapping
        Validated saved training normalizer state.
    task_spec : TaskSpec
        Validated persisted task contract.
    run_name : str
        Canonical completed-run name.
    evaluation : collections.abc.Mapping
        Validated resolved evaluation configuration.

    Returns
    -------
    str
        Canonical SHA-256 digest of complete validated artifact provenance.

    Raises
    ------
    TypeError, ValueError, ComparisonCompatibilityError
        If any schema, run, model, normalizer, evaluator, physics, role,
        dataset, selection, runtime, or payload identity disagrees.

    """
    provenance = dataframe.require_complete_provenance(frame)
    expected_physics = _expected_physics_provenance(config=config, task_spec=task_spec)
    expected_keys = {
        "provenance_schema_version",
        "artifact_schema_version",
        "run",
        "model",
        "split_role",
        "dataset",
        "selection",
        "normalizer",
        "evaluator",
        "generation",
        "runtime",
        "aggregate",
        "outputs",
    }
    if expected_physics is not None:
        expected_keys.add("physics")
    if set(provenance) != expected_keys:
        missing = sorted(expected_keys.difference(provenance))
        unexpected = sorted(set(provenance).difference(expected_keys))
        msg = f"Artifact provenance schema mismatch: missing={missing}, unexpected={unexpected}."
        raise ValueError(msg)

    expected_legacy_run = _expected_run_provenance(
        summary=summary,
        task_spec=task_spec,
        run_name=run_name,
    )
    expected_current_run = _expected_run_provenance(
        summary=summary,
        task_spec=task_spec,
        run_name=run_name,
        evaluation=evaluation,
    )
    actual_run = dict(_mapping(provenance.get("run"), label="artifact provenance run"))
    allowed_runs = (expected_current_run,) if evaluation.get("is_provisional") is True else (expected_legacy_run, expected_current_run)
    if actual_run not in allowed_runs:
        msg = "Artifact run provenance does not match the authoritative evaluable run and selected checkpoint."
        raise ValueError(msg)
    expected_model = _expected_model_provenance(config=config, summary=summary)
    if dict(_mapping(provenance.get("model"), label="artifact provenance model")) != expected_model:
        msg = "Artifact model provenance does not match the evaluable run configuration and capacity."
        raise ValueError(msg)
    expected_normalizer = _expected_normalizer_provenance(
        task_spec=task_spec,
        normalizer_sha256=summary.get("normalizer_sha256"),
        normalizer_state=normalizer_state,
    )
    if dict(_mapping(provenance.get("normalizer"), label="artifact provenance normalizer")) != expected_normalizer:
        msg = "Artifact normalizer provenance does not match the evaluable run."
        raise ValueError(msg)
    expected_evaluator = _expected_evaluator_provenance(config=config, task_spec=task_spec)
    if dict(_mapping(provenance.get("evaluator"), label="artifact provenance evaluator")) != expected_evaluator:
        msg = "Artifact evaluator provenance does not match the resolved task and run evaluation contract."
        raise ValueError(msg)
    if expected_physics is None:
        if "physics" in provenance:
            msg = "Artifact physics provenance is unexpected for this task."
            raise ValueError(msg)
    elif dict(_mapping(provenance.get("physics"), label="artifact provenance physics")) != expected_physics:
        msg = "Artifact physics provenance does not match the current task diagnostic contract."
        raise ValueError(msg)

    if provenance.get("split_role") != role.split_role:
        msg = "Artifact split role does not match its exact ID/OOD root."
        raise ValueError(msg)
    expected_dataset = {
        "name": role.dataset_name,
        "full_case_count": _positive_int(
            role.dataset_identity.get("sample_count"),
            label=f"saved {role.split_role} dataset sample_count",
        ),
        "fingerprint": role.dataset_identity.get("fingerprint"),
        "data_contract_digest": role.dataset_identity.get("data_contract_digest"),
        "saved_membership_digest": role.saved_membership_digest,
    }
    if dict(_mapping(provenance.get("dataset"), label="artifact provenance dataset")) != expected_dataset:
        msg = "Artifact dataset provenance does not match saved dataset identity and membership."
        raise ValueError(msg)
    data_config = _mapping(config.get("data"), label="config.yaml data")
    evaluation_batch_size = _positive_int(data_config.get("batch_size"), label="config.yaml data.batch_size")
    effective_count, effective_indices = _effective_selection(
        role,
        provenance,
        evaluation_batch_size=evaluation_batch_size,
    )
    if len(frame) != effective_count:
        msg = "Artifact Parquet row count contradicts the complete saved membership."
        raise ValueError(msg)
    _validate_payload_membership(
        frame=frame,
        provenance=provenance,
        role=role,
        effective_indices=effective_indices,
    )
    runtime_batch_size = _validate_runtime_provenance(provenance.get("runtime"))
    if runtime_batch_size != evaluation_batch_size:
        msg = "Artifact runtime batch size does not match the evaluable run evaluation batch size."
        raise ValueError(msg)
    return common.serialization.canonical_json_sha256(dict(provenance))


def _load_role(
    *,
    role: _SavedRole,
    task: str,
    run_name: str,
    run_dir: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    normalizer_state: Mapping[str, Any],
    task_spec: TaskSpec,
    evaluation: Mapping[str, Any],
) -> LoadedEvaluationArtifact:
    """
    Load and bind one exact role without NPZ array access or mutation.

    Parameters
    ----------
    role : _SavedRole
        Authoritative saved role identity and target root.
    task, run_name : str
        Canonical completed-run selection.
    run_dir : pathlib.Path
        Exact current run directory owning the artifacts.
    config : collections.abc.Mapping
        Validated resolved run configuration.
    summary : collections.abc.Mapping
        Validated completed-run summary.
    normalizer_state : collections.abc.Mapping
        Validated saved training normalizer state.
    task_spec : TaskSpec
        Validated persisted task contract.
    evaluation : collections.abc.Mapping
        Validated resolved evaluation configuration.

    Returns
    -------
    LoadedEvaluationArtifact
        Role-local frame, root, dataset identity, and canonical digest.

    Raises
    ------
    MissingEvaluationArtifactsError
        If the exact role target is absent or empty.
    IncompatibleEvaluationArtifactsError
        If the target is partial, unreadable, changed, or contradictory.

    """
    root = role.root
    if not root.exists():
        raise _missing(task=task, run_name=run_name, run_dir=run_dir, role=role.split_role, root=root)
    if not root.is_dir():
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason="expected artifact root is not a directory",
        )
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"artifact root cannot be resolved exactly: {error}",
        ) from error
    if resolved_root != root:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason="artifact root uses a symbolic-link alias outside its exact run-owned path",
        )
    provenance_path = contracts.artifact_provenance_path(root)
    if not provenance_path.is_file():
        try:
            has_content = any(root.iterdir())
        except OSError as error:
            raise _incompatible(
                task=task,
                run_name=run_name,
                run_dir=run_dir,
                role=role.split_role,
                root=root,
                reason=f"artifact root cannot be inspected: {error}",
            ) from error
        if not has_content:
            raise _missing(task=task, run_name=run_name, run_dir=run_dir, role=role.split_role, root=root)
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"partial artifact target has no completion marker {provenance_path.name}",
        )
    try:
        frame = dataframe.load_evaluation_artifact(root)
        identity_sha256 = _bind_artifact_to_run(
            frame=frame,
            role=role,
            config=config,
            summary=summary,
            normalizer_state=normalizer_state,
            task_spec=task_spec,
            run_name=run_name,
            evaluation=evaluation,
        )
    except EvaluationArtifactLoadError:
        raise
    except Exception as error:
        raise _incompatible(
            task=task,
            run_name=run_name,
            run_dir=run_dir,
            role=role.split_role,
            root=root,
            reason=f"{type(error).__name__}: {error}",
        ) from error
    return LoadedEvaluationArtifact(
        split_role=role.split_role,
        dataset_name=role.dataset_name,
        root=root,
        frame=frame,
        identity_sha256=identity_sha256,
    )


def _load_admitted_run_artifacts(
    run_dir: Path,
    admitted: Mapping[str, Any],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
) -> LoadedRunArtifacts:
    """Load both artifact roles from one already admitted current run path."""
    run_dir = run_dir.expanduser().resolve()
    config = _mapping(admitted.get("config"), label="evaluable config")
    summary = _mapping(admitted.get("summary"), label="evaluable summary")
    normalizer_state = _mapping(admitted.get("normalizer_state"), label="evaluable normalizer state")
    task_spec = experiments.config.loader.validate_resolved_task_contract(config)
    run_config = _mapping(config.get("run"), label="config.yaml run")
    scientific_run_name = _nonempty_string(run_config.get("name"), label="config.yaml run.name")
    if config.get("task") != task_spec.id or summary.get("task") != task_spec.id:
        msg = "Saved task identity contradicts the authoritative evaluable run."
        raise ValueError(msg)
    if summary.get("run_name") != scientific_run_name:
        msg = "summary.json run_name contradicts config.yaml scientific run identity."
        raise ValueError(msg)
    admitted_dir = admitted.get("run_dir")
    if admitted_dir is not None and Path(admitted_dir).resolve() != run_dir:
        msg = "Run validator returned a different current run directory."
        raise ValueError(msg)

    evidence_summary = dict(summary)
    evidence_summary["effective_config_digest"] = admitted.get(
        "effective_config_digest",
        evidence_summary.get("effective_config_digest"),
    )
    evidence_summary["best_checkpoint_sha256"] = admitted.get(
        "selected_checkpoint_sha256",
        evidence_summary.get("best_checkpoint_sha256"),
    )
    evidence_summary["normalizer_sha256"] = admitted.get(
        "normalizer_sha256",
        evidence_summary.get("normalizer_sha256"),
    )
    evaluation = {
        "lifecycle_status": admitted.get("lifecycle_status", "completed"),
        "is_completed": admitted.get("is_completed", True),
        "is_provisional": admitted.get("is_provisional", False),
        "selected_checkpoint_role": admitted.get("selected_checkpoint_role", "best"),
        "selected_checkpoint_epoch": admitted.get("selected_checkpoint_epoch"),
        "selected_checkpoint_sha256": evidence_summary.get("best_checkpoint_sha256"),
    }
    try:
        id_role, ood_role = _saved_roles(
            completed=admitted,
            run_dir=run_dir,
            task=task_spec.id,
            task_spec=task_spec,
        )
    except Exception as error:
        msg = f"Evaluable run has incompatible saved artifact identity: {type(error).__name__}: {error}"
        raise ValueError(msg) from error

    if not artifact_roles or len(set(artifact_roles)) != len(artifact_roles) or set(artifact_roles).difference({"id", "ood"}):
        msg = "artifact_roles must contain unique values from {'id', 'ood'}."
        raise ValueError(msg)

    def load_role(role: _SavedRole) -> LoadedEvaluationArtifact:
        return _load_role(
            role=role,
            task=task_spec.id,
            run_name=scientific_run_name,
            run_dir=run_dir,
            config=config,
            summary=evidence_summary,
            normalizer_state=normalizer_state,
            task_spec=task_spec,
            evaluation=evaluation,
        )

    id_artifact = load_role(id_role) if "id" in artifact_roles else None
    ood_artifact = load_role(ood_role) if "ood" in artifact_roles else None
    for artifact in (id_artifact, ood_artifact):
        if artifact is not None:
            artifact.frame.attrs[dataframe.COMPLETED_RUN_CONFIG_ATTR] = deepcopy(dict(config))
    return LoadedRunArtifacts(
        task=task_spec.id,
        run_name=scientific_run_name,
        run_dir=run_dir,
        id_artifact=id_artifact,
        ood_artifact=ood_artifact,
        lifecycle_status=str(evaluation["lifecycle_status"]),
        is_completed=bool(evaluation["is_completed"]),
        is_provisional=bool(evaluation["is_provisional"]),
        selected_checkpoint_role=str(evaluation["selected_checkpoint_role"]),
        selected_checkpoint_epoch=(int(evaluation["selected_checkpoint_epoch"]) if evaluation["selected_checkpoint_epoch"] is not None else None),
        selected_checkpoint_sha256=str(evaluation["selected_checkpoint_sha256"] or ""),
        summary=deepcopy(dict(summary)),
    )


def load_run_artifacts(
    run_dir: Path | str,
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
) -> LoadedRunArtifacts:
    """
    Load strict ID and OOD artifacts from one explicit current run directory.

    The directory may be a direct run or Optuna trial at any current location.
    Its leaf is a storage alias; immutable scientific identity comes only from
    validated ``config.yaml`` and ``summary.json`` evidence.
    """
    path = Path(run_dir).expanduser().resolve()
    with experiments.run.evaluable_run_lease(path) as admitted:
        return _load_admitted_run_artifacts(path, admitted, artifact_roles=artifact_roles)


def load_completed_run_artifacts(
    task: str,
    run_name: str,
    *,
    output_root: Path | str | None = None,
) -> LoadedRunArtifacts:
    """
    Load one strictly completed run through the canonical-name convenience path.

    This compatibility resolver remains strict and delegates artifact loading to
    the same path-based implementation used for relocated evaluable bundles.
    """
    canonical_task = common.paths.validate_logical_name(task, label="task")
    canonical_run_name = common.paths.validate_logical_name(run_name, label="run_name")
    run_dir = common.paths.resolve_run_output_dir(
        canonical_task,
        canonical_run_name,
        output_root=output_root,
    ).resolve()
    with experiments.run.run_reader_lease(run_dir):
        completed = experiments.run.validate_completed_run(run_dir)
        config = _mapping(completed.get("config"), label="completed config")
        summary = _mapping(completed.get("summary"), label="completed summary")
        run_config = _mapping(config.get("run"), label="config.yaml run")
        if config.get("task") != canonical_task or summary.get("task") != canonical_task:
            msg = "Requested task contradicts the authoritative completed run."
            raise ValueError(msg)
        if run_config.get("name") != canonical_run_name or summary.get("run_name") != canonical_run_name:
            msg = "Requested run name contradicts the authoritative completed run."
            raise ValueError(msg)
        return _load_admitted_run_artifacts(run_dir, completed)
