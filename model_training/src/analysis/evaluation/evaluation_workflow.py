"""
===============================================================================
evaluation_workflow.py
===============================================================================
Prepare portable evaluation sessions and panels from concise run selections.

Responsibilities:
  - Normalize explicit current run paths and optional presentation labels
  - Invoke shared artifact load-or-build orchestration for selected roles
  - Compose session-owned role contexts for single-run or comparison panels
  - Return concise lifecycle, checkpoint, artifact, and device reporting

Design principles:
  - Notebook cells expose choices while reusable orchestration remains in source
  - Scientific run identity and mutable storage identity remain visibly separate
  - Presentation labels never participate in artifact or numerical identity
  - Failed preparation closes any session created before the failure

This module does NOT:
  - Admit runs, reconstruct models, generate payloads, or publish artifacts itself
  - Choose datasets, roles, labels, sections, devices, or rebuild policy for users
  - Contact tracking services or mutate run lifecycle evidence
===============================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src import analysis

if TYPE_CHECKING:
    import ipywidgets as widgets

    from src.analysis.artifacts.analysis_artifact_service import PreparedRunArtifacts
    from src.analysis.evaluation.evaluation_artifact_loader import LoadedEvaluationArtifact, LoadedRunArtifacts
    from src.analysis.evaluation.evaluation_panel import EvaluationContext
    from src.analysis.evaluation.evaluation_session import EvaluationSession

ArtifactSelectionRole = Literal["id", "ood"]


@dataclass(frozen=True, slots=True)
class EvaluationRunSelection:
    """
    Select one current run directory and optional presentation label.

    Parameters
    ----------
    run_dir : pathlib.Path | str
        Exact current direct-run or Optuna-trial directory.
    label : str | None, optional
        Presentation-only label. The storage alias is used when omitted.

    """

    run_dir: Path | str
    label: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationContextSpec:
    """
    Describe one ordered dataset context backed by an artifact role.

    Parameters
    ----------
    key : str
        Stable panel-local context key.
    label : str
        User-facing dataset context label.
    artifact_role : {"id", "ood"}
        Selected run artifact role supplying each model frame.

    """

    key: str
    label: str
    artifact_role: ArtifactSelectionRole


@dataclass(frozen=True, slots=True)
class PreparedEvaluationWorkflow:
    """
    Hold prepared artifacts, live evaluation state, panel, and report rows.

    Parameters
    ----------
    prepared_runs : tuple[PreparedRunArtifacts, ...]
        Per-run artifact load-or-build outcomes in selection order.
    session : EvaluationSession
        Live bounded numerical and case-reuse session.
    contexts : tuple[EvaluationContext, ...]
        Ordered role-local contexts admitted to the panel.
    panel : ipywidgets.Widget
        Single-model or comparison panel bound to ``session``.
    report : tuple[dict[str, Any], ...]
        JSON-friendly run, lifecycle, checkpoint, artifact, and device records.

    """

    prepared_runs: tuple[PreparedRunArtifacts, ...]
    session: EvaluationSession
    contexts: tuple[EvaluationContext, ...]
    panel: widgets.Widget
    report: tuple[dict[str, Any], ...]

    def close(self) -> None:
        """Release all session-owned case and numerical state."""
        self.session.close()


def _normalized_selections(
    run_selections: Sequence[EvaluationRunSelection],
) -> tuple[tuple[EvaluationRunSelection, Path], ...]:
    """Validate selections and resolve explicit current paths."""
    if isinstance(run_selections, (str, bytes)) or not isinstance(run_selections, Sequence) or not run_selections:
        msg = "run_selections must be a non-empty ordered sequence."
        raise TypeError(msg)
    normalized: list[tuple[EvaluationRunSelection, Path]] = []
    for position, selection in enumerate(run_selections):
        if not isinstance(selection, EvaluationRunSelection):
            msg = f"run_selections[{position}] must be an EvaluationRunSelection."
            raise TypeError(msg)
        if selection.label is not None and (not isinstance(selection.label, str) or not selection.label.strip()):
            msg = f"run_selections[{position}].label must be non-blank text or None."
            raise ValueError(msg)
        normalized.append(
            (
                selection,
                Path(selection.run_dir).expanduser().resolve(),
            )
        )
    return tuple(normalized)


def _normalized_context_specs(
    context_specs: Sequence[EvaluationContextSpec],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...],
) -> tuple[EvaluationContextSpec, ...]:
    """Validate ordered context declarations and retain selected roles."""
    if isinstance(context_specs, (str, bytes)) or not isinstance(context_specs, Sequence) or not context_specs:
        msg = "context_specs must be a non-empty ordered sequence."
        raise TypeError(msg)
    normalized: list[EvaluationContextSpec] = []
    for position, spec in enumerate(context_specs):
        if not isinstance(spec, EvaluationContextSpec):
            msg = f"context_specs[{position}] must be an EvaluationContextSpec."
            raise TypeError(msg)
        if not spec.key.strip() or not spec.label.strip():
            msg = f"context_specs[{position}] requires non-blank key and label text."
            raise ValueError(msg)
        if spec.artifact_role in artifact_roles:
            normalized.append(spec)
    if not normalized:
        msg = "No evaluation context selects one of the requested artifact roles."
        raise ValueError(msg)
    keys = tuple(spec.key for spec in normalized)
    labels = tuple(spec.label for spec in normalized)
    if len(keys) != len(set(keys)) or len(labels) != len(set(labels)):
        msg = "Selected evaluation context keys and labels must each be unique."
        raise ValueError(msg)
    return tuple(normalized)


def _selected_artifact(
    loaded: LoadedRunArtifacts,
    role: ArtifactSelectionRole,
) -> LoadedEvaluationArtifact:
    """Return one role guaranteed by artifact preparation."""
    artifact = loaded.id_artifact if role == "id" else loaded.ood_artifact
    if artifact is None:
        msg = f"Prepared run is missing its selected {role!r} artifact: {loaded.run_dir}"
        raise RuntimeError(msg)
    return artifact


def _display_label(
    selection: EvaluationRunSelection,
    loaded: LoadedRunArtifacts,
) -> str:
    """Return the presentation label with truthful provisional status."""
    base_label = selection.label.strip() if selection.label is not None else loaded.storage_alias
    return f"{base_label} [provisional]" if loaded.is_provisional else base_label


def _report_row(
    prepared: PreparedRunArtifacts,
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...],
    display_label: str,
) -> dict[str, Any]:
    """Build one concise JSON-friendly notebook report row."""
    loaded = prepared.loaded_run
    artifacts = {role: _selected_artifact(loaded, role) for role in artifact_roles}
    return {
        "display_label": display_label,
        "scientific_run_name": loaded.scientific_run_name,
        "storage_alias": loaded.storage_alias,
        "run_dir": str(loaded.run_dir),
        "lifecycle_status": loaded.lifecycle_status,
        "is_completed": loaded.is_completed,
        "is_provisional": loaded.is_provisional,
        "selected_checkpoint_role": loaded.selected_checkpoint_role,
        "selected_checkpoint_epoch": loaded.selected_checkpoint_epoch,
        "selected_checkpoint_sha256": loaded.selected_checkpoint_sha256,
        "artifact_roles": list(artifact_roles),
        "artifact_roots": {role: str(artifact.root) for role, artifact in artifacts.items()},
        "artifact_actions": dict(prepared.role_actions),
        "artifact_device": prepared.artifact_device,
    }


def prepare_evaluation_workflow(
    run_selections: Sequence[EvaluationRunSelection],
    context_specs: Sequence[EvaluationContextSpec],
    *,
    artifact_roles: tuple[ArtifactSelectionRole, ...] = ("id", "ood"),
    dataset_root: Path | str | None = None,
    metadata_root: Path | str | None = None,
    auto_build_missing: bool = True,
    rebuild_incompatible: bool = False,
    device_policy: str = "cpu",
    sections: Sequence[str] | str = "all",
) -> PreparedEvaluationWorkflow:
    """
    Prepare one portable single-run or comparison evaluation workflow.

    Parameters
    ----------
    run_selections : collections.abc.Sequence[EvaluationRunSelection]
        Explicit current run paths and optional presentation labels.
    context_specs : collections.abc.Sequence[EvaluationContextSpec]
        Ordered ID/OOD context declarations for the panel.
    artifact_roles : tuple[{"id", "ood"}, ...], optional
        Unique roles to load or generate. Defaults to both roles.
    dataset_root, metadata_root : pathlib.Path | str | None, optional
        Current dataset and validated metadata roots used only for generation.
    auto_build_missing : bool, optional
        Generate absent selected roles before normal loading.
    rebuild_incompatible : bool, optional
        Deliberately replace an incompatible exact target when true.
    device_policy : {"cpu", "cuda", "auto"}, optional
        Generation device policy. Notebook callers should retain the CPU default.
    sections : collections.abc.Sequence[str] | str, optional
        Panel section selection forwarded unchanged to the panel builder.

    Returns
    -------
    PreparedEvaluationWorkflow
        Live session, role contexts, panel, artifact outcomes, and report rows.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError, RuntimeError
        If selections, run evidence, artifacts, comparison contracts, device
        resolution, generation, or panel construction fail validation.

    Notes
    -----
    Valid artifacts are loaded before device resolution or model construction.
    Missing-role generation delegates to the local-only artifact service. This
    workflow never invokes completed-run upload orchestration or W&B.

    """
    normalized_selections = _normalized_selections(run_selections)
    roles = tuple(artifact_roles)
    specs = _normalized_context_specs(context_specs, artifact_roles=roles)
    prepared_runs = tuple(
        analysis.artifacts.service.load_or_build_run_artifacts(
            path,
            artifact_roles=roles,
            dataset_root=dataset_root,
            metadata_root=metadata_root,
            auto_build_missing=auto_build_missing,
            rebuild_incompatible=rebuild_incompatible,
            device_policy=device_policy,
        )
        for _selection, path in normalized_selections
    )
    loaded_runs = tuple(prepared.loaded_run for prepared in prepared_runs)
    display_labels = tuple(_display_label(selection, loaded) for (selection, _path), loaded in zip(normalized_selections, loaded_runs, strict=True))
    if len(display_labels) != len(set(display_labels)):
        msg = "Evaluation presentation labels must be unique after provisional labelling."
        raise ValueError(msg)

    artifacts = {(loaded.run_dir, role): _selected_artifact(loaded, role) for loaded in loaded_runs for role in roles}
    session = analysis.evaluation.session.EvaluationSession(
        {f"{loaded.run_dir}:{role}": artifacts[(loaded.run_dir, role)].frame for loaded in loaded_runs for role in roles}
    )
    try:
        contexts = tuple(
            analysis.evaluation.panel.EvaluationContext(
                key=spec.key,
                label=spec.label,
                datasets={
                    display_label: artifacts[(loaded.run_dir, spec.artifact_role)].frame
                    for display_label, loaded in zip(display_labels, loaded_runs, strict=True)
                },
            )
            for spec in specs
        )
        if len(loaded_runs) == 1:
            panel = analysis.evaluation.panel.build_single_model_panel(
                session=session,
                contexts=contexts,
                sections=sections,
            )
        else:
            panel = analysis.evaluation.panel.build_comparison_panel(
                session=session,
                contexts=contexts,
                sections=sections,
            )
        report = tuple(
            _report_row(prepared, artifact_roles=roles, display_label=display_label)
            for prepared, display_label in zip(prepared_runs, display_labels, strict=True)
        )
    except Exception:
        session.close()
        raise
    return PreparedEvaluationWorkflow(
        prepared_runs=prepared_runs,
        session=session,
        contexts=contexts,
        panel=panel,
        report=report,
    )
