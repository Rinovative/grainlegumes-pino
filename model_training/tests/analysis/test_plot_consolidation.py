# ruff: noqa: S101
"""
Exercise scientific plot formulas and public analysis usability on current fixtures.

Compact ID/OOD frames cover primary aggregation, dual continuity, spatial and
spectral questions, linked case navigation, lazy panels, and the five-item
curated renderer. Artifact cache concurrency and full provenance generation are
covered by the dedicated identity/provenance modules; notebooks are not executed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from src import analysis, experiments

_FIELDS = ("p", "u", "v")
_UNITS = ("Pa", "m/s", "m/s")
_INPUT_FIELDS = ("x", "y", "kxx", "kxy", "kyy", "eps", "p_bc")


def _provenance(raw: pd.DataFrame, *, role: str, label: str) -> dict[str, object]:
    """
    Build current provenance around exact fixture aggregate evidence.

    ``role`` and ``label`` distinguish ID/OOD identity while the shared schema
    supplies every scientific field required by presentation and renderer tests.
    """
    aggregate = analysis.artifacts.aggregate_normalized_macro_rmse(raw, output_fields=_FIELDS)
    return {
        "provenance_schema_version": analysis.artifacts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
        "split_role": role,
        "run": {
            "name": label,
            "task": "steady_flow",
            "task_contract_digest": "task-digest",
            "effective_config_digest": f"config-{label}",
            "best_checkpoint_sha256": f"checkpoint-{label}",
            "normalizer_sha256": f"normalizer-{label}",
        },
        "model": {
            "kind": "fno",
            "architecture": {"hidden_channels": 8, "n_layers": 2, "n_modes": [4, 4]},
            "parameter_counts": {"total": 1300, "trainable": 1200},
            "physics_enabled": True,
        },
        "dataset": {
            "name": f"fixture-{role}",
            "fingerprint": f"fingerprint-{role}",
            "task_contract_digest": "task-digest",
            "saved_membership_digest": f"saved-{role}",
        },
        "selection": {
            "effective_case_count": len(raw),
            "effective_ordered_source_indices_sha256": f"effective-{role}",
        },
        "evaluator": {
            "input_fields": list(_INPUT_FIELDS),
            "input_units": dict(zip(_INPUT_FIELDS, ("m", "m", "m^2", "m^2", "m^2", "1", "Pa"), strict=True)),
            "output_fields": list(_FIELDS),
            "output_units": dict(zip(_FIELDS, _UNITS, strict=True)),
            "objective": {
                **analysis.evaluation.dataframe.PRIMARY_OBJECTIVE_DEFINITION,
                "fields": list(_FIELDS),
            },
            "predictive_metrics": ["rel_l2", "rel_h1"],
        },
        "aggregate": aggregate,
        "physics": {
            "residual_schema_version": analysis.artifacts.RESIDUAL_SCHEMA_VERSION,
            "equation_kind": "steady_brinkman",
            "boundary_condition_kind": "pressure_inlet_outlet",
            "derivatives": "central_difference",
            "interior_crop": 1,
            "scalar_definitions": {"dual_continuity": True},
            "array_definitions": {"Rx": "full_grid", "Ry": "full_grid", "div_u": "full_grid", "div_eps_u": "full_grid"},
            "residual_evaluation_region": "interior_crop",
            "selected_training_continuity": "div_eps_velocity",
        },
    }


def _evaluation_frame(root: Path, *, role: str, bias: float, admit: bool = True) -> pd.DataFrame:
    """
    Create two current-schema rows and their matching NPZ case payloads.

    ``bias`` controls predictive and physics error magnitude, giving ID/OOD plots
    compact but nondegenerate fields, metadata, residuals, and aggregate evidence.
    """
    artifact_root = root / role
    npz_root = artifact_root / "npz"
    npz_root.mkdir(parents=True)
    height, width = 8, 10
    y_values = np.linspace(0.0, 1.0, height)
    x_values = np.linspace(0.0, 2.0, width)
    y_grid, x_grid = np.meshgrid(y_values, x_values, indexing="ij")
    rows: list[dict[str, object]] = []
    for position in range(2):
        source_index = position + (10 if role == "ood" else 0)
        reference = np.stack(
            (
                2.0 - y_grid + 0.1 * position,
                np.sin(np.pi * x_grid) * (1.0 + 0.1 * position),
                np.cos(np.pi * y_grid) * 0.5,
            )
        )
        prediction = reference + bias * (position + 1) * np.stack((np.ones_like(x_grid), np.sin(2.0 * np.pi * x_grid), np.cos(2.0 * np.pi * y_grid)))
        error = prediction - reference
        eps = 0.35 + 0.05 * x_grid
        p_bc = 2.0 - y_grid
        inputs = np.stack((x_grid, y_grid, np.full_like(x_grid, 1e-4), np.zeros_like(x_grid), np.full_like(x_grid, 2e-4), eps, p_bc))
        residual_scale = bias * (position + 1)
        residuals = {
            "Rx": residual_scale * np.sin(np.pi * x_grid),
            "Ry": residual_scale * np.cos(np.pi * y_grid),
            "div_u": residual_scale * np.sin(np.pi * y_grid),
            "div_eps_u": 1.5 * residual_scale * np.cos(np.pi * x_grid),
        }
        metadata = {"reynolds": float(position + 1 + bias), "roughness": float(0.2 + position)}
        npz_path = npz_root / f"case_{source_index + 1:04d}.npz"
        np.savez_compressed(
            npz_path,
            case_index=np.asarray(source_index + 1),
            source_index=np.asarray(source_index),
            split_local_index=np.asarray(position),
            pred=prediction,
            gt=reference,
            err=error,
            artifact_fields=np.asarray(_FIELDS),
            artifact_units=np.asarray(_UNITS),
            input_fields=np.asarray(_INPUT_FIELDS),
            output_fields=np.asarray(_FIELDS),
            output_units=np.asarray(_UNITS),
            x_raw=inputs,
            y_raw=reference,
            meta=np.asarray(json.dumps(metadata)),
            kappa_encoded=np.stack((inputs[2], inputs[3], inputs[4])),
            kappa=np.stack((inputs[2], inputs[3], inputs[4])),
            kappa_names=np.asarray(("kxx", "kxy", "kyy")),
            p_bc=p_bc[None],
            coordinates=np.stack((x_grid, y_grid)),
            Rx=residuals["Rx"],
            Ry=residuals["Ry"],
            div_u=residuals["div_u"],
            div_eps_u=residuals["div_eps_u"],
        )
        pressure_inlet_mse = float(np.mean(error[0, 0] ** 2))
        pressure_outlet_mean_square = float(np.mean(prediction[0, -1]) ** 2)
        row: dict[str, object] = {
            "artifact_schema_version": analysis.artifacts.ARTIFACT_SCHEMA_VERSION,
            "task_id": "steady_flow",
            "output_fields": list(_FIELDS),
            "output_units": list(_UNITS),
            "case_index": source_index + 1,
            "source_index": source_index,
            "split_local_index": position,
            "npz_path": str(npz_path),
            "meta": json.dumps(metadata),
            "inference_time_ms": 1.0 + position,
            "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
            "rel_h1": float(1.2 * np.linalg.norm(error) / np.linalg.norm(reference)),
            "rmse_U": float(np.sqrt(np.mean(error[1:] ** 2))),
            "kappa_names": ["kxx", "kxy", "kyy"],
            "momentum_residual_mse": float(np.mean(residuals["Rx"] ** 2 + residuals["Ry"] ** 2)),
            "div_velocity_mse": float(np.mean(residuals["div_u"] ** 2)),
            "div_eps_velocity_mse": float(np.mean(residuals["div_eps_u"] ** 2)),
            "pressure_inlet_mse": pressure_inlet_mse,
            "pressure_outlet_mean_square": pressure_outlet_mean_square,
        }
        row["pressure_boundary_mse"] = pressure_inlet_mse + pressure_outlet_mean_square
        for field_index, field in enumerate(_FIELDS):
            field_error = error[field_index]
            sse = float(np.sum(field_error**2))
            count = int(field_error.size)
            row[f"rmse_{field}"] = float(np.sqrt(sse / count))
            row[f"normalized_sse_{field}"] = sse
            row[f"normalized_count_{field}"] = count
            row[f"normalized_rmse_{field}"] = float(np.sqrt(sse / count))
        rows.append(row)
    raw = pd.DataFrame(rows)
    raw.attrs["artifact_root"] = str(artifact_root)
    raw.attrs["artifact_provenance"] = _provenance(raw, role=role, label=f"run-{role}")
    return analysis.evaluation.dataframe.build_eval_df(raw) if admit else raw


@pytest.fixture
def evaluation_datasets(tmp_path: Path) -> dict[str, pd.DataFrame]:
    """
    Return compact ID and OOD frames under separate artifact roots.

    Distinct roots and biases exercise comparison and output-containment behavior
    while reusing the same complete current-schema construction.
    """
    return {
        "Fixture ID": _evaluation_frame(tmp_path / "artifacts", role="eval", bias=0.03),
        "Fixture OOD": _evaluation_frame(tmp_path / "artifacts", role="ood", bias=0.06),
    }


@pytest.mark.parametrize(
    "schema_version",
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_artifact_residual_schema_requires_integer_version_one(
    schema_version: object,
    tmp_path: Path,
) -> None:
    """Reject alternate residual schema representations during provenance admission."""
    raw = _evaluation_frame(tmp_path / "invalid-residual", role="eval", bias=0.03, admit=False)
    provenance = raw.attrs["artifact_provenance"]
    assert isinstance(provenance, dict)
    physics = provenance["physics"]
    assert isinstance(physics, dict)
    physics["residual_schema_version"] = schema_version

    with pytest.raises(ValueError, match="residual_schema_version"):
        analysis.evaluation.dataframe.build_eval_df(raw)


def test_public_presentation_labels_are_consecutive_and_unique() -> None:
    """
    Derive consecutive, unique notebook labels from public registries.

    Both EDA and evaluation sections must produce gap-free section numbering and
    globally unique plot labels without a second hand-maintained index.
    """
    for registry in (analysis.presentation.EDA_SECTIONS, analysis.presentation.EVALUATION_SECTIONS):
        numbered = tuple(analysis.presentation.numbered_registry(registry))
        section_labels = [section_label for _section, section_label, _plots in numbered]
        assert section_labels == [analysis.presentation.section_display_label(index, section.name) for index, section in enumerate(registry, start=1)]
        plot_labels = [label for _section, _section_label, plots in numbered for _plot, label in plots]
        assert len(plot_labels) == len(set(plot_labels))
        for section_index, (_section, section_label, plots) in enumerate(numbered, start=1):
            assert section_label.startswith(f"{section_index}. ")
            assert [label for _plot, label in plots] == [
                analysis.presentation.plot_display_label(section_index, plot_index, plot.name)
                for plot_index, (plot, _label) in enumerate(plots, start=1)
            ]


def test_curated_media_keys_match_the_tracking_allowlist() -> None:
    """
    The five approved scientific results are the complete W&B media contract.

    Matching renderer and tracking keys prevents a meaningful local figure from
    being silently omitted or an unapproved cache/data payload from being uploaded.
    """
    assert {
        "run_summary_table",
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    } == analysis.curated_renderer.CURATED_ANALYSIS_KEYS
    assert analysis.curated_renderer.CURATED_ANALYSIS_KEYS == experiments.tracking.POST_ARTIFACT_MEDIA_KEYS


def test_every_retained_evaluation_view_renders(evaluation_datasets: dict[str, pd.DataFrame]) -> None:
    """
    Render every retained evaluation view on one current-schema fixture.

    Table, mapping, and figure results must all remain usable; drawing each figure
    catches deferred Matplotlib errors while avoiding notebook execution.
    """
    plots = analysis.evaluation.plots
    results = [
        plots.run_summary.plot_run_summary_table(datasets=evaluation_datasets),
        plots.run_summary.plot_accuracy_physics_pareto(datasets=evaluation_datasets),
        plots.error_behavior.plot_predictive_error_distributions(datasets=evaluation_datasets),
        plots.error_behavior.plot_error_maps(datasets=evaluation_datasets),
        plots.error_behavior.plot_mean_spatial_fields(datasets=evaluation_datasets),
        plots.error_behavior.plot_mean_field_bias(datasets=evaluation_datasets),
        plots.error_behavior.plot_error_vs_target_magnitude(datasets=evaluation_datasets),
        plots.error_behavior.plot_boundary_error_decomposition(datasets=evaluation_datasets),
        plots.physical_consistency.plot_residual_distributions(datasets=evaluation_datasets),
        plots.physical_consistency.plot_spatial_residuals(datasets=evaluation_datasets),
        plots.physical_consistency.plot_pressure_boundary_summary(datasets=evaluation_datasets),
        plots.spectral_fidelity.plot_spectral_fidelity(datasets=evaluation_datasets),
        plots.sensitivity_capacity.plot_capacity_accuracy(datasets=evaluation_datasets),
        plots.sensitivity_capacity.plot_metadata_error_heatmap(datasets=evaluation_datasets),
        plots.sensitivity_capacity.plot_metadata_error_trends(datasets=evaluation_datasets),
        plots.samples_outliers.plot_task_aware_sample(datasets=evaluation_datasets),
        plots.samples_outliers.plot_permeability_error_overlay(datasets=evaluation_datasets, field="p"),
        plots.samples_outliers.plot_outlier_extreme_tables(datasets=evaluation_datasets),
        plots.samples_outliers.plot_linked_outlier_cases(datasets=evaluation_datasets),
        plots.samples_outliers.plot_linked_input_extreme_cases(
            datasets=evaluation_datasets,
            parameter="reynolds",
        ),
    ]
    assert isinstance(results[0], pd.DataFrame)
    tables = [result for result in results if isinstance(result, dict)]
    assert tables
    assert set(tables[0]) == {"metric_outliers", "input_extremes"}
    figures = [result for result in results if isinstance(result, Figure)]
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)


def test_evaluation_panel_construction_is_lazy_and_selectable(
    evaluation_datasets: dict[str, pd.DataFrame],
) -> None:
    """
    Building the full panel creates controls without loading a case or figure.

    The returned widget exposes reviewed tabs while deferred dropdown and render
    actions prevent notebook startup from reading every NPZ artifact.
    """
    panel = analysis.evaluation.panel.build_evaluation_panel(
        datasets_eval=evaluation_datasets,
        title="Fixture comparison",
        sections="all",
    )
    assert panel is not None
    assert plt.get_fignums() == []


def test_known_frequency_and_safe_curated_rendering(
    tmp_path: Path,
    evaluation_datasets: dict[str, pd.DataFrame],
) -> None:
    """
    Recover a known spectral mode and render five outputs safely.

    The numerical spectrum must peak near frequency five, and every curated media
    file must remain in the requested output directory outside artifact caches.
    """
    samples = 64
    x_values = np.arange(samples, dtype=float) / samples
    field = np.sin(2.0 * np.pi * 5.0 * x_values)[None, :] * np.ones((samples, 1))
    k_values, energy = analysis.evaluation.plots.spectral_fidelity.radial_power_spectrum(
        field,
        dx=1.0 / samples,
        dy=1.0 / samples,
        n_bins=45,
    )
    assert k_values[int(np.argmax(energy[1:])) + 1] == pytest.approx(5.0, abs=1.1)

    output_dir = tmp_path / "rendered"
    bundle = analysis.curated_renderer.render_curated_analysis(
        datasets=evaluation_datasets,
        output_dir=output_dir,
    )
    assert set(bundle.media_files).union(bundle.tables) == analysis.curated_renderer.CURATED_ANALYSIS_KEYS
    assert all(path.is_file() and path.parent == output_dir for path in bundle.media_files.values())
    for frame in evaluation_datasets.values():
        assert not output_dir.is_relative_to(Path(frame.attrs["artifact_root"]))


def test_retained_eda_spectral_questions_render() -> None:
    """
    Render retained EDA spectra and construct the numbered panel without display.

    Isotropic, directional, and height-resolved figures must draw successfully.
    The same compact task-aware frames must also build every lazy EDA tab without
    rendering a hidden figure during notebook startup.
    """
    y_values, x_values = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 2.0, 14), indexing="ij")
    rows = [{"x": x_values, "y": y_values, "p": np.sin((index + 1) * np.pi * x_values), "meta": {}} for index in range(3)]
    datasets: dict[str, pd.DataFrame] = {}
    for label, scale in (("ID", 1.0), ("OOD", 1.2)):
        frame = pd.DataFrame([{**row, "p": scale * row["p"]} for row in rows])
        frame.attrs.update(
            {
                "task_id": "steady_flow",
                "field_names": ("x", "y", "p"),
                "field_units": {"x": "m", "y": "m", "p": "Pa"},
                "field_roles": {"x": "coordinate", "y": "coordinate", "p": "state"},
            }
        )
        datasets[label] = frame
    figures = (
        analysis.eda.plots.spectral.plot_isotropic_spectral_summary(datasets=datasets, max_cases=3),
        analysis.eda.plots.spectral.plot_directional_spectral_summary(datasets=datasets, max_cases=3),
        analysis.eda.plots.spectral.plot_vertical_spectral_evolution(datasets=datasets, max_cases=3),
    )
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)
    panel = analysis.eda.panel.build_eda_panel(datasets=datasets, title="Fixture EDA")
    assert panel is not None
    assert plt.get_fignums() == []
