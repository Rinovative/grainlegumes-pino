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
from typing import cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from src import analysis, experiments
from src.domain.tasks.domain_task_steady_flow import STEADY_FLOW
from traitlets import TraitError

_FIELDS = ("p", "u", "v")
_UNITS = ("Pa", "m/s", "m/s")
_INPUT_FIELDS = ("x", "y", "kxx", "kxy", "kyy", "eps", "p_bc")
_MAX_PARAMETER_FIGURE_HEIGHT = 10.0
_READABLE_LEGEND_FONT_SIZE = 10


def _figure_title(figure: Figure) -> str:
    """Return the sole public figure-level title text."""
    titles = tuple(text.get_text() for text in figure.texts if text.get_text())
    assert len(titles) == 1
    return titles[0]


def _provenance(raw: pd.DataFrame, *, role: str, label: str) -> dict[str, object]:
    """
    Build current provenance around exact fixture aggregate evidence.

    ``role`` and ``label`` distinguish ID/OOD identity while the shared schema
    supplies every scientific field required by presentation and renderer tests.
    """
    aggregate = analysis.artifacts.contracts.aggregate_normalized_macro_rmse(raw, output_fields=_FIELDS)
    return {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
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
            "residual_schema_version": analysis.artifacts.contracts.RESIDUAL_SCHEMA_VERSION,
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
            "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
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
    for registry in (analysis.presentation.registry.EDA_SECTIONS, analysis.presentation.registry.EVALUATION_SECTIONS):
        numbered = tuple(analysis.presentation.registry.numbered_registry(registry))
        section_labels = [section_label for _section, section_label, _plots in numbered]
        assert section_labels == [
            analysis.presentation.registry.section_display_label(index, section.name) for index, section in enumerate(registry, start=1)
        ]
        plot_labels = [label for _section, _section_label, plots in numbered for _plot, label in plots]
        assert len(plot_labels) == len(set(plot_labels))
        for section_index, (_section, section_label, plots) in enumerate(numbered, start=1):
            assert section_label.startswith(f"{section_index}. ")
            assert [label for _plot, label in plots] == [
                analysis.presentation.registry.plot_display_label(section_index, plot_index, plot.name)
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
    } == analysis.presentation.curated.CURATED_ANALYSIS_KEYS
    assert analysis.presentation.curated.CURATED_ANALYSIS_KEYS == experiments.tracking.POST_ARTIFACT_MEDIA_KEYS


def test_every_retained_evaluation_view_renders(evaluation_datasets: dict[str, pd.DataFrame]) -> None:
    """
    Render every retained evaluation view on one current-schema fixture.

    Table, mapping, and figure results must all remain usable; drawing each figure
    catches deferred Matplotlib errors while avoiding notebook execution.
    """
    plots = analysis.evaluation.plots
    results = [
        plots.run_summary.build_run_summary_table(datasets=evaluation_datasets),
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
    bundle = analysis.presentation.curated.render_curated_analysis(
        datasets=evaluation_datasets,
        output_dir=output_dir,
    )
    assert set(bundle.media_files).union(bundle.tables) == analysis.presentation.curated.CURATED_ANALYSIS_KEYS
    assert all(path.is_file() and path.parent == output_dir for path in bundle.media_files.values())
    for frame in evaluation_datasets.values():
        assert not output_dir.is_relative_to(Path(frame.attrs["artifact_root"]))


def _eda_selection_frames(*, count: int = 3) -> dict[str, pd.DataFrame]:
    """Return two task-compatible manifest-indexed spectral frames."""
    y_grid, x_grid = np.meshgrid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 2.0, 10), indexing="ij")
    datasets: dict[str, pd.DataFrame] = {}
    for label, scale in (("ID", 1.0), ("OOD", 1.2)):
        frame = pd.DataFrame(
            [
                {
                    "x": x_grid,
                    "y": y_grid,
                    "p": scale * (index + 1) * (np.sin(np.pi * x_grid) + 0.1 * np.cos((index + 1) * np.pi * y_grid)),
                    "meta": {"seed": 1000 * (1 if label == "ID" else 2) + index},
                }
                for index in range(count)
            ],
            index=pd.Index([f"case_{index + 1:04d}" for index in range(count)], name="sample_id"),
        )
        frame.attrs.update(
            {
                "task_id": "steady_flow",
                "task_contract_digest": "steady-flow-contract",
                "field_names": ("x", "y", "p"),
                "field_units": {"x": "m", "y": "m", "p": "Pa"},
                "field_representations": {
                    "x": "identity",
                    "y": "identity",
                    "p": "identity_before_train_normalization",
                },
                "field_roles": {"x": "coordinate", "y": "coordinate", "p": "state"},
            }
        )
        datasets[label] = frame
    return datasets


def _walk_widgets(widget: widgets.Widget) -> list[widgets.Widget]:
    """Return one widget tree in display order."""
    found = [widget]
    for child in getattr(widget, "children", ()):
        found.extend(_walk_widgets(child))
    return found


def test_retained_eda_spectral_questions_render() -> None:
    """Render retained aggregate questions plus their new exact-case views."""
    datasets = _eda_selection_frames()
    figures = (
        analysis.eda.plots.spectral.plot_isotropic_spectral_summary(datasets=datasets, max_cases=3),
        analysis.eda.plots.spectral.plot_isotropic_spectral_case(datasets=datasets, case_number=1),
        analysis.eda.plots.spectral.plot_directional_spectral_summary(datasets=datasets, max_cases=3),
        analysis.eda.plots.spectral.plot_directional_spectral_case(datasets=datasets, case_number=1),
        analysis.eda.plots.spectral.plot_vertical_spectral_case(datasets=datasets, case_number=1),
    )
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)
    panel = analysis.eda.panel.build_eda_panel(datasets=datasets, title="Fixture EDA")
    assert panel is not None
    assert plt.get_fignums() == []


def test_eda_spectral_labels_use_stored_representation_not_physical_unit() -> None:
    """Disclose transformed dimensionless values instead of implying squared m²."""
    datasets = _eda_selection_frames()
    for frame in datasets.values():
        frame.attrs["field_units"]["p"] = "m^2"
        frame.attrs["field_representations"]["p"] = "dimensionless_log10_ratio_to_1_m2"

    figure = analysis.eda.plots.spectral.plot_isotropic_spectral_summary(
        datasets=datasets,
        max_cases=2,
    )
    power_axis = next(axis for axis in figure.axes if axis.get_title() == "p: isotropic power")
    assert power_axis.get_ylabel() == "Mean spectral power [-]\nStored: log10(k / 1 m²)"
    assert "m^2" not in power_axis.get_ylabel()
    plt.close(figure)


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("task_contract_digest", "different-contract"),
        (
            "field_representations",
            {"x": "identity", "y": "identity", "p": "dimensionless_log10_ratio_to_1_m2"},
        ),
    ],
)
def test_eda_spectral_comparison_rejects_contract_or_representation_drift(
    attribute: str,
    replacement: object,
) -> None:
    """Reject same-task frames governed by different scientific contracts."""
    datasets = _eda_selection_frames()
    datasets["OOD"].attrs[attribute] = replacement
    with pytest.raises(ValueError, match="one TaskSpec contract"):
        analysis.eda.plots.spectral.plot_isotropic_spectral_summary(
            datasets=datasets,
            max_cases=2,
        )


def test_eda_metadata_flattening_excludes_only_rng_implementation_state() -> None:
    """Keep sampled science and seeds while removing exact RNG-state branches."""
    parameters = {
        "seed": 3001,
        "sample_seed": 17,
        "background": {"base_len_rel": 0.05, "ms_weight": [0.4, 0.6]},
        "rng_state": {"Seed": 3001, "State": list(range(625)), "Type": "twister"},
        "python_rng_state": [1, 2, 3],
        "numpy_rng_state": [4, 5, 6],
        "torch_rng_state": [7, 8, 9],
        "rng_type": 2,
        "rng_counter": 11,
    }
    flattened = analysis.eda.plots.case_statistics._flatten_dict_raw(parameters)  # noqa: SLF001
    assert flattened == {
        "seed": 3001.0,
        "sample_seed": 17.0,
        "background_base_len_rel": 0.05,
        "background_ms_weight_0": 0.4,
        "background_ms_weight_1": 0.6,
    }


def test_eda_plot_1_2_layout_uses_only_filtered_parameter_panels() -> None:
    """Derive plot-1.2 rows from retained parameters with no unused subplot axes."""
    parameters = {
        "seed": 3001,
        "background": {"base_len_rel": 0.05, "ms_weight": [0.4, 0.6]},
        "noise": {"level": 0.2, "bias": 0.5},
        "rng_state": {"Seed": 3001, "State": list(range(625)), "Type": "twister"},
    }
    flattened = analysis.eda.plots.case_statistics._flatten_dict_raw(parameters)  # noqa: SLF001
    columns = list(flattened)
    data = {"Batch": {key: np.asarray([value, value + 0.1]) for key, value in flattened.items()}}
    figure = analysis.eda.plots.case_statistics._hist_grid(  # noqa: SLF001
        data_by_dataset=data,
        active_datasets=["Batch"],
        columns=columns,
        title="Meta parameter distributions (first 2 cases)",
    )
    subplot_axes = [axis for axis in figure.axes if axis.axison]
    assert len(subplot_axes) == len(columns)
    assert [axis.get_title() for axis in subplot_axes] == columns
    assert len(figure.axes) == len(columns) + 1
    assert figure.get_size_inches()[1] < _MAX_PARAMETER_FIGURE_HEIGHT
    figure.canvas.draw()
    plt.close(figure)


@pytest.mark.parametrize(
    ("plot_name", "argument_name", "argument_value", "disclosure_kind"),
    [
        ("plot_isotropic_spectral_summary", "max_cases", 3, "legend"),
        ("plot_directional_spectral_summary", "max_cases", 3, "legend"),
        ("plot_vertical_spectral_case", "case_number", 1, "title"),
    ],
)
def test_eda_spectral_plots_share_dataset_selection(
    plot_name: str,
    argument_name: str,
    argument_value: int,
    disclosure_kind: str,
) -> None:
    """Apply one ordered dataset-selection contract to plots 2.1, 2.2, and 2.3."""
    datasets = _eda_selection_frames()
    plot_function = getattr(analysis.eda.plots.spectral, plot_name)
    arguments = {argument_name: argument_value}

    both = plot_function(datasets=datasets, dataset_names=("OOD", "ID"), **arguments)
    if disclosure_kind == "legend":
        labels = [item.get_text() for axis in both.axes if (legend := axis.get_legend()) is not None for item in legend.texts]
        assert any(label == "ID" or label.startswith("ID ") for label in labels)
        assert any(label == "OOD" or label.startswith("OOD ") for label in labels)
    else:
        titles = [axis.get_title() for axis in both.axes if axis.get_title()]
        assert any(title.startswith("ID:") for title in titles)
        assert any(title.startswith("OOD:") for title in titles)
    plt.close(both)

    single = plot_function(datasets=datasets, dataset_names=("OOD",), **arguments)
    titles = [axis.get_title() for axis in single.axes if axis.get_title()]
    if disclosure_kind == "legend":
        labels = [item.get_text() for axis in single.axes if (legend := axis.get_legend()) is not None for item in legend.texts]
        assert labels
        assert all(label != "ID" and not label.startswith("ID ") for label in labels)
    else:
        assert titles == ["OOD: p (identity_before_train_normalization)"]
    plt.close(single)

    with pytest.raises(ValueError, match="Select at least one dataset"):
        plot_function(datasets=datasets, dataset_names=(), **arguments)
    with pytest.raises(ValueError, match="Unknown EDA dataset selection"):
        plot_function(datasets=datasets, dataset_names=("missing",), **arguments)


def test_eda_spectral_flow_order_counts_labels_and_single_bands_are_correct() -> None:
    """Route y/flow left and x/cross-stream right in aggregate and single scopes."""
    datasets = _eda_selection_frames()
    isotropic = analysis.eda.plots.spectral.plot_isotropic_spectral_summary(datasets=datasets, max_cases=50)
    directional = analysis.eda.plots.spectral.plot_directional_spectral_summary(datasets=datasets, max_cases=50)
    single_isotropic = analysis.eda.plots.spectral.plot_isotropic_spectral_case(datasets=datasets, case_number=2)
    single_directional = analysis.eda.plots.spectral.plot_directional_spectral_case(datasets=datasets, case_number=2)
    evolution = analysis.eda.plots.spectral.plot_vertical_spectral_case(datasets=datasets, case_number=2)

    legends = [
        legend
        for figure in (isotropic, directional, single_isotropic, single_directional)
        for axis in figure.axes
        if (legend := axis.get_legend()) is not None
    ]
    assert legends
    font_sizes = [item.get_fontsize() for legend in legends for item in legend.texts]
    assert all(isinstance(size, int | float) and float(size) >= _READABLE_LEGEND_FONT_SIZE for size in font_sizes)
    assert _figure_title(isotropic) == "Isotropic spectra and cumulative energy — first 3 ordered cases"
    assert _figure_title(single_isotropic) == "Isotropic spectra and cumulative energy — case 2"
    assert all(not axis.collections for axis in single_isotropic.axes)

    directional_axes = [axis for axis in directional.axes if axis.get_title()]
    flow_axis, cross_stream_axis = directional_axes
    assert flow_axis.get_title() == "p: flow-direction spectrum (y)"
    assert cross_stream_axis.get_title() == "p: cross-stream spectrum (x)"
    assert flow_axis.get_xlabel() == "Spatial frequency ky [1/m]"
    assert cross_stream_axis.get_xlabel() == "Spatial frequency kx [1/m]"
    assert flow_axis.get_ylabel() == cross_stream_axis.get_ylabel() == "Mean spectral power [(Pa)²]"
    assert all(axis.get_ylabel() == "Cumulative energy [-]" for axis in directional.axes if not axis.get_title())
    assert _figure_title(directional) == "Directional spectra in flow and cross-stream directions — first 3 ordered cases"

    _radial_k, _radial, x_k, x_energy, y_k, y_energy, _count, _unit = analysis.eda.plots.spectral._case_spectra(  # noqa: SLF001
        datasets["ID"], "p", max_cases=50
    )
    x_median = np.quantile(x_energy, 0.5, axis=0)
    y_median = np.quantile(y_energy, 0.5, axis=0)
    x_valid = (x_k > 0.0) & (x_median > 0.0)
    y_valid = (y_k > 0.0) & (y_median > 0.0)
    np.testing.assert_allclose(np.asarray(flow_axis.lines[0].get_xdata(), dtype=float), y_k[y_valid])
    np.testing.assert_allclose(np.asarray(flow_axis.lines[0].get_ydata(), dtype=float), y_median[y_valid])
    np.testing.assert_allclose(np.asarray(cross_stream_axis.lines[0].get_xdata(), dtype=float), x_k[x_valid])
    np.testing.assert_allclose(np.asarray(cross_stream_axis.lines[0].get_ydata(), dtype=float), x_median[x_valid])

    single_directional_axes = [axis for axis in single_directional.axes if axis.get_title()]
    single_flow_axis, single_cross_stream_axis = single_directional_axes
    row = datasets["ID"].loc["case_0002"]
    assert isinstance(row, pd.Series)
    _single_radial_k, _single_radial, single_x_k, single_x_energy, single_y_k, single_y_energy, _single_unit = (
        analysis.eda.plots.spectral._row_spectra(row, "p")  # noqa: SLF001
    )
    single_x_valid = (single_x_k > 0.0) & (single_x_energy > 0.0)
    single_y_valid = (single_y_k > 0.0) & (single_y_energy > 0.0)
    np.testing.assert_allclose(np.asarray(single_flow_axis.lines[0].get_xdata(), dtype=float), single_y_k[single_y_valid])
    np.testing.assert_allclose(np.asarray(single_flow_axis.lines[0].get_ydata(), dtype=float), single_y_energy[single_y_valid])
    np.testing.assert_allclose(np.asarray(single_cross_stream_axis.lines[0].get_xdata(), dtype=float), single_x_k[single_x_valid])
    np.testing.assert_allclose(np.asarray(single_cross_stream_axis.lines[0].get_ydata(), dtype=float), single_x_energy[single_x_valid])
    assert all(not axis.collections for axis in single_directional_axes)
    assert _figure_title(single_directional).endswith("— case 2")
    evolution_axes = [axis for axis in evolution.axes if axis.get_title()]
    assert all(axis.get_xlabel() == "Cross-stream spatial frequency kx [1/m]" for axis in evolution_axes)
    assert all("(identity_before_train_normalization)" in axis.get_title() for axis in evolution_axes)
    assert all(axis.get_ylabel() == "Flow-direction position y [m]" for axis in evolution_axes)
    assert all("frequency" not in axis.get_ylabel().casefold() for axis in evolution_axes)
    assert _figure_title(evolution) == "Cross-stream spectral evolution along the flow direction — case 2"
    assert all("<=" not in _figure_title(figure) for figure in (isotropic, directional, single_isotropic, single_directional, evolution))

    for figure in (isotropic, directional, single_isotropic, single_directional, evolution):
        figure.canvas.draw()
        plt.close(figure)


def test_case_numbers_use_manifest_ids_without_positional_pairing_or_clamping() -> None:
    """Resolve reordered shared IDs exactly while treating realizations independently."""
    base = _eda_selection_frames(count=4)
    id_frame = base["ID"].iloc[[1, 3]].copy()
    ood_frame = base["OOD"].iloc[[3, 1]].copy()
    id_frame.index = pd.Index(["case_0002", "case_0004"], name="sample_id")
    ood_frame.index = pd.Index(["case_0004", "case_0002"], name="sample_id")
    datasets = {"ID": id_frame, "OOD": ood_frame}

    assert analysis.eda.plots.spectral.available_case_numbers(id_frame) == (2, 4)
    assert analysis.eda.plots.spectral.available_case_numbers(ood_frame) == (4, 2)
    figure = analysis.eda.plots.spectral.plot_isotropic_spectral_case(datasets=datasets, case_number=2)
    row = ood_frame.loc["case_0002"]
    assert isinstance(row, pd.Series)
    radial_k, radial, *_rest = analysis.eda.plots.spectral._row_spectra(row, "p")  # noqa: SLF001
    valid = (radial_k > 0.0) & (radial > 0.0)
    np.testing.assert_allclose(np.asarray(figure.axes[0].lines[1].get_xdata(), dtype=float), radial_k[valid])
    np.testing.assert_allclose(np.asarray(figure.axes[0].lines[1].get_ydata(), dtype=float), radial[valid])
    id_metadata = cast("dict[str, object]", id_frame.loc["case_0002", "meta"])
    ood_metadata = cast("dict[str, object]", ood_frame.loc["case_0002", "meta"])
    assert id_metadata["seed"] != ood_metadata["seed"]
    with pytest.raises(ValueError, match="Requested case 3 is unavailable"):
        analysis.eda.plots.spectral.plot_vertical_spectral_case(datasets=datasets, case_number=3)
    plt.close(figure)


def test_synchronized_case_viewer_uses_compact_sparse_navigation_once_per_action() -> None:
    """Use committed numeric entry while navigating actual sparse shared IDs."""
    first_case = 2
    second_case = 4
    extra_case = 6
    base = _eda_selection_frames(count=second_case)
    frames = {
        "ID": base["ID"].iloc[[1, 3]].copy(),
        "OOD": base["OOD"].iloc[[3, 1]].copy(),
        "AUX": base["ID"].iloc[[3, 1, 0]].copy(),
    }
    frames["ID"].index = pd.Index(["case_0002", "case_0004"], name="sample_id")
    frames["OOD"].index = pd.Index(["case_0004", "case_0002"], name="sample_id")
    frames["AUX"].index = pd.Index(["case_0004", "case_0002", "case_0006"], name="sample_id")
    case_numbers = {name: analysis.eda.plots.spectral.available_case_numbers(frame) for name, frame in frames.items()}
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_single(*, datasets: dict[str, pd.DataFrame], case_number: int) -> Figure:
        calls.append((tuple(datasets), case_number))
        return plt.figure()

    viewer = analysis.ui.viewers.make_dataset_case_scope_viewer(
        datasets=frames,
        case_numbers_by_dataset=case_numbers,
        single_plot_func=fake_single,
    )
    tree = _walk_widgets(viewer)
    case_control = next(widget for widget in tree if type(widget) is widgets.IntText and widget.description == "Case:")
    checkboxes = [widget for widget in tree if isinstance(widget, widgets.Checkbox)]
    previous = next(widget for widget in tree if isinstance(widget, widgets.Button) and widget.description == "←")
    following = next(widget for widget in tree if isinstance(widget, widgets.Button) and widget.description == "→")
    assert not any(
        isinstance(widget, widgets.Dropdown | widgets.SelectionSlider | widgets.RadioButtons) and getattr(widget, "description", "") == "Case:"
        for widget in tree
    )
    assert case_control.continuous_update is False
    assert case_control.value == first_case
    assert previous.disabled is True
    assert following.disabled is False
    assert calls == [(("ID", "OOD", "AUX"), first_case)]

    before = len(calls)
    case_control.value = second_case
    assert len(calls) == before + 1
    assert calls[-1] == (("ID", "OOD", "AUX"), second_case)
    assert previous.disabled is False
    assert following.disabled is True

    before = len(calls)
    previous.click()
    assert len(calls) == before + 1
    assert case_control.value == first_case
    assert previous.disabled is True
    assert following.disabled is False

    before = len(calls)
    case_control.value = 3
    assert len(calls) == before
    assert case_control.value == first_case
    with pytest.raises(TraitError):
        case_control.set_trait("value", "not-an-integer")

    case_control.value = second_case
    before = len(calls)
    checkboxes[0].value = False
    assert len(calls) == before + 1
    assert case_control.value == second_case
    assert previous.disabled is True
    assert following.disabled is False
    assert calls[-1] == (("OOD", "AUX"), second_case)

    before = len(calls)
    checkboxes[1].value = False
    assert len(calls) == before + 1
    assert case_control.value == second_case
    assert previous.disabled is True
    assert following.disabled is False

    before = len(calls)
    following.click()
    assert len(calls) == before + 1
    assert case_control.value == first_case
    before = len(calls)
    following.click()
    assert len(calls) == before + 1
    assert case_control.value == extra_case
    assert following.disabled is True
    assert calls[-1] == (("AUX",), extra_case)

    before = len(calls)
    checkboxes[1].value = True
    assert len(calls) == before + 1
    assert case_control.value == second_case
    assert calls[-1] == (("OOD", "AUX"), second_case)

    before = len(calls)
    checkboxes[0].value = True
    assert len(calls) == before + 1
    assert case_control.value == second_case
    assert calls[-1] == (("ID", "OOD", "AUX"), second_case)
    visible_text = " ".join(str(value) for widget in tree for value in (getattr(widget, "description", ""), getattr(widget, "value", ""))).casefold()
    assert "seed" not in visible_text
    assert "paired" not in visible_text
    assert all(
        not {"render", "update", "apply", "confirm"}.intersection(button.description.casefold().split())
        for button in tree
        if isinstance(button, widgets.Button)
    )
    plt.close("all")


@pytest.mark.parametrize("count", [1, 37, 50, 64, 100, 125])
def test_aggregate_scope_defaults_to_minimum_of_100_and_available_cases(count: int) -> None:
    """Use one consistent aggregate default with no hidden spectral 64-case rule."""
    frame = pd.DataFrame({"value": range(count)}, index=pd.Index([f"case_{index + 1:04d}" for index in range(count)]))
    calls: list[int] = []

    def fake_aggregate(*, datasets: dict[str, pd.DataFrame], max_cases: int) -> Figure:
        assert tuple(datasets) == ("Batch",)
        calls.append(max_cases)
        return plt.figure()

    viewer = analysis.ui.viewers.make_dataset_case_scope_viewer(
        datasets={"Batch": frame},
        case_numbers_by_dataset={"Batch": tuple(range(1, count + 1))},
        single_plot_func=lambda **_kwargs: plt.figure(),
        aggregate_plot_func=fake_aggregate,
    )
    slider = next(widget for widget in _walk_widgets(viewer) if isinstance(widget, widgets.IntSlider))
    assert slider.value == min(100, count)
    assert calls == [min(100, count)]
    plt.close("all")


@pytest.mark.parametrize(
    ("viewer_name", "aggregate_name", "single_name"),
    [
        ("_isotropic_viewer", "plot_isotropic_spectral_summary", "plot_isotropic_spectral_case"),
        ("_directional_viewer", "plot_directional_spectral_summary", "plot_directional_spectral_case"),
    ],
)
def test_eda_2_1_and_2_2_switch_automatically_between_aggregate_and_single_case(
    viewer_name: str,
    aggregate_name: str,
    single_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain aggregate evidence while exposing exact individual-case navigation."""
    datasets = _eda_selection_frames()
    calls: list[tuple[str, tuple[str, ...], int]] = []

    def fake_aggregate(*, datasets: dict[str, pd.DataFrame], max_cases: int) -> Figure:
        calls.append(("aggregate", tuple(datasets), max_cases))
        return plt.figure()

    def fake_single(*, datasets: dict[str, pd.DataFrame], case_number: int) -> Figure:
        calls.append(("single", tuple(datasets), case_number))
        return plt.figure()

    monkeypatch.setattr(analysis.eda.plots.spectral, aggregate_name, fake_aggregate)
    monkeypatch.setattr(analysis.eda.plots.spectral, single_name, fake_single)
    viewer = getattr(analysis.eda.panel, viewer_name)(datasets=datasets)
    tree = _walk_widgets(viewer)
    buttons = [widget for widget in tree if isinstance(widget, widgets.Button)]
    assert all("render" not in button.description.casefold() and "update" not in button.description.casefold() for button in buttons)
    scope = next(widget for widget in tree if isinstance(widget, widgets.ToggleButtons))
    controls = viewer.children[0]
    assert isinstance(controls, widgets.HBox)
    assert controls.children[0] is scope
    assert scope.description == ""
    assert [label for label, _value in scope.options] == ["Aggregate", "Single case"]
    assert not any(getattr(widget, "description", "") == "Scope:" for widget in tree)
    case_count = next(widget for widget in tree if isinstance(widget, widgets.IntSlider))
    expected_case_count = len(next(iter(datasets.values())))
    assert case_count.value == expected_case_count
    assert calls == [("aggregate", ("ID", "OOD"), expected_case_count)]
    scope.value = "single"
    assert calls[-1] == ("single", ("ID", "OOD"), 1)
    case_control = next(widget for widget in _walk_widgets(viewer) if type(widget) is widgets.IntText and widget.description == "Case:")
    assert controls.children[0] is scope
    assert controls.children[1] is case_control
    assert not any(
        isinstance(widget, widgets.Dropdown | widgets.SelectionSlider | widgets.RadioButtons) and getattr(widget, "description", "") == "Case:"
        for widget in _walk_widgets(viewer)
    )
    before = len(calls)
    case_control.value = 2
    assert len(calls) == before + 1
    assert calls[-1] == ("single", ("ID", "OOD"), 2)
    plt.close("all")


def test_eda_2_3_is_synchronized_single_case_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep 2.3 case-specific with shared checkboxes and no aggregate count."""
    datasets = _eda_selection_frames()
    calls: list[tuple[tuple[str, ...], int]] = []

    def fake_single(*, datasets: dict[str, pd.DataFrame], case_number: int) -> Figure:
        calls.append((tuple(datasets), case_number))
        return plt.figure()

    monkeypatch.setattr(analysis.eda.plots.spectral, "plot_vertical_spectral_case", fake_single)
    viewer = analysis.eda.panel._evolution_viewer(datasets=datasets)  # noqa: SLF001
    tree = _walk_widgets(viewer)
    assert not any(isinstance(widget, widgets.IntSlider) for widget in tree)
    assert not any(isinstance(widget, widgets.ToggleButtons) for widget in tree)
    case_control = next(widget for widget in tree if type(widget) is widgets.IntText and widget.description == "Case:")
    checkboxes = [widget for widget in tree if isinstance(widget, widgets.Checkbox)]
    controls = viewer.children[0]
    assert isinstance(controls, widgets.HBox)
    assert controls.children[0] is case_control
    assert not any(
        isinstance(widget, widgets.Dropdown | widgets.SelectionSlider | widgets.RadioButtons) and getattr(widget, "description", "") == "Case:"
        for widget in tree
    )
    assert calls == [(("ID", "OOD"), 1)]
    case_control.value = 3
    assert calls[-1] == (("ID", "OOD"), 3)
    checkboxes[0].value = False
    assert calls[-1] == (("OOD",), 3)
    plt.close("all")


def test_eda_dataframe_derives_speed_once_without_mutating_velocity_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose EDA-only speed from compatible Cartesian velocity components."""
    u = np.asarray([[3.0, 0.0], [5.0, 8.0]])
    v = np.asarray([[4.0, 7.0], [12.0, 15.0]])
    original_u = u.copy()
    original_v = v.copy()

    def fake_load(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "task": STEADY_FLOW,
            "sample_ids": ["case_0001"],
            "rows": [{"x": np.zeros_like(u), "u": u, "v": v, "meta": {}}],
            "available_case_count": 1,
            "generated_batch_identity": {},
            "manifest_sha256": "manifest",
            "generated_data_root": Path("generated"),
        }

    monkeypatch.setattr("data_generation.load_generated_batch", fake_load)
    frame, _logs = analysis.eda.dataframe.generate_eda_dataframe("fixture", task=STEADY_FLOW, show_progress=False)
    np.testing.assert_allclose(np.asarray(frame.loc["case_0001", "U"], dtype=float), np.hypot(original_u, original_v))
    np.testing.assert_array_equal(frame.loc["case_0001", "u"], original_u)
    np.testing.assert_array_equal(frame.loc["case_0001", "v"], original_v)
    assert frame.attrs["field_names"][-1] == "U"
    assert frame.attrs["field_units"]["U"] == "m/s"
    assert frame.attrs["field_representations"]["kxx"] == "dimensionless_log10_ratio_to_1_m2"
    assert frame.attrs["field_representations"]["kxy"] == "dimensionless_cross_component_ratio_to_geometric_mean"
    assert frame.attrs["field_representations"]["U"] == "derived_speed_magnitude"
    assert frame.attrs["field_roles"]["U"] == "derived_speed"


def test_eda_panel_activates_first_view_when_each_tab_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open tabs on 1.1 and 2.1 without retaining a prompt as a real option."""
    datasets = _eda_selection_frames()
    calls: list[tuple[str, object]] = []
    displayed: list[object] = []

    def fake_metadata(*, datasets: dict[str, pd.DataFrame]) -> widgets.Label:
        calls.append(("1.1", tuple(datasets)))
        return widgets.Label("metadata")

    def fake_isotropic(*, datasets: dict[str, pd.DataFrame], max_cases: int) -> widgets.Label:
        calls.append(("2.1", (tuple(datasets), max_cases)))
        return widgets.Label("spectral")

    monkeypatch.setattr(analysis.eda.plots.case_statistics, "plot_meta_statistics", fake_metadata)
    monkeypatch.setattr(analysis.eda.plots.spectral, "plot_isotropic_spectral_summary", fake_isotropic)
    monkeypatch.setattr(analysis.ui.notebook, "display", displayed.append)
    monkeypatch.setattr(analysis.ui.viewers, "display", lambda _value: None)

    panel_output = analysis.eda.panel.build_eda_panel(datasets=datasets, title="Fixture EDA")
    assert panel_output is not None
    assert calls == []
    open_button = next(item for item in displayed if isinstance(item, widgets.Button))
    open_button.click()
    assert calls == [("1.1", ("ID", "OOD"))]

    expanded_panel = displayed[-1]
    assert isinstance(expanded_panel, widgets.VBox)
    tabs = expanded_panel.children[2]
    assert isinstance(tabs, widgets.Tab)
    first_dropdown = tabs.children[0].children[0]
    second_dropdown = tabs.children[1].children[0]
    assert isinstance(first_dropdown, widgets.Dropdown)
    assert isinstance(second_dropdown, widgets.Dropdown)
    assert first_dropdown.value == 0
    assert [label for label, _value in first_dropdown.options] == [
        "1-1. Metadata statistics",
        "1-2. Parameter distributions",
        "1-3. Field value distributions",
    ]
    assert second_dropdown.value is None
    assert all("Choose a view" not in label for label, _value in (*first_dropdown.options, *second_dropdown.options))

    tabs.selected_index = 1
    assert second_dropdown.value == 0
    assert calls[-1] == ("2.1", (("ID", "OOD"), 3))
    plt.close("all")
