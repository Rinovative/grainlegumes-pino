# ruff: noqa: S101
"""Protect numerical preparation and public rendering behavior, not plot layout."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from src import analysis
from src.domain.tasks.domain_task_steady_flow import STEADY_FLOW


def _spectral_frames(*, count: int = 3) -> dict[str, pd.DataFrame]:
    """Return compact, task-compatible EDA frames with artificial fields."""
    y_grid, x_grid = np.meshgrid(
        np.linspace(0.0, 1.0, 8),
        np.linspace(0.0, 2.0, 10),
        indexing="ij",
    )
    datasets: dict[str, pd.DataFrame] = {}
    for label, scale in (("ID", 1.0), ("OOD", 1.2)):
        frame = pd.DataFrame(
            [
                {
                    "x": x_grid.copy(),
                    "y": y_grid.copy(),
                    "p": scale * (index + 1) * (np.sin(np.pi * x_grid) + 0.1 * np.cos((index + 1) * np.pi * y_grid)),
                    "meta": {"seed": 1000 * (1 if label == "ID" else 2) + index},
                }
                for index in range(count)
            ],
            index=pd.Index(
                [f"case_{index + 1:04d}" for index in range(count)],
                name="sample_id",
            ),
        )
        frame.attrs.update(
            {
                "task_id": STEADY_FLOW.id,
                "task_contract_digest": "synthetic-task-contract",
                "field_names": ("x", "y", "p"),
                "field_units": {"x": "m", "y": "m", "p": "Pa"},
                "field_representations": {
                    "x": "identity",
                    "y": "identity",
                    "p": "identity_before_train_normalization",
                },
                "field_roles": {
                    "x": "coordinate",
                    "y": "coordinate",
                    "p": "state",
                },
            }
        )
        datasets[label] = frame
    return datasets


def test_relative_scoreboard_handles_zero_evidence_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render finite relative values when one explicit synthetic metric is zero."""
    datasets: dict[str, pd.DataFrame] = {}
    for label, aggregate_value, group_values in (
        ("First", 0.0, (0.0, 2.0)),
        ("Second", 1.5, (3.0, 1.0)),
    ):
        frame = pd.DataFrame({"placeholder": [1]})
        frame.attrs["normalized_group_macro_rmse"] = {
            "value": aggregate_value,
            "group_statistics": {
                "pressure": {"normalized_rmse": group_values[0]},
                "velocity": {"normalized_rmse": group_values[1]},
            },
        }
        datasets[label] = frame
    before = {label: deepcopy(frame.attrs) for label, frame in datasets.items()}
    monkeypatch.setattr(
        analysis.evaluation.dataframe,
        "validate_comparison",
        lambda _datasets: None,
    )

    figure = analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(datasets=datasets)

    assert isinstance(figure, Figure)
    heights = [float(patch.get_height()) for axis in figure.axes for patch in axis.patches if isinstance(patch, Rectangle)]
    assert heights
    assert np.isfinite(heights).all()
    figure.canvas.draw()
    data_axis = next(axis for axis in figure.axes if axis.patches)
    explanation = next(text for text in figure.texts if "descriptive comparison" in text.get_text().lower())
    explanation_bounds = explanation.get_window_extent().transformed(figure.transFigure.inverted())
    assert explanation_bounds.y0 > data_axis.get_position().y1
    assert data_axis.get_position().height > explanation_bounds.height
    assert data_axis.get_position().height > 1.0 - data_axis.get_position().y1
    assert {label: frame.attrs for label, frame in datasets.items()} == before
    plt.close(figure)


def test_map_axes_use_global_grid_edges_without_changing_map_geometry() -> None:
    """Decorate only global edges while preserving map data and geometry."""
    figure, axes = plt.subplots(2, 3, squeeze=False)
    values = np.arange(6, dtype=float).reshape(2, 3)
    for index, axis in enumerate(axes.flat):
        axis.imshow(
            values + index,
            origin="lower",
            extent=(1.0, 4.0, 2.0, 4.0),
            aspect="equal",
        )
    axes[1, 2].axis("off")
    figure.canvas.draw()
    x_ticks = [[axis.get_xticks().copy() for axis in row] for row in axes]
    y_ticks = [[axis.get_yticks().copy() for axis in row] for row in axes]
    map_states = [
        [
            (
                axis.images[0].get_array().copy(),
                axis.get_xlim(),
                axis.get_ylim(),
                axis.get_aspect(),
            )
            for axis in row
        ]
        for row in axes
    ]

    analysis.evaluation.plots.layout.apply_map_grid_axis_labels(
        axes,
        x_label="x [m]",
        y_label="y [m]",
    )
    manual_decorations = analysis.evaluation.plots.layout.add_shortened_column_x_decorations(
        axes,
        x_label="x [m]",
    )
    figure.canvas.draw()

    assert [[axis.get_ylabel() for axis in row] for row in axes] == [
        ["y [m]", "", ""],
        ["y [m]", "", ""],
    ]
    assert [[axis.get_xlabel() for axis in row] for row in axes] == [
        ["", "", ""],
        ["x [m]", "x [m]", ""],
    ]
    manual_axis_label = next(artist for artist in manual_decorations if artist.get_text() == "x [m]")
    manual_tick_labels = tuple(artist for artist in manual_decorations if artist is not manual_axis_label)
    shortened_axis = axes[0, 2]
    tick_locations = shortened_axis.get_xticks()
    formatted_ticks = shortened_axis.xaxis.get_major_formatter().format_ticks(tick_locations)
    x_limits = sorted(shortened_axis.get_xlim())
    expected_manual_ticks = [
        (location, label) for location, label in zip(tick_locations, formatted_ticks, strict=True) if label and x_limits[0] <= location <= x_limits[1]
    ]
    assert manual_axis_label.axes is shortened_axis
    assert not manual_axis_label.get_in_layout()
    assert [artist.get_text() for artist in manual_tick_labels] == [label for _location, label in expected_manual_ticks]
    assert [artist.get_position()[0] for artist in manual_tick_labels] == [location for location, _label in expected_manual_ticks]
    assert all(artist.axes is shortened_axis for artist in manual_tick_labels)
    assert all(not artist.get_in_layout() for artist in manual_tick_labels)
    expected_x_tick_labels = ((False, False, False), (True, True, False))
    expected_y_tick_labels = ((True, False, False), (True, False, False))
    for row, axis_row in enumerate(axes):
        for column, axis in enumerate(axis_row):
            expected_data, expected_xlim, expected_ylim, expected_aspect = map_states[row][column]
            np.testing.assert_array_equal(axis.images[0].get_array(), expected_data)
            np.testing.assert_array_equal(axis.get_xticks(), x_ticks[row][column])
            np.testing.assert_array_equal(axis.get_yticks(), y_ticks[row][column])
            assert axis.get_xlim() == expected_xlim
            assert axis.get_ylim() == expected_ylim
            assert axis.get_aspect() == expected_aspect
            if axis.axison:
                assert any(label.get_visible() for label in axis.get_xticklabels()) is expected_x_tick_labels[row][column]
                assert any(label.get_visible() for label in axis.get_yticklabels()) is expected_y_tick_labels[row][column]
    plt.close(figure)


def test_standalone_map_keeps_both_coordinate_labels() -> None:
    """Keep complete coordinate context on a standalone map."""
    figure, axes = plt.subplots(1, 1, squeeze=False)

    analysis.evaluation.plots.layout.apply_map_grid_axis_labels(
        axes,
        x_label="x [m]",
        y_label="y [m]",
    )

    figure.canvas.draw()
    assert axes[0, 0].get_xlabel() == "x [m]"
    assert axes[0, 0].get_ylabel() == "y [m]"
    assert any(label.get_visible() for label in axes[0, 0].get_xticklabels())
    assert any(label.get_visible() for label in axes[0, 0].get_yticklabels())
    plt.close(figure)


def test_outlier_ranking_uses_canonical_case_identity_for_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank equal metric values deterministically by saved case identity."""
    frame = pd.DataFrame(
        {
            "case_index": [2, 1],
            "source_index": [1, 0],
            "rel_l2": [1.0, 1.0],
            "rel_h1": [0.5, 0.5],
            "normalized_rmse_p": [0.25, 0.25],
        }
    )
    frame.attrs.update(
        {
            "output_fields": ("p",),
            "output_units": ("Pa",),
            "output_groups": (),
        }
    )
    monkeypatch.setattr(
        analysis.evaluation.dataframe,
        "validate_comparison",
        lambda _datasets: None,
    )

    table = analysis.evaluation.plots.samples_outliers.build_outlier_table(
        {"Synthetic": frame},
        top_k=2,
    )

    tied = table.loc[table["metric"] == "rel_l2"].sort_values("rank")
    assert tuple(tied["case_index"]) == (1, 2)


def test_spectral_summary_renders_compatible_data_without_mutation() -> None:
    """Represent finite spectral series from compatible synthetic EDA frames."""
    datasets = _spectral_frames()
    before = {label: [np.asarray(value).copy() for value in frame["p"]] for label, frame in datasets.items()}

    figure = analysis.eda.plots.spectral.plot_isotropic_spectral_summary(
        datasets=datasets,
        max_cases=2,
    )

    assert isinstance(figure, Figure)
    plotted: list[np.ndarray] = []
    for axis in figure.axes:
        for line in axis.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            if values.size:
                plotted.append(values)
    assert plotted
    assert any(np.isfinite(values).any() for values in plotted)
    for label, frame in datasets.items():
        for actual, expected in zip(frame["p"], before[label], strict=True):
            np.testing.assert_array_equal(actual, expected)
    plt.close(figure)


def test_spectral_summary_rejects_incompatible_scientific_contract() -> None:
    """Fail before comparing frames governed by different task contracts."""
    datasets = _spectral_frames()
    datasets["OOD"].attrs["task_contract_digest"] = "different-contract"

    with pytest.raises(ValueError, match="one TaskSpec contract"):
        analysis.eda.plots.spectral.plot_isotropic_spectral_summary(
            datasets=datasets,
            max_cases=2,
        )


def test_eda_dataframe_derives_speed_without_mutating_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive speed magnitude from explicit components while preserving inputs."""
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
            "manifest_sha256": "synthetic-manifest",
            "generated_data_root": Path("synthetic-generated-root"),
        }

    monkeypatch.setattr("data_generation.load_generated_batch", fake_load)
    frame, _logs = analysis.eda.dataframe.generate_eda_dataframe(
        "synthetic-batch",
        task=STEADY_FLOW,
        show_progress=False,
    )

    np.testing.assert_allclose(
        np.asarray(frame.loc["case_0001", "U"], dtype=float),
        np.hypot(original_u, original_v),
    )
    np.testing.assert_array_equal(frame.loc["case_0001", "u"], original_u)
    np.testing.assert_array_equal(frame.loc["case_0001", "v"], original_v)
