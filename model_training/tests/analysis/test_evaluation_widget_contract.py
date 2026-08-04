# ruff: noqa: S101
"""Exercise generic evaluation widget interaction without freezing composition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.ui import analysis_ui_viewers as viewers

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.figure import Figure


def _descendants(widget: widgets.Widget) -> Iterator[widgets.Widget]:
    """Yield nested widget descendants without depending on child positions."""
    for child in getattr(widget, "children", ()):
        yield child
        yield from _descendants(child)


def test_outlier_dataset_control_remains_close_to_upper_section_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the minimally adjusted lower selector aligned and usable."""
    datasets = {
        "First": pd.DataFrame({"value": [1.0]}),
        "Second": pd.DataFrame({"value": [2.0]}),
    }
    rendered: list[tuple[str, ...]] = []

    def render_figure(**kwargs: Any) -> None:
        selected = kwargs["kwargs"]["datasets"]
        rendered.append(tuple(selected))

    monkeypatch.setattr(panel.analysis.ui.viewers, "render_figure", render_figure)
    viewer = panel._make_outlier_table_viewer(  # noqa: SLF001
        lambda **_kwargs: plt.figure(),
        datasets=datasets,
    )

    upper_section = panel.analysis.ui.notebook.make_dropdown_section(
        [("Synthetic view", lambda: None, "synthetic-view")],
        select_first=True,
    )
    descendants = tuple(_descendants(viewer))
    selector = next(widget for widget in descendants if isinstance(widget, widgets.Dropdown))
    upper_selector = next(widget for widget in _descendants(upper_section) if isinstance(widget, widgets.Dropdown))
    assert tuple(selector.options) == tuple(datasets)
    assert selector.value == "First"
    assert selector.description == "Select:"
    assert viewer.layout.align_items == "flex-start"
    assert selector.layout.width.endswith("px")
    assert upper_selector.layout.width.endswith("px")
    lower_width = float(selector.layout.width.removesuffix("px"))
    upper_width = float(upper_selector.layout.width.removesuffix("px"))
    assert lower_width <= upper_width
    assert upper_width - lower_width < upper_width * 0.01
    selector.value = "Second"
    assert rendered == [("First",), ("Second",)]


def test_controlled_viewer_rerenders_and_tracks_the_current_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward selected data and control values on initial and changed renders."""
    datasets = {
        "First": pd.DataFrame({"value": [1.0, 2.0]}),
        "Second": pd.DataFrame({"value": [3.0, 4.0]}),
    }
    before = {label: frame.copy(deep=True) for label, frame in datasets.items()}
    mode = widgets.Dropdown(options=("mean", "median"), value="mean")
    rendered: list[tuple[tuple[str, ...], str, Figure]] = []
    displayed: list[object] = []
    export_state: dict[str, Any] = {}

    def plot(
        *,
        datasets: dict[str, pd.DataFrame],
        statistic: str,
    ) -> Figure:
        figure = plt.figure()
        rendered.append((tuple(datasets), statistic, figure))
        return figure

    monkeypatch.setattr(viewers, "display", displayed.append)
    result = viewers.make_controlled_viewer(
        plot,
        datasets=datasets,
        controls={"statistic": mode},
        allow_dataset_selection=False,
        export_state=export_state,
        export_plot_name="synthetic-view",
    )

    assert isinstance(result, widgets.VBox)
    assert [(labels, value) for labels, value, _figure in rendered] == [(("First", "Second"), "mean")]
    mode.value = "median"
    assert [(labels, value) for labels, value, _figure in rendered] == [
        (("First", "Second"), "mean"),
        (("First", "Second"), "median"),
    ]
    assert export_state["fig"] is rendered[-1][2]
    assert export_state["plot_name"] == "synthetic-view"
    assert displayed == [rendered[0][2], rendered[1][2]]
    for label, frame in datasets.items():
        pd.testing.assert_frame_equal(frame, before[label])
    assert plt.get_fignums() == []


def test_controlled_viewer_rejects_an_empty_dataset_mapping() -> None:
    """Reject a viewer that cannot perform any intended rendering operation."""
    with pytest.raises(ValueError, match="at least one"):
        viewers.make_controlled_viewer(
            lambda **_kwargs: plt.figure(),
            datasets={},
        )
