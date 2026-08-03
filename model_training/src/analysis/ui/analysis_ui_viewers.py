"""
===============================================================================
analysis_ui_viewers.py
===============================================================================
Build interactive viewers for case-level and aggregate analysis plots.

Responsibilities:
  - Render case-by-case Matplotlib figures inside widgets
  - Manage dataset and case navigation controls
  - Store current figures for notebook export

Design principles:
  - Viewer callbacks receive explicit datasets and render functions
  - Widget construction delegates to analysis_ui_components
  - Export context is updated only around rendered figures

This module does NOT:
  - Compose numbered notebook sections or choose scientific control vocabularies
  - Load artifacts directly or implement domain-specific plot mathematics
===============================================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure

from . import analysis_ui_components as components

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pandas as pd


# =============================================================================
# INTERNAL HELPERS (viewer-agnostic, no semantics)
# =============================================================================


def _render_figure(
    *,
    out: widgets.Output,
    plot_func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Invoke and display one plot result inside an output widget.

    Parameters
    ----------
    out : ipywidgets.Output
        Output area cleared immediately before invocation.
    plot_func : Callable[..., Any]
        Callable returning a Matplotlib figure, a tuple whose first item is a
        figure, another displayable value, or ``None``.
    args : tuple[Any, ...], optional
        Positional arguments forwarded unchanged.
    kwargs : dict[str, Any] | None, optional
        Keyword arguments forwarded unchanged.

    Notes
    -----
    Recognized figures update the current module export context, are displayed,
    and are then closed to release GUI resources. Other non-``None`` results are
    displayed without changing the export figure.

    """
    kwargs = kwargs or {}

    with out:
        out.clear_output(wait=True)

        result = plot_func(*args, **kwargs)

        # Accept (fig, ...) as well
        fig: Figure | None = None
        if isinstance(result, Figure):
            fig = result
        elif isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], Figure):
            fig = result[0]

        if fig is not None:
            # Update export target if available
            export_state = _EXPORT_CTX.get("export_state")
            if isinstance(export_state, dict):
                export_state["fig"] = fig

                pn = _EXPORT_CTX.get("plot_name")
                tt = _EXPORT_CTX.get("title")

                if isinstance(pn, str) and pn:
                    export_state["plot_name"] = pn
                if isinstance(tt, str) and tt:
                    export_state["title"] = tt

            display(fig)
            plt.close(fig)
            return

        # Non-figure results (rare): still display them
        if result is not None:
            display(result)


def _attach_widget_rerender(
    widgets_list: list[widgets.Widget],
    render_func: Callable[[], None],
) -> None:
    """
    Register one render callback on heterogeneous semantic controls.

    Standard value widgets observe their ``value`` trait. Checkbox-group
    containers are recognized through the public ``boxes`` mapping and each
    checkbox is observed; widgets without either contract are ignored.
    """
    for w in widgets_list:
        # ---------------------------------------------
        # Case 1: standard ValueWidget (Dropdown, Radio)
        # ---------------------------------------------
        if hasattr(w, "observe") and hasattr(w, "value"):
            w.observe(lambda _: render_func(), names="value")
            continue

        # ---------------------------------------------
        # Case 2: checkbox group (VBox with .boxes)
        # ---------------------------------------------
        boxes = getattr(w, "boxes", None)
        if isinstance(boxes, dict):
            for checkbox in boxes.values():
                if isinstance(checkbox, widgets.Checkbox):
                    checkbox.observe(lambda _: render_func(), names="value")


# =============================================================================
# EXPORT CONTEXT (set by analysis_ui_notebook before running a plot)
# =============================================================================
_EXPORT_CTX: dict[str, Any] = {}


def set_export_context(export_state: dict | None, *, plot_name: str | None = None, title: str | None = None) -> None:
    """
    Replace the module-global export context used by subsequent viewer renders.

    Parameters
    ----------
    export_state : dict | None
        Shared panel state to populate, or ``None`` to disable figure capture.
    plot_name : str | None, optional
        Filename stem associated with the active dropdown entry.
    title : str | None, optional
        User-facing title associated with the active entry.

    Notes
    -----
    The context is process-global mutable notebook state, not thread-safe or
    panel-isolated. Dropdown selection must set it immediately before rendering;
    viewer callbacks then update the referenced state with their latest figure.

    """
    _EXPORT_CTX.clear()
    _EXPORT_CTX.update(
        {
            "export_state": export_state,
            "plot_name": plot_name,
            "title": title,
        }
    )


# =============================================================================
# CONTROLLED COMPARISON VIEWER
# =============================================================================


def make_controlled_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    controls: Mapping[str, widgets.ValueWidget] | None = None,
    plot_kwargs: Mapping[str, Any] | None = None,
    allow_dataset_selection: bool = True,
) -> widgets.VBox:
    """
    Build a lazy renderer with model/dataset and semantic argument controls.

    Parameters
    ----------
    plot_func : callable
        Plot function accepting ``datasets=...`` plus keyword arguments named by
        `controls`. It may return a Matplotlib figure or displayable table.
    datasets : dict[str, pandas.DataFrame]
        Labelled artifact frames available to the viewer.
    controls : Mapping[str, ipywidgets.ValueWidget] | None, optional
        Mapping from plot keyword to a widget whose current ``value`` is passed
        on each render. Labels and option vocabularies remain owned by callers.
    plot_kwargs : Mapping[str, Any] | None, optional
        Fixed keyword arguments forwarded unchanged.
    allow_dataset_selection : bool, optional
        Add model/dataset checkboxes when more than one frame is available.

    Returns
    -------
    ipywidgets.VBox
        A non-rendering control surface. The first plot is created only after the
        user selects ``Render / update``; subsequent control changes rerender.

    Raises
    ------
    ValueError
        If `datasets` is empty. An empty checkbox selection is reported inside
        the viewer without invoking the plot function.

    Notes
    -----
    Figure rendering flows through the shared export context, so the panel's PDF
    action always targets the latest controlled figure.

    """
    if not datasets:
        msg = "Controlled analysis viewers require at least one labelled dataset."
        raise ValueError(msg)
    semantic_controls = dict(controls or {})
    fixed_kwargs = dict(plot_kwargs or {})
    selector = components.ui_checkbox_datasets(dataset_names=list(datasets)) if allow_dataset_selection and len(datasets) > 1 else None
    selector_boxes = {} if selector is None else cast("components.CheckboxGroup", selector).boxes
    output = components.ui_output_plot()
    render_button = widgets.Button(
        description="Render / update",
        button_style="primary",
        layout=widgets.Layout(width="145px"),
    )
    state = {"rendered": False}

    def _selected_datasets() -> dict[str, pd.DataFrame]:
        """Return the currently enabled labelled artifact frames."""
        if selector is None:
            return dict(datasets)
        return {name: datasets[name] for name, checkbox in selector_boxes.items() if checkbox.value}

    def _render(_: object = None) -> None:
        """
        Render selected frames with current semantic control values.

        An empty checkbox selection is disclosed in the output without invoking
        scientific code; successful invocation marks the viewer as initialized.
        """
        selected = _selected_datasets()
        if not selected:
            with output:
                output.clear_output(wait=True)
                print("Select at least one model/dataset before rendering.")
            return
        kwargs = {name: widget.value for name, widget in semantic_controls.items()}
        _render_figure(
            out=output,
            plot_func=plot_func,
            kwargs={"datasets": selected, **fixed_kwargs, **kwargs},
        )
        state["rendered"] = True

    def _rerender_after_first(_: object = None) -> None:
        """Apply live control changes only after an explicit first render."""
        if state["rendered"]:
            _render()

    render_button.on_click(_render)
    for widget in semantic_controls.values():
        widget.observe(_rerender_after_first, names="value")
    if selector is not None:
        for checkbox in selector_boxes.values():
            checkbox.observe(_rerender_after_first, names="value")

    controls_row = widgets.HBox([*semantic_controls.values(), render_button])
    children: list[widgets.Widget] = [controls_row]
    if selector is not None:
        children.insert(0, widgets.VBox([widgets.HTML("<b>Models / datasets</b>"), selector]))
    children.append(output)
    return widgets.VBox(children)


# =============================================================================
# 1) CASE VIEWER (single-case visualisations)
# =============================================================================


def make_interactive_case_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_idx: int = 0,
    enable_dataset_dropdown: bool = True,
    extra_widgets: list[widgets.Widget] | None = None,
    n_cases_fn: Callable[[str, pd.DataFrame], int] | None = None,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Build and immediately render a case-indexed notebook viewer.

    Parameters
    ----------
    plot_func : Callable[..., Any]
        Called as ``plot_func(case_idx, df=..., dataset_name=..., **plot_kwargs)``;
        the internal index is zero-based although the control displays one-based.
    datasets : dict[str, pandas.DataFrame]
        Labelled frames available to the viewer; first insertion order is initial.
    start_idx : int, optional
        Initial zero-based case position.
    enable_dataset_dropdown : bool, optional
        Show a dataset selector when more than one frame is available.
    extra_widgets : list[ipywidgets.Widget] | None, optional
        Additional controls whose changes trigger rerendering.
    n_cases_fn : Callable[[str, pandas.DataFrame], int] | None, optional
        Per-frame case-count resolver; defaults to ``len(frame)``.
    **plot_kwargs : Any
        Fixed plot arguments forwarded on every render.

    Returns
    -------
    ipywidgets.VBox
        Navigation controls and output containing the initial rendered result.

    Notes
    -----
    Dataset changes rebind the case maximum and preserve the nearest valid
    one-based control value. Rendering participates in the shared export context.

    """
    dataset_names = list(datasets.keys())
    active_dataset = dataset_names[0]

    # ------------------------------------------------------------------
    # Dataset selector
    # ------------------------------------------------------------------
    dataset_dropdown = components.ui_dropdown_dataset(dataset_names) if enable_dataset_dropdown and len(dataset_names) > 1 else None

    # ------------------------------------------------------------------
    # Case index step control
    # ------------------------------------------------------------------
    df_active = datasets[active_dataset]

    n_cases_active = n_cases_fn(active_dataset, df_active) if n_cases_fn is not None else len(df_active)

    case_index, prev_btn, next_btn = components.ui_step_case_index(
        n_cases=n_cases_active,
        start_idx=start_idx,
    )
    if not isinstance(case_index, widgets.BoundedIntText):
        msg = "Contiguous case navigation must provide a bounded integer text control."
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Output container
    # ------------------------------------------------------------------
    out = components.ui_output_plot()
    extra_widgets = extra_widgets or []

    # ------------------------------------------------------------------
    # Render logic
    # ------------------------------------------------------------------
    def _render() -> None:
        """
        Clamp and render the current dataset/case selection.

        The display control is one-based; the plotting callable receives a
        zero-based index plus the selected frame and label.
        """
        if dataset_dropdown is not None:
            selected_name = dataset_dropdown.value
            if not isinstance(selected_name, str):
                msg = "Dataset dropdown must contain string values."
                raise TypeError(msg)
            name = selected_name
        else:
            name = active_dataset

        df = datasets[name]

        n_cases = n_cases_fn(name, df) if n_cases_fn is not None else len(df)

        case_idx = case_index.value - 1
        case_idx = max(0, min(n_cases - 1, case_idx))

        _render_figure(
            out=out,
            plot_func=plot_func,
            args=(case_idx,),
            kwargs={
                "df": df,
                "dataset_name": name,
                **plot_kwargs,
            },
        )

    def _step(delta: int) -> None:
        """Move the one-based case control without crossing dataset bounds."""
        case_index.value = max(
            1,
            min(case_index.max, case_index.value + delta),
        )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_index.observe(lambda _: _render(), names="value")

    if dataset_dropdown is not None:

        def _on_dataset_change(change: dict) -> None:
            """Rebind case bounds and rerender after a dataset selection change."""
            df_new = datasets[change["new"]]

            n_cases_new = n_cases_fn(change["new"], df_new) if n_cases_fn is not None else len(df_new)

            case_index.max = n_cases_new
            case_index.value = min(case_index.value, n_cases_new)
            _render()

        dataset_dropdown.observe(_on_dataset_change, names="value")

    _attach_widget_rerender(extra_widgets, _render)

    # Initial render
    _render()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    header_items: list[widgets.Widget] = [
        case_index,
        prev_btn,
        next_btn,
        *extra_widgets,
    ]

    if dataset_dropdown is not None:
        header_items.append(dataset_dropdown)

    header = widgets.HBox(header_items)

    return widgets.VBox([header, out])


# =============================================================================
# 2) CASECOUNT VIEWER (multi-case aggregations)
# =============================================================================


def make_casecount_viewer(
    plot_func: Callable[..., Any],
    *,
    datasets: dict[str, pd.DataFrame],
    start_cases: int = 100,
    step_size: int = 50,
    extra_widgets: list[widgets.Widget] | None = None,
    **plot_kwargs: Any,
) -> widgets.VBox:
    """
    Build and immediately render a shared-prefix aggregate viewer.

    Parameters
    ----------
    plot_func : Callable[..., Any]
        Called with ``datasets`` and integer ``max_cases`` keyword arguments.
    datasets : dict[str, pandas.DataFrame]
        Labelled frames; the shortest frame defines the shared maximum prefix.
    start_cases : int, optional
        Initial prefix count, capped by the shared maximum (default 100).
    step_size : int, optional
        Positive navigation increment passed to the slider (default 50).
    extra_widgets : list[ipywidgets.Widget] | None, optional
        Additional controls whose changes trigger rerendering.
    **plot_kwargs : Any
        Fixed keyword arguments forwarded on every render.

    Returns
    -------
    ipywidgets.VBox
        Prefix controls and output containing the initial aggregate render.

    Notes
    -----
    Navigation clamps button-driven changes to one through the shortest frame,
    while the underlying slider retains its configured zero minimum. Dataset
    selection semantics, caching, and scientific reduction belong to ``plot_func``.

    """
    max_cases_global = min(len(df) for df in datasets.values())

    case_count, prev_btn, next_btn = components.ui_step_case_count(
        start_cases=min(start_cases, max_cases_global),
        min_cases=0,
        max_cases=max_cases_global,
        step_size=step_size,
    )

    out = components.ui_output_plot()
    extra_widgets = extra_widgets or []

    # ------------------------------------------------------------------
    # Render logic
    # ------------------------------------------------------------------
    def _render() -> None:
        """
        Render the current shared ordered-prefix count across all frames.

        The viewer forwards state only; prefix interpretation and aggregation
        remain owned by the supplied plotting callable.
        """
        _render_figure(
            out=out,
            plot_func=plot_func,
            kwargs={
                "datasets": datasets,
                "max_cases": int(case_count.value),
                **plot_kwargs,
            },
        )

    def _step(delta: int) -> None:
        """Change prefix size by one configured step within the shared bound."""
        new_val = case_count.value + delta * step_size
        case_count.value = max(1, min(max_cases_global, new_val))

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------
    prev_btn.on_click(lambda _: _step(-1))
    next_btn.on_click(lambda _: _step(1))
    case_count.observe(lambda _: _render(), names="value")

    _attach_widget_rerender(extra_widgets, _render)

    # Initial render
    _render()

    header = widgets.HBox(
        [
            case_count,
            prev_btn,
            next_btn,
            *extra_widgets,
        ],
        layout=widgets.Layout(
            align_items="center",
        ),
    )

    return widgets.VBox([header, out])


# =============================================================================
# 3) DATASET CASE-SCOPE VIEWER (aggregate and synchronized single cases)
# =============================================================================


def make_dataset_case_scope_viewer(  # noqa: C901, PLR0915
    *,
    datasets: dict[str, pd.DataFrame],
    case_numbers_by_dataset: Mapping[str, Sequence[int]],
    single_plot_func: Callable[..., Any],
    aggregate_plot_func: Callable[..., Any] | None = None,
    start_cases: int = 100,
    step_size: int = 50,
) -> widgets.VBox:
    """Render selected datasets in aggregate or synchronized case-number scope."""
    if not datasets or any(frame.empty for frame in datasets.values()):
        msg = "Dataset case viewers require non-empty labelled frames."
        raise ValueError(msg)
    if tuple(case_numbers_by_dataset) != tuple(datasets):
        msg = "Case-number metadata must match dataset labels and order."
        raise ValueError(msg)
    normalized_numbers = {name: tuple(case_numbers_by_dataset[name]) for name in datasets}
    if any(not numbers for numbers in normalized_numbers.values()):
        msg = "Every dataset must expose at least one navigable case number."
        raise ValueError(msg)

    selector = cast("components.CheckboxGroup", components.ui_checkbox_datasets(dataset_names=list(datasets)))
    selected_names = list(datasets)

    def _shared_numbers(names: Sequence[str]) -> tuple[int, ...]:
        """Preserve the first selected manifest order while intersecting IDs."""
        if not names:
            return ()
        shared = set(normalized_numbers[names[0]])
        for name in names[1:]:
            shared.intersection_update(normalized_numbers[name])
        return tuple(number for number in normalized_numbers[names[0]] if number in shared)

    shared_numbers = _shared_numbers(selected_names)
    if not shared_numbers:
        msg = "Initially selected datasets have no shared case numbers."
        raise ValueError(msg)
    case_number, previous_case, next_case = components.ui_step_case_index(case_numbers=shared_numbers)
    max_cases = min(len(datasets[name]) for name in selected_names)
    case_count, fewer_cases, more_cases = components.ui_step_case_count(
        start_cases=min(start_cases, max_cases),
        min_cases=1,
        max_cases=max_cases,
        step_size=step_size,
    )
    scope = None
    if aggregate_plot_func is not None:
        scope = widgets.ToggleButtons(
            options=(("Aggregate", "aggregate"), ("Single case", "single")),
            value="aggregate",
        )
    controls = widgets.HBox(layout=widgets.Layout(align_items="center"))
    output = components.ui_output_plot()
    state = {"updating": False}
    case_state = {"options": shared_numbers, "last_valid": case_number.value}

    def _active_names() -> list[str]:
        """Return checked dataset labels in stable input order."""
        return [name for name, checkbox in selector.boxes.items() if checkbox.value]

    def _show_message(message: str) -> None:
        """Replace the current result with one actionable selection message."""
        with output:
            output.clear_output(wait=True)
            print(message)

    def _single_scope() -> bool:
        """Return whether the current viewer renders individual cases."""
        return scope is None or scope.value == "single"

    def _sync_case_buttons() -> None:
        """Disable case arrows exactly at the current sparse-option bounds."""
        options = case_state["options"]
        if not options or case_number.value not in options:
            previous_case.disabled = True
            next_case.disabled = True
            return
        position = options.index(case_number.value)
        previous_case.disabled = position == 0
        next_case.disabled = position == len(options) - 1

    def _update_controls() -> None:
        """Expose only controls relevant to the active scientific scope."""
        prefix: list[widgets.Widget] = [] if scope is None else [scope]
        if _single_scope():
            controls.children = (*prefix, case_number, previous_case, next_case, selector)
        else:
            controls.children = (*prefix, case_count, fewer_cases, more_cases, selector)

    def _render() -> None:
        """Render once from the current validated selection state."""
        if state["updating"]:
            return
        names = _active_names()
        if not names:
            _show_message("Select at least one dataset.")
            return
        selected = {name: datasets[name] for name in names}
        if _single_scope():
            shared = _shared_numbers(names)
            if not shared:
                _show_message("Selected datasets have no shared case numbers. Choose a different dataset combination.")
                return
            if case_number.value not in shared:
                _show_message("Enter a case number shared by every selected dataset, or use the arrows.")
                return
            _render_figure(
                out=output,
                plot_func=single_plot_func,
                kwargs={"datasets": selected, "case_number": int(case_number.value)},
            )
            return
        if aggregate_plot_func is None:
            msg = "Aggregate rendering is unavailable for this view."
            raise RuntimeError(msg)
        _render_figure(
            out=output,
            plot_func=aggregate_plot_func,
            kwargs={"datasets": selected, "max_cases": int(case_count.value)},
        )

    def _rebind_dataset_state(_: object = None) -> None:
        """Recompute valid common IDs and aggregate bounds, then render once."""
        names = _active_names()
        if not names:
            case_state["options"] = ()
            previous_case.disabled = True
            next_case.disabled = True
            _render()
            return
        state["updating"] = True
        try:
            shared = _shared_numbers(names)
            case_state["options"] = shared
            old_number = case_number.value
            if shared:
                selected_number = old_number if old_number in shared else shared[0]
                case_state["last_valid"] = selected_number
                case_number.value = selected_number
            _sync_case_buttons()
            maximum = min(len(datasets[name]) for name in names)
            case_count.max = maximum
            case_count.value = min(case_count.value, maximum)
        finally:
            state["updating"] = False
        _render()

    def _step_case(delta: int) -> None:
        """Move through the current sparse shared-number options."""
        options = case_state["options"]
        if not options or case_number.value not in options:
            return
        current = options.index(case_number.value)
        case_number.value = options[max(0, min(len(options) - 1, current + delta))]

    def _step_count(delta: int) -> None:
        """Move the aggregate prefix count within the selected-frame bound."""
        case_count.value = max(case_count.min, min(case_count.max, case_count.value + delta * step_size))

    def _on_case_change(_: object = None) -> None:
        """Validate a typed case number, then update navigation and render once."""
        if state["updating"]:
            return
        requested_number = case_number.value
        if requested_number not in case_state["options"]:
            state["updating"] = True
            try:
                case_number.value = case_state["last_valid"]
            finally:
                state["updating"] = False
            _sync_case_buttons()
            _show_message(f"Case {requested_number} is unavailable for the selected datasets. Enter a shared case number or use the arrows.")
            return
        case_state["last_valid"] = requested_number
        _sync_case_buttons()
        _render()

    previous_case.on_click(lambda _: _step_case(-1))
    next_case.on_click(lambda _: _step_case(1))
    fewer_cases.on_click(lambda _: _step_count(-1))
    more_cases.on_click(lambda _: _step_count(1))
    case_number.observe(_on_case_change, names="value")
    case_count.observe(lambda _: _render(), names="value")
    for checkbox in selector.boxes.values():
        checkbox.observe(_rebind_dataset_state, names="value")
    if scope is not None:

        def _on_scope_change(_: object = None) -> None:
            """Update the active control surface and render the new scope once."""
            _update_controls()
            _render()

        scope.observe(_on_scope_change, names="value")

    _sync_case_buttons()
    _update_controls()
    _render()
    return widgets.VBox([controls, output])
