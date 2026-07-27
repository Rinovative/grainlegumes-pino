"""
===============================================================================
analysis_curated_renderer.py
===============================================================================
Render the fixed scientific media inventory accepted by optional W&B tracking.

Responsibilities:
  - Build four figures and one neutral run-summary table from validated frames
  - Save rendered files below an explicit caller-owned output directory
  - Return an exact five-key bundle without importing or calling W&B

Design principles:
  - Input frames already satisfy current artifact/provenance compatibility
  - Output media remains separate from immutable ID/OOD artifact cache directories
  - Rendering never mutates artifact caches or run state
  - The inventory is identical to the tracking upload allowlist

This module does NOT:
  - Import, initialize, or call W&B
  - Admit artifact provenance or decide whether frames are comparable
  - Own the scientific mathematics implemented by evaluation plot modules
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation.evaluation_plot import (
    evaluation_plot_physical_consistency as physical_consistency,
)
from src.analysis.evaluation.evaluation_plot import evaluation_plot_run_summary as run_summary
from src.analysis.evaluation.evaluation_plot import evaluation_plot_spectral_fidelity as spectral_fidelity

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd
    from matplotlib.figure import Figure

CURATED_ANALYSIS_KEYS = frozenset(
    {
        "run_summary_table",
        "accuracy_physics_pareto",
        "dual_continuity_diagnostics",
        "pressure_boundary_summary",
        "spectral_fidelity",
    }
)


@dataclass(frozen=True, slots=True)
class CuratedAnalysisBundle:
    """
    Hold the exact local scientific bundle accepted by W&B tracking.

    Parameters
    ----------
    media_files : Mapping[str, pathlib.Path]
        Four explicitly rendered files outside immutable artifact caches.
    tables : Mapping[str, Any]
        Neutral ``run_summary_table`` columns/data payload.

    Raises
    ------
    ValueError
        If keys are missing, additional, or shared by both mappings.

    Notes
    -----
    The dataclass is frozen and slotted, but the supplied mappings are not copied
    or deeply frozen. Callers retain ownership of rendered files and mappings.

    """

    media_files: Mapping[str, Path]
    tables: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the exact fixed five-key inventory."""
        names = set(self.media_files).union(self.tables)
        if names != CURATED_ANALYSIS_KEYS or set(self.media_files).intersection(self.tables):
            msg = f"Curated analysis bundle must contain exactly {sorted(CURATED_ANALYSIS_KEYS)}."
            raise ValueError(msg)


def _neutral_table(frame: pd.DataFrame) -> dict[str, object]:
    """
    Convert one summary frame to a path-free, W&B-neutral table payload.

    The index is materialized as columns; NumPy scalars become Python scalars,
    supported primitives remain unchanged, and other values use readable text.
    No W&B type is imported or constructed.
    """
    table = frame.reset_index()
    columns = [str(column) for column in table.columns]
    data: list[list[object]] = []
    for row in table.itertuples(index=False, name=None):
        values: list[object] = []
        for value in row:
            scalar = value.item() if isinstance(value, np.generic) else value
            if scalar is None or isinstance(scalar, (str, bool, int, float)):
                values.append(scalar)
            else:
                values.append(str(scalar))
        data.append(values)
    return {"columns": columns, "data": data}


def _save_figure(figure: Figure, path: Path) -> None:
    """
    Save or replace one figure path, then close the Matplotlib figure.

    A non-figure-like object fails before writing. Matplotlib owns format and
    overwrite behavior for an existing caller-owned target.
    """
    savefig = getattr(figure, "savefig", None)
    if not callable(savefig):
        msg = f"Curated renderer expected a Matplotlib figure for {path.name}."
        raise TypeError(msg)
    savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_curated_analysis(
    *,
    datasets: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> CuratedAnalysisBundle:
    """
    Render the exact curated inventory without importing W&B or mutating caches.

    Parameters
    ----------
    datasets : Mapping[str, pandas.DataFrame]
        Provenance-compatible ID or OOD evaluation frames with physics evidence.
    output_dir : str | pathlib.Path
        Caller-owned render directory outside every immutable artifact root. It
        is created when absent; existing fixed-name PNG targets are overwritten.

    Returns
    -------
    CuratedAnalysisBundle
        Four PNG paths and one path-neutral summary-table payload under the exact
        five-key tracking allowlist.

    Raises
    ------
    ComparisonCompatibilityError
        If frames are scientifically incompatible or lack required physics data.
    ValueError
        If the output directory is inside an immutable artifact cache.
    TypeError
        If a renderer does not return a Matplotlib figure-like object.

    Notes
    -----
    Rendering owns only the four output figures. The caller owns directory
    lifecycle and cleanup; evaluation readers and plot modules own admission and
    scientific calculations.

    """
    dataframe.validate_comparison(datasets, require_physics=True)
    target = Path(output_dir).resolve()
    artifact_roots = {
        Path(str(frame.attrs["artifact_root"])).resolve() for frame in datasets.values() if frame.attrs.get("artifact_root") is not None
    }
    if any(target == root or target.is_relative_to(root) for root in artifact_roots):
        msg = "Curated analysis output must be outside every immutable artifact cache."
        raise ValueError(msg)
    target.mkdir(parents=True, exist_ok=True)

    media_files = {
        "accuracy_physics_pareto": target / "accuracy_physics_pareto.png",
        "dual_continuity_diagnostics": target / "dual_continuity_diagnostics.png",
        "pressure_boundary_summary": target / "pressure_boundary_summary.png",
        "spectral_fidelity": target / "spectral_fidelity.png",
    }
    renderers = {
        "accuracy_physics_pareto": run_summary.plot_accuracy_physics_pareto,
        "dual_continuity_diagnostics": physical_consistency.plot_spatial_residuals,
        "pressure_boundary_summary": physical_consistency.plot_pressure_boundary_summary,
        "spectral_fidelity": spectral_fidelity.plot_spectral_fidelity,
    }
    for name, renderer in renderers.items():
        _save_figure(renderer(datasets=datasets), media_files[name])

    summary = run_summary.build_run_summary_table(datasets)
    return CuratedAnalysisBundle(
        media_files=media_files,
        tables={"run_summary_table": _neutral_table(summary)},
    )
