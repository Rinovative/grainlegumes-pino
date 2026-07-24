"""
===============================================================================
evaluation_plot_spectral_analysis.py
===============================================================================
Plot spectral diagnostics for evaluated model predictions.

Responsibilities:
  - Compare prediction and target power spectra
  - Plot spectral transfer ratios and error spectra

Design principles:
  - Spectral analysis uses Hann-windowed 2D FFTs
  - Wavenumber plots expose scale-dependent model behavior
  - Dataset comparisons are shown side by side

Boundaries:
  - Dataset-only spectral EDA belongs to eda_plot_spectral_analysis
  - Spatial error decomposition belongs to evaluation_plot_error_decomposition
===============================================================================

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import analysis, domain

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = list(domain.fields.ANALYSIS_FIELDS)
CHANNEL_INDICES = {name: i for i, name in enumerate(CHANNELS)}

SPECTRAL_NUMERICAL_FLOOR = 1e-12
WhichSignal = Literal["gt", "pred", "err"]


# =============================================================================
# TYPES
# =============================================================================


class _SpectraAcc(TypedDict):
    """
    Accumulator for per-channel spectra (GT / Pred / Err) across cases.

    Attributes
    ----------
    loaded_until : int
        Number of cases loaded so far.
    shape : tuple[int, int] | None
        Spatial shape (H, W) of loaded fields, or None if not yet known.
    k : numpy.ndarray | None
        Radial frequency bins, or None if not yet known.
    sum_gt : dict[str, numpy.ndarray]
        Summed ground-truth spectra per channel.

    """

    loaded_until: int
    shape: tuple[int, int] | None
    k: np.ndarray | None
    sum_gt: dict[str, np.ndarray]
    sum_pred: dict[str, np.ndarray]
    sum_err: dict[str, np.ndarray]
    count: dict[str, int]


# =============================================================================
# HELPERS: field contracts, geometry, and NPZ loading
# =============================================================================


def _has_npz_path_col(df: pd.DataFrame) -> bool:
    """
    Check if a DataFrame contains an NPZ path column.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation DataFrame.

    Returns
    -------
    bool
        True if 'npz_path' exists in df.columns.

    """
    return "npz_path" in df.columns


def _to_pos_float_from_df_col(df: pd.DataFrame, col: str, default: float) -> float:
    """
    Read a positive float from a DataFrame column or use the supplied default.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation DataFrame.
    col : str
        Column name.
    default : float
        Default value.

    Returns
    -------
    float
        Positive float value.

    """
    if col not in df.columns or len(df) == 0:
        return default

    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    v = float(arr[0]) if arr.size else np.nan

    if not np.isfinite(v) or v <= 0:
        return default
    return v


def _get_dx_dy_from_df(df: pd.DataFrame) -> tuple[float, float]:
    """
    Extract dx, dy from evaluation DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Evaluation DataFrame (should contain geometry_dx / geometry_dy).

    Returns
    -------
    tuple[float, float]
        ``(dx, dy)`` with explicit defaults when geometry is absent.

    """
    dx = _to_pos_float_from_df_col(df, "geometry_dx", 1.0)
    dy = _to_pos_float_from_df_col(df, "geometry_dy", 1.0)
    return dx, dy


def _safe_load_npz(path: str | Path) -> dict[str, Any] | None:
    """
    Load an NPZ file into a dict, or return None if missing.

    Parameters
    ----------
    path : str | pathlib.Path
        NPZ file path.

    Returns
    -------
    dict[str, Any] | None
        Loaded NPZ content (arrays), or None if file does not exist.

    """
    p = Path(path)
    if not p.is_file():
        return None

    with np.load(str(p), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# =============================================================================
# HELPERS: colors, UI selections, misc
# =============================================================================


def _channel_color_map() -> dict[str, str]:
    """
    Construct a stable matplotlib color mapping per channel.

    Returns
    -------
    dict[str, str]
        Mapping channel_name -> matplotlib color string.

    """
    colors = plt.rcParams.get("axes.prop_cycle", None)
    seq = colors.by_key().get("color", []) if colors is not None else []

    if not seq:
        seq = [f"C{i}" for i in range(10)]

    return {ch: seq[i % len(seq)] for i, ch in enumerate(CHANNELS)}


def _active_channels_from_selector(channel_selector: Any) -> list[str]:
    """
    Get active channels from analysis.ui.components checkbox widget.

    Parameters
    ----------
    channel_selector : Any
        Checkbox group widget (expects .boxes dict[str, checkbox]).

    Returns
    -------
    list[str]
        List of enabled channels.

    """
    return [name for name, cb in channel_selector.boxes.items() if cb.value]


def _message_figure(msg: str) -> Figure:
    """
    Render a simple message as a Matplotlib figure.

    Parameters
    ----------
    msg : str
        Message to show.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with text only.

    """
    fig = plt.figure(figsize=(7.0, 2.2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.55, msg, fontsize=12, va="center", ha="left")
    fig.tight_layout()
    return fig


# =============================================================================
# HELPERS: spectra (field-based, physical k-axis)
# =============================================================================


def _radial_spectrum_fft2(
    field: np.ndarray,
    *,
    dx: float,
    dy: float,
    nbins: int | None = None,
    remove_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial (azimuthally averaged) power spectrum from a 2D field.

    Parameters
    ----------
    field : numpy.ndarray
        2D array [H, W].
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    nbins : int | None
        Number of radial bins. If None, uses max(8, min(H, W) // 2).
    remove_mean : bool
        If True, subtract spatial mean before FFT.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (k_centers, spectrum) where spectrum is power averaged per radial bin.

    """
    if field.ndim != 2:  # noqa: PLR2004
        return np.array([0.0]), np.array([0.0])

    f = np.asarray(field, dtype=float)

    if remove_mean:
        f = f - float(np.nanmean(f))

    ny, nx = f.shape

    if nbins is None:
        nbins = max(8, min(ny, nx) // 2)

    fhat = np.fft.fft2(f)
    power2d = np.abs(fhat) ** 2

    ky = np.fft.fftfreq(ny, d=dy)
    kx = np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX**2 + KY**2)

    kr_flat = kr.ravel()
    p_flat = power2d.ravel()

    k_max = float(np.max(kr_flat))
    if not np.isfinite(k_max) or k_max <= 0:
        return np.array([0.0]), np.array([0.0])

    bins = np.linspace(0.0, k_max, int(nbins) + 1)
    idx = np.digitize(kr_flat, bins) - 1

    valid = (idx >= 0) & (idx < nbins)
    idx_v = idx[valid]
    p_v = p_flat[valid]

    sum_p = np.bincount(idx_v, weights=p_v, minlength=nbins).astype(float)
    cnt = np.bincount(idx_v, minlength=nbins).astype(float)

    spec = np.zeros(nbins, dtype=float)
    m = cnt > 0
    spec[m] = sum_p[m] / cnt[m]

    k_centers = 0.5 * (bins[:-1] + bins[1:])
    return k_centers.astype(float), spec.astype(float)


def _normalise_spectrum(spec: np.ndarray) -> np.ndarray:
    """
    Normalise a spectrum to unit sum.

    Parameters
    ----------
    spec : numpy.ndarray
        Spectrum values.

    Returns
    -------
    numpy.ndarray
        Normalised spectrum, or unchanged if sum is invalid.

    """
    s = float(np.nansum(spec))
    if not np.isfinite(s) or s <= 0:
        return spec
    return spec / (s + SPECTRAL_NUMERICAL_FLOOR)


# =============================================================================
# HELPERS: signal extraction (gt / pred / err) with [1,C,H,W] support
# =============================================================================


def _squeeze_case(arr: np.ndarray) -> np.ndarray:
    """
    Squeeze a single-case batch dimension if present.

    Parameters
    ----------
    arr : numpy.ndarray
        Array, possibly [1, C, H, W] or [C, H, W] or [H, W].

    Returns
    -------
    numpy.ndarray
        Squeezed array with batch removed if it was a single element.

    """
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:  # noqa: PLR2004
        return np.asarray(a[0])
    return a


def _channel_field(arr: np.ndarray, ch: str) -> np.ndarray | None:
    """
    Extract a specific channel as a 2D field.

    Parameters
    ----------
    arr : numpy.ndarray
        Array [C, H, W] or [H, W].
    ch : str
        Channel name.

    Returns
    -------
    numpy.ndarray | None
        2D field [H, W], or None if not compatible.

    """
    a = _squeeze_case(arr)

    if a.ndim == 2:  # noqa: PLR2004
        return np.asarray(a, dtype=float)

    if a.ndim != 3:  # noqa: PLR2004
        return None

    idx = CHANNEL_INDICES.get(ch)
    if idx is None or idx >= a.shape[0]:
        return None
    return np.asarray(a[idx], dtype=float)


def _extract_field(npz: dict[str, Any], *, ch: str, which: WhichSignal) -> np.ndarray | None:
    """
    Extract a 2D field for a given channel and signal type.

    Parameters
    ----------
    npz : dict[str, Any]
        NPZ content dictionary, expected to contain 'gt' and 'pred' (and optionally 'err').
    ch : str
        Channel name.
    which : {"gt", "pred", "err"}
        Signal type to extract.

    Returns
    -------
    numpy.ndarray | None
        2D field [H, W], or None if missing / incompatible.

    """
    gt_raw = npz.get("gt")
    pr_raw = npz.get("pred")
    er_raw = npz.get("err")

    if gt_raw is None or pr_raw is None:
        return None

    gt = _squeeze_case(np.asarray(gt_raw))
    pr = _squeeze_case(np.asarray(pr_raw))

    if which == "gt":
        base = gt
    elif which == "pred":
        base = pr
    elif er_raw is None:
        base = pr - gt
    else:
        base = _squeeze_case(np.asarray(er_raw))

    return _channel_field(base, ch)


# =============================================================================
# Streaming spectra cache lifecycle
# =============================================================================


def _init_spectra_cache(names: list[str]) -> dict[str, _SpectraAcc]:
    """
    Create a spectra cache per dataset.

    Parameters
    ----------
    names : list[str]
        Dataset names.

    Returns
    -------
    dict[str, _SpectraAcc]
        Per-dataset accumulators.

    """
    return {
        name: {
            "loaded_until": 0,
            "shape": None,
            "k": None,
            "sum_gt": {},
            "sum_pred": {},
            "sum_err": {},
            "count": {},
        }
        for name in names
    }


def _reset_spectra_cache(cache: dict[str, _SpectraAcc]) -> None:
    """
    Reset spectra cache in-place.

    Parameters
    ----------
    cache : dict[str, _SpectraAcc]
        Cache to reset.

    Returns
    -------
    None

    """
    for entry in cache.values():
        entry["loaded_until"] = 0
        entry["shape"] = None
        entry["k"] = None
        entry["sum_gt"] = {}
        entry["sum_pred"] = {}
        entry["sum_err"] = {}
        entry["count"] = {}


def _accumulate_spectra(
    entry: _SpectraAcc,
    df: pd.DataFrame,
    *,
    max_cases: int,
    normalise: bool,
) -> None:
    """
    Accumulate mean spectra (GT, Pred, Err) across up to max_cases.

    Parameters
    ----------
    entry : _SpectraAcc
        Dataset accumulator.
    df : pandas.DataFrame
        Evaluation DataFrame with 'npz_path'.
    max_cases : int
        Number of cases to include.
    normalise : bool
        If True, each case spectrum is normalised to unit sum before accumulation.

    Returns
    -------
    None

    """
    if not _has_npz_path_col(df):
        entry["loaded_until"] = int(min(max_cases, len(df)))
        return

    max_cases_i = int(min(max_cases, len(df)))
    loaded = int(entry["loaded_until"])
    if max_cases_i <= loaded:
        return

    dx, dy = _get_dx_dy_from_df(df)
    df_i = df.reset_index(drop=True)

    for path in df_i["npz_path"].iloc[loaded:max_cases_i]:
        npz = _safe_load_npz(path)
        if npz is None:
            continue

        for ch in CHANNELS:
            gt_field = _extract_field(npz, ch=ch, which="gt")
            pr_field = _extract_field(npz, ch=ch, which="pred")
            er_field = _extract_field(npz, ch=ch, which="err")

            if gt_field is None or pr_field is None or er_field is None:
                continue

            if entry["shape"] is None:
                entry["shape"] = (int(gt_field.shape[0]), int(gt_field.shape[1]))

            if entry["shape"] is None:
                continue

            if gt_field.shape != entry["shape"] or pr_field.shape != entry["shape"] or er_field.shape != entry["shape"]:
                continue

            nbins = max(8, min(entry["shape"][0], entry["shape"][1]) // 2)

            k, s_gt = _radial_spectrum_fft2(gt_field, dx=dx, dy=dy, nbins=nbins)
            _, s_pr = _radial_spectrum_fft2(pr_field, dx=dx, dy=dy, nbins=nbins)
            _, s_er = _radial_spectrum_fft2(er_field, dx=dx, dy=dy, nbins=nbins)

            if entry["k"] is None:
                entry["k"] = k

            if entry["k"] is None or len(k) != len(entry["k"]):
                continue

            if normalise:
                s_gt = _normalise_spectrum(s_gt)
                s_pr = _normalise_spectrum(s_pr)
                s_er = _normalise_spectrum(s_er)

            if ch not in entry["sum_gt"]:
                entry["sum_gt"][ch] = np.zeros_like(s_gt, dtype=float)
                entry["sum_pred"][ch] = np.zeros_like(s_pr, dtype=float)
                entry["sum_err"][ch] = np.zeros_like(s_er, dtype=float)
                entry["count"][ch] = 0

            entry["sum_gt"][ch] += s_gt
            entry["sum_pred"][ch] += s_pr
            entry["sum_err"][ch] += s_er
            entry["count"][ch] += 1

    entry["loaded_until"] = max_cases_i


def _mean(sum_arr: np.ndarray | None, count: int) -> np.ndarray | None:
    """
    Compute a mean given a sum array and a count.

    Parameters
    ----------
    sum_arr : numpy.ndarray | None
        Summed array.
    count : int
        Number of samples.

    Returns
    -------
    numpy.ndarray | None
        Mean array, or None if invalid.

    """
    if sum_arr is None or count <= 0:
        return None
    return sum_arr / float(count)


# =============================================================================
# Demand vs prediction and error (channels overlayed, datasets side by side)
# =============================================================================


def plot_spectral_demand_prediction_error(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive demand vs prediction and error spectra.

    One column per dataset. Two rows:
    - top: GT demand and prediction capacity (overlayed channels)
    - bottom: error spectrum (overlayed channels)

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name -> evaluation DataFrame.
        Each DataFrame should contain:
            - 'npz_path' : str (path to .npz with 'gt' and 'pred', optionally 'err')
            - 'geometry_dx', 'geometry_dy' (optional, for k-axis scaling)

    Returns
    -------
    ipywidgets.VBox
        Interactive viewer widget with a case-count slider and channel toggles.

    """
    names = list(datasets.keys())
    if not names:
        return widgets.VBox([widgets.HTML("<b>No datasets provided.</b>")])

    cache = _init_spectra_cache(names)
    state: dict[str, Any] = {"normalise": True}

    norm_cb = analysis.ui.components.ui_checkbox_normalise(
        description="Normalise",
        default=True,
        width="160px",
    )
    channel_selector = analysis.ui.components.ui_checkbox_channels(default_on=CHANNELS)
    controls = widgets.HBox([norm_cb])

    ch_colors = _channel_color_map()

    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        norm_cb: widgets.Checkbox,
        channel_selector: Any,
    ) -> Figure:
        """
        Plot demand vs prediction and error spectra.

        Parameters
        ----------
        max_cases : int
            Number of cases to include from each dataset.
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name -> evaluation DataFrame.
        norm_cb : ipywidgets.Checkbox
            Normalise toggle.
        channel_selector : Any
            Channel selection checkbox widget.

        Returns
        -------
        matplotlib.figure.Figure
            Demand vs prediction (top) and error spectra (bottom), per dataset.

        """
        normalise = bool(norm_cb.value)
        active_channels = _active_channels_from_selector(channel_selector)
        if not active_channels:
            return _message_figure("Select at least one channel.")

        if normalise != state["normalise"]:
            _reset_spectra_cache(cache)
            state["normalise"] = normalise

        for name, df in datasets.items():
            _accumulate_spectra(cache[name], df, max_cases=max_cases, normalise=normalise)

        fig = plt.figure(figsize=(6.0 * len(names) + 2.8, 7.0))
        gs = fig.add_gridspec(
            2,
            len(names) + 1,
            width_ratios=[1.0] * len(names) + [0.35],
            wspace=0.25,
            hspace=0.25,
        )

        axes_top = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
        axes_bot = [fig.add_subplot(gs[1, i]) for i in range(len(names))]
        ax_leg = fig.add_subplot(gs[:, -1])
        ax_leg.axis("off")

        for i, name in enumerate(names):
            ax_top = axes_top[i]
            ax_bot = axes_bot[i]

            entry = cache[name]
            k = entry["k"]

            if k is None:
                ax_top.set_title(name)
                ax_top.axis("off")
                ax_bot.axis("off")
                continue

            any_plotted = False

            for ch in active_channels:
                cnt = int(entry["count"].get(ch, 0))
                if cnt <= 0:
                    continue

                gt = _mean(entry["sum_gt"].get(ch), cnt)
                pr = _mean(entry["sum_pred"].get(ch), cnt)
                er = _mean(entry["sum_err"].get(ch), cnt)

                if gt is None or pr is None or er is None:
                    continue

                col = ch_colors.get(ch, "C0")

                ax_top.plot(k, gt, lw=2.0, ls="--", color=col, alpha=0.95)
                ax_top.plot(k, pr, lw=2.2, ls="-", color=col, alpha=0.95)
                ax_bot.plot(k, er, lw=2.2, ls="-", color=col, alpha=0.95)

                any_plotted = True

            if not any_plotted:
                ax_top.set_title(name)
                ax_top.axis("off")
                ax_bot.axis("off")
                continue

            for ax in (ax_top, ax_bot):
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both", linestyle="--", alpha=0.3)

            ax_top.set_title(name)

            if i == 0:
                ax_top.set_ylabel("Spectral power" + (" (normalised)" if normalise else ""))
                ax_bot.set_ylabel("Error spectral power" + (" (normalised)" if normalise else ""))

            ax_bot.set_xlabel("Spatial frequency k [1/length]")

        curve_handles = [
            Line2D([0], [0], color="black", lw=2.2, ls="--", label="GT demand"),
            Line2D([0], [0], color="black", lw=2.2, ls="-", label="Prediction"),
            Line2D([0], [0], color="black", lw=2.2, ls="-", label="Error"),
        ]
        channel_handles: list[Line2D] = [Line2D([0], [0], color=ch_colors.get(ch, "C0"), lw=2.6, label=ch) for ch in active_channels]
        curve_labels: list[str] = [str(h.get_label()) for h in curve_handles]
        channel_labels: list[str] = [str(h.get_label()) for h in channel_handles]

        leg1 = ax_leg.legend(
            curve_handles,
            curve_labels,
            title="Curves",
            loc="upper left",
        )
        ax_leg.add_artist(leg1)

        ax_leg.legend(
            channel_handles,
            channel_labels,
            title="Channels",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.80),
        )

        fig.suptitle("Spectral demand vs prediction and error", y=0.97)
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=50,
        step_size=50,
        extra_widgets=[controls, channel_selector],
        norm_cb=norm_cb,
        channel_selector=channel_selector,
    )


# =============================================================================
# Transfer ratio R(k) = Pred / GT (channels overlayed, datasets side by side)
# =============================================================================


def plot_spectral_transfer_ratio(*, datasets: dict[str, pd.DataFrame]) -> widgets.VBox:
    """
    Interactive transfer ratio plot R(k) = Pred / GT.

    One subplot per dataset (side by side).
    All selected channels are overlayed in the same axis.
    Legend is in a dedicated right column.

    Parameters
    ----------
    datasets : dict[str, pandas.DataFrame]
        Mapping dataset_name -> evaluation DataFrame.

    Returns
    -------
    ipywidgets.VBox
        Interactive viewer widget with a case-count slider and channel toggles.

    """
    names = list(datasets.keys())
    if not names:
        return widgets.VBox([widgets.HTML("<b>No datasets provided.</b>")])

    cache = _init_spectra_cache(names)
    channel_selector = analysis.ui.components.ui_checkbox_channels(default_on=CHANNELS)
    ch_colors = _channel_color_map()

    def _plot(
        max_cases: int,
        *,
        datasets: dict[str, pd.DataFrame],
        channel_selector: Any,
    ) -> Figure:
        """
        Plot spectral transfer ratio.

        Parameters
        ----------
        max_cases : int
            Number of cases to include from each dataset.
        datasets : dict[str, pandas.DataFrame]
            Mapping dataset_name -> evaluation DataFrame.
        channel_selector : Any
            Channel selection checkbox widget.

        Returns
        -------
        matplotlib.figure.Figure
            Overlayed R(k) curves per dataset.

        """
        active_channels = _active_channels_from_selector(channel_selector)
        if not active_channels:
            return _message_figure("Select at least one channel.")

        for name, df in datasets.items():
            _accumulate_spectra(cache[name], df, max_cases=max_cases, normalise=False)

        fig = plt.figure(figsize=(6.0 * len(names) + 2.8, 4.6))
        gs = fig.add_gridspec(
            1,
            len(names) + 1,
            width_ratios=[1.0] * len(names) + [0.35],
            wspace=0.25,
        )

        axes = [fig.add_subplot(gs[0, i]) for i in range(len(names))]
        ax_leg = fig.add_subplot(gs[0, -1])
        ax_leg.axis("off")

        for i, name in enumerate(names):
            ax = axes[i]
            entry = cache[name]
            k = entry["k"]

            if k is None:
                ax.set_title(name)
                ax.axis("off")
                continue

            any_plotted = False

            for ch in active_channels:
                cnt = int(entry["count"].get(ch, 0))
                if cnt <= 0:
                    continue

                gt = _mean(entry["sum_gt"].get(ch), cnt)
                pr = _mean(entry["sum_pred"].get(ch), cnt)

                if gt is None or pr is None:
                    continue

                ratio = pr / (gt + SPECTRAL_NUMERICAL_FLOOR)
                col = ch_colors.get(ch, "C0")
                ax.plot(k, ratio, lw=2.2, color=col, alpha=0.95)

                any_plotted = True

            if not any_plotted:
                ax.set_title(name)
                ax.axis("off")
                continue

            ax.axhline(1.0, lw=1.6, ls="--", color="black", alpha=0.6)
            ax.set_xscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.set_xlabel("Spatial frequency k [1/length]")
            if i == 0:
                ax.set_ylabel("Transfer ratio R(k)")
            ax.set_title(name)

        channel_handles: list[Line2D] = [Line2D([0], [0], color=ch_colors.get(ch, "C0"), lw=2.6, label=ch) for ch in active_channels]
        style_handles: list[Line2D] = [
            Line2D([0], [0], color="black", lw=1.6, ls="--", label="R(k)=1"),
        ]

        style_labels: list[str] = [str(h.get_label()) for h in style_handles]
        channel_labels: list[str] = [str(h.get_label()) for h in channel_handles]

        leg1 = ax_leg.legend(
            style_handles,
            style_labels,
            title="Reference",
            loc="upper left",
        )
        ax_leg.add_artist(leg1)

        ax_leg.legend(
            channel_handles,
            channel_labels,
            title="Channels",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.65),
        )

        fig.suptitle("Spectral transfer ratio (Pred / GT)", y=0.98)
        fig.subplots_adjust(top=0.86, bottom=0.18, left=0.06, right=0.98)
        return fig

    return analysis.ui.viewers.make_casecount_viewer(
        plot_func=_plot,
        datasets=datasets,
        start_cases=50,
        step_size=50,
        extra_widgets=[channel_selector],
        channel_selector=channel_selector,
    )
