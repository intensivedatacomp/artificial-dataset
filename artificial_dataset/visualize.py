"""Visualization utilities for SyntheticSeries instances.

Provides a single entry point, `plot_series`, for plotting a generated
1D time series alongside its injected anomaly points/spans. Matplotlib is
an optional, plot-only dependency: it is imported lazily so importing
`artificial_dataset` never requires it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

import torch

from artificial_dataset.series import (
    SyntheticSeries,
    SyntheticSeriesSplits,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import matplotlib.pyplot as plt


def _anomaly_spans(is_anomaly: torch.Tensor) -> list[tuple[int, int]]:
    """Return (start, end_inclusive) index pairs for contiguous anomaly runs."""
    idx = torch.where(is_anomaly)[0]
    if idx.numel() == 0:
        return []

    spans = []
    start = prev = int(idx[0])
    for i in idx[1:].tolist():
        if i == prev + 1:
            prev = i
            continue
        spans.append((start, prev))
        start = prev = i
    spans.append((start, prev))
    return spans


def plot_series(
    series: SyntheticSeries,
    title: str | None = None,
    ax: Axes | None = None,
) -> Figure:
    """
    Plot a SyntheticSeries: the 1D signal against time (a.u.), with anomalies marked.

    Single anomalous points are drawn as scatter markers; contiguous
    anomalous runs (e.g. a level shift or dropout span) are additionally
    shaded to make their extent visible.

    Builds and returns the figure without displaying it. In a notebook,
    the figure is shown automatically (either as the cell's returned value,
    or, with `%matplotlib inline`, because it is still open at the end of
    cell execution); in a script, call `plt.show()` on the result if you
    want to display it.

    Parameters
    ----------
    series : SyntheticSeries
        The series to visualize (e.g. as returned by `make_series` or after
        one or more `add_*` injectors have been applied).
    title : str, optional
        Plot title. Defaults to the `function_type` recorded in
        `series.meta`, if present.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure/axes is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    x = series.x.detach().cpu().numpy()
    y = series.y.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()
    fig = cast("Figure", fig)

    ax.plot(x, y, color="tab:blue", linewidth=1.0, label="signal", zorder=1)

    anomaly_idx = torch.where(series.is_anomaly)[0]
    if anomaly_idx.numel() > 0:
        ax.scatter(
            x[anomaly_idx.numpy()],
            y[anomaly_idx.numpy()],
            color="tab:red",
            s=18,
            zorder=3,
            label="anomaly",
        )
        for start, end in _anomaly_spans(series.is_anomaly):
            ax.axvspan(x[start], x[end], color="tab:red", alpha=0.12, zorder=0)

    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("value (a.u.)")
    ax.set_title(title or series.meta.get("function_type", "Synthetic series"))
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig


def plot_splits(
    splits: SyntheticSeriesSplits,
    titles: tuple[str, str, str] = ("Train", "Validation", "Test"),
    figsize: tuple[float, float] = (11, 12),
    sharey: bool = True,
    save_path: str | os.PathLike[str] | None = None,
) -> Figure:
    """
    Plot the train/val/test partitions of a SyntheticSeriesSplits, stacked vertically.

    Each partition is drawn with `plot_series` into its own subplot of a
    single figure, so the three segments (and any anomalies within them)
    can be compared at a glance. Builds and returns the figure without
    displaying it; see `plot_series` for notebook/script display notes.

    Parameters
    ----------
    splits : SyntheticSeriesSplits
        The partition to visualize, e.g. as returned by
        `SyntheticSeries.split`.
    titles : tuple of str, default ("Train", "Validation", "Test")
        Subplot titles, in `(train, val, test)` order.
    figsize : tuple of float, default (11, 12)
        Overall figure size, in inches.
    sharey : bool, default True
        Whether all three subplots share the same y-axis scale, so the
        partitions are directly comparable.
    save_path : str or os.PathLike, optional
        If given, the figure is saved to this path via `Figure.savefig`,
        with `bbox_inches="tight"`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the three stacked subplots.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharey=sharey)

    series_by_part = (splits.train, splits.val, splits.test)
    for ax, series, title in zip(axes, series_by_part, titles, strict=True):
        plot_series(series, title=title, ax=ax)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
