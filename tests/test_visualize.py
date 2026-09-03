"""Unit tests for plotting utilities."""

from collections.abc import Generator
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure

matplotlib.use("Agg")

from artificial_dataset.injectors import add_level_shift, add_point_anomalies
from artificial_dataset.series import (
    SyntheticSeries,
    SyntheticSeriesSplits,
    make_series,
)
from artificial_dataset.visualize import _anomaly_spans, plot_series, plot_splits


@pytest.fixture
def base_series() -> SyntheticSeries:
    """Fixture returning a standard, non-anomalous synthetic series."""
    return make_series(
        series_length=100,
        function_type="sinusoidal",
        function_params={"amplitude": 2.0, "frequency": 0.05},
        noise_std=0.05,
        random_state=0,
    )


@pytest.fixture
def base_splits() -> SyntheticSeriesSplits:
    """Fixture returning an even 3-way split with one anomaly per partition."""
    series = make_series(
        series_length=90,
        function_type="sinusoidal",
        function_params={"amplitude": 2.0, "frequency": 0.05},
        noise_std=0.05,
        random_state=0,
    )
    # One level-shift anomaly at the same relative offset in each third of
    # the series, so every partition ends up with exactly one anomaly.
    series = add_level_shift(series, start_idx=5, duration=3, random_state=0)
    series = add_level_shift(series, start_idx=35, duration=3, random_state=0)
    series = add_level_shift(series, start_idx=65, duration=3, random_state=0)
    return series.split((1 / 3, 1 / 3, 1 / 3))


@pytest.fixture(autouse=True)
def _close_figures() -> Generator[None, None, None]:
    """Close any figures left open by a test to avoid matplotlib state leaks."""
    yield
    plt.close("all")


# ---------- _anomaly_spans ----------


def test_anomaly_spans_empty_mask_returns_no_spans() -> None:
    """An all-False mask yields an empty span list."""
    mask = torch.zeros(10, dtype=torch.bool)
    assert _anomaly_spans(mask) == []


def test_anomaly_spans_single_point() -> None:
    """A single True index yields one (start, end) span equal to itself."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[4] = True
    assert _anomaly_spans(mask) == [(4, 4)]


def test_anomaly_spans_single_contiguous_run() -> None:
    """A contiguous run of True values collapses into one span."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[2:6] = True
    assert _anomaly_spans(mask) == [(2, 5)]


def test_anomaly_spans_multiple_disjoint_runs() -> None:
    """Non-adjacent True runs are returned as separate spans, in order."""
    mask = torch.zeros(20, dtype=torch.bool)
    mask[1:3] = True
    mask[10] = True
    mask[15:18] = True
    assert _anomaly_spans(mask) == [(1, 2), (10, 10), (15, 17)]


def test_anomaly_spans_all_true() -> None:
    """A fully anomalous mask yields a single span covering the whole range."""
    mask = torch.ones(6, dtype=torch.bool)
    assert _anomaly_spans(mask) == [(0, 5)]


def test_anomaly_spans_adjacent_indices_merge() -> None:
    """Two runs separated by a single gap stay separate; touching ones merge."""
    mask = torch.zeros(10, dtype=torch.bool)
    mask[0] = True
    mask[1] = True  # adjacent to index 0 -> should merge into one span
    mask[3] = True  # gap at index 2 -> separate span
    assert _anomaly_spans(mask) == [(0, 1), (3, 3)]


# ---------- plot_series ----------


def test_plot_series_returns_figure(base_series: SyntheticSeries) -> None:
    """plot_series returns a matplotlib Figure instance."""
    fig = plot_series(base_series)
    assert isinstance(fig, Figure)


def test_plot_series_no_anomalies_has_no_scatter(base_series: SyntheticSeries) -> None:
    """With no anomalies, no scatter collection or shaded span is drawn."""
    fig = plot_series(base_series)
    ax = fig.axes[0]
    assert len(ax.collections) == 0


def test_plot_series_with_point_anomalies_adds_scatter(
    base_series: SyntheticSeries,
) -> None:
    """Injected point anomalies are drawn as a scatter collection."""
    series = add_point_anomalies(base_series, n_anomalies=3, random_state=0)
    fig = plot_series(series)
    ax = fig.axes[0]
    assert len(ax.collections) >= 1


def test_plot_series_with_contiguous_anomaly_adds_shaded_span(
    base_series: SyntheticSeries,
) -> None:
    """A contiguous anomaly (level shift) adds an axvspan patch next to the scatter."""
    series = add_level_shift(base_series, start_idx=10, duration=15, random_state=0)
    fig = plot_series(series)
    ax = fig.axes[0]
    assert len(ax.patches) >= 1


def test_plot_series_default_title_uses_function_type(
    base_series: SyntheticSeries,
) -> None:
    """When no title is given, the axes title falls back to meta['function_type']."""
    fig = plot_series(base_series)
    ax = fig.axes[0]
    assert ax.get_title() == base_series.meta["function_type"]


def test_plot_series_explicit_title_overrides_default(
    base_series: SyntheticSeries,
) -> None:
    """An explicit title argument takes precedence over series.meta."""
    fig = plot_series(base_series, title="Custom Title")
    ax = fig.axes[0]
    assert ax.get_title() == "Custom Title"


def test_plot_series_missing_function_type_falls_back_to_default_string() -> None:
    """When meta has no 'function_type' key, the hardcoded fallback title is used."""
    series = make_series(series_length=20, function_type="constant")
    series.meta.pop("function_type", None)
    fig = plot_series(series)
    ax = fig.axes[0]
    assert ax.get_title() == "Synthetic series"


def test_plot_series_creates_new_figure_when_ax_omitted(
    base_series: SyntheticSeries,
) -> None:
    """Without an ax argument, plot_series creates its own figure."""
    n_figures_before = len(plt.get_fignums())
    fig = plot_series(base_series)
    assert len(plt.get_fignums()) == n_figures_before + 1
    assert fig.number in plt.get_fignums()


def test_plot_series_draws_into_provided_axes(base_series: SyntheticSeries) -> None:
    """When ax is provided, plot_series draws into it and returns its parent figure."""
    fig, ax = plt.subplots()
    returned_fig = plot_series(base_series, ax=ax)
    assert returned_fig is fig
    assert len(ax.lines) == 1


def test_plot_series_line_data_matches_series_values(
    base_series: SyntheticSeries,
) -> None:
    """The plotted line's x/y data matches the series' x/y tensors."""
    fig = plot_series(base_series)
    ax = fig.axes[0]
    line = ax.lines[0]
    x_plotted, y_plotted = line.get_data()

    assert x_plotted == pytest.approx(base_series.x.detach().cpu().numpy())
    assert y_plotted == pytest.approx(base_series.y.detach().cpu().numpy())


def test_plot_series_sets_axis_labels(base_series: SyntheticSeries) -> None:
    """X and y axis labels are set to the expected fixed strings."""
    fig = plot_series(base_series)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time (a.u.)"
    assert ax.get_ylabel() == "value (a.u.)"


def test_plot_series_does_not_call_plt_show(
    base_series: SyntheticSeries, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_series never calls plt.show() itself; display is left to the caller.

    This is what avoids the double-render seen in notebooks when a figure
    is both explicitly shown and then auto-displayed again by the
    notebook's own inline backend.
    """
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    plot_series(base_series)
    assert calls == []


# ---------- plot_splits ----------


def test_plot_splits_returns_figure(base_splits: SyntheticSeriesSplits) -> None:
    """plot_splits returns a matplotlib Figure instance."""
    fig = plot_splits(base_splits)
    assert isinstance(fig, Figure)


def test_plot_splits_creates_three_stacked_subplots(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """plot_splits produces exactly three axes, one per partition."""
    fig = plot_splits(base_splits)
    assert len(fig.axes) == 3


def test_plot_splits_default_titles_are_train_val_test(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """Without explicit titles, subplots are labeled Train/Validation/Test in order."""
    fig = plot_splits(base_splits)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Train", "Validation", "Test"]


def test_plot_splits_custom_titles_are_applied_in_order(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """Custom titles are assigned to subplots in (train, val, test) order."""
    fig = plot_splits(base_splits, titles=("Tanító", "Validáció", "Teszt"))
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Tanító", "Validáció", "Teszt"]


def test_plot_splits_each_subplot_shows_its_own_partition(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """Each subplot's line data matches the corresponding partition, in order."""
    fig = plot_splits(base_splits)
    parts = (base_splits.train, base_splits.val, base_splits.test)

    for ax, series in zip(fig.axes, parts, strict=True):
        x_plotted, y_plotted = ax.lines[0].get_data()
        assert x_plotted == pytest.approx(series.x.detach().cpu().numpy())
        assert y_plotted == pytest.approx(series.y.detach().cpu().numpy())


def test_plot_splits_each_subplot_shows_its_own_anomaly(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """Each partition's anomaly is drawn in its own subplot, not smeared across all."""
    fig = plot_splits(base_splits)
    for ax in fig.axes:
        assert len(ax.collections) >= 1  # scatter marker
        assert len(ax.patches) >= 1  # shaded span


def test_plot_splits_sharey_true_gives_matching_ylim_by_default(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """With the default sharey=True, all three subplots share the same y-limits."""
    fig = plot_splits(base_splits)
    ylims = {ax.get_ylim() for ax in fig.axes}
    assert len(ylims) == 1


def test_plot_splits_sharey_false_is_accepted(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """sharey=False is accepted and still produces a complete 3-axes figure."""
    fig = plot_splits(base_splits, sharey=False)
    assert len(fig.axes) == 3


def test_plot_splits_figsize_is_applied(base_splits: SyntheticSeriesSplits) -> None:
    """The figsize argument controls the resulting figure's size in inches."""
    fig = plot_splits(base_splits, figsize=(7.0, 9.0))
    assert fig.get_size_inches() == pytest.approx((7.0, 9.0))


def test_plot_splits_save_path_writes_a_file(
    base_splits: SyntheticSeriesSplits, tmp_path: Path
) -> None:
    """When save_path is given, the figure is saved to that path."""
    out_path = tmp_path / "splits.png"
    plot_splits(base_splits, save_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_splits_without_save_path_writes_no_file(
    base_splits: SyntheticSeriesSplits, tmp_path: Path
) -> None:
    """Without save_path, plot_splits has no filesystem side effects."""
    plot_splits(base_splits)
    assert list(tmp_path.iterdir()) == []


def test_plot_splits_does_not_call_plt_show(
    base_splits: SyntheticSeriesSplits, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plot_splits never calls plt.show() itself; display is left to the caller."""
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    plot_splits(base_splits)
    assert calls == []


def test_plot_splits_mismatched_titles_length_raises(
    base_splits: SyntheticSeriesSplits,
) -> None:
    """A titles sequence that isn't length 3 raises, rather than silently truncating."""
    with pytest.raises(ValueError, match="zip"):
        plot_splits(base_splits, titles=("Only", "Two"))  # type: ignore[arg-type]
