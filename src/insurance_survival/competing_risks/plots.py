"""
plots.py — Visualisation utilities for competing risks analysis.

Functions
---------
plot_cif_comparison    Plot CIFs for multiple fitted models or groups side by side
plot_forest            Subdistribution hazard ratio forest plot
plot_cumulative_hazard Plot the baseline cumulative subdistribution hazard

These are convenience wrappers. The core plotting methods are also available
directly on ``AalenJohansenFitter.plot()`` and
``FineGrayFitter.plot_partial_effects_on_outcome()``.

All functions require matplotlib. Install with: pip install insurance-survival[plot]
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

_MATPLOTLIB_MSG = (
    "matplotlib is required for plotting. "
    "Install with: pip install insurance-survival[plot]"
)


def plot_cif_comparison(
    fitters: dict,
    times: Optional[np.ndarray] = None,
    cause_labels: Optional[dict] = None,
    ax: Optional[Any] = None,
    title: str = "Cumulative Incidence Functions",
) -> Any:
    """Plot CIFs from multiple ``AalenJohansenFitter`` instances.

    Parameters
    ----------
    fitters:
        Dictionary mapping label to fitted ``AalenJohansenFitter``.
    times:
        Time grid for evaluation (optional, uses each fitter's own times).
    cause_labels:
        Ignored; use the keys of ``fitters`` as labels.
    ax:
        Matplotlib axes.
    title:
        Plot title.

    Returns
    -------
    plt.Axes

    Examples
    --------
    >>> fitters = {"Group A": aj_a, "Group B": aj_b}
    >>> plot_cif_comparison(fitters)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MATPLOTLIB_MSG)

    if ax is None:
        _, ax = plt.subplots()

    for label, fitter in fitters.items():
        t = fitter.times_
        cif = fitter.cumulative_incidence_.values
        ax.step(t, cif, where="post", label=label)
        lower = fitter.confidence_intervals_["lower"].values
        upper = fitter.confidence_intervals_["upper"].values
        colour = ax.lines[-1].get_color()
        ax.fill_between(t, lower, upper, alpha=0.15, color=colour, step="post")

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Incidence")
    ax.set_ylim(0, None)
    ax.set_title(title)
    ax.legend()
    return ax


def plot_forest(
    model: "FineGrayFitter",
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    exponentiate: bool = True,
) -> Any:
    """Subdistribution hazard ratio forest plot.

    Plots each covariate as a row with its SHR (or log-SHR) and confidence
    interval.

    Parameters
    ----------
    model:
        A fitted ``FineGrayFitter``.
    ax:
        Matplotlib axes.
    title:
        Plot title. Defaults to "Subdistribution Hazard Ratios".
    exponentiate:
        If ``True`` (default), plot exp(coef) = SHR. If ``False``, plot the
        raw log-scale coefficient.

    Returns
    -------
    plt.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MATPLOTLIB_MSG)

    from .fine_gray import FineGrayFitter

    if not model._fitted:
        raise RuntimeError("Model must be fitted before plotting.")

    if title is None:
        title = f"Subdistribution Hazard Ratios (cause {model.event_of_interest})"

    summary = model.summary.copy()
    alpha = model.alpha
    ci_cols = [c for c in summary.columns if "lower" in c or "upper" in c]

    if len(ci_cols) < 2:
        raise ValueError("Could not locate CI columns in model summary.")

    lower_col, upper_col = ci_cols[0], ci_cols[1]

    if exponentiate:
        point = summary["exp(coef)"]
        lower = np.exp(summary[lower_col])
        upper = np.exp(summary[upper_col])
        null_value = 1.0
        xlabel = "Subdistribution Hazard Ratio"
    else:
        point = summary["coef"]
        lower = summary[lower_col]
        upper = summary[upper_col]
        null_value = 0.0
        xlabel = "Log Subdistribution Hazard Ratio"

    n_cov = len(summary)
    y_positions = np.arange(n_cov)
    covariates = summary.index.tolist()

    if ax is None:
        fig_height = max(3, n_cov * 0.5 + 1)
        _, ax = plt.subplots(figsize=(7, fig_height))

    ax.scatter(point, y_positions, color="black", zorder=3, s=30)
    for i in range(n_cov):
        ax.hlines(y_positions[i], lower.iloc[i], upper.iloc[i], color="black", linewidth=1.5)
        # Caps
        cap_size = 0.15
        ax.vlines(lower.iloc[i], y_positions[i] - cap_size, y_positions[i] + cap_size,
                  color="black", linewidth=1)
        ax.vlines(upper.iloc[i], y_positions[i] - cap_size, y_positions[i] + cap_size,
                  color="black", linewidth=1)

    ax.axvline(null_value, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(covariates)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    return ax


def plot_cumulative_hazard(
    model: "FineGrayFitter",
    ax: Optional[Any] = None,
    title: Optional[str] = None,
) -> Any:
    """Plot the Breslow baseline cumulative subdistribution hazard.

    Parameters
    ----------
    model:
        A fitted ``FineGrayFitter``.
    ax:
        Matplotlib axes.
    title:
        Plot title.

    Returns
    -------
    plt.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MATPLOTLIB_MSG)

    if not model._fitted:
        raise RuntimeError("Model must be fitted before plotting.")

    if title is None:
        title = (
            f"Baseline Cumulative Subdistribution Hazard "
            f"(cause {model.event_of_interest})"
        )

    if ax is None:
        _, ax = plt.subplots()

    bch = model.baseline_cumulative_hazard_
    ax.step(bch.index, bch.values, where="post", color="steelblue")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Hazard")
    ax.set_title(title)
    return ax


def plot_brier_score(
    times: np.ndarray,
    brier_scores: "pd.Series | np.ndarray",
    null_score: float = 0.25,
    ax: Optional[Any] = None,
    title: str = "Brier Score Over Time",
) -> Any:
    """Plot the IPCW Brier score curve.

    Parameters
    ----------
    times:
        Evaluation times.
    brier_scores:
        Brier score at each time point.
    null_score:
        Benchmark Brier score for a model predicting constant 0.5 (default
        0.25). Plotted as a dashed reference line.
    ax:
        Matplotlib axes.
    title:
        Plot title.

    Returns
    -------
    plt.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(_MATPLOTLIB_MSG)

    if ax is None:
        _, ax = plt.subplots()

    bs_vals = np.asarray(brier_scores)
    ax.plot(times, bs_vals, label="Model", color="steelblue")
    ax.axhline(null_score, color="grey", linestyle="--", label="Null (0.25)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Brier Score")
    ax.set_title(title)
    ax.set_ylim(0, None)
    ax.legend()
    return ax
