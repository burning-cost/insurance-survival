"""
_plots.py — Visualisation for censored forecast evaluation.

Murphy diagram and reliability diagram for threshold-weighted scoring under
right-censoring. Follows Taggart, Loveday & Louis (2026), arXiv:2603.14835.

Both functions require matplotlib (insurance-survival[plot]).
"""

from __future__ import annotations

import warnings
from typing import Callable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.axes


def murphy_diagram(
    forecasters: dict[
        str,
        tuple[list[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray],
    ],
    tau: float,
    alpha: float = 0.5,
    thresholds: np.ndarray | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """Murphy diagram for comparing survival model quantile forecasts.

    Plots the mean elementary Murphy score at each threshold theta in [0, tau]
    for each named forecaster. The area under each curve equals the mean
    quantile loss (threshold-weighted quantile score at horizon tau).

    Where model A's curve lies below model B's at threshold theta, model A
    is better calibrated for events near time theta.

    Parameters
    ----------
    forecasters:
        Dict mapping {name: (surv_fns, T_obs, event)} where:
        - surv_fns: list of S(t) callables, one per observation
        - T_obs: observed times, shape (n,)
        - event: event indicators, shape (n,)
        All forecasters should cover the same set of observations.
    tau:
        Evaluation horizon. Must be fixed (same for all observations).
    alpha:
        Quantile level for Murphy decomposition.
    thresholds:
        Grid of theta values. Default: linspace(0, tau, 200).
    ax:
        Existing Axes to draw on. If None, a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for Murphy diagrams. "
            "Install with: pip install insurance-survival[plot]"
        )

    from ._scoring import murphy_elementary_score

    if thresholds is None:
        thresholds = np.linspace(0.0, tau, 200)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for name, (surv_fns, T_obs, event) in forecasters.items():
        T_obs = np.asarray(T_obs, dtype=float)
        n = len(T_obs)
        if len(surv_fns) != n:
            raise ValueError(
                f"Forecaster '{name}': len(surv_fns) != len(T_obs)"
            )

        # Derive alpha-quantile forecasts from survival functions.
        # Q_alpha(F) = inf{t >= 0 : F(t) >= alpha} = inf{t : S(t) <= 1-alpha}
        # We evaluate on thresholds grid for efficiency.
        t_eval = np.unique(np.concatenate([thresholds, [0.0, tau]]))
        q_preds = np.array([
            _quantile_from_surv(surv_fns[i], alpha, t_eval)
            for i in range(n)
        ])

        mean_es = np.array([
            murphy_elementary_score(q_preds, T_obs, tau, theta, alpha).mean()
            for theta in thresholds
        ])
        ax.plot(thresholds, mean_es, label=name, linewidth=1.5)

    ax.set_xlabel("Threshold theta")
    ax.set_ylabel(f"Mean elementary score (alpha={alpha:.2f})")
    ax.set_title(
        f"Murphy Diagram (tau={tau:.2f})\n"
        "Area under curve = mean quantile loss. Lower curve = better model."
    )
    ax.legend()
    ax.set_xlim(0, tau)
    fig.tight_layout()
    return fig


def reliability_diagram(
    surv_fns: list[Callable[[np.ndarray], np.ndarray]],
    T_obs: np.ndarray,
    event: np.ndarray,
    eval_time: float,
    n_bins: int = 10,
    ax: "matplotlib.axes.Axes | None" = None,
) -> "matplotlib.figure.Figure":
    """Reliability diagram for survival probability forecasts under censoring.

    Bins observations by their predicted P(T > eval_time), then estimates the
    observed probability P(T > eval_time) within each bin using the
    Kaplan-Meier estimator (handles right-censoring).

    Overlays an isotonic regression recalibration curve. Perfect calibration
    lies on the 45-degree diagonal.

    Parameters
    ----------
    surv_fns:
        List of S(t) callables, one per observation.
    T_obs:
        Observed times, shape (n,).
    event:
        Event indicators, shape (n,).
    eval_time:
        Time point t* at which to assess calibration.
    n_bins:
        Number of equally-sized bins (approximate — some may be merged if small).
    ax:
        Existing Axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for reliability diagrams. "
            "Install with: pip install insurance-survival[plot]"
        )

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        raise ImportError(
            "lifelines is required for reliability diagrams. "
            "Install with: pip install insurance-survival"
        )

    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        raise ImportError(
            "scikit-learn is required for isotonic regression overlay. "
            "Install with: pip install scikit-learn"
        )

    T_obs = np.asarray(T_obs, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(T_obs)

    if len(surv_fns) != n:
        raise ValueError(f"len(surv_fns) != len(T_obs): {len(surv_fns)} vs {n}")

    # Predicted P(T > eval_time) = S(eval_time)
    t_arr = np.array([eval_time])
    p_pred = np.array([float(np.clip(surv_fns[i](t_arr)[0], 0.0, 1.0)) for i in range(n)])

    # Sort by predicted probability and bin
    sort_idx = np.argsort(p_pred)
    bin_edges = np.array_split(sort_idx, n_bins)

    mean_pred = []
    obs_surv = []
    bin_sizes = []

    for bin_idx in bin_edges:
        if len(bin_idx) < 3:
            continue
        T_bin = T_obs[bin_idx]
        E_bin = event[bin_idx]
        p_bin = p_pred[bin_idx]

        # KM estimate of S(eval_time) within this bin
        km = KaplanMeierFitter()
        try:
            km.fit(T_bin, event_observed=E_bin)
            # KM survival at eval_time
            km_sf = km.survival_function_
            # Interpolate: find S(eval_time)
            times_km = km_sf.index.values
            surv_km = km_sf["KM_estimate"].values
            km_val = float(np.interp(eval_time, times_km, surv_km, left=1.0, right=surv_km[-1]))
        except Exception:
            km_val = np.nan

        mean_pred.append(float(p_bin.mean()))
        obs_surv.append(km_val)
        bin_sizes.append(len(bin_idx))

    mean_pred = np.array(mean_pred)
    obs_surv = np.array(obs_surv)
    bin_sizes = np.array(bin_sizes)

    # Isotonic regression (ignores NaN bins)
    valid = ~np.isnan(obs_surv)
    iso = IsotonicRegression(out_of_bounds="clip")
    if valid.sum() >= 2:
        iso_fitted = iso.fit_transform(mean_pred[valid], obs_surv[valid])
        iso_x = mean_pred[valid]
        iso_y = iso_fitted
    else:
        iso_x = iso_y = np.array([])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Scatter: size proportional to bin size
    s = np.clip(bin_sizes / bin_sizes.max() * 200, 20, 200)
    scatter = ax.scatter(
        mean_pred[valid],
        obs_surv[valid],
        s=s[valid],
        alpha=0.8,
        zorder=3,
        label="Bins (sized by n)",
    )

    if len(iso_x) >= 2:
        sort_iso = np.argsort(iso_x)
        ax.plot(iso_x[sort_iso], iso_y[sort_iso], "r--", linewidth=1.5, label="Isotonic recalibration")

    # 45-degree diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.5)

    ax.set_xlabel(f"Mean predicted P(T > {eval_time:.2f})")
    ax.set_ylabel(f"Observed P(T > {eval_time:.2f}) [KM]")
    ax.set_title(f"Reliability Diagram at t = {eval_time:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _quantile_from_surv(
    surv_fn: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    t_grid: np.ndarray,
) -> float:
    """Extract alpha-quantile from a survival function via grid search.

    Q_alpha(F) = inf{t : F(t) >= alpha} = inf{t : S(t) <= 1 - alpha}

    Parameters
    ----------
    surv_fn:
        S(t) callable.
    alpha:
        Quantile level (0 < alpha < 1).
    t_grid:
        Time grid to search over (must be sorted, ascending).

    Returns
    -------
    float
        Estimated alpha-quantile. Returns t_grid[-1] if not reached.
    """
    t_grid = np.sort(t_grid)
    S_vals = np.clip(surv_fn(t_grid), 0.0, 1.0)
    threshold = 1.0 - alpha
    # Find first t where S(t) <= threshold
    idx = np.searchsorted(-S_vals, -threshold, side="right")
    if idx >= len(t_grid):
        return float(t_grid[-1])
    return float(t_grid[idx])
