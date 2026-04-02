"""
_scoring.py — Core scoring functions for censored time-to-event forecasts.

Implements threshold-weighted CRPS (twCRPS) and related scores under
right-censoring, based on Taggart, Loveday & Louis (2026), arXiv:2603.14835.

The key idea is provisional propriety: we truncate both the forecast F and the
observed outcome at horizon tau. This is valid when tau is a FIXED evaluation
horizon common to all subjects. When subjects have individual random censoring
times, IPCW adjustment is required (see paper Section 5 for the caveat).

The integral formulas derive from Equations 18-19 of the paper:

  twCRPS_tau(F, t) = integral_0^tau (1{s >= t} - F(s))^2 ds

For uncensored obs (event=1):
  = integral_0^min(t,tau) F(s)^2 ds + integral_min(t,tau)^tau (1 - F(s))^2 ds

For censored obs (event=0, censored at c <= tau):
  = integral_0^c F(s)^2 ds + integral_c^tau (1 - F(s))^2 ds
  (treating min(T, tau) = c, i.e., we know T > c but not by how much)

Note: For a censored obs where c > tau, the observation effectively becomes
uncensored at tau — the formula handles this via min(c, tau).
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

# numpy<2.0 compat: trapezoid was added in 2.0
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


def _twcrps_single(
    surv_fn: Callable[[np.ndarray], np.ndarray],
    T_obs: float,
    event: int,
    tau: float,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None,
    n_grid: int,
) -> float:
    """twCRPS for a single observation.

    Parameters
    ----------
    surv_fn:
        Survival function S(t) = P(T > t). Will be evaluated on a time grid.
    T_obs:
        Observed time (event or censoring).
    event:
        1 if the event was observed, 0 if censored.
    tau:
        Evaluation horizon.
    weight_fn:
        Weight function w(t) applied to the integrand. None = uniform weight 1.
    n_grid:
        Number of quadrature points in [0, tau].

    Returns
    -------
    float
        twCRPS_tau(F, T_obs).
    """
    grid = np.linspace(0.0, tau, n_grid)
    S = surv_fn(grid)
    # Clip to [0, 1] for numerical safety
    S = np.clip(S, 0.0, 1.0)
    F = 1.0 - S

    # The split point is where the indicator 1{s >= T_obs} changes value.
    # For censored obs, we treat T_obs as the point beyond which the indicator
    # is ambiguous — but we use 1 (we know T > T_obs, so event hasn't occurred).
    split = min(T_obs, tau)

    # Build integrand: (indicator(s) - F(s))^2
    # indicator(s) = 0 for s < split, 1 for s >= split
    indicator = (grid >= split).astype(float)
    integrand = (indicator - F) ** 2

    if weight_fn is not None:
        w = np.clip(weight_fn(grid), 0.0, None)
        integrand = integrand * w

    return float(_trapz(integrand, grid))


def twcrps_obs(
    surv_fn: Callable[[np.ndarray], np.ndarray],
    T_obs: float,
    event: int,
    tau: float,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_grid: int = 200,
) -> float:
    """Threshold-weighted CRPS for a single observation.

    This is the public single-observation version of the scoring function,
    exported for users who want per-observation scores.

    Parameters
    ----------
    surv_fn:
        S(t) callable — returns array of survival probabilities.
    T_obs:
        Observed time.
    event:
        1 = event observed, 0 = censored.
    tau:
        Evaluation horizon (must be fixed across all observations for
        provisional propriety to hold).
    weight_fn:
        Optional weight function w(t). Must be non-negative on [0, tau].
        Default: uniform (w = 1).
    n_grid:
        Quadrature resolution.

    Returns
    -------
    float
        twCRPS_tau(F, T_obs).
    """
    return _twcrps_single(surv_fn, T_obs, event, tau, weight_fn, n_grid)


def twcrps_mean(
    surv_fns: list[Callable[[np.ndarray], np.ndarray]],
    T_obs: np.ndarray,
    event: np.ndarray,
    tau: float,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_grid: int = 200,
) -> float:
    """Mean twCRPS across a set of observations.

    Parameters
    ----------
    surv_fns:
        List of survival functions, one per observation.
    T_obs:
        Observed times, shape (n,).
    event:
        Event indicators, shape (n,). 1 = event, 0 = censored.
    tau:
        Fixed evaluation horizon.
    weight_fn:
        Optional weight function w(t).
    n_grid:
        Quadrature resolution.

    Returns
    -------
    float
        Mean twCRPS_tau.
    """
    T_obs = np.asarray(T_obs, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(T_obs)
    if len(surv_fns) != n:
        raise ValueError(
            f"len(surv_fns)={len(surv_fns)} must equal len(T_obs)={n}"
        )
    scores = np.array([
        _twcrps_single(surv_fns[i], T_obs[i], event[i], tau, weight_fn, n_grid)
        for i in range(n)
    ])
    return float(scores.mean())


def twcrps_profile(
    surv_fns: list[Callable[[np.ndarray], np.ndarray]],
    T_obs: np.ndarray,
    event: np.ndarray,
    thresholds: np.ndarray,
    weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    n_grid: int = 200,
) -> np.ndarray:
    """twCRPS evaluated at each threshold in thresholds.

    Returns the mean twCRPS(tau') for each tau' in thresholds. This is the
    profile of the scoring function over the time horizon — useful for
    diagnosing where errors accumulate.

    Parameters
    ----------
    surv_fns:
        List of survival functions.
    T_obs:
        Observed times.
    event:
        Event indicators.
    thresholds:
        Array of tau' values to evaluate at.
    weight_fn:
        Weight function.
    n_grid:
        Quadrature resolution.

    Returns
    -------
    np.ndarray
        Mean twCRPS at each threshold, shape (len(thresholds),).
    """
    T_obs = np.asarray(T_obs, dtype=float)
    event = np.asarray(event, dtype=int)
    thresholds = np.asarray(thresholds, dtype=float)

    return np.array([
        twcrps_mean(surv_fns, T_obs, event, tau=th, weight_fn=weight_fn, n_grid=n_grid)
        for th in thresholds
    ])


def quantile_score(
    q_forecasts: np.ndarray,
    T_obs: np.ndarray,
    event: np.ndarray,
    tau: float,
    alpha: float = 0.5,
) -> float:
    """Threshold-weighted quantile score (provisional pinball loss).

    Implements the provisional alpha-quantile scoring rule from Theorem 3 of
    Taggart et al. (2026). The censored realization is [T]_tau = min(T, tau).

    Score for observation i:
        S(x_i, [T_i]_tau) = (1{[T_i]_tau >= x_i} - alpha) * (x_i - [T_i]_tau)
                           = (1 - alpha) * max(x_i - [T_i]_tau, 0)
                           + alpha * max([T_i]_tau - x_i, 0)

    This is the standard pinball loss applied to the censored realization.

    Parameters
    ----------
    q_forecasts:
        Predicted alpha-quantile of T, one per observation. Shape (n,).
    T_obs:
        Observed times.
    event:
        Event indicators.
    tau:
        Evaluation horizon.
    alpha:
        Quantile level.

    Returns
    -------
    float
        Mean quantile score.
    """
    q_forecasts = np.asarray(q_forecasts, dtype=float)
    T_obs = np.asarray(T_obs, dtype=float)
    event = np.asarray(event, dtype=int)

    # Censored realization: min(T_obs, tau)
    t_cens = np.minimum(T_obs, tau)

    # Pinball loss
    scores = (1.0 - alpha) * np.maximum(q_forecasts - t_cens, 0.0) + alpha * np.maximum(
        t_cens - q_forecasts, 0.0
    )
    return float(scores.mean())


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    T_obs: np.ndarray,
    event: np.ndarray,
    tau: float,
    alpha: float = 0.1,
) -> float:
    """Threshold-weighted interval score (provisional twIS).

    Implements the provisional interval scoring rule from Theorem 4 of
    Taggart et al. (2026). The central (1-alpha) prediction interval [l, u]
    is scored against the censored realization [T]_tau.

    twIS_alpha(l, u, t) = (u - l)
                         + (2/alpha) * max(l - [T]_tau, 0)   # left miss
                         + (2/alpha) * max([T]_tau - u, 0)   # right miss

    Parameters
    ----------
    lower:
        Lower bound of prediction interval, shape (n,). Corresponds to the
        alpha/2 quantile forecast.
    upper:
        Upper bound of prediction interval, shape (n,). Corresponds to the
        (1 - alpha/2) quantile forecast.
    T_obs:
        Observed times.
    event:
        Event indicators.
    tau:
        Evaluation horizon.
    alpha:
        Miscoverage rate for the interval (e.g., 0.1 for 90% interval).

    Returns
    -------
    float
        Mean interval score.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    T_obs = np.asarray(T_obs, dtype=float)

    t_cens = np.minimum(T_obs, tau)

    width = upper - lower
    left_miss = (2.0 / alpha) * np.maximum(lower - t_cens, 0.0)
    right_miss = (2.0 / alpha) * np.maximum(t_cens - upper, 0.0)

    scores = width + left_miss + right_miss
    return float(scores.mean())


def murphy_elementary_score(
    q_forecast: np.ndarray,
    T_obs: np.ndarray,
    tau: float,
    theta: float,
    alpha: float = 0.5,
) -> np.ndarray:
    """Elementary Murphy score for each observation at a given threshold theta.

    The Murphy decomposition of the quantile loss:

        ES_{alpha,theta}(x, t) = {
            (1 - alpha)  if [T]_tau <= theta < x
            alpha        if x <= theta < [T]_tau
            0            otherwise
        }

    Integrating over theta from 0 to tau gives the quantile loss.
    The Murphy diagram plots mean ES_{alpha,theta} vs theta.

    Parameters
    ----------
    q_forecast:
        Predicted quantile, shape (n,).
    T_obs:
        Observed times, shape (n,).
    tau:
        Evaluation horizon.
    theta:
        Current threshold value.
    alpha:
        Quantile level.

    Returns
    -------
    np.ndarray
        Elementary score per observation, shape (n,).
    """
    q_forecast = np.asarray(q_forecast, dtype=float)
    T_obs = np.asarray(T_obs, dtype=float)
    t_cens = np.minimum(T_obs, tau)

    scores = np.where(
        (t_cens <= theta) & (theta < q_forecast),
        1.0 - alpha,
        np.where((q_forecast <= theta) & (theta < t_cens), alpha, 0.0),
    )
    return scores
