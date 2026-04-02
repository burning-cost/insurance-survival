"""
_evaluator.py — CensoredForecastEvaluator: evaluation hub for right-censored
time-to-event forecasts.

Based on Taggart, Loveday & Louis (2026), arXiv:2603.14835. Implements
threshold-weighted CRPS and related proper scoring rules under right-censoring
via provisional propriety.

IMPORTANT — PROVISIONAL PROPRIETY WARNING:
These scores are provisionally proper (not unconditionally proper) relative to
the fixed horizon tau. This means:

1. tau must be FIXED and IDENTICAL for all observations. A single evaluation
   horizon — e.g., 90-day settlement, 12-month lapse — is the intended use
   case.

2. If subjects have DIFFERENT random censoring times (common in observational
   insurance data), the provisional propriety arguments do not hold exactly.
   The rankings are heuristic in that setting. IPCW adjustment (which this
   class does NOT implement) would be required for asymptotically unbiased
   model comparison.

3. The censored observations that are administratively censored at tau
   (i.e., T_obs >= tau with event=0) are handled correctly: they contribute
   the full [0, tau] integral.

References
----------
Taggart, R.J., Loveday, N. & Louis, S. (2026). On the evaluation of time-to-
event, survival time and first passage time forecasts. arXiv:2603.14835.
"""

from __future__ import annotations

import warnings
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

from ._scoring import (
    twcrps_mean,
    twcrps_profile,
    quantile_score as _quantile_score,
    interval_score as _interval_score,
)
from ._plots import murphy_diagram as _murphy_diagram, reliability_diagram as _reliability_diagram

if TYPE_CHECKING:
    import matplotlib.figure


_PROVISIONAL_WARNING = (
    "CensoredForecastEvaluator uses provisionally proper scoring rules "
    "(Taggart et al., 2026, arXiv:2603.14835). "
    "tau must be FIXED and IDENTICAL for all observations. "
    "If subjects have different random censoring times, rankings are heuristic. "
    "IPCW adjustment is not implemented."
)


class CensoredForecastEvaluator:
    """Evaluation hub for right-censored time-to-event forecasts.

    Implements threshold-weighted CRPS (twCRPS), quantile score, interval
    score, Murphy diagram, and reliability diagram from Taggart, Loveday &
    Louis (2026), arXiv:2603.14835.

    The scoring framework is based on provisional strict propriety at a fixed
    evaluation horizon tau. For proper use, tau should be the same for all
    observations (e.g., the policy term length, or a fixed administrative
    censoring date expressed as time since start).

    Parameters
    ----------
    tau:
        Evaluation horizon. All scoring is truncated at tau. Must be positive.
        The same value must apply to all observations for provisional propriety
        to hold.
    alpha:
        Quantile level for Murphy diagram and quantile score. Default: 0.5
        (median). Use alpha=0.1 for 90th-percentile quantile evaluation.
    weight_fn:
        Optional weight function w(t) applied to the twCRPS integrand. If None,
        uniform weight (w=1) is used, giving the standard twCRPS_tau.
        Must be non-negative on [0, tau]. Example: ``lambda t: t`` for
        time-weighted scoring that emphasises later failures.
    warn:
        If True (default), emit a UserWarning about provisional propriety.
        Set to False to suppress after you've acknowledged the limitation.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import weibull_min
    >>> from insurance_survival.evaluation import CensoredForecastEvaluator

    >>> # Simulate some survival data
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
    >>> tau = 8.0
    >>> # Fixed administrative censoring
    >>> T_obs = np.minimum(T_true, tau)
    >>> event = (T_true <= tau).astype(int)

    >>> # True model: Weibull(2, 5)
    >>> def make_surv(c, scale):
    ...     def S(t):
    ...         return weibull_min.sf(t, c=c, scale=scale)
    ...     return S

    >>> true_surv = [make_surv(2, 5)] * n   # correct model
    >>> wrong_surv = [make_surv(1, 5)] * n  # exponential (misspecified)

    >>> ev = CensoredForecastEvaluator(tau=tau, warn=False)
    >>> score_true = ev.twcrps(true_surv, T_obs, event)
    >>> score_wrong = ev.twcrps(wrong_surv, T_obs, event)
    >>> assert score_true < score_wrong, "True model should score better"
    """

    def __init__(
        self,
        tau: float,
        alpha: float = 0.5,
        weight_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        warn: bool = True,
    ) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.weight_fn = weight_fn
        if warn:
            warnings.warn(_PROVISIONAL_WARNING, UserWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def twcrps(
        self,
        surv_fns: list[Callable[[np.ndarray], np.ndarray]],
        T_obs: np.ndarray,
        event: np.ndarray,
        n_grid: int = 200,
    ) -> float:
        """Mean threshold-weighted CRPS across all observations.

        Lower is better. 0 = perfect forecast.

        Parameters
        ----------
        surv_fns:
            List of survival functions S_i(t), one per observation. Each must
            accept a numpy array of times and return survival probabilities in
            [0, 1].
        T_obs:
            Observed times (event time or censoring time), shape (n,).
        event:
            Event indicators: 1 if event occurred, 0 if censored. Shape (n,).
        n_grid:
            Number of quadrature points for numerical integration over [0, tau].
            200 is sufficient for most smooth survival distributions.

        Returns
        -------
        float
            Mean twCRPS_tau across n observations.
        """
        return twcrps_mean(
            surv_fns,
            np.asarray(T_obs, dtype=float),
            np.asarray(event, dtype=int),
            tau=self.tau,
            weight_fn=self.weight_fn,
            n_grid=n_grid,
        )

    def twcrps_profile(
        self,
        surv_fns: list[Callable[[np.ndarray], np.ndarray]],
        T_obs: np.ndarray,
        event: np.ndarray,
        thresholds: np.ndarray | None = None,
        n_grid: int = 200,
    ) -> pd.Series:
        """Mean twCRPS at each threshold tau' in [0, tau].

        Returns the score as a function of the evaluation horizon. Useful for
        seeing whether model A beats model B early in the horizon or late.

        Parameters
        ----------
        surv_fns:
            Survival functions.
        T_obs:
            Observed times.
        event:
            Event indicators.
        thresholds:
            Array of tau' values. Default: linspace(0, tau, 100).
        n_grid:
            Quadrature resolution.

        Returns
        -------
        pd.Series
            Index = thresholds, values = mean twCRPS(tau').
        """
        if thresholds is None:
            thresholds = np.linspace(0.0, self.tau, 100)
        thresholds = np.asarray(thresholds, dtype=float)
        scores = twcrps_profile(
            surv_fns,
            np.asarray(T_obs, dtype=float),
            np.asarray(event, dtype=int),
            thresholds=thresholds,
            weight_fn=self.weight_fn,
            n_grid=n_grid,
        )
        return pd.Series(scores, index=thresholds, name="twcrps")

    def quantile_score(
        self,
        q_forecasts: np.ndarray,
        T_obs: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Provisional threshold-weighted quantile score (pinball loss).

        Scores the alpha-quantile forecast against the censored realization
        min(T, tau). This is the integral of the Murphy elementary score.

        Parameters
        ----------
        q_forecasts:
            Predicted alpha-quantile of T, shape (n,). These are the
            forecast times by which alpha fraction of events are expected
            to have occurred.
        T_obs:
            Observed times.
        event:
            Event indicators.

        Returns
        -------
        float
            Mean quantile score. Lower is better.
        """
        return _quantile_score(
            np.asarray(q_forecasts, dtype=float),
            np.asarray(T_obs, dtype=float),
            np.asarray(event, dtype=int),
            tau=self.tau,
            alpha=self.alpha,
        )

    def interval_score(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        T_obs: np.ndarray,
        event: np.ndarray,
        alpha: float | None = None,
    ) -> float:
        """Provisional threshold-weighted interval score (twIS).

        Scores a central prediction interval [lower, upper] against the
        censored realization min(T, tau). Penalises width and coverage failures.

        Parameters
        ----------
        lower:
            Lower bound (alpha/2 quantile forecast), shape (n,).
        upper:
            Upper bound (1 - alpha/2 quantile forecast), shape (n,).
        T_obs:
            Observed times.
        event:
            Event indicators.
        alpha:
            Miscoverage rate. Default: 1 - 2 * self.alpha (so if self.alpha=0.5,
            this gives the symmetric 0% interval; more usefully, pass alpha=0.1
            for a 90% interval). If you're using this seriously, pass alpha
            explicitly.

        Returns
        -------
        float
            Mean interval score. Lower is better.
        """
        if alpha is None:
            # Default: use the complement of the quantile alpha for a symmetric interval
            alpha = float(2.0 * min(self.alpha, 1.0 - self.alpha))
            if alpha <= 0.0:
                alpha = 0.1
        return _interval_score(
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            np.asarray(T_obs, dtype=float),
            np.asarray(event, dtype=int),
            tau=self.tau,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        forecasters: dict[
            str,
            tuple[list[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray],
        ],
        n_grid: int = 200,
    ) -> pd.DataFrame:
        """Compare multiple forecasters on twCRPS and quantile score.

        Parameters
        ----------
        forecasters:
            Dict of {name: (surv_fns, T_obs, event)}.
        n_grid:
            Quadrature resolution for twCRPS.

        Returns
        -------
        pd.DataFrame
            Columns: twCRPS, quantile_score. Sorted by twCRPS ascending.
            Lower is better for all metrics.
        """
        from ._plots import _quantile_from_surv

        rows = []
        for name, (surv_fns, T_obs, event) in forecasters.items():
            T_obs_arr = np.asarray(T_obs, dtype=float)
            event_arr = np.asarray(event, dtype=int)

            score_tw = twcrps_mean(
                surv_fns, T_obs_arr, event_arr,
                tau=self.tau, weight_fn=self.weight_fn, n_grid=n_grid,
            )

            # Derive quantile forecasts from survival functions
            t_grid = np.linspace(0.0, self.tau * 1.5, 400)
            q_preds = np.array([
                _quantile_from_surv(surv_fns[i], self.alpha, t_grid)
                for i in range(len(surv_fns))
            ])
            score_q = _quantile_score(q_preds, T_obs_arr, event_arr, tau=self.tau, alpha=self.alpha)

            rows.append({
                "model": name,
                "twCRPS": score_tw,
                "quantile_score": score_q,
            })

        df = pd.DataFrame(rows).set_index("model").sort_values("twCRPS")
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def murphy_diagram(
        self,
        forecasters: dict[
            str,
            tuple[list[Callable[[np.ndarray], np.ndarray]], np.ndarray, np.ndarray],
        ],
        thresholds: np.ndarray | None = None,
        ax: "matplotlib.axes.Axes | None" = None,
    ) -> "matplotlib.figure.Figure":
        """Murphy diagram for model comparison.

        Plots the mean elementary Murphy score vs threshold theta for each
        forecaster. X-axis: threshold theta in [0, tau]. Y-axis: mean
        elementary score. Area under each line = mean quantile loss.

        A model that dominates another at all thresholds has uniformly better
        quantile calibration over the entire time horizon.

        Parameters
        ----------
        forecasters:
            Dict of {name: (surv_fns, T_obs, event)}.
        thresholds:
            Grid of theta values. Default: linspace(0, tau, 200).
        ax:
            Existing Axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        return _murphy_diagram(
            forecasters,
            tau=self.tau,
            alpha=self.alpha,
            thresholds=thresholds,
            ax=ax,
        )

    def reliability_diagram(
        self,
        surv_fns: list[Callable[[np.ndarray], np.ndarray]],
        T_obs: np.ndarray,
        event: np.ndarray,
        eval_time: float,
        n_bins: int = 10,
        ax: "matplotlib.axes.Axes | None" = None,
    ) -> "matplotlib.figure.Figure":
        """Reliability diagram: predicted vs observed survival probability.

        Bins observations by their predicted P(T > eval_time), estimates the
        empirical frequency via Kaplan-Meier, and plots observed vs predicted.
        Perfect calibration = 45-degree diagonal.

        Parameters
        ----------
        surv_fns:
            List of S_i(t) callables.
        T_obs:
            Observed times.
        event:
            Event indicators.
        eval_time:
            Time point t* for calibration check. Should be <= tau.
        n_bins:
            Number of probability bins.
        ax:
            Existing Axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if eval_time > self.tau:
            warnings.warn(
                f"eval_time={eval_time} > tau={self.tau}. "
                "Reliability at times beyond tau involves extrapolation.",
                UserWarning,
                stacklevel=2,
            )
        return _reliability_diagram(
            surv_fns,
            np.asarray(T_obs, dtype=float),
            np.asarray(event, dtype=int),
            eval_time=eval_time,
            n_bins=n_bins,
            ax=ax,
        )

    def __repr__(self) -> str:
        weight_desc = "uniform" if self.weight_fn is None else "custom"
        return (
            f"CensoredForecastEvaluator("
            f"tau={self.tau}, alpha={self.alpha}, weight={weight_desc})"
        )


# ------------------------------------------------------------------
# Helper: wrap a 2D survival matrix into a list of callables
# ------------------------------------------------------------------

def from_matrix(
    S_matrix: np.ndarray,
    time_grid: np.ndarray,
) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Wrap a 2D survival probability matrix into a list of callables.

    Many survival models (pycox, scikit-survival) output survival estimates
    as a matrix of shape (n_samples, n_times). This helper converts that to
    the list-of-callables format expected by CensoredForecastEvaluator.

    Interpolation is linear. Values are clamped to [0, 1].

    Parameters
    ----------
    S_matrix:
        Survival probabilities, shape (n_samples, n_times). Rows are subjects,
        columns correspond to ``time_grid``.
    time_grid:
        Time points corresponding to columns of S_matrix. Shape (n_times,).

    Returns
    -------
    list[Callable]
        List of n_samples callables, each taking a time array and returning
        survival probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import weibull_min
    >>> from insurance_survival.evaluation import from_matrix, CensoredForecastEvaluator

    >>> n, n_times = 100, 50
    >>> t_grid = np.linspace(0, 10, n_times)
    >>> # Simulate a matrix (e.g., from a pycox model)
    >>> S_matrix = weibull_min.sf(t_grid[None, :], c=2, scale=5)
    >>> S_matrix = np.repeat(S_matrix, n, axis=0)  # identical predictions

    >>> surv_fns = from_matrix(S_matrix, t_grid)
    >>> ev = CensoredForecastEvaluator(tau=8.0, warn=False)
    >>> score = ev.twcrps(surv_fns, T_obs=np.ones(n) * 5.0, event=np.ones(n))
    """
    S_matrix = np.asarray(S_matrix, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)

    if S_matrix.ndim != 2:
        raise ValueError(f"S_matrix must be 2D (n_samples, n_times), got shape {S_matrix.shape}")
    if S_matrix.shape[1] != len(time_grid):
        raise ValueError(
            f"S_matrix.shape[1]={S_matrix.shape[1]} must equal len(time_grid)={len(time_grid)}"
        )

    sort_idx = np.argsort(time_grid)
    time_sorted = time_grid[sort_idx]
    S_sorted = S_matrix[:, sort_idx]

    def _make_fn(row: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        def surv_fn(t: np.ndarray) -> np.ndarray:
            t = np.asarray(t, dtype=float)
            scalar = t.ndim == 0
            t = np.atleast_1d(t)
            result = np.clip(
                np.interp(t, time_sorted, row, left=row[0], right=0.0),
                0.0,
                1.0,
            )
            return float(result[0]) if scalar else result
        return surv_fn

    return [_make_fn(S_sorted[i]) for i in range(S_sorted.shape[0])]
