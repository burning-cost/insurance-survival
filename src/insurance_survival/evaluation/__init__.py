"""
insurance_survival.evaluation — Proper scoring rules for right-censored
time-to-event forecasts.

Based on Taggart, Loveday & Louis (2026), arXiv:2603.14835.

This subpackage implements threshold-weighted CRPS (twCRPS) and related
scoring functions under right-censoring via provisional strict propriety.

The key gap filled: no existing Python library (pycox, scikit-survival,
lifelines, properscoring) provides proper scoring rules for predictive
survival distributions under censoring. The IPCW Brier score used by
scikit-survival and pycox is NOT a proper scoring rule for F(t).

Main class
----------
CensoredForecastEvaluator:
    Evaluation hub. Takes tau (horizon), alpha (quantile level), and
    optional weight function. Methods:

    .twcrps(surv_fns, T_obs, event)          -> float
    .twcrps_profile(surv_fns, T_obs, event)  -> pd.Series
    .quantile_score(q_forecasts, T_obs, event) -> float
    .interval_score(lower, upper, T_obs, event) -> float
    .compare(forecasters: dict)              -> pd.DataFrame
    .murphy_diagram(forecasters: dict)       -> Figure
    .reliability_diagram(surv_fns, T_obs, event, eval_time) -> Figure

Helper
------
from_matrix(S_matrix, time_grid):
    Converts a (n_samples, n_times) survival matrix into the list-of-callables
    format expected by CensoredForecastEvaluator. Use this to wrap output from
    pycox, scikit-survival, or any model that returns a matrix.

Provisional propriety caveat
-----------------------------
tau must be FIXED and IDENTICAL for all observations. With random per-subject
censoring times, rankings are heuristic (IPCW adjustment would be needed).
See the CensoredForecastEvaluator docstring and warn=True behaviour.

Typical usage
-------------
>>> import numpy as np
>>> from scipy.stats import weibull_min
>>> from insurance_survival.evaluation import CensoredForecastEvaluator

>>> # Suppose you have two survival models to compare
>>> n = 500
>>> rng = np.random.default_rng(0)
>>> T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
>>> tau = 8.0
>>> T_obs = np.minimum(T_true, tau)
>>> event = (T_true <= tau).astype(int)

>>> def make_surv(c, scale):
...     return lambda t: weibull_min.sf(np.asarray(t), c=c, scale=scale)

>>> true_fns = [make_surv(2, 5)] * n      # correct model
>>> wrong_fns = [make_surv(1, 5)] * n     # misspecified model

>>> ev = CensoredForecastEvaluator(tau=tau, warn=False)
>>> print(ev.compare({
...     "Weibull(2,5)": (true_fns, T_obs, event),
...     "Expo(5)":      (wrong_fns, T_obs, event),
... }))
"""

from ._evaluator import CensoredForecastEvaluator, from_matrix
from ._scoring import twcrps_obs, quantile_score, interval_score, murphy_elementary_score

__all__ = [
    "CensoredForecastEvaluator",
    "from_matrix",
    "twcrps_obs",
    "quantile_score",
    "interval_score",
    "murphy_elementary_score",
]
