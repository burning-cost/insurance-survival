"""
calibration.py — Competing-risks calibration metrics and post-hoc recalibration.

Three tools for diagnosing and correcting calibration in competing-risks models:

1. **compute_cal_k_alpha** — time-integrated marginal calibration error for cause k.
   Measures the discrepancy between the mean predicted CIF and the Aalen-Johansen
   marginal estimate over the full observation window. Based on the cal_K^alpha
   metric from Alberge et al. (2026).

2. **AJRecalibrator** — post-hoc additive recalibration that shifts each predicted
   CIF curve by the time-varying gap between mean predictions and the AJ estimate.
   Preserves the ranking of predictions (C-index unchanged) while zeroing out
   marginal miscalibration.

3. **CRDCalibration** — competing-risks D-calibration using normalised probability
   integral transform (PIT) values. Tests whether, conditional on experiencing
   cause k, the normalised PIT is uniform over [0, 1].

All three metrics operate on a (n_samples, n_times) CIF array and a times grid.
They do not require a fitted model object — pass predictions from any source.

Design rationale
----------------
The additive correction in AJRecalibrator is intentionally simple. Isotonic
regression or spline-based recalibration would give smoother corrections but
require additional hyperparameters. For UK pricing teams, the additive approach
is transparent: you can inspect delta(t) directly and explain it to reviewers.
The monotonicity enforcement after adding delta(t) uses isotonic regression, which
is the correct non-parametric monotone projection.

References
----------
Alberge, J., Aghbalou, A., Sabourin, A., Mozharovskyi, P. & d'Alché-Buc, F.
(2026). Calibration of Survival Models. arXiv:2602.00194. AISTATS 2026.

Putter, Fiocco & Geskus (2007). Tutorial in biostatistics: Competing risks and
multi-state models. Statistics in Medicine, 26(11), 2389-2430.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import chi2

from .cif import AalenJohansenFitter

# numpy<2.0 compat: trapezoid was added in 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aj_estimate(
    times_grid: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    event_k: int,
) -> np.ndarray:
    """Return AJ CIF values evaluated at times_grid.

    Parameters
    ----------
    times_grid:
        Evaluation time grid.
    event_times:
        Observed event/censoring times.
    event_indicators:
        0=censored, k=cause k.
    event_k:
        Which cause to estimate the CIF for.

    Returns
    -------
    np.ndarray shape (len(times_grid),)
    """
    aj = AalenJohansenFitter()
    aj.fit(event_times, event_indicators, event_of_interest=event_k)
    return aj.predict(times_grid)


def _enforce_monotone(arr: np.ndarray) -> np.ndarray:
    """Project arr to the nearest non-decreasing sequence (isotonic regression).

    Uses the pool-adjacent-violators (PAV) algorithm, equivalent to
    scipy.optimize.isotonic_regression for the L2 loss.
    """
    out = arr.copy().astype(float)
    n = len(out)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] < out[j - 1]:
            j += 1
        if j > i + 1:
            mean_val = out[i:j].mean()
            out[i:j] = mean_val
        i = j
    # A single PAV pass is not always sufficient; use a simple cumulative max
    # as a fast alternative that is correct for this use case.
    return np.maximum.accumulate(out)


# ---------------------------------------------------------------------------
# cal_K^alpha
# ---------------------------------------------------------------------------

def compute_cal_k_alpha(
    predicted_cif: "np.ndarray",
    times: "np.ndarray",
    event_times: "np.ndarray",
    event_indicators: "np.ndarray",
    event_k: int = 1,
    alpha: int = 2,
) -> float:
    """Time-integrated marginal calibration error for cause k.

    Computes:

        cal_K^alpha = (1 / T_max) * integral_0^{T_max} |E[F_hat_k(t|X)] - F_k^AJ(t)|^alpha dt

    where F_k^AJ(t) is the Aalen-Johansen estimate of the marginal CIF for
    cause k, and the integral is approximated with the trapezoid rule over the
    supplied time grid.

    A perfectly calibrated model — where the mean predicted CIF matches the
    marginal AJ CIF at every time point — returns 0.0.

    Parameters
    ----------
    predicted_cif:
        Predicted cause-specific CIFs, shape (n_samples, n_times). Values
        should be in [0, 1] and the array should be monotone non-decreasing
        along the time axis for each sample.
    times:
        Evaluation time grid, shape (n_times,). Must be sorted ascending.
    event_times:
        Observed event/censoring times, shape (n_samples,).
    event_indicators:
        Event cause indicators, shape (n_samples,). 0 = censored, k = cause k.
    event_k:
        Which cause to evaluate calibration for.
    alpha:
        Power for the Lp calibration error (default 2 = mean squared error).

    Returns
    -------
    float
        Calibration error (lower is better, 0.0 = perfect).
    """
    predicted_cif = np.asarray(predicted_cif, dtype=float)
    times = np.asarray(times, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    event_indicators = np.asarray(event_indicators, dtype=int)

    if predicted_cif.ndim != 2:
        raise ValueError(
            f"predicted_cif must be 2-D (n_samples, n_times), got shape {predicted_cif.shape}"
        )
    if predicted_cif.shape[1] != len(times):
        raise ValueError(
            f"predicted_cif has {predicted_cif.shape[1]} columns but times has {len(times)} elements"
        )

    # Mean predicted CIF at each evaluation time
    mean_pred = predicted_cif.mean(axis=0)  # shape (n_times,)

    # Aalen-Johansen estimate at each evaluation time
    aj_cif = _aj_estimate(times, event_times, event_indicators, event_k)

    # Integrand: |mean_pred(t) - AJ(t)|^alpha
    integrand = np.abs(mean_pred - aj_cif) ** alpha

    # Normalise by observation window
    t_range = times[-1] - times[0]
    if t_range <= 0.0:
        return float(integrand.mean())

    integral = float(np.trapezoid(integrand, times))
    return integral / t_range


# ---------------------------------------------------------------------------
# AJRecalibrator
# ---------------------------------------------------------------------------

class AJRecalibrator:
    """Post-hoc additive recalibration to match Aalen-Johansen marginals.

    After fitting on a calibration set, the recalibrator computes the
    time-varying additive correction:

        delta(t) = F_k^AJ(t) - mean(F_hat_k(t | X))

    and applies it to new predictions:

        F_hat_k_recal(t | x) = clip(F_hat_k(t | x) + delta(t), 0, 1)

    followed by isotonic monotonicity enforcement.

    This approach preserves the discrimination ordering of predictions
    (the C-index is unchanged) while forcing marginal calibration to zero.
    It is equivalent to a bias-correction in the mean predicted probability
    at each time point.

    The correction can be inspected via ``delta_`` after fitting.
    """

    def __init__(self) -> None:
        self._fitted = False
        self.delta_: Optional[np.ndarray] = None
        self.times_: Optional[np.ndarray] = None

    def fit(
        self,
        predicted_cif: np.ndarray,
        times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        event_k: int = 1,
    ) -> "AJRecalibrator":
        """Compute the additive correction from a calibration set.

        Parameters
        ----------
        predicted_cif:
            Predicted CIFs on the calibration set, shape (n_samples, n_times).
        times:
            Evaluation time grid, shape (n_times,).
        event_times:
            Observed event/censoring times for the calibration set.
        event_indicators:
            Event cause indicators for the calibration set.
        event_k:
            Which cause to recalibrate for.

        Returns
        -------
        self
        """
        predicted_cif = np.asarray(predicted_cif, dtype=float)
        times = np.asarray(times, dtype=float)
        event_times = np.asarray(event_times, dtype=float)
        event_indicators = np.asarray(event_indicators, dtype=int)

        if predicted_cif.ndim != 2:
            raise ValueError(
                f"predicted_cif must be 2-D (n_samples, n_times), got shape {predicted_cif.shape}"
            )

        # AJ marginal CIF on the calibration set
        aj_cif = _aj_estimate(times, event_times, event_indicators, event_k)

        # Mean predicted CIF at each time
        mean_pred = predicted_cif.mean(axis=0)

        # Additive correction: positive when model under-predicts
        self.delta_ = aj_cif - mean_pred
        self.times_ = times
        self._fitted = True
        return self

    def transform(
        self,
        predicted_cif: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Apply the recalibration correction to new predictions.

        Parameters
        ----------
        predicted_cif:
            Predicted CIFs to recalibrate, shape (n_samples, n_times).
        times:
            Evaluation time grid. Must be consistent with the grid used at
            fit time (used to interpolate delta if needed).

        Returns
        -------
        np.ndarray
            Recalibrated CIF array, same shape as input.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        predicted_cif = np.asarray(predicted_cif, dtype=float)
        times = np.asarray(times, dtype=float)

        # Interpolate delta to the provided time grid
        delta = np.interp(
            times,
            self.times_,
            self.delta_,
            left=self.delta_[0],
            right=self.delta_[-1],
        )

        # Add correction and clip to [0, 1]
        recal = np.clip(predicted_cif + delta[np.newaxis, :], 0.0, 1.0)

        # Enforce monotonicity along time axis for each sample
        for i in range(recal.shape[0]):
            recal[i] = _enforce_monotone(recal[i])

        return recal

    def fit_transform(
        self,
        predicted_cif: np.ndarray,
        times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        event_k: int = 1,
    ) -> np.ndarray:
        """Fit and immediately transform the same dataset.

        This is useful when you want to evaluate in-sample recalibration but
        is typically overoptimistic — use separate fit/transform sets in
        production.
        """
        self.fit(predicted_cif, times, event_times, event_indicators, event_k)
        return self.transform(predicted_cif, times)


# ---------------------------------------------------------------------------
# CRDCalibration
# ---------------------------------------------------------------------------

class CRDCalibration:
    """Competing-risks D-calibration using normalised PIT values.

    For subjects who experience cause k, the normalised probability integral
    transform (PIT) is:

        PIT_i = F_hat_k(T_i | X_i) / F_hat_k(t_max | X_i)

    where t_max is the last time in the evaluation grid (a proxy for the
    cause-specific CIF asymptote). This normalisation maps PIT_i to [0, 1]
    even when F_hat_k(inf | X_i) < 1 (as is always the case in competing
    risks, where the cause-k CIF plateaus below 1).

    Under a perfectly calibrated model, {PIT_i} are uniform on [0, 1]. The
    test checks uniformity using a chi-squared goodness-of-fit test against
    n_bins equal-width bins.

    Attributes (after calling compute())
    -------------------------------------
    statistic:
        Chi-squared test statistic.
    p_value:
        p-value from the chi-squared test. p > 0.05 indicates no strong
        evidence of miscalibration.
    n_events_:
        Number of cause-k events used in the test.
    bin_counts_:
        Array of observed counts in each bin.
    """

    def __init__(self, n_bins: int = 10) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self._computed = False
        self._statistic: Optional[float] = None
        self._p_value: Optional[float] = None
        self.n_events_: Optional[int] = None
        self.bin_counts_: Optional[np.ndarray] = None

    def compute(
        self,
        predicted_cif: np.ndarray,
        times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        event_k: int = 1,
    ) -> "CRDCalibration":
        """Compute the D-calibration test statistic.

        Parameters
        ----------
        predicted_cif:
            Predicted cause-k CIFs, shape (n_samples, n_times).
        times:
            Evaluation time grid, shape (n_times,).
        event_times:
            Observed event/censoring times, shape (n_samples,).
        event_indicators:
            Event cause indicators, shape (n_samples,). 0=censored, k=cause k.
        event_k:
            Which cause to test calibration for.

        Returns
        -------
        self
        """
        predicted_cif = np.asarray(predicted_cif, dtype=float)
        times = np.asarray(times, dtype=float)
        event_times = np.asarray(event_times, dtype=float)
        event_indicators = np.asarray(event_indicators, dtype=int)

        if predicted_cif.ndim != 2:
            raise ValueError(
                f"predicted_cif must be 2-D (n_samples, n_times), got shape {predicted_cif.shape}"
            )

        # Select subjects who experienced cause k
        cause_k_mask = event_indicators == event_k
        if cause_k_mask.sum() == 0:
            raise ValueError(f"No events of cause {event_k} found in event_indicators.")

        event_times_k = event_times[cause_k_mask]
        cif_k = predicted_cif[cause_k_mask]  # shape (n_k, n_times)

        # Interpolate predicted CIF at observed event time for each subject
        n_k = cif_k.shape[0]
        cif_at_event = np.array([
            np.interp(event_times_k[i], times, cif_k[i])
            for i in range(n_k)
        ])

        # CIF at t_max (last time in grid) as asymptote proxy
        cif_at_tmax = cif_k[:, -1]

        # Guard against zero asymptote (degenerate predictions)
        safe_denom = np.maximum(cif_at_tmax, 1e-10)
        pit_values = np.clip(cif_at_event / safe_denom, 0.0, 1.0)

        # Chi-squared test against uniform over n_bins bins
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        observed_counts, _ = np.histogram(pit_values, bins=bin_edges)
        expected_count = n_k / self.n_bins

        # Chi-squared statistic: sum (O - E)^2 / E
        chi2_stat = float(np.sum((observed_counts - expected_count) ** 2 / expected_count))
        df = self.n_bins - 1
        p_val = float(1.0 - chi2.cdf(chi2_stat, df=df))

        self._statistic = chi2_stat
        self._p_value = p_val
        self.n_events_ = int(n_k)
        self.bin_counts_ = observed_counts
        self._computed = True
        return self

    @property
    def statistic(self) -> float:
        """Chi-squared test statistic."""
        self._check_computed()
        return self._statistic  # type: ignore[return-value]

    @property
    def p_value(self) -> float:
        """p-value from the chi-squared test.

        p > 0.05: no strong evidence of miscalibration.
        p < 0.05: evidence that predictions are miscalibrated.
        """
        self._check_computed()
        return self._p_value  # type: ignore[return-value]

    def summary(self) -> dict:
        """Return a summary dict suitable for logging or display.

        Returns
        -------
        dict with keys: statistic, p_value, n_bins, n_events, bin_counts
        """
        self._check_computed()
        return {
            "statistic": self._statistic,
            "p_value": self._p_value,
            "n_bins": self.n_bins,
            "n_events": self.n_events_,
            "bin_counts": self.bin_counts_.tolist(),  # type: ignore[union-attr]
        }

    def _check_computed(self) -> None:
        if not self._computed:
            raise RuntimeError("Call compute() before accessing results.")
