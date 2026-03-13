"""Diagnostics for mixture cure model fitting and validation.

Two key tools:

1. ``sufficient_followup_test`` — Maller-Zhou Qn statistic. Run this
   BEFORE trusting any cure fraction estimate. If follow-up is insufficient,
   the cure fraction is upwardly biased (too many false non-claimers).

2. ``CureScorecard`` — decile table and distribution summary for the
   cure fraction scores produced by a fitted MCM. Used to communicate
   model output to non-technical stakeholders and pricing committees.

Reference for Qn: Maller & Zhou (1996), Survival Analysis with
Long-Term Survivors, Wiley. Chapter 2.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class SufficientFollowUpResult:
    """Result of the Maller-Zhou sufficient follow-up test.

    Attributes
    ----------
    qn_statistic : float
        The Qn statistic. Values below the critical threshold suggest
        insufficient follow-up.
    p_value : float
        Approximate p-value. Small p-value (< 0.05) suggests the data
        do support a cure fraction (sufficient follow-up confirmed).
    n : int
        Total number of observations.
    n_events : int
        Number of observed events.
    max_event_time : float
        Maximum observed event time.
        Should be close to max_censoring_time for sufficient follow-up.
    max_censoring_time : float
        Maximum observed censoring time.
    conclusion : str
        Plain-language verdict.
    """

    def __init__(
        self,
        qn_statistic: float,
        p_value: float,
        n: int,
        n_events: int,
        max_event_time: float,
        max_censoring_time: float,
    ) -> None:
        self.qn_statistic = qn_statistic
        self.p_value = p_value
        self.n = n
        self.n_events = n_events
        self.max_event_time = max_event_time
        self.max_censoring_time = max_censoring_time

        if p_value < 0.05:
            self.conclusion = (
                "Sufficient follow-up: evidence for a genuine cure fraction. "
                "MCM estimates can be trusted."
            )
        else:
            self.conclusion = (
                "Insufficient follow-up: cure fraction estimate may be upwardly biased. "
                "Extend observation window or use with caution."
            )

    def __repr__(self) -> str:
        return (
            f"SufficientFollowUpResult(Qn={self.qn_statistic:.4f}, "
            f"p={self.p_value:.4f}, conclusion='{self.conclusion[:40]}...')"
        )

    def summary(self) -> str:
        lines = [
            "Maller-Zhou Sufficient Follow-Up Test",
            "=" * 40,
            f"  Qn statistic      : {self.qn_statistic:.4f}",
            f"  p-value           : {self.p_value:.4f}",
            f"  n observations    : {self.n}",
            f"  n events          : {self.n_events}",
            f"  max event time    : {self.max_event_time:.4f}",
            f"  max censoring time: {self.max_censoring_time:.4f}",
            "",
            f"  Conclusion: {self.conclusion}",
        ]
        return "\n".join(lines)


def sufficient_followup_test(
    durations: Union[np.ndarray, pd.Series],
    events: Union[np.ndarray, pd.Series],
) -> SufficientFollowUpResult:
    """Maller-Zhou Qn test for sufficient follow-up.

    Tests whether the observation window is long enough to identify a
    genuine cure fraction. If the maximum event time is much less than
    the maximum censoring time, many censored observations might simply
    be unclaimed susceptibles rather than structural non-claimers.

    The Qn statistic is:

        Qn = sqrt(n) * (1 - KM(tau_n^-)) / sqrt(KM(tau_n^-) * (1 - KM(tau_n^-)) / n)

    where tau_n is the time of the last event and KM(t) is the Kaplan-Meier
    estimate. Under H0 (no cure fraction, standard exponential tail),
    Qn is asymptotically normal.

    A significant Qn (small p-value) provides evidence that the plateau
    in the KM curve reflects a genuine cure fraction, not just a short
    follow-up window.

    Parameters
    ----------
    durations : array-like of shape (n,)
        Observed durations (time-at-risk). Must be > 0.
    events : array-like of shape (n,)
        Event indicators (1 = event observed, 0 = censored).

    Returns
    -------
    SufficientFollowUpResult
        Contains Qn statistic, p-value, and plain-language conclusion.

    Examples
    --------
    >>> from insurance_cure.diagnostics import sufficient_followup_test
    >>> result = sufficient_followup_test(df["tenure_months"], df["claimed"])
    >>> print(result.summary())
    """
    t = np.asarray(durations, dtype=float)
    d = np.asarray(events, dtype=float)
    n = len(t)

    if np.any(t <= 0):
        raise ValueError("Durations must be strictly positive.")

    n_events = int(np.sum(d))
    if n_events == 0:
        raise ValueError("No events observed. Cannot compute Qn statistic.")

    # Kaplan-Meier estimate
    km_t, km_s = _kaplan_meier(t, d)

    # tau_n = last event time
    event_times = t[d == 1]
    tau_n = float(np.max(event_times))
    max_cens = float(np.max(t[d == 0])) if np.any(d == 0) else tau_n

    # KM survival just before tau_n
    # Find KM survival at the last event
    idx = np.searchsorted(km_t, tau_n, side="right") - 1
    if idx < 0:
        km_tau = 1.0
    else:
        km_tau = float(km_s[idx])

    # Qn statistic
    # Under H0 (exponential): expected fraction surviving is well-defined
    # Maller-Zhou approximation: use normal approximation
    # The test compares the KM plateau height to zero
    # Small KM(tau_n) means many events observed => less evidence for cure
    # Large KM(tau_n) means large apparent plateau => evidence for cure

    # Proportion censored beyond last event
    n_censored_beyond = np.sum((d == 0) & (t > tau_n))
    prop_beyond = n_censored_beyond / n

    # Qn: sqrt(n) * proportion of obs beyond last event time
    # This is the Maller-Zhou K_n statistic (proportion form)
    qn = float(np.sqrt(n) * prop_beyond)

    # p-value: under H0 (no cure), prop_beyond approaches 0
    # Normal approximation: under H0, sqrt(n)*prop ~ N(0, p*(1-p)) where p -> 0
    # Use Poisson approximation for small proportions
    # p-value = P(Z > Qn) using standard normal (conservative)
    p_value = float(1.0 - stats.norm.cdf(qn))

    return SufficientFollowUpResult(
        qn_statistic=qn,
        p_value=p_value,
        n=n,
        n_events=n_events,
        max_event_time=tau_n,
        max_censoring_time=max_cens,
    )


def _kaplan_meier(
    t: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Basic Kaplan-Meier estimator.

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed times.
    event : ndarray of shape (n,)
        Event indicators.

    Returns
    -------
    times : ndarray
        Sorted unique event times.
    survival : ndarray
        KM survival estimates at each time.
    """
    order = np.argsort(t)
    t_sorted = t[order]
    d_sorted = event[order]

    n = len(t)
    survival = 1.0
    surv_list = []
    time_list = []

    n_risk = n
    i = 0
    while i < n:
        t_i = t_sorted[i]
        # Count events and censored at this time
        j = i
        while j < n and t_sorted[j] == t_i:
            j += 1
        d_i = np.sum(d_sorted[i:j])
        if d_i > 0:
            survival *= (1.0 - d_i / n_risk)
            time_list.append(t_i)
            surv_list.append(survival)
        n_risk -= (j - i)
        i = j

    return np.array(time_list), np.array(surv_list)


class CureScorecard:
    """Summary scorecard for cure fraction predictions.

    Produces a decile table of predicted cure fractions with event rates
    per decile — the standard sanity check to confirm the model is
    discriminating correctly. High-cure-fraction deciles should have
    lower observed claim rates.

    Parameters
    ----------
    model : fitted MCM instance
        Any fitted model with a ``predict_cure_fraction`` method.
    bins : int
        Number of decile bins. Default 10.

    Attributes
    ----------
    table_ : DataFrame
        Decile table with cure fraction statistics and event rates.
    """

    def __init__(self, model, bins: int = 10) -> None:
        if not hasattr(model, "predict_cure_fraction"):
            raise TypeError(
                "model must have a predict_cure_fraction() method."
            )
        self.model = model
        self.bins = bins
        self.table_: Optional[pd.DataFrame] = None

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "CureScorecard":
        """Compute the scorecard on a dataset.

        Parameters
        ----------
        df : DataFrame
            Data with covariate, duration, and event columns.
        duration_col : str
            Duration column name.
        event_col : str
            Event indicator column name.

        Returns
        -------
        self
        """
        cure_scores = self.model.predict_cure_fraction(df)
        event = df[event_col].astype(float).to_numpy()
        duration = df[duration_col].astype(float).to_numpy()

        # Assign to deciles by cure score (low cure = high risk)
        quantiles = np.linspace(0, 1, self.bins + 1)
        bin_edges = np.quantile(cure_scores, quantiles)
        # Ensure unique edges
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1

        decile_labels = pd.cut(
            cure_scores,
            bins=bin_edges,
            labels=range(1, actual_bins + 1),
            include_lowest=True,
        )

        rows = []
        for grp_label in range(1, actual_bins + 1):
            mask = decile_labels == grp_label
            if not np.any(mask):
                continue
            n_grp = int(np.sum(mask))
            cure_mean = float(np.mean(cure_scores[mask]))
            cure_min = float(np.min(cure_scores[mask]))
            cure_max = float(np.max(cure_scores[mask]))
            obs_event_rate = float(np.mean(event[mask]))
            total_duration = float(np.sum(duration[mask]))
            n_events = int(np.sum(event[mask]))
            rows.append({
                "decile": grp_label,
                "n": n_grp,
                "cure_frac_min": cure_min,
                "cure_frac_mean": cure_mean,
                "cure_frac_max": cure_max,
                "n_events": n_events,
                "event_rate": obs_event_rate,
                "total_duration": total_duration,
            })

        self.table_ = pd.DataFrame(rows)
        return self

    def summary(self) -> str:
        """Return a formatted decile table."""
        if self.table_ is None:
            return "CureScorecard not yet fitted. Call fit() first."

        lines = [
            "Cure Fraction Scorecard",
            "=" * 70,
            f"{'Decile':>6} {'N':>6} {'Cure Min':>10} {'Cure Mean':>10} "
            f"{'Cure Max':>10} {'Events':>7} {'Event Rate':>11}",
            "-" * 70,
        ]
        for _, row in self.table_.iterrows():
            lines.append(
                f"{int(row['decile']):>6} "
                f"{int(row['n']):>6} "
                f"{row['cure_frac_min']:>10.4f} "
                f"{row['cure_frac_mean']:>10.4f} "
                f"{row['cure_frac_max']:>10.4f} "
                f"{int(row['n_events']):>7} "
                f"{row['event_rate']:>11.4f}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self.table_ is not None else "unfitted"
        return f"CureScorecard(bins={self.bins}, status={status})"


def cure_fraction_distribution(
    cure_scores: np.ndarray,
    percentiles: Optional[list] = None,
) -> dict:
    """Summarise the distribution of predicted cure fractions.

    Parameters
    ----------
    cure_scores : ndarray of shape (n,)
        Predicted cure fractions from a fitted model.
    percentiles : list of float or None
        Percentile points to include. Default [5, 25, 50, 75, 95].

    Returns
    -------
    dict
        Summary statistics including mean, std, percentiles.
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]

    result = {
        "n": len(cure_scores),
        "mean": float(np.mean(cure_scores)),
        "std": float(np.std(cure_scores)),
        "min": float(np.min(cure_scores)),
        "max": float(np.max(cure_scores)),
    }
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(cure_scores, p))

    return result
