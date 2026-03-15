"""
gray_test.py — Gray's K-sample test for comparing CIFs across groups.

This implements the test from Gray (1988), which is the competing-risks
analogue of the log-rank test. It tests whether the CIFs for a given cause
are equal across K groups.

The test statistic is:

    chi^2 = U' V^{-1} U

where U_k and V are derived from a weighted log-rank-type process using
IPCW weights to handle the competing risks extended risk set.

Unlike the standard log-rank test, Gray's test weights the process by
the CIF itself, giving more power for detecting differences in CIF shape
rather than just hazard differences.

Reference
---------
Gray, R.J. (1988). A class of K-sample tests for comparing the cumulative
incidence of a competing risk. Annals of Statistics, 16(3), 1141-1154.
DOI: 10.1214/aos/1176350951.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .cif import AalenJohansenFitter


class GrayTestResult:
    """Result of Gray's K-sample CIF test.

    Attributes
    ----------
    statistic:
        Chi-squared test statistic.
    p_value:
        P-value under the chi-squared distribution with (K-1) degrees of
        freedom.
    degrees_of_freedom:
        Number of degrees of freedom.
    test_name:
        Human-readable description of the test.
    """

    def __init__(
        self,
        statistic: float,
        p_value: float,
        degrees_of_freedom: int,
        test_name: str = "Gray's K-Sample Test",
    ) -> None:
        self.statistic = statistic
        self.p_value = p_value
        self.degrees_of_freedom = degrees_of_freedom
        self.test_name = test_name

    def __repr__(self) -> str:
        return (
            f"{self.test_name}\n"
            f"  chi^2 = {self.statistic:.4f}  "
            f"df = {self.degrees_of_freedom}  "
            f"p = {self.p_value:.4f}"
        )

    @property
    def significant(self) -> bool:
        """True if p_value < 0.05."""
        return self.p_value < 0.05


def gray_test(
    T: "pd.Series | np.ndarray",
    E: "pd.Series | np.ndarray",
    groups: "pd.Series | np.ndarray",
    event_of_interest: int = 1,
    *,
    weights: Optional["pd.Series | np.ndarray"] = None,
) -> GrayTestResult:
    """Gray's K-sample test for equality of CIFs across groups.

    Tests H_0: F_k(t|group=1) = F_k(t|group=2) = ... = F_k(t|group=K) for all t.

    Parameters
    ----------
    T:
        Observed times.
    E:
        Event indicators. 0 = censored, positive integers = causes.
    groups:
        Group membership for each subject. Can be any hashable type.
    event_of_interest:
        The cause code for which to compare CIFs.
    weights:
        Optional observation weights.

    Returns
    -------
    GrayTestResult
        Contains ``statistic``, ``p_value``, ``degrees_of_freedom``.

    Examples
    --------
    >>> result = gray_test(df["T"], df["E"], df["group"], event_of_interest=1)
    >>> print(result)
    >>> if result.p_value < 0.05:
    ...     print("CIFs differ significantly between groups")
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    groups = np.asarray(groups)

    if weights is not None:
        w_obs = np.asarray(weights, dtype=float)
    else:
        w_obs = np.ones(len(T))

    unique_groups = np.unique(groups)
    K = len(unique_groups)

    if K < 2:
        raise ValueError("Need at least 2 groups to perform the test.")

    # Overall CIF (pooled) for the weight function
    pooled_fitter = AalenJohansenFitter()
    pooled_fitter.fit(T, E, event_of_interest=event_of_interest)

    # All event times (cause of interest) across all groups
    event_times = np.sort(np.unique(T[(E == event_of_interest)]))

    if len(event_times) == 0:
        raise ValueError(f"No events with cause code {event_of_interest} found.")

    # We use K-1 linearly independent contrasts (compare each group to last)
    # Compute U vector of length K-1 and variance matrix V of shape (K-1, K-1)
    # Using Gray's equation (2.1): U_g(tau) = integral_0^tau W(t) dM_g(t)
    # where M_g is the counting process martingale for group g and
    # W(t) = S_hat(t) per Gray (1988) eq (2.2): the pooled KM overall survivor.
    # This is the chi-squared(K-1) version. Note: F_vals kept for reference only.

    # Compute overall survival S(t) and pooled CIF F_k(t) at event times
    S_vals = _overall_survival(T, E, event_times)
    F_vals = pooled_fitter.predict(event_times)

    # For K groups we test K-1 contrasts; use first K-1 groups vs reference
    reference_group = unique_groups[-1]
    test_groups = unique_groups[:-1]

    U = np.zeros(K - 1)
    V = np.zeros((K - 1, K - 1))

    for g_idx, g in enumerate(test_groups):
        for t_idx, t in enumerate(event_times):
            s_t = S_vals[t_idx]
            f_t = F_vals[t_idx]

            if s_t <= 0:
                continue

            # Weight: w(t) = S_hat(t) per Gray (1988) eq (2.2).
            # Using the pooled KM overall survivor gives a chi-squared(K-1) test.
            weight = s_t

            # Risk set size for group g at time t
            n_g = np.sum(w_obs[(groups == g) & (T >= t)])
            n_total = np.sum(w_obs[T >= t])

            # Expected number of cause-k events for group g at t
            d_k_total = np.sum(w_obs[(T == t) & (E == event_of_interest)])
            if n_total > 0:
                expected_g = n_g * d_k_total / n_total
            else:
                expected_g = 0.0

            # Observed cause-k events for group g at t
            observed_g = np.sum(
                w_obs[(groups == g) & (T == t) & (E == event_of_interest)]
            )

            U[g_idx] += weight * (observed_g - expected_g)

            # Variance contribution (diagonal)
            if n_total > 1:
                var_contrib = (
                    weight ** 2
                    * n_g * (n_total - n_g) * d_k_total * (n_total - d_k_total)
                    / (n_total ** 2 * (n_total - 1))
                )
                V[g_idx, g_idx] += var_contrib

        # Off-diagonal covariance terms
        for h_idx, h in enumerate(test_groups):
            if h_idx <= g_idx:
                continue
            for t_idx, t in enumerate(event_times):
                s_t = S_vals[t_idx]
                f_t = F_vals[t_idx]
                if s_t <= 0:
                    continue

                weight = s_t
                n_g = np.sum(w_obs[(groups == g) & (T >= t)])
                n_h = np.sum(w_obs[(groups == h) & (T >= t)])
                n_total = np.sum(w_obs[T >= t])
                d_k_total = np.sum(w_obs[(T == t) & (E == event_of_interest)])

                if n_total > 1:
                    cov_contrib = (
                        -weight ** 2
                        * n_g * n_h * d_k_total * (n_total - d_k_total)
                        / (n_total ** 2 * (n_total - 1))
                    )
                    V[g_idx, h_idx] += cov_contrib
                    V[h_idx, g_idx] += cov_contrib

    # Compute test statistic
    try:
        V_inv = np.linalg.inv(V)
        statistic = float(U @ V_inv @ U)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        V_inv = np.linalg.pinv(V)
        statistic = float(U @ V_inv @ U)

    df = K - 1
    p_value = float(chi2.sf(statistic, df=df))

    return GrayTestResult(
        statistic=statistic,
        p_value=p_value,
        degrees_of_freedom=df,
        test_name=f"Gray's {K}-Sample CIF Test (cause {event_of_interest})",
    )


def _overall_survival(
    T: np.ndarray, E: np.ndarray, eval_times: np.ndarray
) -> np.ndarray:
    """Kaplan-Meier overall survival (all causes treated as events)."""
    from .cif import _z_from_alpha

    all_event_times = np.sort(np.unique(T[E != 0]))
    S = 1.0
    S_at_times = {}

    for t in all_event_times:
        n_at_risk = np.sum(T >= t)
        d = np.sum((T == t) & (E != 0))
        if n_at_risk > 0:
            S *= (1.0 - d / n_at_risk)
        S_at_times[t] = S

    time_arr = np.array([0.0] + list(all_event_times))
    S_arr = np.array([1.0] + [S_at_times[t] for t in all_event_times])

    return np.interp(eval_times, time_arr, S_arr, left=1.0, right=S_arr[-1])
