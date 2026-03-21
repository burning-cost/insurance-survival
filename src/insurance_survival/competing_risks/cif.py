"""
cif.py — Non-parametric cumulative incidence function (CIF) estimation.

Implements the Aalen-Johansen estimator for the CIF in a competing-risks
setting. This is the non-parametric baseline — before fitting any regression
model, you should plot the crude CIFs to understand the marginal event
probabilities.

Theory
------
With K competing event types, the CIF for cause k at time t is:

    F_k(t) = sum_{t_i <= t} [d_{k,i} / n_i] * S(t_{i-})

where:
- d_{k,i} = number of cause-k events at time t_i
- n_i = size of the total risk set just before t_i
- S(t) = Kaplan-Meier overall survival function (treating all events as events)

This is the Aalen-Johansen estimator, equivalent to a special case of the
product-limit estimator for multi-state models.

The confidence intervals use the delta method on the log(-log) scale, which
gives better coverage near 0 and 1 than the naive Wald interval.

References
----------
Putter, Fiocco & Geskus (2007). Tutorial in biostatistics: Competing risks and
multi-state models. Statistics in Medicine, 26(11), 2389-2430.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


class AalenJohansenFitter:
    """Non-parametric CIF estimator using the Aalen-Johansen method.

    This mirrors the API of ``lifelines.AalenJohansenFitter`` but is
    implemented from scratch so the library has no runtime dependency on
    lifelines for the core estimator.

    Parameters
    ----------
    alpha:
        Significance level for confidence intervals (default 0.05 → 95% CIs).

    Examples
    --------
    >>> from insurance_competing_risks import AalenJohansenFitter
    >>> aj = AalenJohansenFitter()
    >>> aj.fit(df["T"], df["E"], event_of_interest=1)
    >>> aj.plot()
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        T: "pd.Series | np.ndarray",
        E: "pd.Series | np.ndarray",
        event_of_interest: int = 1,
        *,
        label: Optional[str] = None,
        weights: Optional["pd.Series | np.ndarray"] = None,
    ) -> "AalenJohansenFitter":
        """Fit the Aalen-Johansen CIF estimator.

        Parameters
        ----------
        T:
            Observed times (non-negative).
        E:
            Event indicators. 0 = censored, any positive integer = an event
            cause. All non-zero, non-``event_of_interest`` values are treated
            as competing events.
        event_of_interest:
            The cause code to estimate the CIF for.
        label:
            Optional label used in plot legends.
        weights:
            Optional observation weights. Must sum to n.

        Returns
        -------
        self
        """
        T = np.asarray(T, dtype=float)
        E = np.asarray(E, dtype=int)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
        else:
            weights = np.ones(len(T))

        self.event_of_interest = event_of_interest
        self.label_ = label or f"Cause {event_of_interest}"

        # Compute on unique event times (all causes)
        event_times = np.unique(T[E != 0])

        n_total = len(T)
        S_prev = 1.0  # overall survival just before current time
        cif_vals = []
        var_vals = []
        times_out = []

        # Greenwood-type variance accumulator
        cumulative_var = 0.0

        for t in event_times:
            # Risk set at t (all subjects with T >= t)
            n_at_risk = np.sum(weights[T >= t])
            if n_at_risk == 0:
                continue

            # All-cause events at t
            d_all = np.sum(weights[(T == t) & (E != 0)])
            # Cause-k events at t
            d_k = np.sum(weights[(T == t) & (E == event_of_interest)])

            if d_all == 0:
                continue

            # Aalen-Johansen increment: dF_k(t) = (d_k / n) * S(t-)
            increment = (d_k / n_at_risk) * S_prev
            cif_prev = sum(cif_vals) if cif_vals else 0.0
            cif_vals.append(increment)
            new_cif = cif_prev + increment

            # Variance via proper Aalen-Johansen delta method (Putter et al. 2007 eq 3)
            # Var(F_k(t)) = sum_{s<=t} S(s-)^2 * [d_k(s)/n(s)^2 * (1 - d_k(s)/n(s))
            #              + (F_k(t) - F_k(s-))^2 * d_all(s)/(n(s)*(n(s)-d_all(s)))]
            # Separates cause-k variance from all-cause variance contribution.
            if n_at_risk > 1:
                # Term 1: cause-k specific variance contribution
                cause_k_contrib = (S_prev ** 2) * (d_k / n_at_risk ** 2) * (1.0 - d_k / n_at_risk)
                # Term 2: all-cause correction (cross term via overall survival)
                if n_at_risk > d_all:
                    all_cause_contrib = (S_prev ** 2) * ((new_cif - cif_prev) ** 2) * (d_all / (n_at_risk * (n_at_risk - d_all)))
                else:
                    all_cause_contrib = 0.0
                cumulative_var += cause_k_contrib + all_cause_contrib
            var_vals.append(cumulative_var)

            times_out.append(t)

            # Update overall survival
            S_prev *= (1.0 - d_all / n_at_risk)

        self.times_ = np.array([0.0] + times_out)
        raw_cif = np.cumsum([0.0] + list(cif_vals))
        # Clip to valid probability range
        self.cumulative_incidence_ = pd.Series(
            np.clip(raw_cif, 0.0, 1.0),
            index=self.times_,
            name=self.label_,
        )

        # Confidence intervals on log(-log) scale for better coverage
        vars_with_zero = [0.0] + list(var_vals)
        se = np.sqrt(np.array(vars_with_zero))
        z = _z_from_alpha(self.alpha)

        cif_arr = np.clip(raw_cif, 1e-10, 1.0 - 1e-10)
        # log(-log(F_k)) transform: correct for CIF (not a survival function).
        # A CIF F_k(t) approaches 1, so we use log(-log(F_k)) which maps (0,1) -> R.
        # The back-transform is F_k = exp(-exp(theta)) — increasing in theta.
        # SE on transformed scale: se / (F_k * |log(F_k)|)  [delta method]
        log_neg_log_cif = np.log(-np.log(cif_arr))
        se_transformed = se / (cif_arr * np.abs(np.log(cif_arr)))
        ll = np.exp(-np.exp(log_neg_log_cif + z * se_transformed))
        ul = np.exp(-np.exp(log_neg_log_cif - z * se_transformed))

        # At t=0 CIF is exactly 0, CI collapses
        ll[0] = 0.0
        ul[0] = 0.0

        self.confidence_intervals_ = pd.DataFrame(
            {"lower": np.clip(ll, 0.0, 1.0), "upper": np.clip(ul, 0.0, 1.0)},
            index=self.times_,
        )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, times: "Sequence[float] | np.ndarray") -> np.ndarray:
        """Return interpolated CIF values at given times.

        Parameters
        ----------
        times:
            Times at which to evaluate the CIF.

        Returns
        -------
        np.ndarray
            CIF values, same length as ``times``.
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        return np.interp(
            times,
            self.times_,
            self.cumulative_incidence_.values,
            left=0.0,
            right=self.cumulative_incidence_.values[-1],
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax: Optional[Any] = None,
        ci: bool = True,
        **kwargs,
    ) -> Any:
        """Plot the CIF with optional confidence band.

        Requires matplotlib. Install with: pip install insurance-survival[plot]

        Parameters
        ----------
        ax:
            Matplotlib axes. Created if not provided.
        ci:
            Whether to shade the confidence interval.
        **kwargs:
            Passed to ``ax.step()``.

        Returns
        -------
        plt.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-survival[plot]"
            )

        self._check_fitted()
        if ax is None:
            _, ax = plt.subplots()

        default_kwargs = {"label": self.label_, "where": "post"}
        default_kwargs.update(kwargs)

        ax.step(self.times_, self.cumulative_incidence_.values, **default_kwargs)
        if ci:
            colour = ax.lines[-1].get_color()
            ax.fill_between(
                self.times_,
                self.confidence_intervals_["lower"].values,
                self.confidence_intervals_["upper"].values,
                alpha=0.2,
                color=colour,
                step="post",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative Incidence")
        ax.set_ylim(0, None)
        return ax

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")


# ---------------------------------------------------------------------------
# Stacked CIF plot (all causes together)
# ---------------------------------------------------------------------------

def plot_stacked_cif(
    T: "pd.Series | np.ndarray",
    E: "pd.Series | np.ndarray",
    causes: Optional[list[int]] = None,
    cause_labels: Optional[dict[int, str]] = None,
    ax: Optional[Any] = None,
    times: Optional[np.ndarray] = None,
) -> Any:
    """Plot stacked cumulative incidence functions for all causes.

    The stacked plot shows, at each time point, the probability of each cause
    having occurred. The total height equals the overall event probability
    (1 - overall survival).

    Requires matplotlib. Install with: pip install insurance-survival[plot]

    Parameters
    ----------
    T:
        Observed times.
    E:
        Event indicators (0 = censored).
    causes:
        List of event cause codes to include. Defaults to all non-zero codes.
    cause_labels:
        Optional mapping from cause code to display label.
    ax:
        Matplotlib axes.
    times:
        Time grid for interpolation. Defaults to observed event times.

    Returns
    -------
    plt.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install insurance-survival[plot]"
        )

    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)

    if causes is None:
        causes = sorted(c for c in np.unique(E) if c != 0)

    if cause_labels is None:
        cause_labels = {c: f"Cause {c}" for c in causes}

    if times is None:
        times = np.unique(T[E != 0])
        times = np.concatenate([[0.0], times])

    fitters = {}
    for cause in causes:
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=cause, label=cause_labels.get(cause, str(cause)))
        fitters[cause] = fitter

    if ax is None:
        _, ax = plt.subplots()

    # Stack the CIFs
    bottom = np.zeros(len(times))
    for cause in causes:
        cif = fitters[cause].predict(times)
        ax.fill_between(
            times,
            bottom,
            bottom + cif,
            label=cause_labels.get(cause, str(cause)),
            alpha=0.7,
            step="post",
        )
        bottom += cif

    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Incidence")
    ax.set_ylim(0, 1)
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _z_from_alpha(alpha: float) -> float:
    """Return the z-score for a two-sided interval at significance ``alpha``."""
    from scipy.stats import norm
    return norm.ppf(1.0 - alpha / 2.0)
