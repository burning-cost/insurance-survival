"""
Core model fitters for recurrent insurance claims.

Three models are provided:

1. AndersenGillFrailty — the workhorse. Cox intensity model with shared gamma
   or lognormal frailty. EM algorithm: the E-step computes posterior frailty
   expectations; the M-step updates beta via weighted partial likelihood and
   theta via marginal likelihood.

2. PWPModel — Prentice-Williams-Peterson stratified by event number. Separate
   baseline hazard for 1st, 2nd, 3rd+ claim. No frailty. Good when the
   intensity genuinely changes with claim history (e.g., motor — NCD resets).

3. NelsonAalenFrailty — non-parametric baseline with gamma frailty via
   profile likelihood. Lighter than AG but same credibility output.

API follows scikit-learn conventions: fit(data), predict(data), summary().
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg, optimize, special

from .data import RecurrentEventData
from .frailty import FrailtyDistribution, GammaFrailty, make_frailty


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class FrailtyFitResult:
    """
    Results from a fitted AndersenGillFrailty model.

    Attributes
    ----------
    coef : np.ndarray, shape (n_covariates,)
        Log hazard ratio coefficients (beta).
    coef_se : np.ndarray, shape (n_covariates,)
        Standard errors for beta (robust sandwich).
    theta : float
        Estimated frailty dispersion. Var[z_i] = 1/theta for gamma frailty.
    log_likelihood : float
        Marginal log-likelihood at convergence.
    n_iter : int
        Number of EM iterations taken.
    converged : bool
        Whether the EM algorithm met the convergence criterion.
    covariate_names : list[str]
        Names matching the coef array.
    frailty_name : str
        Distribution used ("gamma" or "lognormal").
    """

    coef: np.ndarray
    coef_se: np.ndarray
    theta: float
    log_likelihood: float
    n_iter: int
    converged: bool
    covariate_names: list[str]
    frailty_name: str

    def summary(self) -> pd.DataFrame:
        """
        Coefficient table with hazard ratios and 95% confidence intervals.
        """
        z = 1.959964  # 97.5th percentile of standard normal
        hr = np.exp(self.coef)
        hr_lo = np.exp(self.coef - z * self.coef_se)
        hr_hi = np.exp(self.coef + z * self.coef_se)
        p_vals = 2 * (1 - _norm_cdf(np.abs(self.coef / np.maximum(self.coef_se, 1e-300))))
        return pd.DataFrame(
            {
                "coef": self.coef,
                "se": self.coef_se,
                "HR": hr,
                "HR_lower_95": hr_lo,
                "HR_upper_95": hr_hi,
                "p_value": p_vals,
            },
            index=self.covariate_names,
        )

    def __repr__(self) -> str:
        return (
            f"FrailtyFitResult("
            f"frailty={self.frailty_name}, "
            f"theta={self.theta:.4f}, "
            f"log_lik={self.log_likelihood:.2f}, "
            f"converged={self.converged}, "
            f"n_iter={self.n_iter})"
        )


@dataclass
class PWPFitResult:
    """Results from a fitted PWPModel."""

    coef: np.ndarray
    coef_se: np.ndarray
    log_likelihood: float
    covariate_names: list[str]
    n_strata: int
    stratum_event_counts: dict

    def summary(self) -> pd.DataFrame:
        z = 1.959964
        hr = np.exp(self.coef)
        hr_lo = np.exp(self.coef - z * self.coef_se)
        hr_hi = np.exp(self.coef + z * self.coef_se)
        p_vals = 2 * (1 - _norm_cdf(np.abs(self.coef / np.maximum(self.coef_se, 1e-300))))
        return pd.DataFrame(
            {
                "coef": self.coef,
                "se": self.coef_se,
                "HR": hr,
                "HR_lower_95": hr_lo,
                "HR_upper_95": hr_hi,
                "p_value": p_vals,
            },
            index=self.covariate_names,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + special.erf(x / np.sqrt(2)))


def _breslow_cumhaz(
    stop: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    query_times: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Breslow estimator of the cumulative baseline hazard.

    Returns (event_times, cumhaz) arrays. If query_times is given,
    interpolates at those times.
    """
    event_mask = event == 1
    if event_mask.sum() == 0:
        t = np.array([0.0])
        return t, np.array([0.0])

    event_times = np.sort(np.unique(stop[event_mask]))
    cumhaz = np.zeros(len(event_times))

    for k, t in enumerate(event_times):
        at_risk = stop >= t  # intervals that haven't ended yet
        d_k = np.sum((stop == t) & event_mask)
        r_k = np.sum(risk_score[at_risk])
        if r_k > 0:
            cumhaz[k] = d_k / r_k

    cumhaz = np.cumsum(cumhaz)

    if query_times is not None:
        indices = np.searchsorted(event_times, query_times, side="right") - 1
        indices = np.clip(indices, 0, len(cumhaz) - 1)
        result = cumhaz[indices]
        result[query_times < event_times[0]] = 0.0
        return event_times, result

    return event_times, cumhaz


def _partial_log_likelihood(
    beta: np.ndarray,
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Efron-approximation partial log-likelihood for Cox model.

    Parameters
    ----------
    beta : shape (p,)
    X : shape (n, p)
    stop : shape (n,)
    event : shape (n,)
    weights : shape (n,) — frailty weights or None for unweighted.
    """
    if weights is None:
        weights = np.ones(len(stop))

    log_theta = X @ beta
    theta = np.exp(log_theta)
    w_theta = weights * theta

    event_mask = event == 1
    event_times = np.unique(stop[event_mask])
    log_lik = 0.0

    for t in event_times:
        at_t = (stop == t) & event_mask
        at_risk = stop >= t

        d_k = int(at_t.sum())
        r_k = np.sum(w_theta[at_risk])

        if r_k <= 0 or d_k == 0:
            continue

        # Efron correction for ties
        sum_log_theta_events = np.sum(log_theta[at_t] + np.log(weights[at_t]))
        log_lik += sum_log_theta_events

        r_k_event = np.sum(w_theta[at_t])
        for j in range(d_k):
            log_lik -= np.log(r_k - j / d_k * r_k_event)

    return log_lik


def _partial_log_likelihood_grad(
    beta: np.ndarray,
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Numerical gradient of partial log-likelihood."""
    eps = 1e-5
    grad = np.zeros_like(beta)
    ll0 = _partial_log_likelihood(beta, X, stop, event, weights)
    for j in range(len(beta)):
        beta_h = beta.copy()
        beta_h[j] += eps
        grad[j] = (_partial_log_likelihood(beta_h, X, stop, event, weights) - ll0) / eps
    return grad


def _robust_sandwich_se(
    beta: np.ndarray,
    X: np.ndarray,
    stop: np.ndarray,
    event: np.ndarray,
    subject_ids: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Lin-Wei robust sandwich standard errors for clustered data.

    Accounts for within-subject correlation across multiple intervals.
    """
    if weights is None:
        weights = np.ones(len(stop))

    p = len(beta)
    n_subjects = len(np.unique(subject_ids))
    theta = np.exp(X @ beta) * weights

    # Hessian approximation via finite differences
    eps = 1e-4
    H = np.zeros((p, p))
    ll0 = _partial_log_likelihood(beta, X, stop, event, weights)
    for j in range(p):
        for k in range(j, p):
            b_jk = beta.copy()
            b_jk[j] += eps
            b_jk[k] += eps
            b_j = beta.copy()
            b_j[j] += eps
            b_k = beta.copy()
            b_k[k] += eps
            H[j, k] = (
                _partial_log_likelihood(b_jk, X, stop, event, weights)
                - _partial_log_likelihood(b_j, X, stop, event, weights)
                - _partial_log_likelihood(b_k, X, stop, event, weights)
                + ll0
            ) / eps**2
            H[k, j] = H[j, k]

    try:
        inv_H = linalg.inv(-H)
    except linalg.LinAlgError:
        return np.full(p, np.nan)

    # Score residuals per subject
    event_mask = event == 1
    event_times = np.unique(stop[event_mask])

    score_resid = np.zeros((n_subjects, p))
    unique_ids = np.unique(subject_ids)
    id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}

    for t in event_times:
        at_t = (stop == t) & event_mask
        at_risk = stop >= t

        r = np.sum(theta[at_risk])
        if r <= 0:
            continue

        x_bar = X[at_risk].T @ theta[at_risk] / r

        # Contribution for events at t
        for row_idx in np.where(at_t)[0]:
            sid = subject_ids[row_idx]
            score_resid[id_to_idx[sid]] += X[row_idx] - x_bar

        # Subtract expected score for at-risk intervals
        for row_idx in np.where(at_risk)[0]:
            sid = subject_ids[row_idx]
            d_k = float(at_t.sum())
            score_resid[id_to_idx[sid]] -= d_k * theta[row_idx] / r * (X[row_idx] - x_bar)

    B = score_resid.T @ score_resid
    sandwich = inv_H @ B @ inv_H
    return np.sqrt(np.maximum(np.diag(sandwich), 0.0))


# ------------------------------------------------------------------
# AndersenGillFrailty
# ------------------------------------------------------------------


class AndersenGillFrailty:
    """
    Andersen-Gill intensity model with shared gamma or lognormal frailty.

    The model is:
        lambda_i(t) = z_i * lambda_0(t) * exp(X_i(t)' beta)

    where z_i ~ Gamma(theta, theta) or z_i ~ LogNormal(-sigma^2/2, sigma^2)
    is the unobserved frailty for subject i.

    The EM algorithm alternates between:
    - E-step: compute E[z_i | data, beta, theta] and E[log z_i | data, ...]
    - M-step: update beta via weighted partial likelihood; update theta via
      marginal likelihood optimisation.

    Parameters
    ----------
    frailty : str
        "gamma" (default) or "lognormal".
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    verbose : bool
        Print iteration progress.
    min_theta : float
        Lower bound on theta to prevent degenerate solutions.

    Notes
    -----
    The posterior frailty E[z_i | data] equals the Bühlmann-Straub credibility
    premium for subject i. Use ``credibility_scores()`` after fitting to
    retrieve these.
    """

    def __init__(
        self,
        frailty: str = "gamma",
        max_iter: int = 50,
        tol: float = 1e-5,
        verbose: bool = False,
        min_theta: float = 0.01,
    ) -> None:
        self.frailty = frailty
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.min_theta = min_theta

        self._frailty_dist: FrailtyDistribution = make_frailty(frailty)
        self._result: Optional[FrailtyFitResult] = None
        self._data: Optional[RecurrentEventData] = None
        self._lambda_i: Optional[np.ndarray] = None  # cumhaz per subject at fit
        self._n_i: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: RecurrentEventData) -> "AndersenGillFrailty":
        """
        Fit the Andersen-Gill frailty model via EM algorithm.

        Parameters
        ----------
        data : RecurrentEventData
            Counting process data.

        Returns
        -------
        self
        """
        self._data = data
        df = data.df
        X = data.X
        stop = data.stop
        start = data.start
        event = data.event
        n_subjects = data.n_subjects
        subject_ids = df[data.id_col].to_numpy()
        unique_ids = data.subject_ids

        id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        subj_idx = np.array([id_to_idx[sid] for sid in subject_ids])

        p = X.shape[1]

        # Initialise: no-frailty Cox estimates (all z_i = 1)
        beta = np.zeros(p)
        if p > 0:
            beta = self._init_beta(X, stop, event, p)

        theta = 1.0  # start with moderate frailty

        # Per-subject event counts
        n_i = np.array([
            int(event[subj_idx == i].sum()) for i in range(n_subjects)
        ])

        log_lik_prev = -np.inf

        for iteration in range(self.max_iter):
            # -- E-step: compute expected frailty per subject --
            risk_score = np.exp(X @ beta) if p > 0 else np.ones(len(stop))
            lambda_i = self._compute_cumhaz_per_subject(
                stop, event, risk_score, subj_idx, n_subjects
            )

            z_i = self._frailty_dist.posterior_mean(n_i, lambda_i, theta)
            z_i = np.maximum(z_i, 1e-6)

            # -- M-step beta: weighted partial likelihood --
            weights_per_row = z_i[subj_idx]
            if p > 0:
                beta = self._update_beta(beta, X, stop, event, weights_per_row)

            # -- M-step theta: marginal likelihood --
            # Recompute lambda_i with updated beta
            risk_score = np.exp(X @ beta) if p > 0 else np.ones(len(stop))
            lambda_i = self._compute_cumhaz_per_subject(
                stop, event, risk_score, subj_idx, n_subjects
            )

            theta_new = self._frailty_dist.update_theta(n_i, lambda_i, theta)
            theta_new = max(theta_new, self.min_theta)

            # -- Log-likelihood --
            log_lik = float(np.sum(
                self._frailty_dist.log_marginal(n_i, lambda_i, theta_new)
            ))

            if self.verbose:
                print(
                    f"  Iter {iteration+1:3d}: log_lik={log_lik:.4f}, "
                    f"theta={theta_new:.4f}, "
                    f"||beta||={np.linalg.norm(beta):.4f}"
                )

            converged = abs(log_lik - log_lik_prev) < self.tol
            theta = theta_new
            log_lik_prev = log_lik

            if converged:
                break
        else:
            warnings.warn(
                f"AndersenGillFrailty: EM did not converge in {self.max_iter} iterations. "
                "Consider increasing max_iter or checking data quality.",
                RuntimeWarning,
                stacklevel=2,
            )
            converged = False

        # Final SE computation
        risk_score = np.exp(X @ beta) if p > 0 else np.ones(len(stop))
        lambda_i = self._compute_cumhaz_per_subject(
            stop, event, risk_score, subj_idx, n_subjects
        )
        z_i_final = self._frailty_dist.posterior_mean(n_i, lambda_i, theta)
        weights_per_row = z_i_final[subj_idx]

        if p > 0:
            se = _robust_sandwich_se(beta, X, stop, event, subject_ids, weights_per_row)
        else:
            se = np.array([])

        self._lambda_i = lambda_i
        self._n_i = n_i
        self._beta = beta
        self._theta = theta
        self._subj_idx = subj_idx
        self._unique_ids = unique_ids

        self._result = FrailtyFitResult(
            coef=beta,
            coef_se=se,
            theta=theta,
            log_likelihood=log_lik_prev,
            n_iter=iteration + 1,
            converged=converged,
            covariate_names=data.covariate_cols,
            frailty_name=self.frailty,
        )
        return self

    def _init_beta(
        self, X: np.ndarray, stop: np.ndarray, event: np.ndarray, p: int
    ) -> np.ndarray:
        """Warm-start: Cox partial likelihood without frailty."""
        result = optimize.minimize(
            lambda b: -_partial_log_likelihood(b, X, stop, event),
            x0=np.zeros(p),
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-6},
        )
        return result.x

    def _update_beta(
        self,
        beta_init: np.ndarray,
        X: np.ndarray,
        stop: np.ndarray,
        event: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """M-step: maximise weighted partial likelihood over beta."""
        result = optimize.minimize(
            lambda b: -_partial_log_likelihood(b, X, stop, event, weights),
            x0=beta_init,
            method="L-BFGS-B",
            options={"maxiter": 50, "ftol": 1e-6},
        )
        return result.x

    def _compute_cumhaz_per_subject(
        self,
        stop: np.ndarray,
        event: np.ndarray,
        risk_score: np.ndarray,
        subj_idx: np.ndarray,
        n_subjects: int,
    ) -> np.ndarray:
        """
        Compute the cumulative baseline hazard integrated over each subject's
        follow-up: Lambda_i = sum over intervals of dLambda_0(t) * exp(X'beta).
        """
        # Breslow increments at each event time
        event_mask = event == 1
        event_times = np.sort(np.unique(stop[event_mask])) if event_mask.any() else np.array([])

        if len(event_times) == 0:
            return np.zeros(n_subjects)

        # For each event time, compute Breslow increment
        dLambda = np.zeros(len(event_times))
        for k, t in enumerate(event_times):
            at_risk = stop >= t
            d_k = float(np.sum((stop == t) & event_mask))
            r_k = float(np.sum(risk_score[at_risk]))
            if r_k > 0:
                dLambda[k] = d_k / r_k

        # Per-subject cumulative hazard: sum of (risk_score * dLambda) over
        # intervals where the interval contains each event time
        lambda_i = np.zeros(n_subjects)
        for k, t in enumerate(event_times):
            # Intervals that contain time t: start < t <= stop
            in_interval = (stop >= t)
            for j in np.where(in_interval)[0]:
                si = subj_idx[j]
                lambda_i[si] += risk_score[j] * dLambda[k]

        return lambda_i

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    @property
    def result_(self) -> FrailtyFitResult:
        """The fitted result. Raises if not yet fitted."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._result

    def credibility_scores(self) -> pd.DataFrame:
        """
        Posterior frailty estimates — the Bühlmann-Straub credibility scores.

        Returns a DataFrame with columns:
        - id: subject identifier
        - n_events: observed event count
        - lambda_i: expected cumulative hazard (prior expectation)
        - frailty_mean: E[z_i | data] — posterior frailty estimate
        - frailty_var: Var[z_i | data]
        - credibility_weight: lambda_i / (lambda_i + theta)
        """
        if self._lambda_i is None:
            raise RuntimeError("Model has not been fitted.")

        result = self._result
        z_mean = self._frailty_dist.posterior_mean(
            self._n_i, self._lambda_i, result.theta
        )
        z_var = self._frailty_dist.posterior_variance(
            self._n_i, self._lambda_i, result.theta
        )
        z_cred = self._frailty_dist.credibility_weight(
            self._n_i, self._lambda_i, result.theta
        )
        return pd.DataFrame(
            {
                "id": self._unique_ids,
                "n_events": self._n_i,
                "lambda_i": self._lambda_i,
                "frailty_mean": z_mean,
                "frailty_var": z_var,
                "credibility_weight": z_cred,
            }
        )

    def predict_intensity(self, data: RecurrentEventData) -> np.ndarray:
        """
        Predict the intensity (hazard) multiplier for each row in data.

        Returns exp(X' beta) — the covariate-driven component. Multiply by
        frailty_mean and baseline hazard for the full intensity.
        """
        result = self.result_
        if len(result.coef) == 0:
            return np.ones(len(data.df))
        return np.exp(data.X @ result.coef)

    def summary(self) -> pd.DataFrame:
        """Coefficient summary table. See FrailtyFitResult.summary()."""
        return self.result_.summary()

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "unfitted"
        return f"AndersenGillFrailty(frailty={self.frailty!r}, {status})"


# ------------------------------------------------------------------
# PWP Model
# ------------------------------------------------------------------


class PWPModel:
    """
    Prentice-Williams-Peterson model stratified by event number.

    Fits a separate baseline hazard for the 1st, 2nd, and 3rd+ claim.
    Uses either gap time (PWP-GT: time since previous event) or calendar
    time (PWP-CT: time since study start).

    PWP is appropriate when the intensity process genuinely changes after
    each event — for example, in motor insurance where a claim resets NCD
    and changes premium, so the policyholder's propensity to claim changes.

    For pure heterogeneity (some policyholders are just risky), use
    AndersenGillFrailty instead.

    Parameters
    ----------
    time_scale : str
        "gap" (default) for gap time, "calendar" for calendar time.
    max_stratum : int
        Strata are 1, 2, ..., max_stratum (last stratum pools all later events).
    """

    def __init__(
        self,
        time_scale: str = "gap",
        max_stratum: int = 3,
    ) -> None:
        if time_scale not in ("gap", "calendar"):
            raise ValueError("time_scale must be 'gap' or 'calendar'.")
        self.time_scale = time_scale
        self.max_stratum = max_stratum
        self._result: Optional[PWPFitResult] = None
        self._beta: Optional[np.ndarray] = None

    def fit(self, data: RecurrentEventData) -> "PWPModel":
        """
        Fit the stratified PWP model.

        Internally adds a stratum column (event number within subject) and
        fits a shared beta with stratum-specific baselines.
        """
        df = self._prepare(data)
        X = data.X
        p = X.shape[1]
        stop = df["_stop"].to_numpy(dtype=float)
        event = data.event
        strata = df["_stratum"].to_numpy(dtype=int)

        # Fit: shared beta across strata, separate baseline per stratum
        def neg_pll(beta: np.ndarray) -> float:
            total = 0.0
            for s in range(1, self.max_stratum + 1):
                mask = strata == s
                if mask.sum() == 0:
                    continue
                total += _partial_log_likelihood(
                    beta, X[mask], stop[mask], event[mask]
                )
            return -total

        beta0 = np.zeros(p)
        result = optimize.minimize(
            neg_pll,
            x0=beta0,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-7},
        )
        beta = result.x

        # SE: robust sandwich
        subject_ids = data.df[data.id_col].to_numpy()
        if p > 0:
            se = _robust_sandwich_se(beta, X, stop, event, subject_ids)
        else:
            se = np.array([])

        stratum_counts = {}
        for s in range(1, self.max_stratum + 1):
            mask = strata == s
            stratum_counts[s] = int(event[mask].sum())

        self._beta = beta
        self._result = PWPFitResult(
            coef=beta,
            coef_se=se,
            log_likelihood=-result.fun,
            covariate_names=data.covariate_cols,
            n_strata=self.max_stratum,
            stratum_event_counts=stratum_counts,
        )
        return self

    def _prepare(self, data: RecurrentEventData) -> pd.DataFrame:
        """Add stratum (event number) and gap time columns."""
        df = data.df.copy()
        # Count events prior to each row within subject
        df["_cum_events_before"] = (
            df.groupby(data.id_col)[data.event_col]
            .cumsum()
            .shift(1, fill_value=0)
        )
        df["_stratum"] = (df["_cum_events_before"] + 1).clip(upper=self.max_stratum).astype(int)

        if self.time_scale == "gap":
            # Gap time: t_stop - t_start (interval length)
            df["_stop"] = data.df[data.stop_col] - data.df[data.start_col]
        else:
            # Calendar time: use stop as-is
            df["_stop"] = data.df[data.stop_col]

        return df

    @property
    def result_(self) -> PWPFitResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._result

    def summary(self) -> pd.DataFrame:
        return self.result_.summary()

    def predict_hr(self, X_new: np.ndarray) -> np.ndarray:
        """Predicted hazard ratio exp(X' beta) for new data."""
        if self._beta is None or len(self._beta) == 0:
            return np.ones(len(X_new))
        return np.exp(X_new @ self._beta)

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "unfitted"
        return f"PWPModel(time_scale={self.time_scale!r}, {status})"


# ------------------------------------------------------------------
# NelsonAalenFrailty (lightweight alternative)
# ------------------------------------------------------------------


class NelsonAalenFrailty:
    """
    Non-parametric Nelson-Aalen estimator with gamma frailty via
    profile likelihood.

    This is a simpler version of AndersenGillFrailty with no covariates.
    Useful for estimating unobserved heterogeneity without imposing a
    covariate structure, or as a diagnostic tool.

    Parameters
    ----------
    max_iter : int
        Maximum iterations for theta optimisation.
    tol : float
        Convergence tolerance.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._frailty_dist = GammaFrailty()
        self._result: Optional[dict] = None

    def fit(self, data: RecurrentEventData) -> "NelsonAalenFrailty":
        """Fit the non-parametric frailty model."""
        stop = data.stop
        event = data.event
        subject_ids = data.df[data.id_col].to_numpy()
        unique_ids = data.subject_ids
        n_subjects = data.n_subjects
        id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        subj_idx = np.array([id_to_idx[sid] for sid in subject_ids])

        n_i = np.array([int(event[subj_idx == i].sum()) for i in range(n_subjects)])
        theta = 1.0

        for _ in range(self.max_iter):
            # Nelson-Aalen with frailty weights
            event_mask = event == 1
            event_times = np.sort(np.unique(stop[event_mask])) if event_mask.any() else np.array([])

            lambda_i = np.zeros(n_subjects)
            if len(event_times) > 0:
                z_i = self._frailty_dist.posterior_mean(n_i, lambda_i + 1e-6, theta)
                weights = z_i[subj_idx]
                for t in event_times:
                    at_risk = stop >= t
                    d_k = float(np.sum((stop == t) & event_mask))
                    r_k = float(np.sum(weights[at_risk]))
                    if r_k > 0:
                        dL = d_k / r_k
                        for j in np.where(at_risk)[0]:
                            lambda_i[subj_idx[j]] += weights[j] * dL

            theta_new = self._frailty_dist.update_theta(n_i, lambda_i, theta)
            theta_new = max(theta_new, 0.01)

            if abs(theta_new - theta) < self.tol:
                theta = theta_new
                break
            theta = theta_new

        self._n_i = n_i
        self._lambda_i = lambda_i
        self._unique_ids = unique_ids
        self._theta = theta
        self._result = {
            "theta": theta,
            "n_subjects": n_subjects,
            "n_events": int(n_i.sum()),
        }
        return self

    def credibility_scores(self) -> pd.DataFrame:
        """Return posterior frailty estimates."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        theta = self._theta
        z_mean = self._frailty_dist.posterior_mean(self._n_i, self._lambda_i, theta)
        z_cred = self._frailty_dist.credibility_weight(self._n_i, self._lambda_i, theta)
        return pd.DataFrame(
            {
                "id": self._unique_ids,
                "n_events": self._n_i,
                "lambda_i": self._lambda_i,
                "frailty_mean": z_mean,
                "credibility_weight": z_cred,
            }
        )

    @property
    def theta_(self) -> float:
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        return self._theta

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "unfitted"
        return f"NelsonAalenFrailty({status})"
