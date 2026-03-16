"""Log-normal AFT mixture cure model.

Provides a heavier left-tail alternative to the Weibull latency.
Log-normal is sometimes preferred for claim timing data that shows
a mode away from zero — for instance, pet insurance where the first
claim often occurs a few months after policy inception rather than
immediately.

Reference: Boag (1949), JRSS-B 11(1):15-53 (original log-normal MCM).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseMixtureCure
from ._em import (
    compute_loglik,
    compute_pi,
    e_step,
    lognormal_aft_density,
    lognormal_aft_survival,
    m_step_incidence,
    m_step_lognormal,
)


class LogNormalMixtureCure(BaseMixtureCure):
    """Mixture cure model with log-normal AFT latency sub-model.

    Uses the same EM framework as ``WeibullMixtureCure`` but with a
    log-normal latency distribution. The log-normal can fit better
    when the conditional hazard for susceptibles rises and then falls
    (non-monotone hazard), which the Weibull cannot capture.

    Parameters
    ----------
    incidence_formula : str
        Additive formula for incidence covariates.
    latency_formula : str
        Additive formula for latency covariates.
    n_em_starts : int
        Number of random EM restarts. Default 5.
    max_iter : int
        Maximum EM iterations per run. Default 200.
    tol : float
        Convergence tolerance on log-likelihood change. Default 1e-5.
    bootstrap_se : bool
        Whether to compute bootstrap standard errors. Default False.
    n_bootstrap : int
        Number of bootstrap resamples. Default 200.
    n_jobs : int
        Parallel jobs for bootstrap. Default -1 (all cores).
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    result_ : MCMResult
        Detailed fit results, available after ``fit()``.

    Examples
    --------
    >>> from insurance_cure import LogNormalMixtureCure
    >>> model = LogNormalMixtureCure(
    ...     incidence_formula="ncd_years + age_band",
    ...     latency_formula="ncd_years",
    ...     n_em_starts=3,
    ... )
    >>> model.fit(df, duration_col="tenure_months", event_col="claimed")
    """

    def _latency_param_names(self) -> list[str]:
        names = ["mu", "log_sigma"]
        if self._latency_cols:
            names += [f"beta_{c}" for c in self._latency_cols]
        return names

    def _compute_latency_surv_dens(
        self,
        t: np.ndarray,
        x: np.ndarray,
        latency_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = x.shape[1]
        mu = latency_params[0]
        log_sigma = latency_params[1]
        beta = latency_params[2 : 2 + q]
        surv = lognormal_aft_survival(t, x, mu, log_sigma, beta)
        dens = lognormal_aft_density(t, x, mu, log_sigma, beta)
        return surv, dens

    def _smart_init(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a smart initial point from the data."""
        from sklearn.linear_model import LogisticRegression
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500, fit_intercept=True)
            clf.fit(z, event.astype(int))
        gamma0 = clf.coef_[0]
        intercept0 = clf.intercept_

        q = x.shape[1]
        t_ev = t[event == 1]
        if len(t_ev) < 2:
            t_ev = t[t > 0]

        log_t_ev = np.log(np.clip(t_ev, 1e-15, None))
        mu0 = float(np.mean(log_t_ev))
        sigma0 = float(np.std(log_t_ev)) if len(log_t_ev) > 1 else 1.0

        lat0 = np.zeros(2 + q)
        lat0[0] = mu0
        lat0[1] = np.log(max(sigma0, 0.1))
        return gamma0, intercept0, lat0

    def _run_em_single(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool, int]:
        """Run one EM sequence from a randomised starting point."""
        q = x.shape[1]
        p = z.shape[1]

        if rng.random() < 0.5:
            gamma, intercept, lat_params = self._smart_init(t, event, z, x)
        else:
            gamma = rng.normal(0, 0.5, size=p)
            intercept = rng.normal(0, 0.5, size=1)
            lat_params = np.zeros(2 + q)
            log_t = np.log(np.clip(t, 1e-15, None))
            lat_params[0] = float(np.mean(log_t)) + rng.normal(0, 0.3)
            lat_params[1] = rng.normal(0, 0.3)

        prev_loglik = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            surv_u, dens_u = self._compute_latency_surv_dens(t, x, lat_params)
            pi = compute_pi(z, gamma, intercept)
            w = e_step(pi, surv_u, event)

            gamma, intercept = m_step_incidence(z, w)
            lat_params = m_step_lognormal(t, x, event, w, init_params=lat_params.copy())

            surv_u, dens_u = self._compute_latency_surv_dens(t, x, lat_params)
            pi = compute_pi(z, gamma, intercept)
            loglik = compute_loglik(pi, dens_u, surv_u, event)

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        return loglik, gamma, intercept, lat_params, converged, iteration + 1
