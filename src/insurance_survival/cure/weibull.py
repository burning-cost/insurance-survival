"""Weibull AFT mixture cure model.

The primary workhorse of the library. Uses a Weibull accelerated failure
time latency sub-model with logistic incidence. This combination is the
most common parametric MCM in applied work: clean extrapolation beyond
the observation window, interpretable shape and scale parameters, and
good numerical stability in the EM algorithm.

Reference: Farewell (1982), Biometrics 38:1041-1046.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseMixtureCure
from ._em import (
    compute_loglik,
    compute_pi,
    e_step,
    m_step_incidence,
    m_step_weibull,
    weibull_aft_density,
    weibull_aft_survival,
)


class WeibullMixtureCure(BaseMixtureCure):
    """Mixture cure model with Weibull AFT latency sub-model.

    Fits the canonical MCM structure:

        S_pop(t|x,z) = pi(z) * S_u(t|x) + [1 - pi(z)]

    where pi(z) = sigmoid(z' gamma) is the logistic incidence sub-model
    and S_u(t|x) = exp(-(t / scale(x))^rho) is the Weibull AFT latency.

    Parameters
    ----------
    incidence_formula : str
        Additive formula for incidence covariates, e.g.
        ``"ncb_years + age_band + vehicle_age"``.
    latency_formula : str
        Additive formula for latency covariates.
    n_em_starts : int
        Number of random EM restarts to avoid local optima. Default 5.
    max_iter : int
        Maximum EM iterations per run. Default 200.
    tol : float
        Convergence tolerance on log-likelihood change. Default 1e-5.
    bootstrap_se : bool
        Whether to compute bootstrap standard errors. Default False.
        Set True for production runs; adds substantial compute time.
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
    >>> from insurance_cure import WeibullMixtureCure
    >>> model = WeibullMixtureCure(
    ...     incidence_formula="ncb_years + age_band",
    ...     latency_formula="ncb_years",
    ...     n_em_starts=3,
    ... )
    >>> model.fit(df, duration_col="tenure_months", event_col="claimed")
    >>> cure_scores = model.predict_cure_fraction(df)
    """

    def _latency_param_names(self) -> list[str]:
        names = ["log_lambda", "log_rho"]
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
        log_lambda = latency_params[0]
        log_rho = latency_params[1]
        beta = latency_params[2 : 2 + q]
        surv = weibull_aft_survival(t, x, log_lambda, log_rho, beta)
        dens = weibull_aft_density(t, x, log_lambda, log_rho, beta)
        return surv, dens

    def _smart_init(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a smart initial point from the data.

        Incidence: logistic regression on event indicator.
        Latency: Weibull fit to event observations only.
        """
        from sklearn.linear_model import LogisticRegression
        import warnings

        # Incidence init
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500, fit_intercept=True)
            clf.fit(z, event.astype(int))
        gamma0 = clf.coef_[0]
        intercept0 = clf.intercept_

        # Latency init: Weibull on event times
        q = x.shape[1]
        t_ev = t[event == 1]
        x_ev = x[event == 1]
        if len(t_ev) < 2:
            t_ev = t
            x_ev = x

        lat0 = np.zeros(2 + q)
        lat0[0] = np.log(np.mean(t_ev))
        lat0[1] = 0.0  # rho = 1
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

        # 50% chance of smart init, 50% random
        if rng.random() < 0.5:
            gamma, intercept, lat_params = self._smart_init(t, event, z, x)
        else:
            gamma = rng.normal(0, 0.5, size=p)
            intercept = rng.normal(0, 0.5, size=1)
            lat_params = np.zeros(2 + q)
            lat_params[0] = np.log(np.mean(t)) + rng.normal(0, 0.3)
            lat_params[1] = rng.normal(0, 0.3)

        prev_loglik = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E-step
            surv_u, dens_u = self._compute_latency_surv_dens(t, x, lat_params)
            pi = compute_pi(z, gamma, intercept)
            w = e_step(pi, surv_u, event)

            # M-step
            gamma, intercept = m_step_incidence(z, w)
            lat_params = m_step_weibull(t, x, event, w, init_params=lat_params.copy())

            # Log-likelihood check
            surv_u, dens_u = self._compute_latency_surv_dens(t, x, lat_params)
            pi = compute_pi(z, gamma, intercept)
            loglik = compute_loglik(pi, dens_u, surv_u, event)

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        return loglik, gamma, intercept, lat_params, converged, iteration + 1
