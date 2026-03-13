"""Core EM algorithm for mixture cure models.

Implements the E-step (posterior susceptibility weights) and the
M-step wrappers for logistic incidence and parametric latency.
This module contains the shared infrastructure used by all MCM classes.

The canonical reference for the EM algorithm is Peng & Dear (2000),
Biometrics 56:237-243, and Sy & Taylor (2000), Biometrics 56:227-236.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def e_step(
    pi: np.ndarray,
    surv_u: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Compute posterior susceptibility weights.

    For observed events (event=1), the weight is 1: the individual is
    certainly susceptible. For censored observations (event=0), the weight
    is the posterior probability of being susceptible given survival to the
    censoring time.

    Parameters
    ----------
    pi : ndarray of shape (n,)
        Current P(susceptible) from the incidence sub-model.
    surv_u : ndarray of shape (n,)
        Current S_u(t|x) from the latency sub-model for each individual.
    event : ndarray of shape (n,), dtype bool or int
        Indicator: 1 if the event was observed, 0 if censored.

    Returns
    -------
    w : ndarray of shape (n,)
        Posterior susceptibility weights in [0, 1].
        w_i = 1 for events; w_i = pi_i * S_u_i / (pi_i * S_u_i + 1 - pi_i)
        for censored observations.
    """
    w = np.ones(len(pi))
    censored = event == 0
    denom = pi[censored] * surv_u[censored] + (1.0 - pi[censored])
    # Avoid division by zero at boundaries
    safe_denom = np.where(denom < 1e-15, 1e-15, denom)
    w[censored] = pi[censored] * surv_u[censored] / safe_denom
    return w


def m_step_incidence(
    z: np.ndarray,
    w: np.ndarray,
    C: float = 1e4,
) -> tuple[np.ndarray, np.ndarray]:
    """M-step: update logistic incidence parameters.

    Fits logistic regression with soft labels derived from the E-step
    weights. The trick: use w_i as the probability label rather than
    a hard binary label, implemented via sample-weighted logistic regression
    on both classes.

    When all weights are at a boundary (all 1 or all 0 — meaning all
    observations are certainly susceptible or certainly immune), this
    function adds a tiny regularising pseudo-observation of the minority
    class to keep sklearn's solver stable. The weight of the pseudo-obs
    is 1e-6, negligible relative to any real dataset.

    Parameters
    ----------
    z : ndarray of shape (n, p)
        Incidence covariates (already including intercept column if desired,
        or raw features for sklearn which adds its own intercept).
    w : ndarray of shape (n,)
        Posterior susceptibility weights from the E-step.
    C : float
        Inverse regularisation strength for sklearn LogisticRegression.
        Large C = minimal regularisation (matches maximum likelihood).

    Returns
    -------
    gamma : ndarray of shape (p,)
        Logistic regression coefficients (excluding intercept).
    intercept : ndarray of shape (1,)
        Logistic regression intercept.
    """
    n = len(w)
    # Construct doubled dataset: each obs contributes two pseudo-rows
    # with labels 1 (weight w_i) and 0 (weight 1 - w_i)
    # This implements weighted logistic regression with soft labels
    X_doubled = np.vstack([z, z])
    y_doubled = np.concatenate([np.ones(n), np.zeros(n)])
    sw_doubled = np.concatenate([w, 1.0 - w])

    # Filter near-zero weights to keep sklearn stable
    mask = sw_doubled > 1e-10
    X_fit = X_doubled[mask]
    y_fit = y_doubled[mask]
    sw_fit = sw_doubled[mask]

    # Ensure both classes are represented. When all weights are 1 (or all 0),
    # only one class survives after filtering. Add a single tiny pseudo-obs
    # of the missing class so sklearn does not raise a ValueError.
    present_classes = set(np.unique(y_fit))
    if 0.0 not in present_classes:
        # Add a tiny weight-0 class for label=0
        X_fit = np.vstack([X_fit, np.zeros((1, z.shape[1]))])
        y_fit = np.append(y_fit, 0.0)
        sw_fit = np.append(sw_fit, 1e-6)
    elif 1.0 not in present_classes:
        X_fit = np.vstack([X_fit, np.zeros((1, z.shape[1]))])
        y_fit = np.append(y_fit, 1.0)
        sw_fit = np.append(sw_fit, 1e-6)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=500,
            fit_intercept=True,
        )
        clf.fit(X_fit, y_fit, sample_weight=sw_fit)

    return clf.coef_[0], clf.intercept_


def compute_pi(z: np.ndarray, gamma: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Compute P(susceptible) from logistic incidence parameters.

    Parameters
    ----------
    z : ndarray of shape (n, p)
        Incidence covariates.
    gamma : ndarray of shape (p,)
        Logistic coefficients.
    intercept : ndarray of shape (1,)
        Logistic intercept.

    Returns
    -------
    pi : ndarray of shape (n,)
        P(susceptible | z) = sigmoid(z @ gamma + intercept).
    """
    return expit(z @ gamma + intercept[0])


def weibull_aft_survival(
    t: np.ndarray,
    x: np.ndarray,
    log_lambda: float,
    log_rho: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Weibull AFT survival function S_u(t|x).

    Parameterisation: S(t) = exp(-(t / scale)^shape) where
    scale = exp(log_lambda + x @ beta), shape = exp(log_rho).

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations.
    x : ndarray of shape (n, q)
        Latency covariates.
    log_lambda : float
        Log of baseline scale parameter.
    log_rho : float
        Log of shape parameter (rho = exp(log_rho) > 0).
    beta : ndarray of shape (q,)
        Latency regression coefficients.

    Returns
    -------
    surv : ndarray of shape (n,)
        S_u(t | x) values in (0, 1].
    """
    rho = np.exp(log_rho)
    # AFT: accelerated failure time. beta enters the log-scale.
    log_scale = log_lambda + x @ beta
    scale = np.exp(log_scale)
    ratio = np.clip(t / scale, 0, 1e10)
    return np.exp(-ratio ** rho)


def weibull_aft_density(
    t: np.ndarray,
    x: np.ndarray,
    log_lambda: float,
    log_rho: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Weibull AFT density f_u(t|x).

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations (must be > 0).
    x : ndarray of shape (n, q)
        Latency covariates.
    log_lambda : float
        Log of baseline scale parameter.
    log_rho : float
        Log of shape parameter.
    beta : ndarray of shape (q,)
        Latency regression coefficients.

    Returns
    -------
    dens : ndarray of shape (n,)
        f_u(t | x) values >= 0.
    """
    rho = np.exp(log_rho)
    log_scale = log_lambda + x @ beta
    scale = np.exp(log_scale)
    ratio = np.clip(t / scale, 1e-15, 1e10)
    return (rho / scale) * (ratio ** (rho - 1.0)) * np.exp(-(ratio ** rho))


def weibull_neg_loglik(
    params: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    event: np.ndarray,
    w: np.ndarray,
) -> float:
    """Weighted negative log-likelihood for Weibull AFT latency.

    Objective for the M-step latency optimisation. The weights w_i
    are the posterior susceptibility probabilities from the E-step.

    Parameters
    ----------
    params : ndarray
        [log_lambda, log_rho, beta_0, beta_1, ...] flat parameter vector.
    t : ndarray of shape (n,)
        Observed durations.
    x : ndarray of shape (n, q)
        Latency covariates.
    event : ndarray of shape (n,)
        Event indicator.
    w : ndarray of shape (n,)
        E-step weights.

    Returns
    -------
    float
        Weighted negative log-likelihood.
    """
    q = x.shape[1]
    log_lambda = params[0]
    log_rho = params[1]
    beta = params[2 : 2 + q]

    rho = np.exp(log_rho)
    log_scale = log_lambda + x @ beta
    scale = np.exp(log_scale)
    ratio = np.clip(t / scale, 1e-15, 1e10)

    # Log density: log f(t) = log(rho) - log(scale) + (rho-1)*log(ratio) - ratio^rho
    log_dens = np.log(rho) - log_scale + (rho - 1.0) * np.log(ratio) - ratio ** rho
    # Log survival: log S(t) = -ratio^rho
    log_surv = -(ratio ** rho)

    # Weighted complete-data log-likelihood
    ll = np.sum(w * event * log_dens + w * (1.0 - event) * log_surv)
    return -ll


def m_step_weibull(
    t: np.ndarray,
    x: np.ndarray,
    event: np.ndarray,
    w: np.ndarray,
    init_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """M-step: update Weibull AFT latency parameters.

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations.
    x : ndarray of shape (n, q)
        Latency covariates.
    event : ndarray of shape (n,)
        Event indicator.
    w : ndarray of shape (n,)
        E-step weights.
    init_params : ndarray or None
        Initial parameter vector [log_lambda, log_rho, beta...].
        If None, initialised from event observations only.

    Returns
    -------
    params : ndarray
        Optimised [log_lambda, log_rho, beta...] vector.
    """
    q = x.shape[1]
    if init_params is None:
        # Initialise from event observations
        t_ev = t[event == 1]
        if len(t_ev) < 2:
            t_ev = t[t > 0]
        log_lambda0 = np.log(np.mean(t_ev)) if len(t_ev) > 0 else 0.0
        init_params = np.zeros(2 + q)
        init_params[0] = log_lambda0  # log_lambda
        init_params[1] = 0.0          # log_rho = 0 => rho = 1 (exponential)

    result = optimize.minimize(
        weibull_neg_loglik,
        init_params,
        args=(t, x, event, w),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return result.x


def lognormal_aft_survival(
    t: np.ndarray,
    x: np.ndarray,
    mu: float,
    log_sigma: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Log-normal AFT survival function S_u(t|x).

    S(t) = 1 - Phi((log(t) - mu_i) / sigma)
    where mu_i = mu + x @ beta.

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations.
    x : ndarray of shape (n, q)
        Latency covariates.
    mu : float
        Baseline log-mean.
    log_sigma : float
        Log of the log-normal scale (sigma = exp(log_sigma) > 0).
    beta : ndarray of shape (q,)
        Latency regression coefficients.

    Returns
    -------
    surv : ndarray of shape (n,)
        S_u(t | x) values in (0, 1].
    """
    from scipy.special import ndtr
    sigma = np.exp(log_sigma)
    mu_i = mu + x @ beta
    z = (np.log(np.clip(t, 1e-15, None)) - mu_i) / sigma
    return 1.0 - ndtr(z)


def lognormal_aft_density(
    t: np.ndarray,
    x: np.ndarray,
    mu: float,
    log_sigma: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Log-normal AFT density f_u(t|x).

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations (must be > 0).
    x : ndarray of shape (n, q)
        Latency covariates.
    mu : float
        Baseline log-mean.
    log_sigma : float
        Log of the log-normal scale.
    beta : ndarray of shape (q,)
        Latency regression coefficients.

    Returns
    -------
    dens : ndarray of shape (n,)
        f_u(t | x) values >= 0.
    """
    sigma = np.exp(log_sigma)
    mu_i = mu + x @ beta
    log_t = np.log(np.clip(t, 1e-15, None))
    z = (log_t - mu_i) / sigma
    return np.exp(-0.5 * z ** 2) / (t * sigma * np.sqrt(2.0 * np.pi))


def lognormal_neg_loglik(
    params: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    event: np.ndarray,
    w: np.ndarray,
) -> float:
    """Weighted negative log-likelihood for log-normal AFT latency."""
    from scipy.special import ndtr
    q = x.shape[1]
    mu = params[0]
    log_sigma = params[1]
    beta = params[2 : 2 + q]

    sigma = np.exp(log_sigma)
    mu_i = mu + x @ beta
    log_t = np.log(np.clip(t, 1e-15, None))
    z = (log_t - mu_i) / sigma

    log_dens = (
        -0.5 * z ** 2
        - log_t
        - log_sigma
        - 0.5 * np.log(2.0 * np.pi)
    )
    log_surv = np.log(np.clip(1.0 - ndtr(z), 1e-15, None))

    ll = np.sum(w * event * log_dens + w * (1.0 - event) * log_surv)
    return -ll


def m_step_lognormal(
    t: np.ndarray,
    x: np.ndarray,
    event: np.ndarray,
    w: np.ndarray,
    init_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """M-step: update log-normal AFT latency parameters.

    Parameters
    ----------
    t : ndarray of shape (n,)
        Observed durations.
    x : ndarray of shape (n, q)
        Latency covariates.
    event : ndarray of shape (n,)
        Event indicator.
    w : ndarray of shape (n,)
        E-step weights.
    init_params : ndarray or None
        Initial [mu, log_sigma, beta...]. If None, initialised from events.

    Returns
    -------
    params : ndarray
        Optimised [mu, log_sigma, beta...] vector.
    """
    q = x.shape[1]
    if init_params is None:
        t_ev = t[event == 1]
        if len(t_ev) < 2:
            t_ev = t[t > 0]
        log_t_ev = np.log(np.clip(t_ev, 1e-15, None))
        mu0 = np.mean(log_t_ev) if len(log_t_ev) > 0 else 0.0
        sigma0 = np.std(log_t_ev) if len(log_t_ev) > 1 else 1.0
        init_params = np.zeros(2 + q)
        init_params[0] = mu0
        init_params[1] = np.log(max(sigma0, 0.1))

    result = optimize.minimize(
        lognormal_neg_loglik,
        init_params,
        args=(t, x, event, w),
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    return result.x


def compute_loglik(
    pi: np.ndarray,
    dens_u: np.ndarray,
    surv_u: np.ndarray,
    event: np.ndarray,
) -> float:
    """Compute observed-data log-likelihood for convergence checking.

    log L = sum_i [ delta_i * log(pi_i * f_u_i)
                  + (1-delta_i) * log(pi_i * S_u_i + 1 - pi_i) ]

    Parameters
    ----------
    pi : ndarray of shape (n,)
        P(susceptible).
    dens_u : ndarray of shape (n,)
        Latency density f_u(t|x).
    surv_u : ndarray of shape (n,)
        Latency survival S_u(t|x).
    event : ndarray of shape (n,)
        Event indicator.

    Returns
    -------
    float
        Observed-data log-likelihood.
    """
    eps = 1e-15
    log_lik_event = np.log(np.clip(pi * dens_u, eps, None))
    log_lik_cens = np.log(np.clip(pi * surv_u + (1.0 - pi), eps, None))
    ll = np.sum(event * log_lik_event + (1.0 - event) * log_lik_cens)
    return float(ll)
