"""
Data generating process (DGP) simulation for validation and testing.

Simulates recurrent insurance claims under:
1. Andersen-Gill with gamma frailty — the standard model
2. PWP process — event-dependent intensities
3. Joint process — recurrent + terminal (lapse)

The simulations use the conditional intensity representation:
    lambda_i(t) = z_i * lambda_0(t) * exp(X_i' beta)

with z_i drawn from the specified frailty distribution. Events are
generated via thinning of a homogeneous Poisson process (Ogata's
modified thinning algorithm) or direct exponential waiting times
for piecewise-constant baselines.

These simulations are used in tests to verify that:
1. The estimators recover the true parameters (low bias)
2. Coverage of confidence intervals is approximately nominal
3. The EM algorithm converges reliably
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import RecurrentEventData


@dataclass
class SimulationParams:
    """
    Parameters for recurrent event simulation.

    Parameters
    ----------
    n_subjects : int
        Number of policyholders.
    baseline_rate : float
        Baseline claim rate (events per unit time).
    beta : np.ndarray
        True covariate coefficients.
    theta : float
        Frailty dispersion. Var[z_i] = 1/theta for gamma frailty.
    follow_up : float
        Maximum follow-up time per subject.
    frailty_dist : str
        "gamma" or "lognormal".
    covariate_means : np.ndarray or None
        Mean of each covariate (standard normal by default).
    covariate_stds : np.ndarray or None
        Std dev of each covariate.
    random_state : int or None
        Random seed for reproducibility.
    """

    n_subjects: int = 500
    baseline_rate: float = 0.3
    beta: np.ndarray = None  # type: ignore
    theta: float = 2.0
    follow_up: float = 3.0
    frailty_dist: str = "gamma"
    covariate_means: Optional[np.ndarray] = None
    covariate_stds: Optional[np.ndarray] = None
    random_state: Optional[int] = 42

    def __post_init__(self) -> None:
        if self.beta is None:
            self.beta = np.array([0.3, -0.2])


def simulate_ag_frailty(
    params: Optional[SimulationParams] = None,
    **kwargs,
) -> RecurrentEventData:
    """
    Simulate recurrent claims under the Andersen-Gill gamma frailty model.

    Events for subject i follow a non-homogeneous Poisson process with
    intensity z_i * lambda_0 * exp(X_i' beta), where z_i ~ Gamma(theta, theta).

    Parameters
    ----------
    params : SimulationParams, optional
        If None, uses default parameters.
    **kwargs
        Override specific SimulationParams fields.

    Returns
    -------
    RecurrentEventData
        Ready for fitting with AndersenGillFrailty.

    Examples
    --------
    >>> data = simulate_ag_frailty(SimulationParams(n_subjects=200, theta=1.5))
    >>> model = AndersenGillFrailty().fit(data)
    """
    if params is None:
        params = SimulationParams(**kwargs)
    else:
        if kwargs:
            # Apply overrides
            import dataclasses
            params = dataclasses.replace(params, **kwargs)

    rng = np.random.default_rng(params.random_state)
    p = len(params.beta)

    # Covariate generation
    if params.covariate_means is None:
        cov_means = np.zeros(p)
    else:
        cov_means = np.asarray(params.covariate_means)
    if params.covariate_stds is None:
        cov_stds = np.ones(p)
    else:
        cov_stds = np.asarray(params.covariate_stds)

    # Draw covariates
    X = rng.normal(loc=cov_means, scale=cov_stds, size=(params.n_subjects, p))
    linear_pred = X @ params.beta

    # Draw frailties
    if params.frailty_dist == "gamma":
        z = rng.gamma(shape=params.theta, scale=1.0 / params.theta, size=params.n_subjects)
    elif params.frailty_dist == "lognormal":
        sigma2 = 1.0 / params.theta
        z = np.exp(rng.normal(loc=-sigma2 / 2, scale=np.sqrt(sigma2), size=params.n_subjects))
    else:
        raise ValueError(f"Unknown frailty distribution: {params.frailty_dist!r}")

    # Simulate event times via exponential inter-arrival times
    records = []
    covariate_names = [f"x{j+1}" for j in range(p)]

    for i in range(params.n_subjects):
        rate_i = z[i] * params.baseline_rate * np.exp(linear_pred[i])
        t = 0.0
        cov_row = {f"x{j+1}": float(X[i, j]) for j in range(p)}

        while t < params.follow_up:
            # Draw next event time: exponential with rate_i
            wait = rng.exponential(1.0 / max(rate_i, 1e-10))
            t_event = t + wait

            if t_event >= params.follow_up:
                # Censored interval
                records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": params.follow_up,
                    "event": 0,
                    **cov_row,
                })
                break
            else:
                records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": t_event,
                    "event": 1,
                    **cov_row,
                })
                t = t_event

    df = pd.DataFrame(records)
    return RecurrentEventData.from_long_format(
        df,
        id_col="policy_id",
        start_col="t_start",
        stop_col="t_stop",
        event_col="event",
        covariates=covariate_names,
    )


def simulate_pwp(
    n_subjects: int = 500,
    baseline_rates: Sequence[float] = (0.4, 0.5, 0.6),
    beta: Optional[np.ndarray] = None,
    follow_up: float = 3.0,
    random_state: Optional[int] = 42,
) -> RecurrentEventData:
    """
    Simulate from a PWP (Prentice-Williams-Peterson) process.

    The baseline hazard depends on event number:
    - 1st claim: baseline_rates[0]
    - 2nd claim: baseline_rates[1]
    - 3rd+ claims: baseline_rates[2]

    Parameters
    ----------
    n_subjects : int
        Number of subjects.
    baseline_rates : sequence of float
        Baseline rates for each event number stratum.
    beta : np.ndarray or None
        Covariate effects. If None, uses [0.3, -0.2].
    follow_up : float
        Maximum follow-up.
    random_state : int or None
        Random seed.

    Returns
    -------
    RecurrentEventData
    """
    if beta is None:
        beta = np.array([0.3, -0.2])
    p = len(beta)
    rng = np.random.default_rng(random_state)
    baseline_rates = list(baseline_rates)
    max_stratum = len(baseline_rates)

    X = rng.normal(size=(n_subjects, p))
    linear_pred = X @ beta

    records = []
    for i in range(n_subjects):
        t = 0.0
        n_events = 0
        cov_row = {f"x{j+1}": float(X[i, j]) for j in range(p)}

        while t < follow_up:
            stratum = min(n_events, max_stratum - 1)
            rate_i = baseline_rates[stratum] * np.exp(linear_pred[i])
            wait = rng.exponential(1.0 / max(rate_i, 1e-10))
            t_event = t + wait

            if t_event >= follow_up:
                records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": follow_up,
                    "event": 0,
                    **cov_row,
                })
                break
            else:
                records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": t_event,
                    "event": 1,
                    **cov_row,
                })
                t = t_event
                n_events += 1

    df = pd.DataFrame(records)
    return RecurrentEventData.from_long_format(
        df,
        id_col="policy_id",
        start_col="t_start",
        stop_col="t_stop",
        event_col="event",
        covariates=[f"x{j+1}" for j in range(p)],
    )


def simulate_joint(
    n_subjects: int = 400,
    baseline_claim_rate: float = 0.3,
    baseline_lapse_rate: float = 0.2,
    beta_claims: Optional[np.ndarray] = None,
    beta_lapse: Optional[np.ndarray] = None,
    theta: float = 2.0,
    alpha: float = 1.0,
    gamma: float = 0.8,
    follow_up: float = 3.0,
    random_state: Optional[int] = 42,
) -> tuple[RecurrentEventData, pd.DataFrame]:
    """
    Simulate joint recurrent claims + terminal lapse process.

    Returns a tuple of (RecurrentEventData, terminal_df) ready for
    JointData construction.

    Parameters
    ----------
    n_subjects : int
        Number of policyholders.
    baseline_claim_rate : float
        Baseline claim intensity.
    baseline_lapse_rate : float
        Baseline lapse (terminal event) intensity.
    beta_claims : np.ndarray or None
        Covariate effects on claims.
    beta_lapse : np.ndarray or None
        Covariate effects on lapse.
    theta : float
        Frailty dispersion.
    alpha : float
        Frailty power for claims process.
    gamma : float
        Frailty power for lapse process.
    follow_up : float
        Maximum study duration.
    random_state : int or None
        Random seed.

    Returns
    -------
    tuple[RecurrentEventData, pd.DataFrame]
    """
    if beta_claims is None:
        beta_claims = np.array([0.3, -0.2])
    if beta_lapse is None:
        beta_lapse = np.array([0.1, 0.2])

    p = len(beta_claims)
    rng = np.random.default_rng(random_state)

    # Draw shared covariates and frailty
    X = rng.normal(size=(n_subjects, p))
    z = rng.gamma(shape=theta, scale=1.0 / theta, size=n_subjects)

    claim_records = []
    terminal_records = []

    for i in range(n_subjects):
        rate_claims = z[i] ** alpha * baseline_claim_rate * np.exp(X[i] @ beta_claims)
        rate_lapse = z[i] ** gamma * baseline_lapse_rate * np.exp(X[i] @ beta_lapse)
        cov_row = {f"x{j+1}": float(X[i, j]) for j in range(p)}

        t = 0.0
        lapsed = False
        lapse_time = follow_up

        # Draw lapse time first (competing risk)
        t_lapse = rng.exponential(1.0 / max(rate_lapse, 1e-10))
        if t_lapse < follow_up:
            lapse_time = t_lapse
            lapsed = True

        # Generate claims up to lapse
        while t < lapse_time:
            wait = rng.exponential(1.0 / max(rate_claims, 1e-10))
            t_event = t + wait
            if t_event >= lapse_time:
                claim_records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": lapse_time,
                    "event": 0,
                    **cov_row,
                })
                break
            else:
                claim_records.append({
                    "policy_id": i,
                    "t_start": t,
                    "t_stop": t_event,
                    "event": 1,
                    **cov_row,
                })
                t = t_event

        # Handle subjects with no intervals (immediate lapse)
        if not any(r["policy_id"] == i for r in claim_records):
            claim_records.append({
                "policy_id": i,
                "t_start": 0.0,
                "t_stop": max(lapse_time, 1e-6),
                "event": 0,
                **cov_row,
            })

        terminal_records.append({
            "policy_id": i,
            "lapse_time": lapse_time,
            "lapsed": int(lapsed),
            **cov_row,
        })

    claims_df = pd.DataFrame(claim_records)
    terminal_df = pd.DataFrame(terminal_records)

    rec_data = RecurrentEventData.from_long_format(
        claims_df,
        id_col="policy_id",
        start_col="t_start",
        stop_col="t_stop",
        event_col="event",
        covariates=[f"x{j+1}" for j in range(p)],
    )
    return rec_data, terminal_df
