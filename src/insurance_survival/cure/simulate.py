"""Synthetic data generators for mixture cure model validation.

Generates realistic motor insurance panel data with a known true cure
fraction. Used to validate that the EM algorithm recovers the true
parameters and to benchmark the Qn test.

The motor panel structure mirrors real UK private car data:
- Each row is a policy-year observation
- Policyholders are either structurally immune or susceptible (latent)
- Susceptible policyholders have a Weibull-distributed time to first claim
- Censoring occurs at policy cancellation or end of observation window
- NCB years, driver age, and vehicle age are generated as covariates
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def simulate_motor_panel(
    n_policies: int = 2000,
    n_years: int = 5,
    cure_fraction: float = 0.4,
    weibull_shape: float = 1.2,
    weibull_scale: float = 36.0,
    censoring_rate: float = 0.15,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic motor insurance panel dataset.

    Each policyholder is either immune (structural non-claimer) or
    susceptible. Susceptible policyholders have a Weibull time-to-claim.
    Observations are censored at policy cancellation or end of window.

    Covariates generated:
    - ``ncb_years``: No-claims bonus years [0, 9]. Higher NCB => more likely immune.
    - ``age``: Driver age [18, 80]. Younger/older => more likely susceptible.
    - ``vehicle_age``: Vehicle age in years [0, 15]. Older => slightly more susceptible.

    The true cure fraction is modulated by ncb_years so that higher-NCB
    policyholders have higher P(immune), making the incidence sub-model
    identifiable.

    Parameters
    ----------
    n_policies : int
        Number of unique policyholders. Default 2000.
    n_years : int
        Maximum observation years per policyholder. Default 5.
    cure_fraction : float
        Overall (population-level) cure fraction. Default 0.4.
    weibull_shape : float
        Weibull shape parameter for susceptible policyholders. Default 1.2.
    weibull_scale : float
        Weibull scale in months for susceptible policyholders. Default 36.
    censoring_rate : float
        Annual probability of policy lapsing (administrative censoring).
        Default 0.15.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns:
        - policy_id : int
        - ncb_years : int [0, 9]
        - age : int [18, 80]
        - vehicle_age : int [0, 15]
        - is_immune : bool — true latent status (not known in practice)
        - tenure_months : float — observed duration
        - claimed : int — 1 if event observed, 0 if censored
        - true_cure_prob : float — true P(immune) from incidence model
    """
    rng = np.random.default_rng(seed)

    # Generate covariates
    ncb_years = rng.integers(0, 10, size=n_policies)
    age = rng.integers(18, 81, size=n_policies)
    vehicle_age = rng.integers(0, 16, size=n_policies)

    # True incidence sub-model: logit P(susceptible) = intercept + gamma * z
    # Negative gamma_ncb means higher NCB => lower P(susceptible) => higher P(immune)
    # This is the correct actuarial direction: more NCB history = more likely immune.
    gamma_ncb = -0.3    # higher NCB => less susceptible
    gamma_age = 0.02    # slightly older/younger => more susceptible (linear approximation)
    gamma_vehicle = 0.05  # older vehicle => slightly more susceptible

    from scipy.special import expit
    from scipy.optimize import brentq

    linear = (
        gamma_ncb * ncb_years
        + gamma_age * (age - 40)
        + gamma_vehicle * vehicle_age
    )

    def mean_cure(intercept):
        return np.mean(1.0 - expit(intercept + linear)) - cure_fraction

    try:
        gamma_0 = brentq(mean_cure, -100, 100)
    except ValueError:
        gamma_0 = 0.0

    logit_suscept = gamma_0 + linear
    pi_true = expit(logit_suscept)  # P(susceptible)
    true_cure_prob = 1.0 - pi_true   # P(immune)

    # Latent immune/susceptible status
    is_immune = rng.random(size=n_policies) > pi_true

    # Generate time-to-claim for susceptibles (months)
    # Scale modulated slightly by ncb_years: higher NCB => slightly longer latency
    # (experienced drivers may drive more carefully, delaying any eventual claim)
    scale_i = weibull_scale * np.exp(0.05 * ncb_years)
    u = rng.random(size=n_policies)
    # Weibull quantile: t = scale * (-log(u))^(1/shape)
    time_to_claim = scale_i * (-np.log(np.clip(u, 1e-10, 1.0))) ** (1.0 / weibull_shape)

    # Maximum observation time per policyholder
    max_obs_months = n_years * 12.0

    # Administrative censoring: annual dropout
    # Geometric with rate censoring_rate per year
    annual_surv = 1.0 - censoring_rate
    cens_u = rng.random(size=n_policies)
    # Time to censoring (months): geometric in years, convert
    with np.errstate(divide="ignore"):
        cens_years = np.where(
            annual_surv > 0,
            np.log(cens_u) / np.log(annual_surv),
            np.inf,
        )
    cens_months = np.clip(cens_years * 12.0, 0.1, max_obs_months)

    rows = []
    for i in range(n_policies):
        if is_immune[i]:
            # Immune: will never claim. Censored at min(cens, max_obs)
            obs_time = min(cens_months[i], max_obs_months)
            claimed = 0
        else:
            # Susceptible: claim at time_to_claim[i] if within window
            ttc = time_to_claim[i]
            obs_time = min(ttc, cens_months[i], max_obs_months)
            claimed = int(ttc <= cens_months[i] and ttc <= max_obs_months)

        rows.append({
            "policy_id": i,
            "ncb_years": int(ncb_years[i]),
            "age": int(age[i]),
            "vehicle_age": int(vehicle_age[i]),
            "is_immune": bool(is_immune[i]),
            "tenure_months": float(obs_time),
            "claimed": claimed,
            "true_cure_prob": float(true_cure_prob[i]),
        })

    df = pd.DataFrame(rows)
    # Ensure minimum duration
    df["tenure_months"] = df["tenure_months"].clip(lower=0.1)
    return df


def simulate_pet_panel(
    n_policies: int = 1500,
    cure_fraction: float = 0.35,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a synthetic pet insurance dataset.

    Simpler than the motor panel: single observation per policy
    (no multi-year structure). Covariates: pet age, breed risk index,
    indoor/outdoor flag.

    Parameters
    ----------
    n_policies : int
        Number of policies. Default 1500.
    cure_fraction : float
        Population cure fraction. Default 0.35.
    seed : int or None
        Random seed.

    Returns
    -------
    DataFrame with columns:
        - policy_id : int
        - pet_age : int [0, 14]
        - breed_risk : float — breed-level risk index in [0, 1]
        - indoor : int — 1 if indoor pet, 0 if outdoor
        - tenure_months : float — observed duration
        - claimed : int — event indicator
        - true_cure_prob : float
    """
    rng = np.random.default_rng(seed)

    pet_age = rng.integers(0, 15, size=n_policies)
    breed_risk = rng.uniform(0, 1, size=n_policies)
    indoor = rng.integers(0, 2, size=n_policies)

    from scipy.special import expit
    from scipy.optimize import brentq

    # Higher breed risk => more susceptible; indoor => less susceptible
    gamma_age = 0.08
    gamma_breed = 1.5
    gamma_indoor = -0.5

    linear = (
        gamma_age * (pet_age - 7)
        + gamma_breed * (breed_risk - 0.5)
        + gamma_indoor * indoor
    )

    def mean_cure(intercept):
        return np.mean(1.0 - expit(intercept + linear)) - cure_fraction

    try:
        gamma_0 = brentq(mean_cure, -100, 100)
    except ValueError:
        gamma_0 = 0.0

    pi_true = expit(gamma_0 + linear)
    true_cure_prob = 1.0 - pi_true

    is_immune = rng.random(size=n_policies) > pi_true

    # Log-normal time to claim: mu ~ 2.5 (months), sigma ~ 0.8
    mu_i = 2.5 + 0.3 * (pet_age - 7) / 7.0
    log_t = rng.normal(mu_i, 0.8, size=n_policies)
    time_to_claim = np.exp(log_t)

    max_obs = 24.0
    cens_months = rng.uniform(12.0, max_obs, size=n_policies)

    rows = []
    for i in range(n_policies):
        if is_immune[i]:
            obs_time = min(cens_months[i], max_obs)
            claimed = 0
        else:
            ttc = time_to_claim[i]
            obs_time = min(ttc, cens_months[i], max_obs)
            claimed = int(ttc <= cens_months[i] and ttc <= max_obs)

        rows.append({
            "policy_id": i,
            "pet_age": int(pet_age[i]),
            "breed_risk": float(breed_risk[i]),
            "indoor": int(indoor[i]),
            "tenure_months": float(max(obs_time, 0.1)),
            "claimed": claimed,
            "true_cure_prob": float(true_cure_prob[i]),
        })

    return pd.DataFrame(rows)
