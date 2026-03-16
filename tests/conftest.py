"""
Test fixtures for insurance_survival.

All synthetic data generated here follows known DGPs so tests can validate
parameter recovery and output correctness against analytical benchmarks.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.special import expit


# ---------------------------------------------------------------------------
# Cure model DGP
# ---------------------------------------------------------------------------

def make_cure_dgp(
    n: int = 2000,
    cure_intercept: float = -1.5,
    cure_ncd_coef: float = 0.3,
    weibull_scale: float = 2.0,
    weibull_shape: float = 1.5,
    censoring_rate: float = 0.4,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic survival data from a known cure model DGP.

    Parameters
    ----------
    n : int
        Number of policies.
    cure_intercept : float
        Intercept in the cure fraction logistic model. -1.5 -> ~18% baseline
        cure fraction.
    cure_ncd_coef : float
        NCD level coefficient in the cure fraction logistic model.
    weibull_scale : float
        Weibull AFT scale parameter for the uncured subgroup.
    weibull_shape : float
        Weibull AFT shape parameter.
    censoring_rate : float
        Approximate fraction of observations to right-censor.
    seed : int
        Random seed.

    Returns
    -------
    pl.DataFrame
        Columns: policy_id, stop, event, ncd_years, cure_prob_true.
    """
    rng = np.random.default_rng(seed)

    # Draw NCD levels from Poisson(3), clipped [0, 9]
    ncd = rng.poisson(lam=3.0, size=n).clip(0, 9)

    # Cure probability: sigmoid(intercept + coef * ncd)
    cure_logit = cure_intercept + cure_ncd_coef * ncd
    cure_prob = expit(cure_logit)

    # Is each policy cured?
    is_cured = rng.binomial(1, cure_prob).astype(bool)

    # Generate survival times
    # Weibull parameterisation: scale lambda, shape rho
    # Draw U ~ Uniform(0,1), invert CDF: T = lambda * (-log U)^(1/rho)
    u = rng.uniform(size=n)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    weibull_times = weibull_scale * (-np.log(u)) ** (1.0 / weibull_shape)

    # Observation window: max observed time = 8 years
    obs_window = 8.0

    # Censoring times
    censor_times = rng.uniform(obs_window * (1.0 - censoring_rate), obs_window, size=n)

    # Observed time and event indicator
    observed_time = np.where(
        is_cured,
        rng.uniform(obs_window * 0.7, obs_window, size=n),
        np.minimum(weibull_times, censor_times),
    )
    event = np.where(
        is_cured,
        0,
        (weibull_times <= censor_times).astype(int),
    )

    observed_time = np.clip(observed_time, 0.01, obs_window)

    return pl.DataFrame({
        "policy_id": [f"POL{i:05d}" for i in range(n)],
        "stop": observed_time.tolist(),
        "event": event.tolist(),
        "ncd_years": ncd.tolist(),
        "cure_prob_true": cure_prob.tolist(),
    })


# ---------------------------------------------------------------------------
# Transaction table DGP (for ExposureTransformer)
# ---------------------------------------------------------------------------

def make_transaction_dgp(
    n_policies: int = 200,
    seed: int = 42,
    cutoff: date = date(2025, 12, 31),
) -> pl.DataFrame:
    """Generate synthetic policy transaction table.

    Includes:
    - Simple inception-only policies (censored)
    - Multi-renewal policies with eventual lapse
    - Policies with MTA (covariate update mid-year)
    - Left-truncated policies

    Parameters
    ----------
    n_policies : int
        Number of policies to generate.
    seed : int
        Random seed.
    cutoff : date
        Observation cutoff date.

    Returns
    -------
    pl.DataFrame
        Policy transaction table matching ExposureTransformer input schema.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    base_date = date(2020, 1, 1)

    for i in range(n_policies):
        policy_id = f"POL{i:05d}"
        days_offset = int(rng.integers(0, 3 * 365))
        inception = base_date + timedelta(days=days_offset)
        ncd = int(rng.integers(0, 6))
        premium = float(rng.uniform(300, 800))
        vehicle_age = int(rng.integers(0, 10))
        policyholder_age = int(rng.integers(25, 70))
        channel = rng.choice(["direct", "aggregator", "broker"])

        expiry = inception + timedelta(days=365)
        rows.append({
            "policy_id": policy_id,
            "transaction_date": inception,
            "transaction_type": "inception",
            "inception_date": inception,
            "expiry_date": expiry,
            "ncd_years": ncd,
            "annual_premium": premium,
            "vehicle_age": vehicle_age,
            "policyholder_age": policyholder_age,
            "channel": channel,
            "event_type": None,
        })

        fate = rng.choice(
            ["censor", "lapse_yr1", "renew_then_lapse", "mta_then_lapse"],
            p=[0.3, 0.25, 0.30, 0.15],
        )

        if fate == "lapse_yr1":
            lapse_date = inception + timedelta(days=int(rng.integers(30, 350)))
            if lapse_date < cutoff:
                rows.append({
                    "policy_id": policy_id,
                    "transaction_date": lapse_date,
                    "transaction_type": "nonrenewal",
                    "inception_date": inception,
                    "expiry_date": expiry,
                    "ncd_years": ncd,
                    "annual_premium": premium,
                    "vehicle_age": vehicle_age,
                    "policyholder_age": policyholder_age,
                    "channel": channel,
                    "event_type": "lapse",
                })

        elif fate == "renew_then_lapse":
            n_renewals = int(rng.integers(1, 3))
            current_expiry = expiry
            current_ncd = ncd
            for r in range(n_renewals):
                renewal_date = current_expiry
                if renewal_date >= cutoff:
                    break
                next_expiry = renewal_date + timedelta(days=365)
                current_ncd = min(current_ncd + 1, 9)
                rows.append({
                    "policy_id": policy_id,
                    "transaction_date": renewal_date,
                    "transaction_type": "renewal",
                    "inception_date": inception,
                    "expiry_date": next_expiry,
                    "ncd_years": current_ncd,
                    "annual_premium": premium * (1.0 - current_ncd * 0.02),
                    "vehicle_age": vehicle_age + r + 1,
                    "policyholder_age": policyholder_age + r + 1,
                    "channel": channel,
                    "event_type": None,
                })
                current_expiry = next_expiry
            lapse_date = current_expiry
            if lapse_date < cutoff:
                rows.append({
                    "policy_id": policy_id,
                    "transaction_date": lapse_date,
                    "transaction_type": "nonrenewal",
                    "inception_date": inception,
                    "expiry_date": current_expiry,
                    "ncd_years": current_ncd,
                    "annual_premium": premium,
                    "vehicle_age": vehicle_age + n_renewals,
                    "policyholder_age": policyholder_age + n_renewals,
                    "channel": channel,
                    "event_type": "lapse",
                })

        elif fate == "mta_then_lapse":
            mta_date = inception + timedelta(days=int(rng.integers(60, 200)))
            if mta_date < cutoff:
                rows.append({
                    "policy_id": policy_id,
                    "transaction_date": mta_date,
                    "transaction_type": "mta",
                    "inception_date": inception,
                    "expiry_date": expiry,
                    "ncd_years": ncd,
                    "annual_premium": premium * 1.05,
                    "vehicle_age": vehicle_age,
                    "policyholder_age": policyholder_age,
                    "channel": channel,
                    "event_type": None,
                })
            lapse_date = expiry
            if lapse_date < cutoff:
                rows.append({
                    "policy_id": policy_id,
                    "transaction_date": lapse_date,
                    "transaction_type": "nonrenewal",
                    "inception_date": inception,
                    "expiry_date": expiry,
                    "ncd_years": ncd,
                    "annual_premium": premium,
                    "vehicle_age": vehicle_age,
                    "policyholder_age": policyholder_age,
                    "channel": channel,
                    "event_type": "lapse",
                })

    df = pl.DataFrame(rows).with_columns([
        pl.col("transaction_date").cast(pl.Date),
        pl.col("inception_date").cast(pl.Date),
        pl.col("expiry_date").cast(pl.Date),
        pl.col("ncd_years").cast(pl.Int32),
        pl.col("vehicle_age").cast(pl.Int32),
        pl.col("policyholder_age").cast(pl.Int32),
        pl.col("channel").cast(pl.Utf8),
        pl.col("event_type").cast(pl.Utf8),
    ])
    return df


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cure_dgp() -> pl.DataFrame:
    """Standard cure model DGP: 2000 policies, known parameters."""
    return make_cure_dgp(n=2000, seed=42)


@pytest.fixture(scope="session")
def small_cure_dgp() -> pl.DataFrame:
    """Small cure model DGP for fast tests: 500 policies."""
    return make_cure_dgp(n=500, seed=99)


@pytest.fixture(scope="session")
def transaction_dgp() -> pl.DataFrame:
    """Transaction table DGP for ExposureTransformer tests."""
    return make_transaction_dgp(n_policies=200, seed=42, cutoff=date(2025, 12, 31))


@pytest.fixture(scope="session")
def observation_cutoff() -> date:
    return date(2025, 12, 31)


@pytest.fixture(scope="session")
def fitted_cure_fitter(small_cure_dgp: pl.DataFrame) -> object:
    """Pre-fitted WeibullMixtureCureFitter for reuse in multiple tests."""
    from insurance_survival import WeibullMixtureCureFitter
    fitter = WeibullMixtureCureFitter(
        cure_covariates=["ncd_years"],
        uncured_covariates=["ncd_years"],
        penalizer=0.05,
        max_iter=200,
    )
    fitter.fit(small_cure_dgp, duration_col="stop", event_col="event")
    return fitter


@pytest.fixture(scope="session")
def fitted_lifelines_fitter(small_cure_dgp: pl.DataFrame) -> object:
    """Pre-fitted lifelines WeibullAFTFitter for CLV and LapseTable tests."""
    from lifelines import WeibullAFTFitter
    fitter = WeibullAFTFitter()
    # Drop non-numeric string columns that lifelines cannot handle
    fit_df = small_cure_dgp.select(["stop", "event", "ncd_years"]).to_pandas()
    fitter.fit(
        fit_df,
        duration_col="stop",
        event_col="event",
    )
    return fitter


# ---------------------------------------------------------------------------
# Cure subpackage fixtures (insurance_survival.cure)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def motor_df():
    """Small motor dataset with known cure fraction ~0.40."""
    from insurance_survival.cure.simulate import simulate_motor_panel
    return simulate_motor_panel(n_policies=500, cure_fraction=0.40, seed=42)


@pytest.fixture(scope="session")
def motor_df_large():
    """Larger motor dataset for convergence tests."""
    from insurance_survival.cure.simulate import simulate_motor_panel
    return simulate_motor_panel(n_policies=1500, cure_fraction=0.35, seed=99)


@pytest.fixture(scope="session")
def pet_df():
    """Pet insurance dataset."""
    from insurance_survival.cure.simulate import simulate_pet_panel
    return simulate_pet_panel(n_policies=500, cure_fraction=0.35, seed=7)


@pytest.fixture(scope="session")
def fitted_weibull(motor_df):
    """Fitted WeibullMixtureCure model (session-scoped for speed)."""
    from insurance_survival.cure import WeibullMixtureCure
    model = WeibullMixtureCure(
        incidence_formula="ncd_years + age + vehicle_age",
        latency_formula="ncd_years + age",
        n_em_starts=2,
        max_iter=50,
        random_state=42,
    )
    model.fit(motor_df, duration_col="tenure_months", event_col="claimed")
    return model


@pytest.fixture(scope="session")
def fitted_lognormal(motor_df):
    """Fitted LogNormalMixtureCure model."""
    from insurance_survival.cure import LogNormalMixtureCure
    model = LogNormalMixtureCure(
        incidence_formula="ncd_years + age",
        latency_formula="ncd_years",
        n_em_starts=2,
        max_iter=50,
        random_state=42,
    )
    model.fit(motor_df, duration_col="tenure_months", event_col="claimed")
    return model


@pytest.fixture(scope="session")
def fitted_promotion(motor_df):
    """Fitted PromotionTimeCure model."""
    from insurance_survival.cure import PromotionTimeCure
    model = PromotionTimeCure(
        formula="ncd_years + age + vehicle_age",
        random_state=42,
    )
    model.fit(motor_df, duration_col="tenure_months", event_col="claimed")
    return model
