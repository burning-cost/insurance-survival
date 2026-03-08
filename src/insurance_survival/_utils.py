"""
Internal helper functions for insurance_survival.

Not part of the public API. Subject to change without notice.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def weibull_sf(t: np.ndarray, scale: np.ndarray, shape: float) -> np.ndarray:
    """Weibull survival function S(t) = exp(-(t/scale)^shape).

    Parameters
    ----------
    t : np.ndarray
        Time values (must be >= 0).
    scale : np.ndarray
        Scale parameter lambda > 0. Can be scalar or array matching t.
    shape : float
        Shape parameter rho > 0.

    Returns
    -------
    np.ndarray
    """
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return np.exp(-((t / scale) ** shape))


def weibull_pdf(t: np.ndarray, scale: np.ndarray, shape: float) -> np.ndarray:
    """Weibull probability density function.

    f(t) = (rho/lambda) * (t/lambda)^(rho-1) * exp(-(t/lambda)^rho)
    """
    t = np.asarray(t, dtype=float)
    scale = np.asarray(scale, dtype=float)
    sf = weibull_sf(t, scale, shape)
    return (shape / scale) * ((t / scale) ** (shape - 1.0)) * sf


def weibull_median(scale: np.ndarray, shape: float) -> np.ndarray:
    """Median of Weibull distribution: scale * (log 2)^(1/shape)."""
    return scale * (np.log(2.0) ** (1.0 / shape))


# ---------------------------------------------------------------------------
# Polars / pandas conversion helpers
# ---------------------------------------------------------------------------

def to_polars(df: object) -> pl.DataFrame:
    """Accept a Polars or pandas DataFrame; always return Polars.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        Input data.

    Returns
    -------
    pl.DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return df
    try:
        import pandas as pd  # type: ignore
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except ImportError:
        pass
    raise TypeError(
        f"Expected pl.DataFrame or pd.DataFrame, got {type(df).__name__}"
    )


# ---------------------------------------------------------------------------
# NCD transition matrix helpers
# ---------------------------------------------------------------------------

def default_uk_ncd_transitions(max_ncd: int = 9) -> pl.DataFrame:
    """Build the standard UK motor NCD transition table.

    Convention: one step up for no claim, two steps down for a claim,
    floors at 0, ceiling at max_ncd.

    Parameters
    ----------
    max_ncd : int
        Maximum NCD level. Default 9.

    Returns
    -------
    pl.DataFrame
        Columns: from_ncd, to_ncd_no_claim, to_ncd_one_claim, claim_probability.
        claim_probability defaults to 0.10 for all levels.
    """
    rows = []
    for ncd in range(max_ncd + 1):
        no_claim = min(ncd + 1, max_ncd)
        one_claim = max(ncd - 2, 0)
        rows.append({
            "from_ncd": ncd,
            "to_ncd_no_claim": no_claim,
            "to_ncd_one_claim": one_claim,
            "claim_probability": 0.10,
        })
    return pl.DataFrame(rows)


def build_ncd_transition_matrix(
    transitions: pl.DataFrame,
    max_ncd: int,
) -> np.ndarray:
    """Convert a transitions DataFrame to a (max_ncd+1) x (max_ncd+1) matrix.

    T[i, j] = P(next NCD = j | current NCD = i)

    Parameters
    ----------
    transitions : pl.DataFrame
        Must contain: from_ncd, to_ncd_no_claim, to_ncd_one_claim, claim_probability.
    max_ncd : int
        Maximum NCD level index.

    Returns
    -------
    np.ndarray
        Shape (max_ncd+1, max_ncd+1).
    """
    n = max_ncd + 1
    M = np.zeros((n, n), dtype=float)
    for row in transitions.iter_rows(named=True):
        i = row["from_ncd"]
        if i > max_ncd:
            continue
        p_claim = row["claim_probability"]
        j_no = min(row["to_ncd_no_claim"], max_ncd)
        j_one = min(row["to_ncd_one_claim"], max_ncd)
        M[i, j_no] += (1.0 - p_claim)
        M[i, j_one] += p_claim
    return M


def expected_ncd_path(
    ncd_0: int,
    horizon: int,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Compute E[NCD(t)] for t = 1 .. horizon via Markov chain.

    Parameters
    ----------
    ncd_0 : int
        Initial NCD level.
    horizon : int
        Number of periods forward.
    transition_matrix : np.ndarray
        Shape (n_states, n_states). Row i sums to 1.

    Returns
    -------
    np.ndarray
        Shape (horizon,). Expected NCD level at each future period.
    """
    n = transition_matrix.shape[0]
    ncd_levels = np.arange(n, dtype=float)
    # State distribution: one-hot at ncd_0
    state = np.zeros(n, dtype=float)
    state[min(ncd_0, n - 1)] = 1.0
    expected = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        state = state @ transition_matrix
        expected[t] = state @ ncd_levels
    return expected


# ---------------------------------------------------------------------------
# Design matrix helpers
# ---------------------------------------------------------------------------

def build_design_matrix(
    df: pl.DataFrame,
    covariates: list[str],
    fit_intercept: bool = True,
) -> np.ndarray:
    """Extract covariates from a Polars DataFrame as a numpy matrix.

    Parameters
    ----------
    df : pl.DataFrame
    covariates : list[str]
        Column names to include.
    fit_intercept : bool
        If True, prepends a column of ones.

    Returns
    -------
    np.ndarray
        Shape (n_rows, n_covariates + int(fit_intercept)).
    """
    X = df.select(covariates).to_numpy().astype(float)
    if fit_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def coef_names(covariates: list[str], fit_intercept: bool = True) -> list[str]:
    """Return coefficient names including intercept if applicable."""
    if fit_intercept:
        return ["Intercept"] + list(covariates)
    return list(covariates)
