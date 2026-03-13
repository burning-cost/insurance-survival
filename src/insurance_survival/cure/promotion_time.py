"""Promotion time (non-mixture) cure model.

Also known as the bounded cumulative hazard model. Provides a proportional
hazards structure for the full population — unlike the MCM, which does not.
Useful as a comparison model to check whether the MCM's explicit incidence
probability is warranted by the data.

The biological motivation: theta(x) represents the expected number of
competing causes (e.g. accident risk factors). If theta(x) is Poisson
distributed, the population survival is:

    S_pop(t) = exp(-theta(x) * F(t))

where F(t) is a proper CDF (the time-to-first-cause distribution).
Cure fraction = exp(-theta(x)), which approaches 1 as theta -> 0.

Reference: Tsodikov (1998), JRSS-B 60:195-207.
Reference: Chen, Ibrahim & Sinha (1999), JASA 94:909-919.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

from ._base import BaseMixtureCure, MCMResult, _parse_formula


class PromotionTimeCure:
    """Non-mixture (promotion time) cure model.

    Models the population survival as:

        S_pop(t|x) = exp(-theta(x) * F(t))

    where theta(x) = exp(x' beta) > 0 and F(t) is a Weibull CDF.
    The cure fraction is exp(-theta(x)), varying per policyholder.

    Unlike the MCM, this model maintains proportional hazards for the
    full population. There is no explicit incidence sub-model: the
    cure fraction emerges from the asymptote of S_pop as t -> infinity.

    Parameters
    ----------
    formula : str
        Right-hand side formula for covariates entering theta(x).
    max_iter : int
        Maximum iterations for L-BFGS-B optimisation. Default 500.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    result_ : dict
        Fitted parameters and diagnostics, available after ``fit()``.
    """

    def __init__(
        self,
        formula: str,
        max_iter: int = 500,
        random_state: Optional[int] = None,
    ) -> None:
        self.formula = formula
        self.max_iter = max_iter
        self.random_state = random_state

        self._fitted = False
        self._beta: Optional[np.ndarray] = None
        self._intercept: Optional[float] = None
        self._log_lambda: Optional[float] = None
        self._log_rho: Optional[float] = None
        self._col_names: Optional[list[str]] = None
        self.result_: Optional[dict] = None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _neg_loglik(
        self,
        params: np.ndarray,
        t: np.ndarray,
        x: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Negative observed-data log-likelihood for NMCM."""
        q = x.shape[1]
        intercept = params[0]
        beta = params[1 : 1 + q]
        log_lambda = params[1 + q]
        log_rho = params[2 + q]

        # theta(x) = exp(intercept + x @ beta)
        log_theta = intercept + x @ beta
        theta = np.exp(log_theta)

        # Weibull CDF: F(t) = 1 - exp(-(t/scale)^rho)
        rho = np.exp(log_rho)
        scale = np.exp(log_lambda)
        ratio = np.clip(t / scale, 1e-15, 1e10)
        F = 1.0 - np.exp(-(ratio ** rho))
        f = (rho / scale) * (ratio ** (rho - 1)) * np.exp(-(ratio ** rho))

        # S_pop = exp(-theta * F)
        log_S_pop = -theta * F

        # Density of population: f_pop = theta * f * S_pop
        log_f_pop = log_theta + np.log(np.clip(f, 1e-15, None)) + log_S_pop

        eps = 1e-15
        ll_event = np.sum(event * log_f_pop)
        ll_cens = np.sum((1.0 - event) * log_S_pop)
        return -(ll_event + ll_cens)

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "PromotionTimeCure":
        """Fit the promotion time cure model by direct MLE.

        Parameters
        ----------
        df : DataFrame
            Training data.
        duration_col : str
            Duration column name.
        event_col : str
            Event indicator column name.

        Returns
        -------
        self
        """
        t = df[duration_col].astype(float).to_numpy()
        event = df[event_col].astype(float).to_numpy()

        if np.any(t <= 0):
            raise ValueError("Duration column contains non-positive values.")

        x, col_names = _parse_formula(self.formula, df)
        self._col_names = col_names
        q = x.shape[1]

        rng = np.random.default_rng(self.random_state)
        # Initialise
        t_ev = t[event == 1]
        init_params = np.zeros(3 + q)
        init_params[0] = -1.0  # intercept for log(theta)
        init_params[q + 1] = np.log(np.mean(t_ev)) if len(t_ev) > 0 else 0.0
        init_params[q + 2] = 0.0  # log_rho = 0 => rho = 1

        result = optimize.minimize(
            self._neg_loglik,
            init_params,
            args=(t, x, event),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": 1e-9},
        )

        params = result.x
        self._intercept = float(params[0])
        self._beta = params[1 : 1 + q]
        self._log_lambda = float(params[1 + q])
        self._log_rho = float(params[2 + q])
        self._fitted = True

        # Compute cure fractions
        theta = np.exp(self._intercept + x @ self._beta)
        cure_frac = np.exp(-theta)

        self.result_ = {
            "converged": result.success,
            "log_likelihood": float(-result.fun),
            "n_obs": len(t),
            "n_events": int(np.sum(event)),
            "cure_fraction_mean": float(np.mean(cure_frac)),
            "intercept": self._intercept,
            "beta": dict(zip(col_names, self._beta)),
            "log_lambda": self._log_lambda,
            "log_rho": self._log_rho,
            "rho": float(np.exp(self._log_rho)),
            "scale": float(np.exp(self._log_lambda)),
        }
        return self

    def predict_cure_fraction(self, df: pd.DataFrame) -> np.ndarray:
        """Predict cure fraction exp(-theta(x)) per policyholder.

        Parameters
        ----------
        df : DataFrame
            Data with formula covariates.

        Returns
        -------
        cure : ndarray of shape (n,)
        """
        self._check_fitted()
        x, _ = _parse_formula(self.formula, df)
        theta = np.exp(self._intercept + x @ self._beta)
        return np.exp(-theta)

    def predict_susceptibility(self, df: pd.DataFrame) -> np.ndarray:
        """Predict 1 - cure fraction per policyholder.

        Parameters
        ----------
        df : DataFrame
            Data with formula covariates.

        Returns
        -------
        suscept : ndarray of shape (n,)
        """
        return 1.0 - self.predict_cure_fraction(df)

    def predict_population_survival(
        self,
        df: pd.DataFrame,
        times: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Predict S_pop(t|x) = exp(-theta(x) * F(t)) at specified times.

        Parameters
        ----------
        df : DataFrame
            Data with formula covariates.
        times : array-like
            Time points.

        Returns
        -------
        DataFrame of shape (n_rows, n_times)
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        x, _ = _parse_formula(self.formula, df)

        theta = np.exp(self._intercept + x @ self._beta)
        rho = np.exp(self._log_rho)
        scale = np.exp(self._log_lambda)

        results = {}
        for t_val in times:
            ratio = np.clip(t_val / scale, 1e-15, 1e10)
            F = 1.0 - np.exp(-(ratio ** rho))
            s_pop = np.exp(-theta * F)
            results[t_val] = s_pop

        return pd.DataFrame(results, index=df.index)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"PromotionTimeCure(formula='{self.formula}', status={status})"
