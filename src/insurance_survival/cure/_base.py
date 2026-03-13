"""Base class for mixture cure models.

Handles the shared interface: formula parsing, data preparation,
multiple EM restarts, bootstrap standard errors, and prediction.
All concrete MCM classes inherit from BaseMixtureCure.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ._em import compute_pi, e_step


class MCMResult:
    """Results container for a fitted mixture cure model.

    Attributes
    ----------
    converged : bool
        Whether the EM algorithm converged within ``max_iter`` iterations.
    n_iter : int
        Number of EM iterations completed.
    log_likelihood : float
        Observed-data log-likelihood at convergence.
    n_obs : int
        Number of observations used in fitting.
    n_events : int
        Number of observed events (uncensored).
    cure_fraction_mean : float
        Mean estimated cure fraction (1 - mean P(susceptible)) across
        the fitted sample.
    incidence_coef : dict
        Logistic incidence coefficients: {covariate_name: coefficient}.
    incidence_intercept : float
        Logistic incidence intercept.
    latency_params : dict
        Latency sub-model parameters (model-specific).
    se_incidence_coef : dict or None
        Bootstrap standard errors for incidence coefficients, if computed.
    se_latency_params : dict or None
        Bootstrap standard errors for latency parameters, if computed.
    """

    def __init__(
        self,
        converged: bool,
        n_iter: int,
        log_likelihood: float,
        n_obs: int,
        n_events: int,
        cure_fraction_mean: float,
        incidence_coef: dict,
        incidence_intercept: float,
        latency_params: dict,
        se_incidence_coef: Optional[dict] = None,
        se_latency_params: Optional[dict] = None,
    ) -> None:
        self.converged = converged
        self.n_iter = n_iter
        self.log_likelihood = log_likelihood
        self.n_obs = n_obs
        self.n_events = n_events
        self.cure_fraction_mean = cure_fraction_mean
        self.incidence_coef = incidence_coef
        self.incidence_intercept = incidence_intercept
        self.latency_params = latency_params
        self.se_incidence_coef = se_incidence_coef
        self.se_latency_params = se_latency_params

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"MCMResult({status}, "
            f"n={self.n_obs}, events={self.n_events}, "
            f"loglik={self.log_likelihood:.2f}, "
            f"cure_fraction={self.cure_fraction_mean:.3f})"
        )

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "Mixture Cure Model Results",
            "=" * 40,
            f"  Convergence    : {'Yes' if self.converged else 'No'} ({self.n_iter} iterations)",
            f"  Log-likelihood : {self.log_likelihood:.4f}",
            f"  Observations   : {self.n_obs}",
            f"  Events         : {self.n_events}",
            f"  Mean cure frac : {self.cure_fraction_mean:.4f}",
            "",
            "Incidence sub-model (logistic):",
            f"  Intercept : {self.incidence_intercept:.4f}",
        ]
        for name, coef in self.incidence_coef.items():
            se_str = ""
            if self.se_incidence_coef and name in self.se_incidence_coef:
                se_str = f"  (SE {self.se_incidence_coef[name]:.4f})"
            lines.append(f"  {name:20s}: {coef:.4f}{se_str}")

        lines.append("")
        lines.append("Latency sub-model:")
        for name, val in self.latency_params.items():
            se_str = ""
            if self.se_latency_params and name in self.se_latency_params:
                se_str = f"  (SE {self.se_latency_params[name]:.4f})"
            lines.append(f"  {name:20s}: {val:.4f}{se_str}")

        return "\n".join(lines)


def _parse_formula(formula: str, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Parse a Patsy-style formula (right-hand side only) into a design matrix.

    Supports additive formulas: ``"age + ncb_years + vehicle_age"``.
    Does not support interaction terms or transformations.
    Each named column is extracted from df and standardised to float.

    Parameters
    ----------
    formula : str
        Right-hand side formula string, e.g. ``"age + ncb_years"``.
    df : DataFrame
        Data frame containing the named columns.

    Returns
    -------
    X : ndarray of shape (n, p)
        Design matrix without intercept column.
    col_names : list of str
        Column names corresponding to X's columns.
    """
    raw_names = [c.strip() for c in formula.split("+")]
    col_names = []
    cols = []
    for name in raw_names:
        if not name:
            continue
        if name not in df.columns:
            raise ValueError(
                f"Column '{name}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        col_names.append(name)
        cols.append(df[name].astype(float).to_numpy())

    if not cols:
        raise ValueError(f"No valid columns found in formula '{formula}'")

    X = np.column_stack(cols)
    return X, col_names


class BaseMixtureCure(ABC):
    """Abstract base class for mixture cure models.

    Concrete subclasses implement ``_run_em_single()`` which performs one
    EM run from given initial parameters, and ``_compute_latency_surv_dens()``
    which evaluates the latency sub-model at given observations and params.

    Parameters
    ----------
    incidence_formula : str
        Right-hand side formula for the incidence sub-model.
        Example: ``"ncb_years + age_band + vehicle_age"``.
    latency_formula : str
        Right-hand side formula for the latency sub-model.
        Example: ``"ncb_years + age_band"``.
    n_em_starts : int
        Number of random EM restarts. The run with the highest
        log-likelihood is returned. Minimum 1.
    max_iter : int
        Maximum EM iterations per run.
    tol : float
        Convergence tolerance on log-likelihood change.
    bootstrap_se : bool
        Whether to compute bootstrap standard errors after fitting.
    n_bootstrap : int
        Number of bootstrap resamples for standard error estimation.
    n_jobs : int
        Number of parallel jobs for bootstrap. -1 uses all available cores.
    random_state : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        incidence_formula: str,
        latency_formula: str,
        n_em_starts: int = 5,
        max_iter: int = 200,
        tol: float = 1e-5,
        bootstrap_se: bool = False,
        n_bootstrap: int = 200,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
    ) -> None:
        self.incidence_formula = incidence_formula
        self.latency_formula = latency_formula
        self.n_em_starts = max(1, n_em_starts)
        self.max_iter = max_iter
        self.tol = tol
        self.bootstrap_se = bootstrap_se
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Set after fit()
        self._fitted = False
        self._gamma: Optional[np.ndarray] = None       # incidence coefs
        self._intercept: Optional[np.ndarray] = None   # incidence intercept
        self._latency_params: Optional[np.ndarray] = None  # model-specific
        self._incidence_cols: Optional[list[str]] = None
        self._latency_cols: Optional[list[str]] = None
        self.result_: Optional[MCMResult] = None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def _prepare_data(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract arrays from DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input data.
        duration_col : str
            Column with observed durations (must be > 0).
        event_col : str
            Column with event indicator (1 = event, 0 = censored).

        Returns
        -------
        t : ndarray of shape (n,)
        event : ndarray of shape (n,)
        z : ndarray of shape (n, p_inc) — incidence covariates
        x : ndarray of shape (n, p_lat) — latency covariates
        """
        t = df[duration_col].astype(float).to_numpy()
        event = df[event_col].astype(float).to_numpy()

        if np.any(t <= 0):
            raise ValueError(
                f"Duration column '{duration_col}' contains non-positive values. "
                "All durations must be strictly positive."
            )
        if not set(np.unique(event)).issubset({0.0, 1.0}):
            raise ValueError(
                f"Event column '{event_col}' must contain only 0 and 1."
            )

        z, self._incidence_cols = _parse_formula(self.incidence_formula, df)
        x, self._latency_cols = _parse_formula(self.latency_formula, df)
        return t, event, z, x

    @abstractmethod
    def _run_em_single(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool, int]:
        """Run a single EM optimisation from a random starting point.

        Returns
        -------
        loglik : float
        gamma : ndarray — incidence coefficients
        intercept : ndarray — incidence intercept (shape (1,))
        latency_params : ndarray — model-specific latency parameters
        converged : bool
        n_iter : int
        """
        ...

    @abstractmethod
    def _compute_latency_surv_dens(
        self,
        t: np.ndarray,
        x: np.ndarray,
        latency_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate latency sub-model.

        Parameters
        ----------
        t : ndarray of shape (n,)
        x : ndarray of shape (n, q)
        latency_params : ndarray

        Returns
        -------
        surv_u : ndarray of shape (n,) — S_u(t|x)
        dens_u : ndarray of shape (n,) — f_u(t|x)
        """
        ...

    @abstractmethod
    def _latency_param_names(self) -> list[str]:
        """Return names of latency parameters for reporting."""
        ...

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> "BaseMixtureCure":
        """Fit the mixture cure model using the EM algorithm.

        Multiple EM restarts are performed; the solution with the highest
        observed-data log-likelihood is retained.

        Parameters
        ----------
        df : DataFrame
            Training data. Must contain ``duration_col``, ``event_col``,
            and all columns referenced in ``incidence_formula`` and
            ``latency_formula``.
        duration_col : str
            Name of the duration / time-at-risk column.
        event_col : str
            Name of the event indicator column (1 = event, 0 = censored).

        Returns
        -------
        self
        """
        t, event, z, x = self._prepare_data(df, duration_col, event_col)
        rng = np.random.default_rng(self.random_state)

        best_loglik = -np.inf
        best_gamma = None
        best_intercept = None
        best_latency = None
        best_converged = False
        best_n_iter = 0

        for start_idx in range(self.n_em_starts):
            try:
                loglik, gamma, intercept, lat_params, converged, n_iter = (
                    self._run_em_single(t, event, z, x, rng)
                )
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_gamma = gamma
                    best_intercept = intercept
                    best_latency = lat_params
                    best_converged = converged
                    best_n_iter = n_iter
            except Exception as e:
                warnings.warn(
                    f"EM start {start_idx + 1} failed: {e}. Skipping.",
                    stacklevel=2,
                )

        if best_gamma is None:
            raise RuntimeError(
                "All EM starts failed. Check your data and formulae."
            )

        self._gamma = best_gamma
        self._intercept = best_intercept
        self._latency_params = best_latency

        # Final posterior weights and cure fraction
        surv_u, dens_u = self._compute_latency_surv_dens(t, x, best_latency)
        pi = compute_pi(z, best_gamma, best_intercept)
        w_final = e_step(pi, surv_u, event)

        n_obs = len(t)
        n_events = int(np.sum(event))
        cure_frac_mean = float(np.mean(1.0 - pi))

        # Build named coefficient dicts
        inc_coef = dict(zip(self._incidence_cols, best_gamma))
        lat_named = dict(zip(self._latency_param_names(), best_latency))

        # Bootstrap SEs
        se_inc = None
        se_lat = None
        if self.bootstrap_se:
            se_inc, se_lat = self._bootstrap_se(t, event, z, x, rng)

        self.result_ = MCMResult(
            converged=best_converged,
            n_iter=best_n_iter,
            log_likelihood=best_loglik,
            n_obs=n_obs,
            n_events=n_events,
            cure_fraction_mean=cure_frac_mean,
            incidence_coef=inc_coef,
            incidence_intercept=float(best_intercept[0]),
            latency_params=lat_named,
            se_incidence_coef=se_inc,
            se_latency_params=se_lat,
        )

        self._fitted = True
        # Store column lists and data details for later
        self._duration_col = duration_col
        self._event_col = event_col
        return self

    def _bootstrap_single(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        seed: int,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run a single bootstrap resample and return parameter estimates."""
        rng = np.random.default_rng(seed)
        n = len(t)
        idx = rng.integers(0, n, size=n)
        t_b = t[idx]
        event_b = event[idx]
        z_b = z[idx]
        x_b = x[idx]

        # Single EM run per bootstrap resample for speed
        try:
            _, gamma, _, lat_params, _, _ = self._run_em_single(
                t_b, event_b, z_b, x_b, rng
            )
            return gamma, lat_params
        except Exception:
            return None, None

    def _bootstrap_se(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict, dict]:
        """Compute bootstrap standard errors.

        Parameters
        ----------
        t, event, z, x : arrays
            Training data arrays.
        rng : Generator
            Random number generator for seed generation.

        Returns
        -------
        se_inc : dict — SE for incidence coefficients
        se_lat : dict — SE for latency parameters
        """
        seeds = rng.integers(0, 2**31, size=self.n_bootstrap).tolist()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._bootstrap_single)(t, event, z, x, seed)
            for seed in seeds
        )

        gammas = [r[0] for r in results if r[0] is not None]
        lats = [r[1] for r in results if r[1] is not None]

        se_inc: dict = {}
        se_lat: dict = {}

        if gammas:
            gamma_arr = np.array(gammas)
            sds = np.std(gamma_arr, axis=0, ddof=1)
            se_inc = dict(zip(self._incidence_cols, sds))

        if lats:
            lat_arr = np.array(lats)
            sds_lat = np.std(lat_arr, axis=0, ddof=1)
            se_lat = dict(zip(self._latency_param_names(), sds_lat))

        return se_inc, se_lat

    def predict_cure_fraction(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the cure fraction (P(immune)) for each row.

        Parameters
        ----------
        df : DataFrame
            Data with incidence covariate columns.

        Returns
        -------
        cure : ndarray of shape (n,)
            Estimated P(immune | z_i) = 1 - pi(z_i) per policyholder.
        """
        self._check_fitted()
        z, _ = _parse_formula(self.incidence_formula, df)
        pi = compute_pi(z, self._gamma, self._intercept)
        return 1.0 - pi

    def predict_susceptibility(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(susceptible) = 1 - cure fraction for each row.

        Parameters
        ----------
        df : DataFrame
            Data with incidence covariate columns.

        Returns
        -------
        suscept : ndarray of shape (n,)
            Estimated P(susceptible | z_i) = pi(z_i) per policyholder.
        """
        self._check_fitted()
        z, _ = _parse_formula(self.incidence_formula, df)
        return compute_pi(z, self._gamma, self._intercept)

    def predict_population_survival(
        self,
        df: pd.DataFrame,
        times: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Predict population-level survival S_pop(t|x,z) at specified times.

        S_pop(t|x,z) = pi(z) * S_u(t|x) + [1 - pi(z)]

        Parameters
        ----------
        df : DataFrame
            Data with covariate columns.
        times : array-like
            Time points at which to evaluate survival.

        Returns
        -------
        DataFrame of shape (n_rows, n_times)
            Columns named after each element of ``times``.
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        z, _ = _parse_formula(self.incidence_formula, df)
        x, _ = _parse_formula(self.latency_formula, df)
        pi = compute_pi(z, self._gamma, self._intercept)

        results = {}
        for t_val in times:
            t_arr = np.full(len(df), t_val)
            surv_u, _ = self._compute_latency_surv_dens(t_arr, x, self._latency_params)
            s_pop = pi * surv_u + (1.0 - pi)
            results[t_val] = s_pop

        return pd.DataFrame(results, index=df.index)

    def predict_susceptible_survival(
        self,
        df: pd.DataFrame,
        times: Union[list, np.ndarray],
    ) -> pd.DataFrame:
        """Predict conditional survival S_u(t|x) for susceptibles only.

        This is the latency sub-model: the survival function conditional
        on being susceptible (not cured).

        Parameters
        ----------
        df : DataFrame
            Data with latency covariate columns.
        times : array-like
            Time points at which to evaluate survival.

        Returns
        -------
        DataFrame of shape (n_rows, n_times)
        """
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        x, _ = _parse_formula(self.latency_formula, df)

        results = {}
        for t_val in times:
            t_arr = np.full(len(df), t_val)
            surv_u, _ = self._compute_latency_surv_dens(t_arr, x, self._latency_params)
            results[t_val] = surv_u

        return pd.DataFrame(results, index=df.index)

    def __repr__(self) -> str:
        cls = type(self).__name__
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"{cls}(incidence='{self.incidence_formula}', "
            f"latency='{self.latency_formula}', "
            f"n_em_starts={self.n_em_starts}, "
            f"status={status})"
        )
