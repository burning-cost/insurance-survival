"""Cox PH semiparametric mixture cure model.

The most flexible MCM: nonparametric baseline hazard for the latency
sub-model. Useful when you want to avoid distributional assumptions
about the time-to-claim for susceptibles. The trade-off is no
extrapolation beyond the last observed event time — a serious
limitation for pricing projection horizons.

Reference: Peng & Dear (2000), Biometrics 56:237-243.
Reference: Sy & Taylor (2000), Biometrics 56:227-236.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._base import BaseMixtureCure
from ._em import compute_loglik, compute_pi, e_step, m_step_incidence


class CoxMixtureCure(BaseMixtureCure):
    """Mixture cure model with Cox PH semiparametric latency sub-model.

    Uses lifelines ``CoxPHFitter`` with ``weights_col`` for the M-step
    latency update. The baseline hazard is estimated nonparametrically via
    the Breslow estimator.

    Limitations
    -----------
    - No extrapolation beyond the maximum observed event time.
    - ``predict_susceptible_survival`` and ``predict_population_survival``
      will return NaN for query times beyond the observed event range.
    - Slower M-step than parametric alternatives.

    Parameters
    ----------
    incidence_formula : str
        Additive formula for incidence covariates.
    latency_formula : str
        Additive formula for latency covariates.
    n_em_starts : int
        Number of random EM restarts. Default 3.
    max_iter : int
        Maximum EM iterations per run. Default 100.
    tol : float
        Convergence tolerance on log-likelihood change. Default 1e-4.
    bootstrap_se : bool
        Whether to compute bootstrap standard errors. Default False.
    n_bootstrap : int
        Number of bootstrap resamples. Default 100.
    n_jobs : int
        Parallel jobs for bootstrap. Default -1 (all cores).
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    result_ : MCMResult
        Detailed fit results, available after ``fit()``.
    _cox_fitter : lifelines.CoxPHFitter
        The fitted Cox model for the latency sub-model.
    """

    def __init__(self, *args, n_em_starts: int = 3, max_iter: int = 100,
                 tol: float = 1e-4, n_bootstrap: int = 100, **kwargs) -> None:
        super().__init__(
            *args,
            n_em_starts=n_em_starts,
            max_iter=max_iter,
            tol=tol,
            n_bootstrap=n_bootstrap,
            **kwargs,
        )
        self._cox_fitter = None
        self._cox_baseline_hazard = None  # ndarray: baseline cumulative hazard
        self._cox_times = None            # event times from training data
        self._cox_beta = None             # Cox PH coefficients

    def _latency_param_names(self) -> list[str]:
        if self._latency_cols:
            return [f"cox_{c}" for c in self._latency_cols]
        return []

    def _m_step_cox(
        self,
        t: np.ndarray,
        event: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
        lat_col_names: list[str],
    ):
        """M-step: update Cox PH latency using lifelines with weights."""
        try:
            from lifelines import CoxPHFitter
        except ImportError as exc:
            raise ImportError(
                "lifelines is required for CoxMixtureCure. "
                "Install it with: pip install lifelines"
            ) from exc

        # Build a temporary DataFrame for lifelines
        df_cox = pd.DataFrame(x, columns=lat_col_names)
        df_cox["_duration"] = t
        df_cox["_event"] = event
        df_cox["_weight"] = np.clip(w, 1e-6, None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cox = CoxPHFitter(penalizer=0.1)
            cox.fit(
                df_cox,
                duration_col="_duration",
                event_col="_event",
                weights_col="_weight",
                robust=False,
            )

        return cox

    def _cox_survival(
        self,
        t: np.ndarray,
        x: np.ndarray,
        cox_fitter,
    ) -> np.ndarray:
        """Evaluate Cox PH survival S_u(t|x) at individual times."""
        lat_col_names = self._latency_cols if self._latency_cols else []
        n = len(t)
        surv = np.zeros(n)

        for i in range(n):
            xi = pd.DataFrame(x[i : i + 1], columns=lat_col_names)
            # lifelines predict_survival_function returns a DataFrame indexed by times
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = cox_fitter.predict_survival_function(xi, times=[t[i]])
            val = sf.values[0, 0]
            surv[i] = max(val, 0.0)

        return surv

    def _cox_density(
        self,
        t: np.ndarray,
        x: np.ndarray,
        cox_fitter,
        eps: float = 0.01,
    ) -> np.ndarray:
        """Approximate Cox PH density f_u(t|x) via finite differences."""
        lat_col_names = self._latency_cols if self._latency_cols else []
        n = len(t)
        dens = np.zeros(n)

        for i in range(n):
            xi = pd.DataFrame(x[i : i + 1], columns=lat_col_names)
            t_lo = max(t[i] - eps, 1e-6)
            t_hi = t[i] + eps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = cox_fitter.predict_survival_function(xi, times=[t_lo, t_hi])
            s_lo = max(sf.values[0, 0], 1e-15)
            s_hi = max(sf.values[1, 0], 1e-15)
            dens[i] = max((s_lo - s_hi) / (2 * eps), 1e-15)

        return dens

    def _compute_latency_surv_dens(
        self,
        t: np.ndarray,
        x: np.ndarray,
        latency_params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate Cox latency using the stored fitter."""
        if self._cox_fitter is None:
            # Before first M-step: use Kaplan-Meier-like fallback
            # (uniform survival as initialisation)
            n = len(t)
            return np.full(n, 0.5), np.full(n, 0.5 / max(t.max(), 1.0))

        surv = self._cox_survival(t, x, self._cox_fitter)
        dens = self._cox_density(t, x, self._cox_fitter)
        return surv, dens

    def _run_em_single(
        self,
        t: np.ndarray,
        event: np.ndarray,
        z: np.ndarray,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, bool, int]:
        """Run one EM sequence for Cox MCM."""
        from sklearn.linear_model import LogisticRegression

        p = z.shape[1]
        lat_col_names = self._latency_cols if self._latency_cols else []

        # Initialise incidence from logistic on event indicator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500, fit_intercept=True)
            clf.fit(z, event.astype(int))
        gamma = clf.coef_[0] + rng.normal(0, 0.1, size=p)
        intercept = clf.intercept_ + rng.normal(0, 0.1, size=1)

        cox_fitter = None
        prev_loglik = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            # E-step
            if cox_fitter is None:
                # Uniform weight init
                n = len(t)
                w = np.where(event == 1, 1.0, 0.5)
            else:
                pi = compute_pi(z, gamma, intercept)
                surv_u = self._cox_survival(t, x, cox_fitter)
                w = e_step(pi, surv_u, event)

            # M-step
            gamma, intercept = m_step_incidence(z, w)
            try:
                cox_fitter = self._m_step_cox(t, event, x, w, lat_col_names)
            except Exception:
                break

            # Log-likelihood
            pi = compute_pi(z, gamma, intercept)
            surv_u = self._cox_survival(t, x, cox_fitter)
            dens_u = self._cox_density(t, x, cox_fitter)
            loglik = compute_loglik(pi, dens_u, surv_u, event)

            if abs(loglik - prev_loglik) < self.tol:
                converged = True
                break
            prev_loglik = loglik

        # Store the cox fitter for prediction
        self._cox_fitter = cox_fitter
        # Latency params: just the cox coefficients (for SE reporting)
        if cox_fitter is not None:
            lat_params = cox_fitter.params_.values
        else:
            lat_params = np.zeros(len(lat_col_names))

        return prev_loglik, gamma, intercept, lat_params, converged, iteration + 1

    def _compute_latency_surv_dens_with_fitter(
        self,
        t: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate survival and density using current cox fitter."""
        if self._cox_fitter is None:
            n = len(t)
            return np.full(n, 0.5), np.full(n, 0.01)
        surv = self._cox_survival(t, x, self._cox_fitter)
        dens = self._cox_density(t, x, self._cox_fitter)
        return surv, dens

    def predict_susceptible_survival(
        self,
        df: pd.DataFrame,
        times,
    ) -> pd.DataFrame:
        """Predict S_u(t|x) for susceptibles at specified times.

        Note: Returns NaN for times beyond the maximum observed event time
        in the training data, as the Cox model does not extrapolate.
        """
        self._check_fitted()
        import warnings
        times_arr = np.asarray(times, dtype=float)
        lat_col_names = self._latency_cols if self._latency_cols else []
        from ._base import _parse_formula
        x, _ = _parse_formula(self.latency_formula, df)

        results = {}
        for t_val in times_arr:
            t_arr = np.full(len(df), t_val)
            surv = self._cox_survival(t_arr, x, self._cox_fitter)
            results[t_val] = surv

        return pd.DataFrame(results, index=df.index)
