"""
WeibullMixtureCureFitter: covariate-adjusted mixture cure model.

This is the primary gap vs lifelines. lifelines.MixtureCureFitter is univariate
only. This class fits a covariate-adjusted cure model:

    S(t|x) = pi(x) + (1 - pi(x)) * S_u(t|x)

where pi(x) = sigmoid(gamma_0 + x_cure' gamma) is the cure fraction (never-lapse
probability) and S_u(t|x) = exp(-(t/lambda(x))^rho) is Weibull AFT for the
uncured subgroup.

Parameters are estimated by EM initialisation followed by joint L-BFGS-B
maximisation. The R equivalent is flexsurvcure::flexsurvcure(mixture=TRUE,
dist="weibull").
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit  # numerically stable sigmoid

from ._utils import (
    build_design_matrix,
    coef_names,
    to_polars,
    weibull_median,
    weibull_pdf,
    weibull_sf,
)


class WeibullMixtureCureFitter:
    """Covariate-adjusted mixture cure model for insurance lapse data.

    Models the fact that a subgroup of policyholders (high-NCD, direct debit,
    long-tenure) effectively never lapse. The survival function is:

        S(t|x) = pi(x) + (1 - pi(x)) * S_u(t|x)

    where:
    - pi(x) = logistic(gamma_0 + x'gamma) is the cure probability
      ("never-lapse fraction")
    - S_u(t|x) = exp(-(t / lambda(x))^rho) is Weibull AFT survival for the
      uncured subgroup, with lambda(x) = exp(beta_0 + x'beta)

    Parameters are estimated jointly by maximum likelihood via scipy.optimize,
    using the EM algorithm for initialisation. The log-likelihood is:

        l = sum_i [
            d_i * log((1 - pi(x_i)) * f_u(t_i|x_i))
            + (1 - d_i) * log(pi(x_i) + (1 - pi(x_i)) * S_u(t_i|x_i))
        ]

    where d_i = 1 if event (lapse), 0 if censored, and f_u is Weibull density.

    Parameters
    ----------
    cure_covariates : list[str]
        Column names for covariates entering the cure fraction logistic model.
        Typically: ["ncd_years", "tenure", "channel_direct"].
    uncured_covariates : list[str]
        Column names for covariates in the Weibull AFT scale parameter.
        Can overlap with cure_covariates.
    penalizer : float
        L2 penalty on both sets of coefficients. Default 0.01.
    fit_intercept : bool
        Include intercepts in both sub-models. Default True.
    max_iter : int
        Maximum EM + L-BFGS-B iterations. Default 300.
    tol : float
        Convergence tolerance. Default 1e-6.

    Attributes
    ----------
    cure_params_ : pl.DataFrame
        Fitted coefficients for the cure fraction logistic model.
    uncured_params_ : pl.DataFrame
        Fitted coefficients and shape parameter for Weibull AFT.
    convergence_ : dict
        Convergence diagnostics: n_iter, log_likelihood, AIC, BIC.

    Examples
    --------
    >>> from insurance_survival import WeibullMixtureCureFitter
    >>>
    >>> fitter = WeibullMixtureCureFitter(
    ...     cure_covariates=["ncd_years", "tenure"],
    ...     uncured_covariates=["ncd_years", "annual_premium_scaled"],
    ... )
    >>> fitter.fit(survival_df, duration_col="stop", event_col="event")
    >>> cure_probs = fitter.predict_cure(new_df)
    >>> surv = fitter.predict_survival_function(new_df, times=[1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        cure_covariates: list[str],
        uncured_covariates: list[str],
        penalizer: float = 0.01,
        fit_intercept: bool = True,
        max_iter: int = 300,
        tol: float = 1e-6,
    ) -> None:
        self.cure_covariates = list(cure_covariates)
        self.uncured_covariates = list(uncured_covariates)
        self.penalizer = penalizer
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

        # Set after fit()
        self._gamma: np.ndarray | None = None  # cure logistic coefs
        self._beta: np.ndarray | None = None   # uncured AFT coefs
        self._log_rho: float | None = None     # log(shape parameter)
        self._hessian_inv: np.ndarray | None = None
        self._n_params: int = 0
        self._n_obs: int = 0
        self._n_events: int = 0

        self.cure_params_: pl.DataFrame | None = None
        self.uncured_params_: pl.DataFrame | None = None
        self.convergence_: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        duration_col: str = "stop",
        event_col: str = "event",
        entry_col: str | None = None,
    ) -> "WeibullMixtureCureFitter":
        """Fit the mixture cure model by joint MLE with EM initialisation.

        Parameters
        ----------
        df : pl.DataFrame
            Survival data in start/stop format. For time-fixed data (no MTAs),
            provide duration directly in duration_col.
        duration_col : str
            Column containing follow-up time (or stop time).
        event_col : str
            Binary event indicator (1 = lapse, 0 = censored).
        entry_col : str | None
            Left-truncation entry time column. Pass "start" if using start/stop.

        Returns
        -------
        self
        """
        df = to_polars(df)
        t = df[duration_col].to_numpy().astype(float)
        d = df[event_col].to_numpy().astype(float)

        # Guard: filter out zero/negative durations
        mask = t > 0
        if not mask.all():
            warnings.warn(
                f"Dropping {(~mask).sum()} rows with non-positive durations.",
                UserWarning,
                stacklevel=2,
            )
            df = df.filter(pl.Series(mask))
            t = t[mask]
            d = d[mask]

        X_cure = build_design_matrix(df, self.cure_covariates, self.fit_intercept)
        X_uncured = build_design_matrix(df, self.uncured_covariates, self.fit_intercept)

        n = len(t)
        p_cure = X_cure.shape[1]
        p_uncured = X_uncured.shape[1]

        self._n_obs = n
        self._n_events = int(d.sum())
        self._n_params = p_cure + p_uncured + 1  # +1 for log_rho

        # EM initialisation
        gamma_init, beta_init, log_rho_init = self._em_init(
            t, d, X_cure, X_uncured
        )

        # Joint L-BFGS-B optimisation
        theta_init = np.concatenate([gamma_init, beta_init, [log_rho_init]])
        result = self._optimise_joint(theta_init, t, d, X_cure, X_uncured)

        gamma = result.x[:p_cure]
        beta = result.x[p_cure:p_cure + p_uncured]
        log_rho = result.x[p_cure + p_uncured]

        self._gamma = gamma
        self._beta = beta
        self._log_rho = log_rho

        ll = -result.fun
        k = self._n_params
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        self.convergence_ = {
            "n_iter": result.nit,
            "log_likelihood": float(ll),
            "AIC": float(aic),
            "BIC": float(bic),
            "converged": bool(result.success),
            "message": result.message,
        }

        # Attempt to compute standard errors via finite-difference Hessian
        self._hessian_inv = self._compute_se(
            result.x, t, d, X_cure, X_uncured
        )

        # Build summary DataFrames
        self._build_param_tables(gamma, beta, log_rho, X_cure, X_uncured)

        return self

    def _em_init(
        self,
        t: np.ndarray,
        d: np.ndarray,
        X_cure: np.ndarray,
        X_uncured: np.ndarray,
        n_em_iter: int = 15,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """EM algorithm for initialisation.

        Returns initial gamma, beta, log_rho.
        """
        n = len(t)
        p_cure = X_cure.shape[1]
        p_uncured = X_uncured.shape[1]

        # Initial parameters: zeros
        gamma = np.zeros(p_cure)
        beta = np.zeros(p_uncured)
        log_rho = np.log(1.5)  # shape = 1.5 as prior

        for _ in range(n_em_iter):
            rho = np.exp(log_rho)
            lam = np.exp(X_uncured @ beta)
            pi = expit(X_cure @ gamma)

            sf = weibull_sf(t, lam, rho)
            pdf = weibull_pdf(t, lam, rho)

            # E-step: posterior probability of being uncured
            # For events: P(uncured | event) = 1 (events must be uncured)
            # For censored: P(uncured | censored, t) = (1-pi)*S_u / S
            S_mix = pi + (1.0 - pi) * sf
            S_mix = np.clip(S_mix, 1e-300, 1.0)

            w = np.where(
                d == 1,
                1.0,
                (1.0 - pi) * sf / S_mix,
            )
            w = np.clip(w, 1e-6, 1.0 - 1e-6)

            # M-step (cure): logistic regression for pi using weights
            gamma = self._fit_logistic_weighted(
                X_cure, 1.0 - w, gamma  # pi_target = 1 - w_uncured
            )

            # M-step (uncured): Weibull AFT using uncured weights
            beta, log_rho = self._fit_weibull_weighted(
                X_uncured, t, d, w, beta, log_rho
            )

        return gamma, beta, log_rho

    def _fit_logistic_weighted(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gamma_init: np.ndarray,
    ) -> np.ndarray:
        """Weighted logistic regression (one Newton step for speed)."""
        def nll(g: np.ndarray) -> float:
            pi = expit(X @ g)
            pi = np.clip(pi, 1e-9, 1.0 - 1e-9)
            ll = np.sum(y * np.log(pi) + (1.0 - y) * np.log(1.0 - pi))
            penalty = (self.penalizer / 2.0) * np.sum(g[1:] ** 2)
            return -(ll - penalty)

        res = minimize(nll, gamma_init, method="L-BFGS-B",
                       options={"maxiter": 50, "ftol": 1e-6})
        return res.x

    def _fit_weibull_weighted(
        self,
        X: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        w: np.ndarray,
        beta_init: np.ndarray,
        log_rho_init: float,
    ) -> tuple[np.ndarray, float]:
        """Weighted Weibull AFT fit (one optimisation step)."""
        theta_init = np.concatenate([beta_init, [log_rho_init]])

        def nll(theta: np.ndarray) -> float:
            beta = theta[:-1]
            rho = np.exp(theta[-1])
            lam = np.exp(X @ beta)
            sf = weibull_sf(t, lam, rho)
            pdf = weibull_pdf(t, lam, rho)
            log_pdf = np.log(np.clip(pdf, 1e-300, None))
            log_sf = np.log(np.clip(sf, 1e-300, None))
            # Weighted log-likelihood
            ll = np.sum(w * d * log_pdf + w * (1.0 - d) * log_sf)
            penalty = (self.penalizer / 2.0) * np.sum(beta[1:] ** 2)
            return -(ll - penalty)

        res = minimize(nll, theta_init, method="L-BFGS-B",
                       options={"maxiter": 50, "ftol": 1e-6})
        return res.x[:-1], float(res.x[-1])

    def _log_likelihood(
        self,
        theta: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        X_cure: np.ndarray,
        X_uncured: np.ndarray,
    ) -> float:
        """Full joint log-likelihood (negative, for minimisation)."""
        p_cure = X_cure.shape[1]
        p_uncured = X_uncured.shape[1]

        gamma = theta[:p_cure]
        beta = theta[p_cure:p_cure + p_uncured]
        rho = np.exp(theta[p_cure + p_uncured])

        pi = expit(X_cure @ gamma)
        lam = np.exp(X_uncured @ beta)

        sf = weibull_sf(t, lam, rho)
        pdf = weibull_pdf(t, lam, rho)

        log_pdf = np.log(np.clip(pdf, 1e-300, None))
        S_mix = pi + (1.0 - pi) * sf
        log_S_mix = np.log(np.clip(S_mix, 1e-300, None))
        log_f_mix = np.log(np.clip((1.0 - pi) * pdf, 1e-300, None))

        ll = np.sum(d * log_f_mix + (1.0 - d) * log_S_mix)

        # L2 penalty (no penalty on intercepts, which are index 0)
        pen = (self.penalizer / 2.0) * (
            np.sum(gamma[1:] ** 2) + np.sum(beta[1:] ** 2)
        )
        return -(ll - pen)

    def _optimise_joint(
        self,
        theta_init: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        X_cure: np.ndarray,
        X_uncured: np.ndarray,
    ) -> Any:
        """Run L-BFGS-B on the full joint log-likelihood."""
        result = minimize(
            self._log_likelihood,
            theta_init,
            args=(t, d, X_cure, X_uncured),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": 1e-5},
        )
        if not result.success:
            warnings.warn(
                f"Optimisation did not converge: {result.message}",
                UserWarning,
                stacklevel=3,
            )
        return result

    def _compute_se(
        self,
        theta: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        X_cure: np.ndarray,
        X_uncured: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate standard errors via finite-difference Hessian."""
        from scipy.optimize import approx_fprime

        eps = 1e-5
        n = len(theta)

        def f(th: np.ndarray) -> float:
            return self._log_likelihood(th, t, d, X_cure, X_uncured)

        try:
            # Numerical Hessian
            H = np.zeros((n, n))
            for i in range(n):
                ei = np.zeros(n)
                ei[i] = eps
                grad_plus = approx_fprime(theta + ei, f, eps)
                grad_minus = approx_fprime(theta - ei, f, eps)
                H[i] = (grad_plus - grad_minus) / (2 * eps)

            # Symmetrise
            H = (H + H.T) / 2.0
            return np.linalg.inv(H)
        except Exception:
            return None

    def _build_param_tables(
        self,
        gamma: np.ndarray,
        beta: np.ndarray,
        log_rho: float,
        X_cure: np.ndarray,
        X_uncured: np.ndarray,
    ) -> None:
        """Populate cure_params_ and uncured_params_ DataFrames."""
        cure_names = coef_names(self.cure_covariates, self.fit_intercept)
        uncured_names = coef_names(self.uncured_covariates, self.fit_intercept)

        p_cure = len(cure_names)
        p_uncured = len(uncured_names)

        if self._hessian_inv is not None:
            # SEs from diagonal of Hessian inverse
            # Hessian is for *negative* LL, so invert gives covariance
            diag = np.diag(self._hessian_inv)
            diag = np.abs(diag)  # guard against numerical issues
            se_gamma = np.sqrt(diag[:p_cure])
            se_beta = np.sqrt(diag[p_cure:p_cure + p_uncured])
            se_rho = float(np.sqrt(abs(diag[p_cure + p_uncured])))
        else:
            se_gamma = np.full(p_cure, np.nan)
            se_beta = np.full(p_uncured, np.nan)
            se_rho = np.nan

        from scipy.stats import norm as _norm

        def _make_table(
            names: list[str],
            coefs: np.ndarray,
            ses: np.ndarray,
            model: str,
        ) -> pl.DataFrame:
            z = coefs / np.where(ses > 0, ses, np.nan)
            p = 2.0 * _norm.sf(np.abs(z))
            lower = coefs - 1.96 * ses
            upper = coefs + 1.96 * ses
            return pl.DataFrame({
                "model": [model] * len(names),
                "covariate": names,
                "coef": coefs.tolist(),
                "exp_coef": np.exp(coefs).tolist(),
                "se": ses.tolist(),
                "z": z.tolist(),
                "p": p.tolist(),
                "lower_95": lower.tolist(),
                "upper_95": upper.tolist(),
            })

        self.cure_params_ = _make_table(
            cure_names, gamma, se_gamma, "cure_logistic"
        )

        # For uncured, append Weibull shape parameter
        rho = np.exp(log_rho)
        uncured_names_full = uncured_names + ["log_shape"]
        beta_full = np.append(beta, log_rho)
        se_beta_full = np.append(se_beta, se_rho)

        self.uncured_params_ = _make_table(
            uncured_names_full, beta_full, se_beta_full, "uncured_weibull_aft"
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._gamma is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def predict_cure(self, df: pl.DataFrame) -> pl.Series:
        """Predict cure fraction (never-lapse probability) per policy.

        Parameters
        ----------
        df : pl.DataFrame
            Must contain cure_covariates columns.

        Returns
        -------
        pl.Series
            Float64, range [0, 1]. pi(x) for each row in df.
        """
        self._check_fitted()
        df = to_polars(df)
        X = build_design_matrix(df, self.cure_covariates, self.fit_intercept)
        pi = expit(X @ self._gamma)
        return pl.Series("cure_prob", pi)

    def predict_survival_function(
        self,
        df: pl.DataFrame,
        times: list[float],
    ) -> pl.DataFrame:
        """Predict survival probability at each time point for each policy.

        Parameters
        ----------
        df : pl.DataFrame
            Must contain cure_covariates and uncured_covariates columns.
        times : list[float]
            Time points (policy years) at which to evaluate S(t|x).

        Returns
        -------
        pl.DataFrame
            Shape (len(df), len(times)). Columns named "S_t1", "S_t2", etc.
        """
        self._check_fitted()
        df = to_polars(df)
        X_cure = build_design_matrix(df, self.cure_covariates, self.fit_intercept)
        X_uncured = build_design_matrix(df, self.uncured_covariates, self.fit_intercept)

        pi = expit(X_cure @ self._gamma)          # (n,)
        lam = np.exp(X_uncured @ self._beta)      # (n,)
        rho = np.exp(self._log_rho)

        result: dict[str, list[float]] = {}
        for k, t in enumerate(times):
            sf_u = weibull_sf(np.full(len(df), t), lam, rho)
            s_mix = pi + (1.0 - pi) * sf_u
            result[f"S_t{k + 1}"] = s_mix.tolist()

        return pl.DataFrame(result)

    def predict_median_survival(self, df: pl.DataFrame) -> pl.Series:
        """Predict median survival time for uncured fraction.

        Returns
        -------
        pl.Series
            Float64. Median conditional on being in the uncured subgroup.
        """
        self._check_fitted()
        df = to_polars(df)
        X_uncured = build_design_matrix(df, self.uncured_covariates, self.fit_intercept)
        lam = np.exp(X_uncured @ self._beta)
        rho = np.exp(self._log_rho)
        medians = weibull_median(lam, rho)
        return pl.Series("median_survival_uncured", medians)

    def plot_survival(
        self,
        df: pl.DataFrame,
        covariate: str,
        title: str | None = None,
    ) -> None:
        """Plot survival curves stratified by a covariate value.

        Uses matplotlib. Overlays the plateau from the cure fraction.

        Parameters
        ----------
        df : pl.DataFrame
            Must contain cure_covariates and uncured_covariates columns.
        covariate : str
            Column name to stratify by.
        title : str | None
            Plot title.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_survival(). "
                "Install with: pip install insurance-survival[plot]"
            )

        self._check_fitted()
        df = to_polars(df)
        times = np.linspace(0, 7, 200)

        values = df[covariate].unique().sort().to_list()
        _, ax = plt.subplots(figsize=(8, 5))

        for val in values:
            sub = df.filter(pl.col(covariate) == val)
            if len(sub) == 0:
                continue
            # Use first row as representative profile
            profile = sub.head(1)
            X_cure = build_design_matrix(
                profile, self.cure_covariates, self.fit_intercept
            )
            X_uncured = build_design_matrix(
                profile, self.uncured_covariates, self.fit_intercept
            )
            pi = float(expit(X_cure @ self._gamma)[0])
            lam = float(np.exp(X_uncured @ self._beta)[0])
            rho = np.exp(self._log_rho)

            sf = pi + (1.0 - pi) * weibull_sf(times, lam, rho)
            ax.plot(times, sf, label=f"{covariate}={val}")
            ax.axhline(pi, linestyle="--", alpha=0.3)

        ax.set_xlabel("Policy year")
        ax.set_ylabel("Survival probability S(t)")
        ax.set_title(title or "Survival curves by covariate")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def summary(self) -> pl.DataFrame:
        """Return coefficient summary table with confidence intervals.

        Returns
        -------
        pl.DataFrame
            Columns: model, covariate, coef, exp_coef, se, z, p,
            lower_95, upper_95.
        """
        self._check_fitted()
        if self.cure_params_ is None or self.uncured_params_ is None:
            raise RuntimeError("Parameter tables not available. Refit the model.")
        return pl.concat([self.cure_params_, self.uncured_params_])

    # ------------------------------------------------------------------
    # Pickle compatibility
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
