"""
fine_gray.py — Fine-Gray subdistribution hazard regression.

This is the core of the library. It implements the Fine-Gray (1999) model
for competing risks data, fitting subdistribution hazard ratios (SHRs) via
IPCW-weighted partial likelihood.

Algorithm
---------
For a chosen event of interest k with n subjects:

1. Estimate the censoring survival function G_c(t) via Kaplan-Meier on the
   censoring indicator (censored = event, uncensored = censored). This gives
   G_c(t) = P(C > t).

2. Build the extended risk set at each event time t_i:
   - Alive and not yet experienced any event: always in risk set
   - Experienced a competing event before t_i: remain in risk set (this is
     Fine-Gray's key innovation) with weight w_j(t_i) = G_c(t_i) / G_c(T_j)
   - Censored before t_i: removed from risk set (weight 0)
   - Experienced cause-k before t_i: removed from risk set

3. Maximise the IPCW-weighted partial log-likelihood:
   l(beta) = sum over cause-k events of:
     [beta'x_i - log(sum_{j in R_k(t_i)} w_j(t_i) exp(beta'x_j))]

4. Score (gradient) and Hessian are computed analytically; Newton-Raphson
   iteration converges to the MLE.

5. Baseline subdistribution hazard estimated via Breslow estimator.

6. CIF prediction: F_k(t|x) = 1 - exp(-Lambda_k0(t) * exp(beta'x))

References
----------
Fine, J.P. & Gray, R.J. (1999). A proportional hazards model for the
subdistribution of a competing risk. JASA 94(446), 496-509.
DOI: 10.1080/01621459.1999.10474144.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class FineGrayFitter:
    """Fine-Gray subdistribution hazard model.

    Mirrors the ``lifelines.CoxPHFitter`` API as closely as practical.

    Parameters
    ----------
    penaliser:
        L2 regularisation coefficient on the log-likelihood. Default 0.0
        (no regularisation). Increase if covariates are collinear or the
        risk set becomes very small.
    alpha:
        Significance level for confidence intervals (default 0.05 → 95%).

    Attributes
    ----------
    summary:
        ``pd.DataFrame`` after fitting. Columns:
        covariate, coef, exp(coef), se(coef), z, p, lower_ci, upper_ci.
    params_:
        ``pd.Series`` mapping covariate name to fitted coefficient.
    variance_matrix_:
        ``pd.DataFrame`` — estimated covariance matrix of the estimator.
    baseline_cumulative_hazard_:
        ``pd.Series`` — Breslow estimate of baseline cumulative subdistribution
        hazard, indexed by time.
    log_likelihood_:
        Value of the partial log-likelihood at convergence.
    event_col:
        Name of the event column used during fit.
    duration_col:
        Name of the duration column used during fit.
    event_of_interest:
        The cause code modelled.

    Examples
    --------
    >>> from insurance_competing_risks import FineGrayFitter
    >>> fg = FineGrayFitter()
    >>> fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
    >>> print(fg.summary)
    >>> cif = fg.predict_cumulative_incidence(df_new, times=[1.0, 2.0])
    """

    def __init__(
        self,
        penaliser: float = 0.0,
        alpha: float = 0.05,
    ) -> None:
        self.penaliser = penaliser
        self.alpha = alpha
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "T",
        event_col: str = "E",
        event_of_interest: int = 1,
        *,
        weights_col: Optional[str] = None,
        formula: Optional[str] = None,
        fit_options: Optional[dict] = None,
    ) -> "FineGrayFitter":
        """Fit the Fine-Gray subdistribution hazard model.

        Parameters
        ----------
        df:
            DataFrame containing durations, event codes, and covariates. All
            columns not named ``duration_col``, ``event_col``, or
            ``weights_col`` are treated as covariates.
        duration_col:
            Column name for observed times.
        event_col:
            Column name for event indicators. Must be integer-coded with 0
            for censoring and positive integers for distinct causes.
        event_of_interest:
            The cause code to model. Subjects with other non-zero event codes
            are treated as having experienced a competing event.
        weights_col:
            Optional column of observation weights (e.g., exposure).
        formula:
            Optional Patsy formula for covariate selection / transformations.
            If ``None``, all columns except ``duration_col``, ``event_col``,
            and ``weights_col`` are used as covariates.
        fit_options:
            Options passed to ``scipy.optimize.minimize``.

        Returns
        -------
        self
        """
        df = df.copy()

        self.duration_col = duration_col
        self.event_col = event_col
        self.event_of_interest = event_of_interest

        T = df[duration_col].values.astype(float)
        E = df[event_col].values.astype(int)

        if weights_col is not None:
            sample_weights = df[weights_col].values.astype(float)
            covariate_cols = [
                c for c in df.columns
                if c not in (duration_col, event_col, weights_col)
            ]
        else:
            sample_weights = np.ones(len(T))
            covariate_cols = [
                c for c in df.columns
                if c not in (duration_col, event_col)
            ]

        if formula is not None:
            import patsy  # type: ignore
            X_dm = patsy.dmatrix(formula, df, return_type="dataframe")
            # Drop intercept if present
            if "Intercept" in X_dm.columns:
                X_dm = X_dm.drop(columns=["Intercept"])
            X = X_dm.values.astype(float)
            covariate_cols = list(X_dm.columns)
        else:
            X = df[covariate_cols].values.astype(float)

        self._covariate_cols = covariate_cols
        self._X_mean = X.mean(axis=0)
        X_centred = X - self._X_mean  # centre for numerical stability

        n, p = X_centred.shape

        # 1. Estimate censoring survival G_c(t) via Kaplan-Meier
        censoring_event = (E == 0).astype(int)  # censored = "event" for G_c
        G_c_times, G_c_vals = _kaplan_meier(T, censoring_event)

        def gc(t: np.ndarray) -> np.ndarray:
            """Evaluate G_c at times t (left-continuous)."""
            return np.interp(t, G_c_times, G_c_vals, left=1.0, right=G_c_vals[-1])

        # 2. Identify cause-k event times
        cause_k_mask = E == event_of_interest
        event_times = np.sort(np.unique(T[cause_k_mask]))

        if len(event_times) == 0:
            raise ValueError(
                f"No events with cause code {event_of_interest} found in data."
            )

        # 3. Pre-compute IPCW weights for the extended risk set
        # w_j(t) = G_c(t) / G_c(min(T_j, t))
        # For cause-k failures (E==1): w = 1 (they are in risk set, standard)
        # For competing events (E==k' != 0,k): w = G_c(t) / G_c(T_j) if t > T_j
        # For censored (E==0): w = 0 if T_j < t (removed from risk set)

        # 4. Optimise log-likelihood using L-BFGS-B
        def neg_log_lik_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
            return _fg_neg_log_lik_and_grad(
                beta, X_centred, T, E, event_of_interest,
                event_times, gc, sample_weights, self.penaliser
            )

        beta_init = np.zeros(p)
        opts = {"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6}
        if fit_options:
            opts.update(fit_options)

        result = minimize(
            neg_log_lik_and_grad,
            beta_init,
            method="L-BFGS-B",
            jac=True,
            options=opts,
        )

        if not result.success:
            warnings.warn(
                f"Optimisation did not fully converge: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        beta_hat = result.x
        self.log_likelihood_ = -result.fun

        # 5. Standard errors from observed information matrix
        H = _fg_hessian(
            beta_hat, X_centred, T, E, event_of_interest,
            event_times, gc, sample_weights, self.penaliser
        )
        try:
            var_matrix = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Hessian is singular; standard errors may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            var_matrix = np.full((p, p), np.nan)

        se = np.sqrt(np.diag(var_matrix))

        # 6. Breslow baseline cumulative hazard
        baseline_ch = _breslow_baseline(
            beta_hat, X_centred, T, E, event_of_interest,
            event_times, gc, sample_weights
        )

        # Store results
        self.params_ = pd.Series(beta_hat, index=covariate_cols)
        self.variance_matrix_ = pd.DataFrame(
            var_matrix, index=covariate_cols, columns=covariate_cols
        )
        self.baseline_cumulative_hazard_ = pd.Series(
            baseline_ch, index=event_times, name="baseline_cumulative_hazard"
        )

        z_scores = beta_hat / se
        p_values = 2.0 * norm.sf(np.abs(z_scores))
        z_crit = norm.ppf(1.0 - self.alpha / 2.0)

        self.summary = pd.DataFrame({
            "covariate": covariate_cols,
            "coef": beta_hat,
            "exp(coef)": np.exp(beta_hat),
            "se(coef)": se,
            "z": z_scores,
            "p": p_values,
            f"lower_{int(100*(1-self.alpha))}%": beta_hat - z_crit * se,
            f"upper_{int(100*(1-self.alpha))}%": beta_hat + z_crit * se,
        }).set_index("covariate")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_cumulative_incidence(
        self,
        df: pd.DataFrame,
        times: Optional[Union[list[float], np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Predict the CIF for each subject at given times.

        Uses the relationship:
        F_k(t|x) = 1 - exp(-Lambda_k0(t) * exp(beta'(x - x_mean)))

        Parameters
        ----------
        df:
            DataFrame with the same covariate columns used during ``fit()``.
        times:
            Times at which to evaluate the CIF. If ``None``, uses all
            observed event times from the training data.

        Returns
        -------
        pd.DataFrame
            Shape (n_subjects, n_times). Row i, column j is the predicted
            CIF for subject i at ``times[j]``.
        """
        self._check_fitted()

        X = df[self._covariate_cols].values.astype(float)
        X_centred = X - self._X_mean

        lp = X_centred @ self.params_.values  # linear predictor

        if times is None:
            eval_times = self.baseline_cumulative_hazard_.index.values
        else:
            eval_times = np.asarray(times, dtype=float)

        # Interpolate baseline cumulative hazard at eval_times
        bch_times = self.baseline_cumulative_hazard_.index.values
        bch_vals = self.baseline_cumulative_hazard_.values
        bch_at_times = np.interp(
            eval_times, bch_times, bch_vals, left=0.0, right=bch_vals[-1]
        )  # shape (n_times,)

        # CIF_{i,j} = 1 - exp(-bch[j] * exp(lp[i]))
        exp_lp = np.exp(lp)  # (n_subjects,)
        cif = 1.0 - np.exp(-np.outer(exp_lp, bch_at_times))  # (n_subjects, n_times)

        return pd.DataFrame(
            cif,
            columns=eval_times,
            index=df.index,
        )

    def predict_median_time(self, df: pd.DataFrame) -> pd.Series:
        """Predict median time-to-event (cause k) for each subject.

        Returns the time at which F_k(t|x) = 0.5, or ``np.inf`` if the CIF
        never reaches 0.5 within the training data range.

        Parameters
        ----------
        df:
            Covariate DataFrame.

        Returns
        -------
        pd.Series
            Median predicted event times.
        """
        self._check_fitted()
        times = self.baseline_cumulative_hazard_.index.values
        cif_df = self.predict_cumulative_incidence(df, times=times)

        medians = []
        for _, row in cif_df.iterrows():
            exceeds = times[row.values >= 0.5]
            medians.append(exceeds[0] if len(exceeds) > 0 else np.inf)

        return pd.Series(medians, index=df.index, name="median_time")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_partial_effects_on_outcome(
        self,
        covariate: str,
        values: list,
        times: Optional[np.ndarray] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """Plot the CIF at different levels of one covariate, all else at mean.

        Requires matplotlib. Install with: pip install insurance-survival[plot]

        Parameters
        ----------
        covariate:
            Name of the covariate to vary.
        values:
            List of covariate values to plot a CIF for.
        times:
            Time grid. Defaults to all event times.
        ax:
            Matplotlib axes.

        Returns
        -------
        plt.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-survival[plot]"
            )

        self._check_fitted()

        if covariate not in self._covariate_cols:
            raise ValueError(f"Covariate '{covariate}' not found in fitted model.")

        if ax is None:
            _, ax = plt.subplots()

        if times is None:
            times = self.baseline_cumulative_hazard_.index.values

        # Base row: all covariates at their training mean (which is 0 after centring)
        base_row = {col: self._X_mean[i] for i, col in enumerate(self._covariate_cols)}

        for val in values:
            row = base_row.copy()
            row[covariate] = val
            df_point = pd.DataFrame([row])
            cif = self.predict_cumulative_incidence(df_point, times=times)
            ax.step(
                times,
                cif.values[0],
                where="post",
                label=f"{covariate}={val}",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel(f"CIF (cause {self.event_of_interest})")
        ax.legend()
        return ax

    def plot_covariate_groups(
        self,
        df: pd.DataFrame,
        covariate: str,
        times: Optional[np.ndarray] = None,
        ax: Optional[Any] = None,
    ) -> Any:
        """Plot predicted CIF for each unique value of a categorical covariate.

        Requires matplotlib. Install with: pip install insurance-survival[plot]

        Parameters
        ----------
        df:
            DataFrame with the same covariates used in fit (other covariates
            held at their values in df, not at mean).
        covariate:
            Categorical covariate to group by.
        times:
            Time grid.
        ax:
            Matplotlib axes.

        Returns
        -------
        plt.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install insurance-survival[plot]"
            )

        self._check_fitted()

        if ax is None:
            _, ax = plt.subplots()

        if times is None:
            times = self.baseline_cumulative_hazard_.index.values

        for val in sorted(df[covariate].unique()):
            subset = df[df[covariate] == val]
            cif_df = self.predict_cumulative_incidence(subset, times=times)
            mean_cif = cif_df.mean(axis=0)
            ax.step(times, mean_cif.values, where="post", label=f"{covariate}={val}")

        ax.set_xlabel("Time")
        ax.set_ylabel(f"Mean predicted CIF (cause {self.event_of_interest})")
        ax.legend()
        return ax

    # ------------------------------------------------------------------
    # Summary display
    # ------------------------------------------------------------------

    def print_summary(self, decimals: int = 4) -> None:
        """Print a formatted model summary."""
        self._check_fitted()
        print(f"\nFine-Gray Subdistribution Hazard Model")
        print(f"Event of interest: {self.event_of_interest}")
        print(f"Duration column: {self.duration_col}")
        print(f"Event column: {self.event_col}")
        print(f"Log partial-likelihood: {self.log_likelihood_:.4f}")
        print()
        print(self.summary.round(decimals).to_string())
        print()

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before using this method.")


# ---------------------------------------------------------------------------
# Core numerical routines
# ---------------------------------------------------------------------------

def _kaplan_meier(
    T: np.ndarray, E: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimator; returns (times, S(t)) on unique event times.

    Left-continuous convention: S(t) gives survival just before time t.
    """
    event_times = np.sort(np.unique(T[E == 1]))
    n = len(T)
    S = 1.0
    times_out = [0.0]
    S_out = [1.0]

    for t in event_times:
        n_at_risk = np.sum(T >= t)
        d = np.sum((T == t) & (E == 1))
        if n_at_risk == 0:
            break
        S *= (1.0 - d / n_at_risk)
        times_out.append(t)
        S_out.append(S)

    # Extend to infinity so interp beyond last time returns last value
    times_out.append(np.inf)
    S_out.append(S_out[-1])

    return np.array(times_out), np.array(S_out)


def _build_risk_set_weights(
    t: float,
    T: np.ndarray,
    E: np.ndarray,
    event_of_interest: int,
    gc: "callable",
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Compute IPCW weights for the extended risk set at time t.

    Returns an array of weights w_i (0 if subject is outside the risk set).

    Risk set membership at time t:
    - Subject never experienced any event yet (T_i > t or T_i == t): weight 1
    - Subject had cause-k event (T_i <= t, E_i = event_of_interest): NOT in risk
      set (except at the exact event time for tied-time handling)
    - Subject had competing event (T_i <= t, E_i != 0, != k): in risk set with
      IPCW weight G_c(t) / G_c(T_i)
    - Subject was censored before t (T_i < t, E_i = 0): NOT in risk set
    """
    n = len(T)
    w = np.zeros(n)

    gc_t = gc(np.array([t]))[0]

    for i in range(n):
        if T[i] >= t:
            # Alive (or failing now) — always in risk set with full weight
            # For cause-k event at exactly t, included (contributes to partial lik)
            if E[i] == 0 or T[i] > t or (E[i] == event_of_interest and T[i] == t):
                w[i] = sample_weights[i]
            elif E[i] != event_of_interest and T[i] == t:
                # Competing event at exactly t: include with IPCW weight
                gc_ti = gc(np.array([T[i]]))[0]
                if gc_ti > 0:
                    w[i] = sample_weights[i] * gc_t / gc_ti
        else:
            # T[i] < t
            if E[i] != 0 and E[i] != event_of_interest:
                # Competing event in the past: still in extended risk set
                gc_ti = gc(np.array([T[i]]))[0]
                if gc_ti > 0:
                    w[i] = sample_weights[i] * gc_t / gc_ti
            # Censored or cause-k before t: weight = 0

    return w


def _fg_neg_log_lik_and_grad(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    event_of_interest: int,
    event_times: np.ndarray,
    gc: "callable",
    sample_weights: np.ndarray,
    penaliser: float,
) -> tuple[float, np.ndarray]:
    """Fine-Gray negative partial log-likelihood and gradient."""
    p = len(beta)
    log_lik = 0.0
    grad = np.zeros(p)

    cause_k_mask = (E == event_of_interest)

    for t in event_times:
        # Subjects failing from cause k at time t
        fail_at_t = np.where((T == t) & cause_k_mask)[0]
        if len(fail_at_t) == 0:
            continue

        # IPCW weights for the risk set at t
        w = _build_risk_set_weights(t, T, E, event_of_interest, gc, sample_weights)

        in_risk = w > 0
        if not np.any(in_risk):
            continue

        X_risk = X[in_risk]
        w_risk = w[in_risk]

        exp_xb = np.exp(X_risk @ beta)
        wexp = w_risk * exp_xb
        denom = np.sum(wexp)

        if denom <= 0:
            continue

        p_vec = wexp / denom  # normalised weights (n_risk,)
        x_bar = X_risk.T @ p_vec  # weighted mean of covariates

        for i in fail_at_t:
            sw_i = sample_weights[i]
            log_lik += sw_i * (X[i] @ beta - np.log(denom))
            grad += sw_i * (X[i] - x_bar)

    # L2 penalty
    if penaliser > 0:
        log_lik -= 0.5 * penaliser * np.dot(beta, beta)
        grad -= penaliser * beta

    return -log_lik, -grad


def _fg_hessian(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    event_of_interest: int,
    event_times: np.ndarray,
    gc: "callable",
    sample_weights: np.ndarray,
    penaliser: float,
) -> np.ndarray:
    """Observed information matrix (negative Hessian of partial log-likelihood).

    Uses the formula:
    -d²l/dbeta² = sum_over_failures [w_i] * sum_{j in R(t_i)} w_j * e^{b'x_j} *
                  (x_j - x_bar)(x_j - x_bar)' / denom

    This is the standard Cox partial likelihood Hessian adapted for weighted
    risk sets.
    """
    p = len(beta)
    H = np.zeros((p, p))
    cause_k_mask = E == event_of_interest

    for t in event_times:
        fail_at_t = np.where((T == t) & cause_k_mask)[0]
        if len(fail_at_t) == 0:
            continue

        w = _build_risk_set_weights(t, T, E, event_of_interest, gc, sample_weights)
        in_risk = w > 0
        if not np.any(in_risk):
            continue

        X_risk = X[in_risk]
        w_risk = w[in_risk]

        exp_xb = np.exp(X_risk @ beta)
        wexp = w_risk * exp_xb
        denom = np.sum(wexp)
        if denom <= 0:
            continue

        p_vec = wexp / denom
        x_bar = X_risk.T @ p_vec

        # Weighted covariance
        X_centred_risk = X_risk - x_bar
        # Hessian contribution: sum of outer products weighted by p_vec
        V = (X_centred_risk * p_vec[:, None]).T @ X_centred_risk

        n_fail = sum(sample_weights[i] for i in fail_at_t)
        H += n_fail * V

    if penaliser > 0:
        H += penaliser * np.eye(p)

    return H


def _breslow_baseline(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    event_of_interest: int,
    event_times: np.ndarray,
    gc: "callable",
    sample_weights: np.ndarray,
) -> np.ndarray:
    """Breslow estimator for the baseline cumulative subdistribution hazard.

    Returns cumulative hazard values at each of ``event_times``.

    dLambda_0(t) = d_k(t) / sum_{j in R_k(t)} w_j(t) * exp(beta'x_j)
    """
    cause_k_mask = E == event_of_interest
    ch = np.zeros(len(event_times))

    for idx, t in enumerate(event_times):
        fail_at_t = np.where((T == t) & cause_k_mask)[0]
        if len(fail_at_t) == 0:
            if idx > 0:
                ch[idx] = ch[idx - 1]
            continue

        w = _build_risk_set_weights(t, T, E, event_of_interest, gc, sample_weights)
        in_risk = w > 0
        if not np.any(in_risk):
            if idx > 0:
                ch[idx] = ch[idx - 1]
            continue

        exp_xb = np.exp(X[in_risk] @ beta)
        denom = np.sum(w[in_risk] * exp_xb)
        if denom <= 0:
            if idx > 0:
                ch[idx] = ch[idx - 1]
            continue

        d_k = sum(sample_weights[i] for i in fail_at_t)
        increment = d_k / denom
        ch[idx] = (ch[idx - 1] if idx > 0 else 0.0) + increment

    return ch
