"""
Joint frailty model for recurrent events and a terminal event (lapse/cancellation).

The problem: when a policyholder lapses, they're censored from the recurrent
claims process. But lapse is not random — high-risk policyholders are more
likely to be cancelled (by the insurer) or to switch (because they got a claim
and their premium spiked). This is informative censoring.

Ignoring it gives biased frailty estimates. The joint model handles this by
sharing the frailty z_i between the claims process and the lapse process.

Model structure:
    Claims:    lambda_i(t) = z_i^alpha * lambda_c(t) * exp(X_c' beta_c)
    Terminal:  mu_i(t) = z_i^gamma * mu_0(t) * exp(X_t' beta_t)

where z_i ~ Gamma(theta, theta) is the shared frailty.

When alpha != gamma, the two processes have different sensitivity to the
frailty — this is the "proportional frailty" generalisation of Rondeau et al.
(2007). When alpha = gamma = 1, it's the standard joint model.

Fitting: EM algorithm with E-step via Gauss-Laguerre quadrature (for gamma
frailty) and M-step updating beta_c, beta_t, and theta separately.

References
----------
Rondeau V, Mathoulin-Pélissier S, Jacqmin-Gadda H, Brouste V, Soubeyran P
(2007). Joint frailty models for recurring events and death using maximum
penalised likelihood estimation. Biostatistics, 8(4):708-721.

Liu L, Wolfe RA, Huang X (2004). Shared frailty models for recurrent events
and a terminal event. Biometrics, 60(3):747-756.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special

from .data import RecurrentEventData
from .frailty import GammaFrailty
from .models import (
    _partial_log_likelihood,
    _robust_sandwich_se,
    FrailtyFitResult,
)


# ------------------------------------------------------------------
# Data container
# ------------------------------------------------------------------


@dataclass
class JointData:
    """
    Data for joint frailty modelling: recurrent events + terminal event.

    Parameters
    ----------
    recurrent : RecurrentEventData
        Counting process data for the recurrent claims process.
    terminal_df : pd.DataFrame
        One row per subject with columns: id_col, terminal_time_col,
        terminal_event_col (1 = lapsed/cancelled, 0 = still active).
    id_col : str
        Shared subject identifier.
    terminal_time_col : str
        Time of terminal event or end of study.
    terminal_event_col : str
        Binary: 1 = terminal event occurred, 0 = censored.
    terminal_covariates : list[str]
        Covariates for the terminal event model.
    """

    recurrent: RecurrentEventData
    terminal_df: pd.DataFrame
    id_col: str
    terminal_time_col: str
    terminal_event_col: str
    terminal_covariates: list[str]

    def __post_init__(self) -> None:
        rec_ids = set(self.recurrent.subject_ids.tolist())
        term_ids = set(self.terminal_df[self.id_col].tolist())
        if not rec_ids.issubset(term_ids):
            missing = rec_ids - term_ids
            raise ValueError(
                f"Subjects in recurrent data not in terminal_df: {list(missing)[:5]}..."
            )

    @property
    def n_subjects(self) -> int:
        return self.recurrent.n_subjects


# ------------------------------------------------------------------
# Result
# ------------------------------------------------------------------


@dataclass
class JointFrailtyResult:
    """Results from a fitted JointFrailtyModel."""

    coef_recurrent: np.ndarray
    coef_recurrent_se: np.ndarray
    coef_terminal: np.ndarray
    coef_terminal_se: np.ndarray
    theta: float
    alpha: float
    gamma: float
    log_likelihood: float
    n_iter: int
    converged: bool
    recurrent_covariate_names: list[str]
    terminal_covariate_names: list[str]

    def summary_recurrent(self) -> pd.DataFrame:
        return _make_summary(
            self.coef_recurrent,
            self.coef_recurrent_se,
            self.recurrent_covariate_names,
        )

    def summary_terminal(self) -> pd.DataFrame:
        return _make_summary(
            self.coef_terminal,
            self.coef_terminal_se,
            self.terminal_covariate_names,
        )

    def __repr__(self) -> str:
        return (
            f"JointFrailtyResult("
            f"theta={self.theta:.4f}, "
            f"alpha={self.alpha:.3f}, "
            f"gamma={self.gamma:.3f}, "
            f"log_lik={self.log_likelihood:.2f}, "
            f"converged={self.converged})"
        )


def _make_summary(coef, se, names) -> pd.DataFrame:
    z = 1.959964
    hr = np.exp(coef)
    hr_lo = np.exp(coef - z * se)
    hr_hi = np.exp(coef + z * se)
    p_vals = 2 * (1 - 0.5 * (1 + special.erf(np.abs(coef / np.maximum(se, 1e-300)) / np.sqrt(2))))
    return pd.DataFrame(
        {"coef": coef, "se": se, "HR": hr, "HR_lower_95": hr_lo, "HR_upper_95": hr_hi, "p_value": p_vals},
        index=names,
    )


# ------------------------------------------------------------------
# Joint frailty model
# ------------------------------------------------------------------


class JointFrailtyModel:
    """
    Joint frailty model for recurrent insurance claims and policy lapse.

    This model handles informative censoring: policyholders who lapse
    are not independent of those who stay. High-risk policyholders are
    systematically more likely to be cancelled, so the claims data from
    survivors is a biased sample.

    The shared frailty z_i links the two processes:
        Claims intensity:  lambda_i(t) = z_i^alpha * lambda_c(t) * exp(X_c' beta_c)
        Lapse intensity:   mu_i(t) = z_i^gamma * mu_0(t) * exp(X_t' beta_t)

    When gamma > 0: higher-frailty (worse risk) policyholders are more likely
    to lapse. This is the typical direction in motor insurance.

    Parameters
    ----------
    frailty_power_recurrent : float
        Power alpha applied to frailty in the recurrent process. Default 1.0.
    frailty_power_terminal : float
        Power gamma applied to frailty in the terminal process. Default 1.0.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood.
    n_quad : int
        Number of Gauss-Laguerre quadrature points for E-step.
    verbose : bool
        Print iteration progress.
    """

    def __init__(
        self,
        frailty_power_recurrent: float = 1.0,
        frailty_power_terminal: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-4,
        n_quad: int = 15,
        verbose: bool = False,
    ) -> None:
        self.alpha = frailty_power_recurrent
        self.gamma = frailty_power_terminal
        self.max_iter = max_iter
        self.tol = tol
        self.n_quad = n_quad
        self.verbose = verbose
        self._frailty_dist = GammaFrailty()
        self._result: Optional[JointFrailtyResult] = None
        self._quad_nodes: Optional[np.ndarray] = None
        self._quad_weights: Optional[np.ndarray] = None

    def _setup_quadrature(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Gauss-Laguerre quadrature nodes and weights for gamma(theta, theta).

        Transform: z = x / theta, so integral over z is mapped to
        Gauss-Laguerre integral over x.
        """
        nodes, weights = np.polynomial.laguerre.laggauss(self.n_quad)
        z_nodes = nodes / theta
        # Adjusted weights: gamma pdf = theta^theta / Gamma(theta) * z^(theta-1) * exp(-theta*z)
        # After substitution z = x/theta: p(z) dz = x^(theta-1) exp(-x) dx / Gamma(theta)
        # So GH weights already absorb the exp(-x) factor; multiply by z_nodes^(theta-1)/Gamma(theta)
        log_w = (
            np.log(weights + 1e-300)
            + (theta - 1) * np.log(z_nodes + 1e-300)
            - special.gammaln(theta)
            - (theta - 1) * np.log(theta)
        )
        w_adj = np.exp(log_w - log_w.max())
        w_adj /= w_adj.sum()
        return z_nodes, w_adj

    def fit(self, data: JointData) -> "JointFrailtyModel":
        """
        Fit the joint frailty model via EM algorithm.

        Parameters
        ----------
        data : JointData
            Combined recurrent + terminal data.
        """
        rec = data.recurrent
        term_df = data.terminal_df.set_index(data.id_col)

        X_rec = rec.X
        stop_rec = rec.stop
        start_rec = rec.start
        event_rec = rec.event
        subject_ids_rec = rec.df[rec.id_col].to_numpy()

        unique_ids = rec.subject_ids
        n_subjects = rec.n_subjects
        id_to_idx = {sid: i for i, sid in enumerate(unique_ids)}
        subj_idx = np.array([id_to_idx[sid] for sid in subject_ids_rec])

        # Terminal event arrays (one per subject)
        T_term = np.array([float(term_df.loc[sid, data.terminal_time_col]) for sid in unique_ids])
        D_term = np.array([int(term_df.loc[sid, data.terminal_event_col]) for sid in unique_ids])

        X_term_cols = data.terminal_covariates
        if X_term_cols:
            X_term = np.array([
                [float(term_df.loc[sid, c]) for c in X_term_cols]
                for sid in unique_ids
            ])
        else:
            X_term = np.zeros((n_subjects, 0))

        p_rec = X_rec.shape[1]
        p_term = X_term.shape[1]

        # Initialise
        beta_rec = np.zeros(p_rec)
        beta_term = np.zeros(p_term)
        theta = 1.0

        n_i = np.array([int(event_rec[subj_idx == i].sum()) for i in range(n_subjects)])
        log_lik_prev = -np.inf

        for iteration in range(self.max_iter):
            # -- E-step: approximate posterior frailty via quadrature --
            risk_score_rec = np.exp(X_rec @ beta_rec) if p_rec > 0 else np.ones(len(stop_rec))
            risk_score_term = np.exp(X_term @ beta_term) if p_term > 0 else np.ones(n_subjects)

            lambda_i_rec = self._compute_cumhaz_per_subject(
                stop_rec, event_rec, risk_score_rec, subj_idx, n_subjects
            )
            lambda_i_term = self._compute_nelson_aalen(T_term, D_term, risk_score_term)

            # Posterior: E[z^alpha] and E[z^gamma] for weighted updates
            z_nodes, z_weights = self._setup_quadrature(theta)

            # Log-likelihood contribution per subject per quadrature node
            # log p(data_i | z_j) = log Poisson(n_i | z_j^alpha * lambda_i_rec)
            #                      + log Poisson(D_i | z_j^gamma * lambda_i_term)
            log_pois_rec = (
                n_i[:, None] * (self.alpha * np.log(np.maximum(z_nodes[None, :], 1e-300))
                                + np.log(np.maximum(lambda_i_rec[:, None], 1e-300)))
                - z_nodes[None, :] ** self.alpha * lambda_i_rec[:, None]
            )
            log_pois_term = (
                D_term[:, None] * (self.gamma * np.log(np.maximum(z_nodes[None, :], 1e-300))
                                   + np.log(np.maximum(lambda_i_term[:, None], 1e-300)))
                - z_nodes[None, :] ** self.gamma * lambda_i_term[:, None]
            )
            log_joint = log_pois_rec + log_pois_term  # (n_subjects, n_quad)

            # Normalise to get posterior weights
            log_scale = log_joint.max(axis=1, keepdims=True)
            unnorm = np.exp(log_joint - log_scale) * z_weights[None, :]
            post_weights = unnorm / np.maximum(unnorm.sum(axis=1, keepdims=True), 1e-300)

            # Expected z^alpha and z^gamma under posterior (per subject)
            ez_alpha = np.sum(post_weights * z_nodes[None, :] ** self.alpha, axis=1)
            ez_gamma = np.sum(post_weights * z_nodes[None, :] ** self.gamma, axis=1)

            # -- M-step: beta_rec --
            if p_rec > 0:
                weights_per_row = (ez_alpha[subj_idx])
                beta_rec = self._update_beta(beta_rec, X_rec, stop_rec, event_rec, weights_per_row)

            # -- M-step: beta_term --
            if p_term > 0:
                beta_term = self._update_beta_term(
                    beta_term, X_term, T_term, D_term, ez_gamma
                )

            # -- M-step: theta via gamma marginal --
            # Use approximate lambda_i for both processes combined
            ez_hat = np.sum(post_weights * z_nodes[None, :], axis=1)
            theta_new = self._frailty_dist.update_theta(n_i, lambda_i_rec, theta)
            theta_new = max(theta_new, 0.01)

            # Log-likelihood (marginal, approximate via quadrature)
            log_marg = np.log(np.maximum(unnorm.sum(axis=1), 1e-300)) + log_scale[:, 0]
            log_lik = float(log_marg.sum())

            if self.verbose:
                print(
                    f"  Iter {iteration+1:3d}: log_lik={log_lik:.4f}, "
                    f"theta={theta_new:.4f}"
                )

            converged = abs(log_lik - log_lik_prev) < self.tol
            theta = theta_new
            log_lik_prev = log_lik

            if converged:
                break
        else:
            warnings.warn(
                f"JointFrailtyModel: EM did not converge in {self.max_iter} iterations.",
                RuntimeWarning,
                stacklevel=2,
            )
            converged = False

        # Standard errors
        if p_rec > 0:
            ez_alpha = np.sum(post_weights * z_nodes[None, :] ** self.alpha, axis=1)
            se_rec = _robust_sandwich_se(
                beta_rec, X_rec, stop_rec, event_rec, subject_ids_rec, ez_alpha[subj_idx]
            )
        else:
            se_rec = np.array([])

        se_term = np.full(p_term, np.nan)  # TODO: implement terminal SE

        self._result = JointFrailtyResult(
            coef_recurrent=beta_rec,
            coef_recurrent_se=se_rec,
            coef_terminal=beta_term,
            coef_terminal_se=se_term,
            theta=theta,
            alpha=self.alpha,
            gamma=self.gamma,
            log_likelihood=log_lik_prev,
            n_iter=iteration + 1,
            converged=converged,
            recurrent_covariate_names=rec.covariate_cols,
            terminal_covariate_names=X_term_cols,
        )

        self._n_i = n_i
        self._lambda_i_rec = lambda_i_rec
        self._unique_ids = unique_ids
        return self

    def _compute_cumhaz_per_subject(
        self,
        stop: np.ndarray,
        event: np.ndarray,
        risk_score: np.ndarray,
        subj_idx: np.ndarray,
        n_subjects: int,
    ) -> np.ndarray:
        event_mask = event == 1
        event_times = np.sort(np.unique(stop[event_mask])) if event_mask.any() else np.array([])
        lambda_i = np.zeros(n_subjects)
        for t in event_times:
            at_risk = stop >= t
            d_k = float(np.sum((stop == t) & event_mask))
            r_k = float(np.sum(risk_score[at_risk]))
            if r_k > 0:
                dL = d_k / r_k
                for j in np.where(at_risk)[0]:
                    lambda_i[subj_idx[j]] += risk_score[j] * dL
        return lambda_i

    def _compute_nelson_aalen(
        self,
        T: np.ndarray,
        D: np.ndarray,
        risk_score: np.ndarray,
    ) -> np.ndarray:
        """Nelson-Aalen cumulative hazard for terminal event."""
        n = len(T)
        lambda_i = np.zeros(n)
        event_times = np.sort(np.unique(T[D == 1])) if D.any() else np.array([])
        for t in event_times:
            at_risk = T >= t
            d_k = float(np.sum((T == t) & (D == 1)))
            r_k = float(np.sum(risk_score[at_risk]))
            if r_k > 0:
                dL = d_k / r_k
                lambda_i += at_risk.astype(float) * risk_score * dL
        return lambda_i

    def _update_beta(
        self,
        beta_init: np.ndarray,
        X: np.ndarray,
        stop: np.ndarray,
        event: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        result = optimize.minimize(
            lambda b: -_partial_log_likelihood(b, X, stop, event, weights),
            x0=beta_init,
            method="L-BFGS-B",
            options={"maxiter": 50, "ftol": 1e-6},
        )
        return result.x

    def _update_beta_term(
        self,
        beta_init: np.ndarray,
        X_term: np.ndarray,
        T: np.ndarray,
        D: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Update terminal event model coefficients."""
        if len(beta_init) == 0:
            return beta_init

        def neg_pll(beta: np.ndarray) -> float:
            return -_partial_log_likelihood(beta, X_term, T, D, weights)

        result = optimize.minimize(
            neg_pll,
            x0=beta_init,
            method="L-BFGS-B",
            options={"maxiter": 50, "ftol": 1e-6},
        )
        return result.x

    @property
    def result_(self) -> JointFrailtyResult:
        if self._result is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self._result

    def credibility_scores(self) -> pd.DataFrame:
        """Posterior frailty scores for the recurrent process."""
        if self._result is None:
            raise RuntimeError("Model has not been fitted.")
        z_mean = self._frailty_dist.posterior_mean(
            self._n_i, self._lambda_i_rec, self._result.theta
        )
        return pd.DataFrame(
            {
                "id": self._unique_ids,
                "n_events": self._n_i,
                "frailty_mean": z_mean,
            }
        )

    def __repr__(self) -> str:
        status = "fitted" if self._result is not None else "unfitted"
        return f"JointFrailtyModel(alpha={self.alpha}, gamma={self.gamma}, {status})"
