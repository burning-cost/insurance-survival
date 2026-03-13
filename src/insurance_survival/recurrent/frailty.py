"""
Frailty distributions and posterior computations.

In insurance terms, the frailty z_i is a latent multiplier on the baseline
claim intensity for policyholder i. A policy with z_i = 2 has twice the claim
intensity of the average risk. The frailty distribution captures unobserved
heterogeneity that covariates don't explain.

Gamma frailty is the natural choice: conjugate with the Poisson likelihood, so
the posterior has a closed form that maps directly to Bühlmann-Straub
credibility. Lognormal frailty is more flexible but requires numerical
quadrature (Gauss-Hermite).

The connection to credibility:
    E[z_i | data] = (theta + n_i) / (theta + Lambda_i)

where n_i = number of claims for policy i, Lambda_i = expected cumulative
hazard (sum of baseline hazard × exp(X'beta) over risk intervals).

This is exactly the Bühlmann-Straub credibility formula with:
    k = theta  (the credibility parameter)
    z = n_i / (n_i + theta)  (the weight given to observed experience)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy import special, stats


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------


class FrailtyDistribution(ABC):
    """Abstract base for frailty distributions."""

    @abstractmethod
    def log_marginal(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Log marginal likelihood contribution for each subject.

        Parameters
        ----------
        n_i : shape (n_subjects,)
            Observed event counts per subject.
        lambda_i : shape (n_subjects,)
            Cumulative hazard (expected events) per subject.
        theta : float
            Frailty dispersion parameter (variance = 1/theta for gamma).

        Returns
        -------
        log_lik : shape (n_subjects,)
        """

    @abstractmethod
    def posterior_mean(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Posterior mean of the frailty E[z_i | data].

        This is the credibility-weighted claim multiplier for each subject.
        """

    @abstractmethod
    def posterior_variance(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """Posterior variance Var[z_i | data]."""

    def credibility_weight(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Bühlmann-Straub credibility weight z_i = n_i / (n_i + theta).

        This measures how much weight to give observed claim experience
        versus the portfolio prior.
        """
        return n_i / (n_i + theta)

    @abstractmethod
    def update_theta(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta_old: float,
    ) -> float:
        """
        M-step: update theta given posterior expectations.

        Returns the new theta estimate.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Distribution name."""


# ------------------------------------------------------------------
# Gamma frailty
# ------------------------------------------------------------------


class GammaFrailty(FrailtyDistribution):
    """
    Gamma shared frailty: z_i ~ Gamma(theta, theta), so E[z_i] = 1, Var[z_i] = 1/theta.

    Gamma is conjugate with the Poisson likelihood, giving a negative binomial
    marginal. The posterior is also gamma, enabling exact EM updates.

    References
    ----------
    Vaupel, Manton & Stallard (1979); Clayton (1978); Nielsen et al. (1992).
    """

    @property
    def name(self) -> str:
        return "gamma"

    def log_marginal(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Log marginal likelihood: integrate out gamma frailty analytically.

        log p(n_i | theta) = log Gamma(theta + n_i) - log Gamma(theta)
                             + theta * log(theta) - (theta + n_i) * log(theta + lambda_i)
                             + n_i * log(lambda_i) - log(n_i!)

        Note: the log(n_i!) term (constant w.r.t. parameters) is included for
        absolute likelihood but cancels in optimisation.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)

        log_lik = (
            special.gammaln(theta + n_i)
            - special.gammaln(theta)
            + theta * np.log(theta)
            - (theta + n_i) * np.log(theta + lambda_i)
            + n_i * np.log(np.maximum(lambda_i, 1e-300))
            - special.gammaln(n_i + 1)
        )
        return log_lik

    def posterior_mean(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        E[z_i | data] = (theta + n_i) / (theta + lambda_i).

        This is the Bühlmann-Straub credibility premium. The numerator is
        the posterior shape, the denominator is the posterior rate.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)
        return (theta + n_i) / (theta + lambda_i)

    def posterior_variance(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Var[z_i | data] = (theta + n_i) / (theta + lambda_i)^2.

        Posterior is Gamma(theta + n_i, theta + lambda_i).
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)
        return (theta + n_i) / (theta + lambda_i) ** 2

    def credibility_weight(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Classical Bühlmann-Straub weight.

        z = lambda_i / (lambda_i + theta)

        When lambda_i >> theta (much observed exposure), z -> 1 and we trust
        the data. When lambda_i << theta (little exposure), z -> 0 and we
        revert to portfolio mean.
        """
        lambda_i = np.asarray(lambda_i, dtype=float)
        return lambda_i / (lambda_i + theta)

    def update_theta(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta_old: float,
    ) -> float:
        """
        M-step update for theta via Newton-Raphson on the marginal log-likelihood.

        The score function for theta in the gamma-frailty negative binomial is:
            d/dtheta log L = N * [log(theta/(theta+lambda)) + 1 - psi(theta+n) + psi(theta)]

        where psi is the digamma function.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)

        def neg_log_lik(theta: float) -> float:
            if theta <= 0:
                return 1e10
            return -float(np.sum(self.log_marginal(n_i, lambda_i, theta)))

        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            neg_log_lik,
            bounds=(0.01, 1000.0),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(result.x)


# ------------------------------------------------------------------
# Lognormal frailty
# ------------------------------------------------------------------


class LognormalFrailty(FrailtyDistribution):
    """
    Lognormal shared frailty: log(z_i) ~ N(mu, sigma^2).

    With the constraint E[z_i] = 1, we have mu = -sigma^2/2 and
    theta = sigma^2 is the dispersion parameter.

    Lognormal is more flexible than gamma (lighter tails in the lower range,
    heavier in the upper range) but requires numerical integration. We use
    Gauss-Hermite quadrature (15 points — adequate for sigma < 2).

    References
    ----------
    Hougaard (1984), McGilchrist & Aisbett (1991).
    """

    def __init__(self, n_quad: int = 15) -> None:
        self.n_quad = n_quad
        # Gauss-Hermite nodes and weights
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(n_quad)

    @property
    def name(self) -> str:
        return "lognormal"

    def _ghq_frailty_values(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (frailty values, weights) for Gauss-Hermite quadrature.

        For log(z) ~ N(-theta/2, theta), we substitute u = (log(z) + theta/2) / sqrt(theta)
        so that the Gauss-Hermite nodes in u map to frailty values.
        """
        sigma = np.sqrt(theta)
        # log(z) = sqrt(2) * sigma * node - theta/2
        log_z = np.sqrt(2) * sigma * self._nodes - theta / 2.0
        z_vals = np.exp(log_z)
        # Gauss-Hermite weights already include the exp(-x^2) factor
        w_vals = self._weights / np.sqrt(np.pi)
        return z_vals, w_vals

    def log_marginal(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Log marginal via Gauss-Hermite quadrature.

        Integrates p(n_i | z_i, lambda_i) * p(z_i | theta) dz_i numerically.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)
        z_vals, w_vals = self._ghq_frailty_values(theta)

        # (n_subjects, n_quad) log Poisson likelihoods
        # Poisson(n | z * lambda): log p = n*log(z*lambda) - z*lambda - log(n!)
        log_pois = (
            n_i[:, None] * np.log(np.maximum(z_vals[None, :] * lambda_i[:, None], 1e-300))
            - z_vals[None, :] * lambda_i[:, None]
            - special.gammaln(n_i[:, None] + 1)
        )
        # Marginal = sum_q w_q * exp(log_pois_q)
        # Use log-sum-exp for numerical stability
        log_scale = log_pois.max(axis=1, keepdims=True)
        integral = np.sum(w_vals[None, :] * np.exp(log_pois - log_scale), axis=1)
        log_lik = np.log(np.maximum(integral, 1e-300)) + log_scale[:, 0]
        return log_lik

    def posterior_mean(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        E[z_i | data] via GHQ: integral(z * p(n|z,lambda) * p(z)) / marginal.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)
        z_vals, w_vals = self._ghq_frailty_values(theta)

        log_pois = (
            n_i[:, None] * np.log(np.maximum(z_vals[None, :] * lambda_i[:, None], 1e-300))
            - z_vals[None, :] * lambda_i[:, None]
            - special.gammaln(n_i[:, None] + 1)
        )
        log_scale = log_pois.max(axis=1, keepdims=True)
        pois_scaled = np.exp(log_pois - log_scale)

        numerator = np.sum(w_vals[None, :] * z_vals[None, :] * pois_scaled, axis=1)
        denominator = np.sum(w_vals[None, :] * pois_scaled, axis=1)
        return numerator / np.maximum(denominator, 1e-300)

    def posterior_variance(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        """
        Var[z_i | data] = E[z^2|data] - E[z|data]^2 via GHQ.
        """
        n_i = np.asarray(n_i, dtype=float)
        lambda_i = np.asarray(lambda_i, dtype=float)
        z_vals, w_vals = self._ghq_frailty_values(theta)

        log_pois = (
            n_i[:, None] * np.log(np.maximum(z_vals[None, :] * lambda_i[:, None], 1e-300))
            - z_vals[None, :] * lambda_i[:, None]
            - special.gammaln(n_i[:, None] + 1)
        )
        log_scale = log_pois.max(axis=1, keepdims=True)
        pois_scaled = np.exp(log_pois - log_scale)

        denom = np.sum(w_vals[None, :] * pois_scaled, axis=1)
        e_z = np.sum(w_vals[None, :] * z_vals[None, :] * pois_scaled, axis=1) / np.maximum(denom, 1e-300)
        e_z2 = np.sum(w_vals[None, :] * z_vals[None, :] ** 2 * pois_scaled, axis=1) / np.maximum(denom, 1e-300)
        return np.maximum(e_z2 - e_z ** 2, 0.0)

    def update_theta(
        self,
        n_i: np.ndarray,
        lambda_i: np.ndarray,
        theta_old: float,
    ) -> float:
        """M-step: maximise marginal log-likelihood over theta via scalar optimisation."""
        from scipy.optimize import minimize_scalar

        def neg_log_lik(theta: float) -> float:
            if theta <= 0:
                return 1e10
            return -float(np.sum(self.log_marginal(n_i, lambda_i, theta)))

        result = minimize_scalar(
            neg_log_lik,
            bounds=(1e-4, 10.0),
            method="bounded",
            options={"xatol": 1e-5},
        )
        return float(result.x)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def make_frailty(name: str, **kwargs) -> FrailtyDistribution:
    """
    Factory function for frailty distributions.

    Parameters
    ----------
    name : str
        One of "gamma" or "lognormal".
    **kwargs
        Passed to the distribution constructor.

    Returns
    -------
    FrailtyDistribution
    """
    name = name.lower()
    if name == "gamma":
        return GammaFrailty()
    elif name == "lognormal":
        return LognormalFrailty(**kwargs)
    else:
        raise ValueError(f"Unknown frailty distribution: {name!r}. Choose 'gamma' or 'lognormal'.")
