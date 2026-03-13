"""
Tests for frailty distribution classes.
"""

import numpy as np
import pytest
from scipy import stats

from insurance_survival.recurrent.frailty import (
    GammaFrailty,
    LognormalFrailty,
    make_frailty,
)


# ------------------------------------------------------------------
# GammaFrailty
# ------------------------------------------------------------------


class TestGammaFrailty:
    @pytest.fixture
    def gf(self):
        return GammaFrailty()

    def test_name(self, gf):
        assert gf.name == "gamma"

    def test_log_marginal_shape(self, gf):
        n_i = np.array([0, 1, 2, 3])
        lambda_i = np.array([0.5, 1.0, 1.5, 2.0])
        ll = gf.log_marginal(n_i, lambda_i, theta=2.0)
        assert ll.shape == (4,)

    def test_log_marginal_finite(self, gf):
        n_i = np.array([0, 1, 5, 10])
        lambda_i = np.array([0.1, 0.5, 2.0, 5.0])
        ll = gf.log_marginal(n_i, lambda_i, theta=1.0)
        assert np.all(np.isfinite(ll))

    def test_log_marginal_negative(self, gf):
        """Log probabilities should be <= 0."""
        n_i = np.array([1, 2])
        lambda_i = np.array([1.0, 2.0])
        ll = gf.log_marginal(n_i, lambda_i, theta=1.0)
        assert np.all(ll <= 0.0)

    def test_posterior_mean_shape(self, gf):
        n_i = np.array([0, 1, 2])
        lambda_i = np.array([1.0, 1.0, 1.0])
        z = gf.posterior_mean(n_i, lambda_i, theta=2.0)
        assert z.shape == (3,)

    def test_posterior_mean_positive(self, gf):
        n_i = np.array([0, 1, 3])
        lambda_i = np.array([1.0, 1.0, 1.0])
        z = gf.posterior_mean(n_i, lambda_i, theta=2.0)
        assert np.all(z > 0)

    def test_posterior_mean_formula(self, gf):
        """E[z | n=2, lambda=1, theta=3] = (3+2)/(3+1) = 1.25"""
        z = gf.posterior_mean(np.array([2.0]), np.array([1.0]), theta=3.0)
        assert abs(z[0] - 1.25) < 1e-10

    def test_posterior_mean_above_1_when_heavy_claimant(self, gf):
        """High-claim subject should get frailty > 1."""
        z = gf.posterior_mean(np.array([5.0]), np.array([1.0]), theta=2.0)
        assert z[0] > 1.0

    def test_posterior_mean_below_1_when_no_claims(self, gf):
        """Subject with zero claims should get frailty < 1."""
        z = gf.posterior_mean(np.array([0.0]), np.array([2.0]), theta=2.0)
        assert z[0] < 1.0

    def test_posterior_variance_positive(self, gf):
        n_i = np.array([0, 1, 2])
        lambda_i = np.array([1.0, 1.0, 2.0])
        v = gf.posterior_variance(n_i, lambda_i, theta=2.0)
        assert np.all(v > 0)

    def test_credibility_weight_formula(self, gf):
        """z = lambda / (lambda + theta)"""
        w = gf.credibility_weight(np.array([1.0]), np.array([2.0]), theta=2.0)
        assert abs(w[0] - 2.0 / 4.0) < 1e-10

    def test_credibility_weight_bounded(self, gf):
        n_i = np.array([0, 1, 5])
        lambda_i = np.array([0.01, 1.0, 100.0])
        w = gf.credibility_weight(n_i, lambda_i, theta=1.0)
        assert np.all(w >= 0) and np.all(w <= 1)

    def test_credibility_weight_approaches_1_large_exposure(self, gf):
        w = gf.credibility_weight(np.array([0.0]), np.array([1000.0]), theta=1.0)
        assert w[0] > 0.999

    def test_credibility_weight_approaches_0_small_exposure(self, gf):
        w = gf.credibility_weight(np.array([0.0]), np.array([0.001]), theta=1.0)
        assert w[0] < 0.01

    def test_update_theta_returns_positive(self, gf):
        n_i = np.array([0, 1, 2, 0, 3])
        lambda_i = np.array([0.5, 1.0, 1.5, 0.5, 2.0])
        theta_new = gf.update_theta(n_i, lambda_i, theta_old=1.0)
        assert theta_new > 0

    def test_update_theta_scalar(self, gf):
        n_i = np.array([1, 2, 0, 3, 1])
        lambda_i = np.array([1.0] * 5)
        theta_new = gf.update_theta(n_i, lambda_i, theta_old=2.0)
        assert isinstance(theta_new, float)

    def test_log_marginal_matches_negative_binomial(self, gf):
        """
        The gamma-Poisson mixture is the negative binomial distribution.
        Verify against scipy.stats.nbinom at a single point.

        NB parametrisation: p = theta/(theta+lambda), r = theta
        P(n | r, p) = C(n+r-1, n) * p^r * (1-p)^n
        """
        theta = 3.0
        lam = 2.0
        n = 4
        p_nb = theta / (theta + lam)
        expected_log_pmf = stats.nbinom.logpmf(n, theta, p_nb)
        computed = gf.log_marginal(np.array([float(n)]), np.array([lam]), theta)
        assert abs(computed[0] - expected_log_pmf) < 1e-8

    def test_zero_events_zero_lambda(self, gf):
        """Zero claims, zero exposure: should still return finite value."""
        ll = gf.log_marginal(np.array([0.0]), np.array([0.0]), theta=1.0)
        assert np.isfinite(ll[0])


# ------------------------------------------------------------------
# LognormalFrailty
# ------------------------------------------------------------------


class TestLognormalFrailty:
    @pytest.fixture
    def lnf(self):
        return LognormalFrailty(n_quad=15)

    def test_name(self, lnf):
        assert lnf.name == "lognormal"

    def test_log_marginal_shape(self, lnf):
        n_i = np.array([0, 1, 2])
        lambda_i = np.array([1.0, 1.0, 1.5])
        ll = lnf.log_marginal(n_i, lambda_i, theta=0.5)
        assert ll.shape == (3,)

    def test_log_marginal_finite(self, lnf):
        n_i = np.array([0, 1, 2, 5])
        lambda_i = np.array([0.5, 1.0, 1.5, 3.0])
        ll = lnf.log_marginal(n_i, lambda_i, theta=0.5)
        assert np.all(np.isfinite(ll))

    def test_posterior_mean_positive(self, lnf):
        n_i = np.array([0, 1, 3])
        lambda_i = np.array([1.0, 1.0, 1.0])
        z = lnf.posterior_mean(n_i, lambda_i, theta=0.5)
        assert np.all(z > 0)

    def test_posterior_mean_centred(self, lnf):
        """Posterior mean should be close to 1 when n_i ~ lambda_i."""
        n_i = np.ones(100) * 2.0
        lambda_i = np.ones(100) * 2.0
        z = lnf.posterior_mean(n_i, lambda_i, theta=0.5)
        # Should average near 1 (model says exactly average risk)
        assert abs(float(np.mean(z)) - 1.0) < 0.3

    def test_posterior_variance_nonneg(self, lnf):
        n_i = np.array([0, 1, 2])
        lambda_i = np.array([1.0, 1.0, 2.0])
        v = lnf.posterior_variance(n_i, lambda_i, theta=0.5)
        assert np.all(v >= 0)

    def test_update_theta_positive(self, lnf):
        n_i = np.array([0, 1, 2, 0, 3])
        lambda_i = np.array([0.5, 1.0, 1.5, 0.5, 2.0])
        theta_new = lnf.update_theta(n_i, lambda_i, theta_old=0.5)
        assert theta_new > 0

    def test_heavy_claimant_higher_frailty(self, lnf):
        """Subject with 5 claims should have higher posterior than subject with 0."""
        z = lnf.posterior_mean(np.array([0.0, 5.0]), np.array([1.0, 1.0]), theta=0.5)
        assert z[1] > z[0]

    def test_ghq_nodes_count(self, lnf):
        assert len(lnf._nodes) == 15


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


class TestMakeFrailty:
    def test_gamma(self):
        f = make_frailty("gamma")
        assert isinstance(f, GammaFrailty)

    def test_lognormal(self):
        f = make_frailty("lognormal")
        assert isinstance(f, LognormalFrailty)

    def test_lognormal_with_kwargs(self):
        f = make_frailty("lognormal", n_quad=20)
        assert f.n_quad == 20

    def test_case_insensitive(self):
        f = make_frailty("Gamma")
        assert isinstance(f, GammaFrailty)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown frailty distribution"):
            make_frailty("inverse_gaussian")

    def test_gamma_vs_lognormal_ordering(self):
        """Both should rank subjects the same way (monotone in n_i)."""
        gf = GammaFrailty()
        lnf = LognormalFrailty()
        n_i = np.array([0.0, 1.0, 2.0, 5.0])
        lambda_i = np.ones(4)
        z_gamma = gf.posterior_mean(n_i, lambda_i, theta=2.0)
        z_lognormal = lnf.posterior_mean(n_i, lambda_i, theta=0.5)
        # Both should be monotone increasing
        assert np.all(np.diff(z_gamma) > 0)
        assert np.all(np.diff(z_lognormal) > 0)
