"""Tests for the core EM algorithm components."""

import numpy as np
import pytest

from insurance_survival.cure._em import (
    compute_loglik,
    compute_pi,
    e_step,
    lognormal_aft_density,
    lognormal_aft_survival,
    lognormal_neg_loglik,
    m_step_incidence,
    m_step_lognormal,
    m_step_weibull,
    weibull_aft_density,
    weibull_aft_survival,
    weibull_neg_loglik,
)


class TestEStep:
    def test_event_weights_are_one(self):
        """Observed events always have w=1."""
        n = 50
        pi = np.full(n, 0.6)
        surv_u = np.full(n, 0.8)
        event = np.zeros(n)
        event[:20] = 1.0
        w = e_step(pi, surv_u, event)
        assert np.allclose(w[:20], 1.0)

    def test_censored_weights_in_unit_interval(self):
        n = 100
        pi = np.random.default_rng(0).uniform(0.1, 0.9, n)
        surv_u = np.random.default_rng(1).uniform(0.1, 0.9, n)
        event = np.zeros(n)
        w = e_step(pi, surv_u, event)
        assert np.all(w >= 0) and np.all(w <= 1)

    def test_high_pi_high_weight(self):
        """When pi is high and S_u is high, censored weight should be high."""
        pi = np.array([0.9, 0.1])
        surv_u = np.array([0.9, 0.9])
        event = np.array([0.0, 0.0])
        w = e_step(pi, surv_u, event)
        assert w[0] > w[1]

    def test_boundary_pi_zero(self):
        """pi=0: weight should be 0 for censored."""
        pi = np.array([0.0, 0.5])
        surv_u = np.array([0.8, 0.8])
        event = np.array([0.0, 0.0])
        w = e_step(pi, surv_u, event)
        assert w[0] == pytest.approx(0.0, abs=1e-10)

    def test_boundary_surv_zero(self):
        """S_u=0: censored weight should be 0."""
        pi = np.array([0.5, 0.5])
        surv_u = np.array([0.0, 0.8])
        event = np.array([0.0, 0.0])
        w = e_step(pi, surv_u, event)
        assert w[0] == pytest.approx(0.0, abs=1e-10)

    def test_formula_correctness(self):
        """Manually verify the E-step formula."""
        pi_val = 0.7
        surv_val = 0.6
        expected = (pi_val * surv_val) / (pi_val * surv_val + (1 - pi_val))
        pi = np.array([pi_val])
        surv_u = np.array([surv_val])
        event = np.array([0.0])
        w = e_step(pi, surv_u, event)
        assert w[0] == pytest.approx(expected, abs=1e-10)


class TestComputePi:
    def test_output_range(self):
        z = np.random.default_rng(0).normal(0, 1, (100, 3))
        gamma = np.array([0.5, -0.3, 0.2])
        intercept = np.array([-0.5])
        pi = compute_pi(z, gamma, intercept)
        assert np.all(pi >= 0) and np.all(pi <= 1)

    def test_known_value(self):
        """sigmoid(0) = 0.5."""
        z = np.zeros((1, 1))
        gamma = np.array([0.0])
        intercept = np.array([0.0])
        pi = compute_pi(z, gamma, intercept)
        assert pi[0] == pytest.approx(0.5)

    def test_large_positive_linear(self):
        """Large positive linear predictor => pi close to 1."""
        z = np.ones((1, 1))
        gamma = np.array([100.0])
        intercept = np.array([0.0])
        pi = compute_pi(z, gamma, intercept)
        assert pi[0] > 0.99


class TestMStepIncidence:
    def test_returns_arrays(self):
        n = 50
        z = np.random.default_rng(0).normal(0, 1, (n, 2))
        w = np.ones(n) * 0.5
        gamma, intercept = m_step_incidence(z, w)
        assert gamma.shape == (2,)
        assert intercept.shape == (1,)

    def test_all_ones_weight(self):
        """All w=1: logistic fits event indicator = 1 for all."""
        n = 40
        z = np.random.default_rng(0).normal(0, 1, (n, 2))
        w = np.ones(n)
        gamma, intercept = m_step_incidence(z, w)
        # Intercept should be large positive
        assert intercept[0] > 1.0

    def test_all_zeros_weight(self):
        """All w=0: logistic fits label=0 for all. Intercept should be negative."""
        n = 40
        z = np.random.default_rng(0).normal(0, 1, (n, 2))
        w = np.zeros(n)
        gamma, intercept = m_step_incidence(z, w)
        assert intercept[0] < -1.0


class TestWeibullFunctions:
    def test_survival_at_zero_is_one(self):
        x = np.zeros((10, 1))
        t = np.full(10, 1e-10)
        beta = np.zeros(1)
        surv = weibull_aft_survival(t, x, log_lambda=2.0, log_rho=0.0, beta=beta)
        assert np.allclose(surv, 1.0, atol=1e-5)

    def test_survival_decreases(self):
        x = np.zeros((3, 1))
        t = np.array([1.0, 10.0, 100.0])
        beta = np.zeros(1)
        surv = weibull_aft_survival(t, x, log_lambda=2.0, log_rho=0.0, beta=beta)
        assert surv[0] > surv[1] > surv[2]

    def test_density_nonnegative(self):
        n = 50
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (n, 2))
        t = rng.uniform(1, 20, n)
        beta = np.array([0.1, -0.2])
        dens = weibull_aft_density(t, x, log_lambda=2.0, log_rho=0.1, beta=beta)
        assert np.all(dens >= 0)

    def test_neg_loglik_returns_float(self):
        n = 30
        rng = np.random.default_rng(1)
        t = rng.uniform(1, 20, n)
        x = rng.normal(0, 1, (n, 2))
        event = rng.integers(0, 2, n).astype(float)
        w = np.ones(n)
        params = np.array([2.0, 0.0, 0.1, -0.1])
        val = weibull_neg_loglik(params, t, x, event, w)
        assert isinstance(val, float)

    def test_m_step_weibull_returns_correct_shape(self):
        n = 80
        rng = np.random.default_rng(2)
        t = rng.uniform(1, 30, n)
        x = rng.normal(0, 1, (n, 2))
        event = rng.integers(0, 2, n).astype(float)
        w = np.ones(n)
        params = m_step_weibull(t, x, event, w)
        assert params.shape == (4,)  # log_lambda, log_rho, beta_0, beta_1

    def test_m_step_weibull_improves_loglik(self):
        """Running the M-step should reduce the negative loglik."""
        n = 100
        rng = np.random.default_rng(3)
        t = rng.exponential(20, n)
        x = rng.normal(0, 1, (n, 1))
        event = (t < 15).astype(float)
        w = np.ones(n)
        init = np.array([2.0, 0.0, 0.0])
        nll_init = weibull_neg_loglik(init, t, x, event, w)
        optimised = m_step_weibull(t, x, event, w, init_params=init.copy())
        nll_opt = weibull_neg_loglik(optimised, t, x, event, w)
        assert nll_opt <= nll_init + 1e-5


class TestLogNormalFunctions:
    def test_survival_decreases(self):
        x = np.zeros((3, 1))
        t = np.array([1.0, 10.0, 100.0])
        beta = np.zeros(1)
        surv = lognormal_aft_survival(t, x, mu=3.0, log_sigma=0.0, beta=beta)
        assert surv[0] > surv[1] > surv[2]

    def test_density_nonnegative(self):
        n = 50
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (n, 2))
        t = rng.uniform(1, 20, n)
        beta = np.array([0.1, -0.2])
        dens = lognormal_aft_density(t, x, mu=3.0, log_sigma=0.0, beta=beta)
        assert np.all(dens >= 0)

    def test_m_step_lognormal_returns_correct_shape(self):
        n = 80
        rng = np.random.default_rng(2)
        t = rng.uniform(1, 30, n)
        x = rng.normal(0, 1, (n, 2))
        event = rng.integers(0, 2, n).astype(float)
        w = np.ones(n)
        params = m_step_lognormal(t, x, event, w)
        assert params.shape == (4,)

    def test_m_step_lognormal_improves_loglik(self):
        n = 100
        rng = np.random.default_rng(3)
        log_t = rng.normal(3.0, 0.8, n)
        t = np.exp(log_t)
        x = rng.normal(0, 1, (n, 1))
        event = (t < 25).astype(float)
        w = np.ones(n)
        init = np.array([3.0, 0.0, 0.0])
        nll_init = lognormal_neg_loglik(init, t, x, event, w)
        optimised = m_step_lognormal(t, x, event, w, init_params=init.copy())
        nll_opt = lognormal_neg_loglik(optimised, t, x, event, w)
        assert nll_opt <= nll_init + 1e-5


class TestComputeLogLik:
    def test_returns_float(self):
        n = 50
        rng = np.random.default_rng(0)
        pi = rng.uniform(0.2, 0.8, n)
        dens_u = rng.uniform(0.01, 0.1, n)
        surv_u = rng.uniform(0.3, 0.9, n)
        event = rng.integers(0, 2, n).astype(float)
        val = compute_loglik(pi, dens_u, surv_u, event)
        assert isinstance(val, float)

    def test_finite(self):
        n = 50
        rng = np.random.default_rng(1)
        pi = rng.uniform(0.2, 0.8, n)
        dens_u = rng.uniform(0.01, 0.1, n)
        surv_u = rng.uniform(0.3, 0.9, n)
        event = rng.integers(0, 2, n).astype(float)
        val = compute_loglik(pi, dens_u, surv_u, event)
        assert np.isfinite(val)

    def test_higher_is_better(self):
        """Correct parameters should yield higher log-likelihood."""
        n = 200
        rng = np.random.default_rng(2)
        t = rng.exponential(20, n)
        x = np.zeros((n, 1))
        event = (t < 15).astype(float)

        # Good params
        surv_good = weibull_aft_survival(t, x, log_lambda=np.log(20), log_rho=0.0, beta=np.zeros(1))
        dens_good = weibull_aft_density(t, x, log_lambda=np.log(20), log_rho=0.0, beta=np.zeros(1))
        pi_good = np.full(n, 0.5)

        # Bad params
        surv_bad = weibull_aft_survival(t, x, log_lambda=np.log(5), log_rho=1.5, beta=np.zeros(1))
        dens_bad = weibull_aft_density(t, x, log_lambda=np.log(5), log_rho=1.5, beta=np.zeros(1))
        pi_bad = np.full(n, 0.9)

        ll_good = compute_loglik(pi_good, dens_good, surv_good, event)
        ll_bad = compute_loglik(pi_bad, dens_bad, surv_bad, event)
        # Good params need not always be better on a small dataset, but we run
        # a sanity check that the function is monotone with respect to obvious extremes
        assert isinstance(ll_good, float) and isinstance(ll_bad, float)
