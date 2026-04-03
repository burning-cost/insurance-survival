"""
Additional tests for insurance_survival.evaluation._scoring — edge cases
and direct function coverage not captured in test_censored_eval.py.

Covers:
- twcrps_obs with weight functions
- twcrps_mean length mismatch validation
- murphy_elementary_score boundary behaviour
- quantile_score boundary values
- interval_score boundary values
- murphy_diagram plot function (requires matplotlib)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import weibull_min

from insurance_survival.evaluation._scoring import (
    murphy_elementary_score,
    twcrps_obs,
    twcrps_mean,
    quantile_score,
    interval_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_surv_weibull(c: float = 2.0, scale: float = 5.0):
    def S(t):
        return weibull_min.sf(np.asarray(t, dtype=float), c=c, scale=scale)
    return S


def make_surv_constant(p: float):
    """S(t) = p for all t. Pathological but numerically valid."""
    def S(t):
        return np.full(np.asarray(t).shape or (1,), p, dtype=float)
    return S


# ---------------------------------------------------------------------------
# twcrps_obs — weight function
# ---------------------------------------------------------------------------

class TestTwcrpsObsWeightFunction:
    def test_weight_fn_changes_score(self):
        """Applying a non-trivial weight function changes the score."""
        surv_fn = make_surv_weibull()
        tau = 6.0
        T_obs = 3.0

        score_uniform = twcrps_obs(surv_fn, T_obs, event=1, tau=tau)
        score_weighted = twcrps_obs(
            surv_fn, T_obs, event=1, tau=tau,
            weight_fn=lambda t: t  # upweight later times
        )
        assert score_uniform != pytest.approx(score_weighted, rel=0.01)

    def test_weight_fn_non_negative_score(self):
        """Weighted twCRPS is non-negative."""
        surv_fn = make_surv_weibull()
        score = twcrps_obs(
            surv_fn, 2.0, event=1, tau=5.0,
            weight_fn=lambda t: np.ones_like(t) * 2.0
        )
        assert score >= 0.0

    def test_weight_fn_zero_gives_zero(self):
        """Zero weight function -> score = 0."""
        surv_fn = make_surv_weibull()
        score = twcrps_obs(
            surv_fn, 2.0, event=1, tau=5.0,
            weight_fn=lambda t: np.zeros_like(t)
        )
        assert abs(score) < 1e-10

    def test_censored_with_weight(self):
        """Censored observation with weight function is finite."""
        surv_fn = make_surv_weibull()
        score = twcrps_obs(
            surv_fn, 1.5, event=0, tau=5.0,
            weight_fn=lambda t: np.sqrt(t + 0.1)
        )
        assert np.isfinite(score)
        assert score >= 0.0

    def test_obs_beyond_tau_treated_as_censored(self):
        """T_obs > tau: split = tau, entire [0,tau] integral contributes."""
        surv_fn = make_surv_weibull()
        tau = 3.0
        # Event at t=10, but tau=3 so treated as censored at tau
        score_event_after_tau = twcrps_obs(surv_fn, 10.0, event=1, tau=tau)
        score_censored_at_tau = twcrps_obs(surv_fn, 10.0, event=0, tau=tau)
        # Both split at tau — scores should be equal (both see full [0,tau] domain)
        assert score_event_after_tau == pytest.approx(score_censored_at_tau, rel=1e-6)

    def test_n_grid_parameter(self):
        """Coarser vs finer grid should give close results for smooth S."""
        surv_fn = make_surv_weibull()
        coarse = twcrps_obs(surv_fn, 2.0, event=1, tau=5.0, n_grid=50)
        fine = twcrps_obs(surv_fn, 2.0, event=1, tau=5.0, n_grid=1000)
        assert abs(coarse - fine) < 0.01


# ---------------------------------------------------------------------------
# twcrps_mean — validation
# ---------------------------------------------------------------------------

class TestTwcrpsMean:
    def test_length_mismatch_raises(self):
        """Mismatched surv_fns length raises ValueError."""
        surv_fns = [make_surv_weibull()] * 5
        T_obs = np.ones(7)
        event = np.ones(7, dtype=int)
        with pytest.raises(ValueError, match="len\\(surv_fns\\)"):
            twcrps_mean(surv_fns, T_obs, event, tau=5.0)

    def test_single_observation(self):
        """Mean of one observation equals twcrps_obs."""
        surv_fn = make_surv_weibull()
        T_obs = np.array([2.5])
        event = np.array([1])
        mean_score = twcrps_mean([surv_fn], T_obs, event, tau=5.0)
        single_score = twcrps_obs(surv_fn, 2.5, event=1, tau=5.0)
        assert mean_score == pytest.approx(single_score, rel=1e-10)

    def test_returns_float(self):
        n = 10
        surv_fns = [make_surv_weibull()] * n
        T_obs = np.linspace(0.5, 4.5, n)
        event = np.ones(n, dtype=int)
        result = twcrps_mean(surv_fns, T_obs, event, tau=5.0)
        assert isinstance(result, float)

    def test_finite_for_mixed_events(self):
        """Mix of events and censored observations."""
        n = 20
        rng = np.random.default_rng(42)
        surv_fns = [make_surv_weibull()] * n
        T_obs = rng.uniform(0.5, 4.5, n)
        event = rng.choice([0, 1], n)
        result = twcrps_mean(surv_fns, T_obs, event, tau=5.0)
        assert np.isfinite(result)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# murphy_elementary_score
# ---------------------------------------------------------------------------

class TestMurphyElementaryScore:
    def test_output_shape(self):
        """Output shape matches input."""
        n = 50
        q = np.ones(n) * 3.0
        T_obs = np.ones(n) * 2.0
        scores = murphy_elementary_score(q, T_obs, tau=5.0, theta=2.5, alpha=0.5)
        assert scores.shape == (n,)

    def test_non_negative(self):
        """Elementary scores are non-negative."""
        rng = np.random.default_rng(7)
        n = 100
        q = rng.uniform(0.5, 4.5, n)
        T_obs = rng.uniform(0.5, 5.0, n)
        scores = murphy_elementary_score(q, T_obs, tau=5.0, theta=2.0, alpha=0.5)
        assert np.all(scores >= 0.0)

    def test_zero_when_theta_outside_bracket(self):
        """When theta >= max(q, T_cens) or theta < min(q, T_cens), score = 0."""
        # theta way above both q and T_cens
        q = np.array([1.0, 2.0])
        T_obs = np.array([1.5, 2.5])
        scores = murphy_elementary_score(q, T_obs, tau=10.0, theta=9.0, alpha=0.5)
        np.testing.assert_array_equal(scores, np.zeros(2))

    def test_alpha_effect(self):
        """Different alpha levels give different scores."""
        q = np.array([3.0])
        T_obs = np.array([1.0])  # T_cens = min(1.0, 5.0) = 1.0
        # T_cens <= theta < q: score = 1 - alpha
        scores_5 = murphy_elementary_score(q, T_obs, tau=5.0, theta=2.0, alpha=0.5)
        scores_2 = murphy_elementary_score(q, T_obs, tau=5.0, theta=2.0, alpha=0.2)
        # score = 1 - alpha, so 0.5 and 0.8 respectively
        assert scores_5[0] == pytest.approx(0.5)
        assert scores_2[0] == pytest.approx(0.8)

    def test_tau_censoring_applied(self):
        """T_obs > tau is censored at tau, not at T_obs."""
        q = np.array([3.0])
        T_obs_beyond_tau = np.array([100.0])
        T_obs_at_tau = np.array([5.0])
        tau = 5.0
        theta = 4.0
        # Both should give same score since both get censored at tau=5
        s1 = murphy_elementary_score(q, T_obs_beyond_tau, tau=tau, theta=theta, alpha=0.5)
        s2 = murphy_elementary_score(q, T_obs_at_tau, tau=tau, theta=theta, alpha=0.5)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# quantile_score — boundary and edge cases
# ---------------------------------------------------------------------------

class TestQuantileScoreBoundary:
    def test_perfect_median_forecast(self):
        """Score is 0 when q_forecast = T_cens for all obs."""
        tau = 5.0
        T_obs = np.array([2.0, 3.0, 4.0])
        event = np.ones(3, dtype=int)
        q = T_obs.copy()  # perfect forecast
        score = quantile_score(q, T_obs, event, tau=tau, alpha=0.5)
        assert score == pytest.approx(0.0, abs=1e-12)

    def test_non_negative(self):
        """Quantile score is always non-negative."""
        rng = np.random.default_rng(1)
        n = 100
        tau = 8.0
        T_obs = rng.uniform(0.5, 10.0, n)
        event = rng.choice([0, 1], n)
        q = rng.uniform(0.5, 8.0, n)
        score = quantile_score(q, T_obs, event, tau=tau, alpha=0.3)
        assert score >= 0.0

    def test_censored_obs_censored_at_tau(self):
        """Censored obs with T_obs > tau: T_cens = tau."""
        tau = 5.0
        # Censored observation beyond tau: treated as tau
        T_obs = np.array([10.0])
        event = np.array([0])
        q = np.array([3.0])  # forecast below tau
        score = quantile_score(q, T_obs, event, tau=tau, alpha=0.5)
        # t_cens = min(10, 5) = 5; loss = alpha * (5 - 3) = 0.5 * 2 = 1.0
        assert score == pytest.approx(1.0)

    def test_single_observation(self):
        tau = 5.0
        T_obs = np.array([2.0])
        event = np.array([1])
        q = np.array([4.0])  # overpredict
        # t_cens = 2.0; loss = (1 - alpha) * (4 - 2) = 0.5 * 2 = 1.0
        score = quantile_score(q, T_obs, event, tau=tau, alpha=0.5)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# interval_score — boundary and edge cases
# ---------------------------------------------------------------------------

class TestIntervalScoreBoundary:
    def test_perfect_interval_no_misses(self):
        """When all T_cens fall within [lower, upper], score = width only."""
        tau = 10.0
        T_obs = np.array([3.0, 5.0, 7.0])
        event = np.ones(3, dtype=int)
        lower = np.array([1.0, 3.0, 5.0])
        upper = np.array([5.0, 7.0, 9.0])
        score = interval_score(lower, upper, T_obs, event, tau=tau, alpha=0.1)
        # Width = 4 for all; no misses
        assert score == pytest.approx(4.0)

    def test_left_miss_penalised(self):
        """Left miss increases score."""
        tau = 10.0
        T_obs = np.array([1.0])  # below lower bound
        event = np.ones(1, dtype=int)
        lower = np.array([3.0])
        upper = np.array([7.0])
        score_miss = interval_score(lower, upper, T_obs, event, tau=tau, alpha=0.2)
        # width=4, left_miss = (2/0.2) * (3 - 1) = 10 * 2 = 20, total = 24
        assert score_miss == pytest.approx(24.0)

    def test_right_miss_penalised(self):
        """Right miss increases score."""
        tau = 10.0
        T_obs = np.array([9.0])  # above upper bound
        event = np.ones(1, dtype=int)
        lower = np.array([3.0])
        upper = np.array([7.0])
        score_miss = interval_score(lower, upper, T_obs, event, tau=tau, alpha=0.2)
        # width=4, right_miss = (2/0.2) * (9 - 7) = 10 * 2 = 20, total = 24
        assert score_miss == pytest.approx(24.0)

    def test_non_negative(self):
        """Interval score is always non-negative."""
        rng = np.random.default_rng(99)
        n = 50
        tau = 8.0
        T_obs = rng.uniform(0.5, 9.0, n)
        event = rng.choice([0, 1], n)
        lower = rng.uniform(0.0, 3.0, n)
        upper = lower + rng.uniform(0.5, 4.0, n)
        score = interval_score(lower, upper, T_obs, event, tau=tau, alpha=0.1)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# murphy_diagram — requires matplotlib
# ---------------------------------------------------------------------------

class TestMurphyDiagram:
    def test_murphy_diagram_runs(self):
        """murphy_diagram should return a Figure without error."""
        pytest.importorskip("matplotlib")
        from insurance_survival.evaluation._plots import murphy_diagram

        rng = np.random.default_rng(42)
        n = 50
        tau = 6.0
        T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
        T_obs = np.minimum(T_true, tau)
        event = (T_true <= tau).astype(int)

        surv_fn = make_surv_weibull()
        forecasters = {
            "ModelA": ([surv_fn] * n, T_obs, event),
        }
        fig = murphy_diagram(forecasters, tau=tau, alpha=0.5)
        assert fig is not None

    def test_murphy_diagram_two_forecasters(self):
        """murphy_diagram with two forecasters should run without error."""
        pytest.importorskip("matplotlib")
        from insurance_survival.evaluation._plots import murphy_diagram

        rng = np.random.default_rng(43)
        n = 50
        tau = 6.0
        T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
        T_obs = np.minimum(T_true, tau)
        event = (T_true <= tau).astype(int)

        good_fn = make_surv_weibull(c=2, scale=5)
        bad_fn = make_surv_weibull(c=1, scale=5)
        forecasters = {
            "Correct": ([good_fn] * n, T_obs, event),
            "Wrong":   ([bad_fn] * n, T_obs, event),
        }
        fig = murphy_diagram(forecasters, tau=tau)
        assert fig is not None

    def test_murphy_diagram_mismatch_raises(self):
        """Mismatch between surv_fns and T_obs raises ValueError."""
        pytest.importorskip("matplotlib")
        from insurance_survival.evaluation._plots import murphy_diagram

        n = 10
        tau = 5.0
        T_obs = np.ones(n)
        event = np.ones(n, dtype=int)

        # surv_fns has n+1 items but T_obs has n
        forecasters = {"Bad": ([make_surv_weibull()] * (n + 1), T_obs, event)}
        with pytest.raises(ValueError, match="len\\(surv_fns\\)"):
            murphy_diagram(forecasters, tau=tau)
