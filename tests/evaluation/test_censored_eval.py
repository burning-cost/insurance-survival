"""
Tests for insurance_survival.evaluation.CensoredForecastEvaluator.

Test strategy follows the build spec (KB 5544):

1. Proper score test: true Weibull model beats misspecified model on twCRPS.
2. Uncensored equivalence: twCRPS with event=1 for all obs matches standard
   CRPS formula (integral of (1{s>=t} - F(s))^2 over [0, tau]).
3. Censoring invariance: fixed-tau censoring preserves model ranking.
4. Murphy diagram area: area under Murphy curve equals mean quantile loss.
5. Reliability diagram runs without error for well-calibrated model.
6. Edge cases: all censored, all uncensored, tiny tau.
7. from_matrix helper: produces correct callables.
8. compare() returns sorted DataFrame with expected columns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import weibull_min, expon

from insurance_survival.evaluation import (
    CensoredForecastEvaluator,
    from_matrix,
    twcrps_obs,
    quantile_score,
    interval_score,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_surv_weibull(c: float, scale: float):
    """Return S(t) = P(T > t) for Weibull(c, scale)."""
    def S(t):
        return weibull_min.sf(np.asarray(t, dtype=float), c=c, scale=scale)
    return S


def make_surv_expo(scale: float):
    """Return S(t) for Exponential(scale)."""
    def S(t):
        return expon.sf(np.asarray(t, dtype=float), scale=scale)
    return S


@pytest.fixture
def weibull_data():
    """500 Weibull(2, 5) samples with fixed tau=8 censoring."""
    rng = np.random.default_rng(2024)
    n = 500
    T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
    tau = 8.0
    T_obs = np.minimum(T_true, tau)
    event = (T_true <= tau).astype(int)
    return T_obs, event, tau


# ---------------------------------------------------------------------------
# Test 1: Proper score — true model beats misspecified model
# ---------------------------------------------------------------------------

def test_proper_score_true_beats_misspecified(weibull_data):
    """True Weibull(2,5) should score better (lower twCRPS) than Expo(5)."""
    T_obs, event, tau = weibull_data
    n = len(T_obs)

    true_fns = [make_surv_weibull(2, 5)] * n
    wrong_fns = [make_surv_expo(5)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    score_true = ev.twcrps(true_fns, T_obs, event)
    score_wrong = ev.twcrps(wrong_fns, T_obs, event)

    assert score_true < score_wrong, (
        f"Expected true model score ({score_true:.4f}) < "
        f"misspecified score ({score_wrong:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 2: Uncensored equivalence — manually compute integral and compare
# ---------------------------------------------------------------------------

def test_uncensored_twcrps_matches_manual_integral():
    """twCRPS with event=1 matches direct numerical integration."""
    tau = 6.0
    T_obs_val = 3.0
    surv_fn = make_surv_weibull(2, 5)

    grid = np.linspace(0.0, tau, 2000)
    F = 1.0 - surv_fn(grid)
    # indicator is 1 for s >= T_obs_val
    indicator = (grid >= T_obs_val).astype(float)
    expected = float(np.trapezoid((indicator - F) ** 2, grid))

    result = twcrps_obs(surv_fn, T_obs_val, event=1, tau=tau, n_grid=2000)

    assert abs(result - expected) < 1e-4, (
        f"twCRPS {result:.6f} doesn't match manual integral {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Censoring invariance — ranking preserved under fixed-tau censoring
# ---------------------------------------------------------------------------

def test_censoring_invariance_ranking():
    """Fixed-tau censoring preserves model ranking (correct > misspecified)."""
    rng = np.random.default_rng(999)
    n = 300
    tau = 5.0
    T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)

    # Heavy censoring: tau = 5 means ~1 - S(5) of events observed
    T_obs = np.minimum(T_true, tau)
    event = (T_true <= tau).astype(int)
    cens_frac = 1.0 - event.mean()

    assert cens_frac > 0.1, f"Expected some censoring, got {cens_frac:.0%}"

    true_fns = [make_surv_weibull(2, 5)] * n
    good_fns = [make_surv_weibull(2.2, 5)] * n  # slightly wrong shape
    bad_fns = [make_surv_expo(5)] * n             # clearly wrong

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    s_true = ev.twcrps(true_fns, T_obs, event)
    s_good = ev.twcrps(good_fns, T_obs, event)
    s_bad = ev.twcrps(bad_fns, T_obs, event)

    assert s_true < s_bad, f"True ({s_true:.4f}) should beat bad ({s_bad:.4f})"
    assert s_good < s_bad, f"Near-true ({s_good:.4f}) should beat bad ({s_bad:.4f})"


# ---------------------------------------------------------------------------
# Test 4: Murphy diagram area equals mean quantile loss
# ---------------------------------------------------------------------------

def test_murphy_diagram_area_equals_quantile_loss(weibull_data):
    """Area under Murphy diagram curve should equal mean quantile loss."""
    T_obs, event, tau = weibull_data
    n = len(T_obs)
    surv_fns = [make_surv_weibull(2, 5)] * n

    ev = CensoredForecastEvaluator(tau=tau, alpha=0.5, warn=False)

    # Mean quantile loss
    qs = ev.quantile_score(
        np.full(n, weibull_min.ppf(0.5, c=2, scale=5)),
        T_obs, event
    )

    # Murphy diagram area
    thresholds = np.linspace(0.0, tau, 500)
    from insurance_survival.evaluation._scoring import murphy_elementary_score
    q_preds = np.full(n, weibull_min.ppf(0.5, c=2, scale=5))
    mean_es = np.array([
        murphy_elementary_score(q_preds, T_obs, tau, theta, alpha=0.5).mean()
        for theta in thresholds
    ])
    area = float(np.trapezoid(mean_es, thresholds))

    # The area under the Murphy curve equals the quantile score (pinball loss)
    assert abs(area - qs) < 0.05, (
        f"Murphy area ({area:.4f}) should match quantile score ({qs:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 5: Reliability diagram runs without error
# ---------------------------------------------------------------------------

def test_reliability_diagram_runs(weibull_data):
    """Reliability diagram should run without error for a valid model."""
    pytest.importorskip("matplotlib")
    T_obs, event, tau = weibull_data
    n = len(T_obs)
    surv_fns = [make_surv_weibull(2, 5)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    fig = ev.reliability_diagram(surv_fns, T_obs, event, eval_time=3.0, n_bins=8)
    assert fig is not None


# ---------------------------------------------------------------------------
# Test 6: Edge cases
# ---------------------------------------------------------------------------

def test_all_censored_at_zero():
    """When all obs are censored at t=0, twCRPS approaches integral of (1-F)^2."""
    tau = 5.0
    n = 20
    # All censored immediately (event=0, T_obs=0.0)
    T_obs = np.zeros(n)
    event = np.zeros(n, dtype=int)
    surv_fn = make_surv_weibull(2, 5)
    surv_fns = [surv_fn] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    score = ev.twcrps(surv_fns, T_obs, event)

    # Should be finite and non-negative
    assert np.isfinite(score), "Score should be finite"
    assert score >= 0.0, "Score should be non-negative"


def test_all_uncensored():
    """All event=1 (no censoring) — score should be positive and finite."""
    tau = 10.0
    n = 50
    rng = np.random.default_rng(42)
    T_obs = weibull_min.rvs(c=2, scale=3, size=n, random_state=rng)
    T_obs = np.clip(T_obs, 0.01, tau - 0.01)  # ensure all within (0, tau)
    event = np.ones(n, dtype=int)
    surv_fns = [make_surv_weibull(2, 3)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    score = ev.twcrps(surv_fns, T_obs, event)
    assert np.isfinite(score)
    assert score >= 0.0


def test_tiny_tau():
    """Very small tau: score should be near 0 for a reasonable model."""
    tau = 0.01  # Essentially no time horizon
    n = 20
    T_obs = np.full(n, 0.1)  # All beyond tau, so all censored at tau
    event = np.zeros(n, dtype=int)
    surv_fns = [make_surv_weibull(2, 5)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    score = ev.twcrps(surv_fns, T_obs, event)
    assert score >= 0.0
    assert np.isfinite(score)
    assert score < 0.01, f"Score {score} should be near 0 for tiny tau"


def test_quantile_score_basic():
    """Quantile score: perfect quantile forecast should score lower than wrong."""
    n = 200
    tau = 8.0
    rng = np.random.default_rng(7)
    T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
    T_obs = np.minimum(T_true, tau)
    event = (T_true <= tau).astype(int)

    # Correct median
    q_correct = np.full(n, weibull_min.ppf(0.5, c=2, scale=5))
    # Wrong quantile — way off
    q_wrong = np.full(n, 0.1)

    ev = CensoredForecastEvaluator(tau=tau, alpha=0.5, warn=False)
    s_correct = ev.quantile_score(q_correct, T_obs, event)
    s_wrong = ev.quantile_score(q_wrong, T_obs, event)

    assert s_correct < s_wrong, (
        f"Correct quantile ({s_correct:.4f}) should score better than wrong ({s_wrong:.4f})"
    )


def test_interval_score_basic():
    """Interval score: tighter correct interval beats wide wrong interval."""
    n = 200
    tau = 8.0
    rng = np.random.default_rng(8)
    T_true = weibull_min.rvs(c=2, scale=5, size=n, random_state=rng)
    T_obs = np.minimum(T_true, tau)
    event = (T_true <= tau).astype(int)

    # Correct 90% interval
    lower_correct = np.full(n, weibull_min.ppf(0.05, c=2, scale=5))
    upper_correct = np.full(n, weibull_min.ppf(0.95, c=2, scale=5))

    # Very wide interval (technically better coverage but similar width penalty)
    lower_wide = np.full(n, 0.0)
    upper_wide = np.full(n, tau * 5)

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    s_correct = ev.interval_score(lower_correct, upper_correct, T_obs, event, alpha=0.1)
    s_wide = ev.interval_score(lower_wide, upper_wide, T_obs, event, alpha=0.1)

    # Wide interval should score worse (larger) due to large width penalty
    assert s_correct < s_wide, (
        f"Correct interval ({s_correct:.4f}) should score better than wide ({s_wide:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 7: from_matrix helper
# ---------------------------------------------------------------------------

def test_from_matrix_roundtrip():
    """from_matrix produces callables that match the original matrix values."""
    n, n_times = 20, 50
    t_grid = np.linspace(0.0, 10.0, n_times)
    # Create survival matrix from Weibull
    S_matrix = weibull_min.sf(t_grid[None, :], c=2, scale=5) * np.ones((n, 1))

    surv_fns = from_matrix(S_matrix, t_grid)
    assert len(surv_fns) == n

    # Check that a callable returns correct values at grid points
    t_test = np.array([1.0, 3.0, 5.0, 8.0])
    expected = weibull_min.sf(t_test, c=2, scale=5)
    result = surv_fns[0](t_test)

    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_from_matrix_shape_error():
    """from_matrix raises ValueError on shape mismatch."""
    S_matrix = np.ones((10, 30))
    t_grid = np.linspace(0, 10, 20)  # wrong length

    with pytest.raises(ValueError, match="must equal len"):
        from_matrix(S_matrix, t_grid)


# ---------------------------------------------------------------------------
# Test 8: compare() method
# ---------------------------------------------------------------------------

def test_compare_returns_sorted_dataframe(weibull_data):
    """compare() returns a DataFrame sorted by twCRPS ascending."""
    T_obs, event, tau = weibull_data
    n = len(T_obs)

    true_fns = [make_surv_weibull(2, 5)] * n
    wrong_fns = [make_surv_expo(5)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    result = ev.compare({
        "Expo(5)": (wrong_fns, T_obs, event),
        "Weibull(2,5)": (true_fns, T_obs, event),
    })

    assert "twCRPS" in result.columns
    assert "quantile_score" in result.columns
    assert result.index[0] == "Weibull(2,5)", (
        "Weibull(2,5) should rank first (lower twCRPS)"
    )
    # Scores should be sorted ascending
    scores = result["twCRPS"].values
    assert scores[0] <= scores[1]


# ---------------------------------------------------------------------------
# Test 9: UserWarning is raised by default
# ---------------------------------------------------------------------------

def test_provisional_warning_emitted():
    """CensoredForecastEvaluator should warn about provisional propriety."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CensoredForecastEvaluator(tau=5.0)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "provisional" in str(w[0].message).lower()


def test_warn_false_suppresses_warning():
    """warn=False should suppress the provisional propriety warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CensoredForecastEvaluator(tau=5.0, warn=False)
        assert len(w) == 0


# ---------------------------------------------------------------------------
# Test 10: twcrps_profile returns correct shape and is non-decreasing
# ---------------------------------------------------------------------------

def test_twcrps_profile_shape_and_monotonicity(weibull_data):
    """twcrps_profile should return Series of correct length and be non-decreasing."""
    T_obs, event, tau = weibull_data
    n = len(T_obs)
    surv_fns = [make_surv_weibull(2, 5)] * n

    ev = CensoredForecastEvaluator(tau=tau, warn=False)
    thresholds = np.linspace(0.0, tau, 20)
    profile = ev.twcrps_profile(surv_fns, T_obs, event, thresholds=thresholds)

    assert len(profile) == 20
    # twCRPS(tau') should be non-decreasing as tau' increases
    # (more horizon = more room for error to accumulate)
    diffs = np.diff(profile.values)
    assert np.all(diffs >= -1e-6), "twCRPS profile should be non-decreasing"
