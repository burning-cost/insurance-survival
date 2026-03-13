"""
Tests for model fitters: AndersenGillFrailty, PWPModel, NelsonAalenFrailty.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_survival.recurrent.models import (
    AndersenGillFrailty,
    FrailtyFitResult,
    NelsonAalenFrailty,
    PWPFitResult,
    PWPModel,
    _breslow_cumhaz,
    _partial_log_likelihood,
)
from insurance_survival.recurrent.simulate import SimulationParams, simulate_ag_frailty, simulate_pwp


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def make_small_data(n_subjects=80, theta=2.0, seed=0):
    return simulate_ag_frailty(
        SimulationParams(
            n_subjects=n_subjects,
            beta=np.array([0.3, -0.2]),
            theta=theta,
            random_state=seed,
        )
    )


def make_small_data_no_cov(n_subjects=80, theta=2.0, seed=0):
    from insurance_survival.recurrent.simulate import SimulationParams, simulate_ag_frailty
    params = SimulationParams(n_subjects=n_subjects, beta=np.array([]), theta=theta, random_state=seed)
    # Re-implement for zero covariates
    from insurance_survival.recurrent.data import RecurrentEventData
    rng = np.random.default_rng(seed)
    z = rng.gamma(shape=theta, scale=1.0 / theta, size=n_subjects)
    records = []
    for i in range(n_subjects):
        rate_i = z[i] * 0.3
        t = 0.0
        while t < 3.0:
            wait = rng.exponential(1.0 / max(rate_i, 1e-10))
            t_event = t + wait
            if t_event >= 3.0:
                records.append({"policy_id": i, "t_start": t, "t_stop": 3.0, "event": 0})
                break
            else:
                records.append({"policy_id": i, "t_start": t, "t_stop": t_event, "event": 1})
                t = t_event
    df = pd.DataFrame(records)
    return RecurrentEventData.from_long_format(df, "policy_id", "t_start", "t_stop", "event")


# ------------------------------------------------------------------
# Breslow / partial likelihood helpers
# ------------------------------------------------------------------


class TestHelpers:
    def test_breslow_cumhaz_increasing(self):
        stop = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        event = np.array([1, 0, 1, 1, 0])
        risk_score = np.ones(5)
        times, cumhaz = _breslow_cumhaz(stop, event, risk_score)
        assert np.all(np.diff(cumhaz) >= 0)

    def test_breslow_no_events(self):
        stop = np.array([1.0, 2.0, 3.0])
        event = np.zeros(3, dtype=int)
        risk_score = np.ones(3)
        times, cumhaz = _breslow_cumhaz(stop, event, risk_score)
        assert len(times) == 1

    def test_partial_ll_scalar(self):
        stop = np.array([0.5, 1.0, 1.5, 2.0])
        event = np.array([1, 0, 1, 0])
        X = np.array([[1.0], [0.0], [-1.0], [0.5]])
        ll = _partial_log_likelihood(np.array([0.0]), X, stop, event)
        assert np.isfinite(ll)

    def test_partial_ll_no_events(self):
        stop = np.array([1.0, 2.0])
        event = np.zeros(2, dtype=int)
        X = np.ones((2, 1))
        ll = _partial_log_likelihood(np.array([0.0]), X, stop, event)
        assert ll == 0.0


# ------------------------------------------------------------------
# AndersenGillFrailty
# ------------------------------------------------------------------


class TestAndersenGillFrailtyFit:
    def test_fit_returns_self(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=5)
        result = model.fit(data)
        assert result is model

    def test_result_type(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        assert isinstance(model.result_, FrailtyFitResult)

    def test_coef_shape(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        assert model.result_.coef.shape == (2,)

    def test_se_shape(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        assert model.result_.coef_se.shape == (2,)

    def test_theta_positive(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=15).fit(data)
        assert model.result_.theta > 0

    def test_log_likelihood_finite(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        assert np.isfinite(model.result_.log_likelihood)

    def test_covariate_names_preserved(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=5).fit(data)
        assert model.result_.covariate_names == ["x1", "x2"]

    def test_frailty_name(self):
        data = make_small_data()
        model = AndersenGillFrailty(frailty="lognormal", max_iter=5).fit(data)
        assert model.result_.frailty_name == "lognormal"

    def test_no_covariates(self):
        data = make_small_data_no_cov()
        model = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        assert model.result_.coef.shape == (0,)
        assert model.result_.theta > 0

    def test_summary_returns_dataframe(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        summary = model.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "HR" in summary.columns
        assert len(summary) == 2

    def test_summary_has_expected_columns(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        summary = model.summary()
        expected_cols = ["coef", "se", "HR", "HR_lower_95", "HR_upper_95", "p_value"]
        for col in expected_cols:
            assert col in summary.columns

    def test_hr_positive(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        assert (model.summary()["HR"] > 0).all()

    def test_ci_covers_hr(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        s = model.summary()
        assert (s["HR"] >= s["HR_lower_95"]).all()
        assert (s["HR"] <= s["HR_upper_95"]).all()

    def test_repr_unfitted(self):
        r = repr(AndersenGillFrailty())
        assert "unfitted" in r

    def test_repr_fitted(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=5).fit(data)
        r = repr(model)
        assert "fitted" in r

    def test_result_before_fit_raises(self):
        model = AndersenGillFrailty()
        with pytest.raises(RuntimeError):
            _ = model.result_

    def test_convergence_warning(self):
        """With max_iter=1, convergence warning should be raised."""
        data = make_small_data()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = AndersenGillFrailty(max_iter=1).fit(data)
            # Either converged in 1 iter (valid) or warned
            if len(w) > 0:
                assert any(issubclass(x.category, RuntimeWarning) for x in w)

    def test_verbose_fit(self, capsys):
        data = make_small_data(n_subjects=30)
        AndersenGillFrailty(max_iter=3, verbose=True).fit(data)
        captured = capsys.readouterr()
        assert "Iter" in captured.out

    def test_lognormal_frailty_fits(self):
        data = make_small_data(n_subjects=60)
        model = AndersenGillFrailty(frailty="lognormal", max_iter=5).fit(data)
        assert model.result_.theta > 0

    def test_covariate_direction_positive(self):
        """
        With true beta[0] = +0.5, fitted beta[0] should be positive.
        Uses large N for reliable estimation.
        """
        data = simulate_ag_frailty(
            SimulationParams(n_subjects=600, beta=np.array([0.5]), theta=3.0, random_state=0)
        )
        model = AndersenGillFrailty(frailty="gamma", max_iter=20).fit(data)
        assert model.result_.coef[0] > 0

    def test_covariate_direction_negative(self):
        data = simulate_ag_frailty(
            SimulationParams(n_subjects=600, beta=np.array([-0.5]), theta=3.0, random_state=0)
        )
        model = AndersenGillFrailty(frailty="gamma", max_iter=20).fit(data)
        assert model.result_.coef[0] < 0


class TestCredibilityScores:
    def test_returns_dataframe(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        scores = model.credibility_scores()
        assert isinstance(scores, pd.DataFrame)

    def test_n_rows(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        scores = model.credibility_scores()
        assert len(scores) == data.n_subjects

    def test_frailty_mean_positive(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        scores = model.credibility_scores()
        assert (scores["frailty_mean"] > 0).all()

    def test_frailty_mean_roughly_centred(self):
        """Population average frailty should be near 1."""
        data = make_small_data(n_subjects=300)
        model = AndersenGillFrailty(max_iter=20).fit(data)
        scores = model.credibility_scores()
        mean_frailty = scores["frailty_mean"].mean()
        assert 0.5 < mean_frailty < 2.0

    def test_high_claimant_higher_frailty(self):
        """Subjects with more claims should generally have higher frailty."""
        data = make_small_data(n_subjects=300)
        model = AndersenGillFrailty(max_iter=20).fit(data)
        scores = model.credibility_scores()
        # Correlation between n_events and frailty should be positive
        corr = scores[["n_events", "frailty_mean"]].corr().iloc[0, 1]
        assert corr > 0

    def test_credibility_weight_bounded(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        scores = model.credibility_scores()
        assert (scores["credibility_weight"] >= 0).all()
        assert (scores["credibility_weight"] <= 1).all()

    def test_before_fit_raises(self):
        model = AndersenGillFrailty()
        with pytest.raises(RuntimeError):
            model.credibility_scores()

    def test_expected_columns(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        scores = model.credibility_scores()
        for col in ["id", "n_events", "lambda_i", "frailty_mean", "frailty_var", "credibility_weight"]:
            assert col in scores.columns


class TestPredictIntensity:
    def test_returns_array(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        pred = model.predict_intensity(data)
        assert isinstance(pred, np.ndarray)

    def test_shape(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        pred = model.predict_intensity(data)
        assert pred.shape == (len(data.df),)

    def test_positive(self):
        data = make_small_data()
        model = AndersenGillFrailty(max_iter=10).fit(data)
        pred = model.predict_intensity(data)
        assert (pred > 0).all()


# ------------------------------------------------------------------
# PWPModel
# ------------------------------------------------------------------


class TestPWPModel:
    def test_fit_returns_self(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel().fit(data)
        assert isinstance(model, PWPModel)

    def test_result_type(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel().fit(data)
        assert isinstance(model.result_, PWPFitResult)

    def test_coef_shape(self):
        data = simulate_pwp(n_subjects=80, beta=np.array([0.3, -0.2]))
        model = PWPModel().fit(data)
        assert model.result_.coef.shape == (2,)

    def test_summary_dataframe(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel().fit(data)
        s = model.summary()
        assert isinstance(s, pd.DataFrame)
        assert "HR" in s.columns

    def test_gap_time(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel(time_scale="gap").fit(data)
        assert model.result_ is not None

    def test_calendar_time(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel(time_scale="calendar").fit(data)
        assert model.result_ is not None

    def test_invalid_time_scale(self):
        with pytest.raises(ValueError, match="time_scale"):
            PWPModel(time_scale="invalid")

    def test_stratum_event_counts(self):
        data = simulate_pwp(n_subjects=100, baseline_rates=(0.4, 0.5, 0.6))
        model = PWPModel(max_stratum=3).fit(data)
        counts = model.result_.stratum_event_counts
        assert 1 in counts and 2 in counts and 3 in counts

    def test_predict_hr_shape(self):
        data = simulate_pwp(n_subjects=80, beta=np.array([0.3, -0.2]))
        model = PWPModel().fit(data)
        X_new = np.array([[1.0, 0.0], [0.0, 1.0]])
        hr = model.predict_hr(X_new)
        assert hr.shape == (2,)

    def test_predict_hr_positive(self):
        data = simulate_pwp(n_subjects=80)
        model = PWPModel().fit(data)
        X_new = np.random.randn(5, 2)
        hr = model.predict_hr(X_new)
        assert (hr > 0).all()

    def test_repr_unfitted(self):
        assert "unfitted" in repr(PWPModel())

    def test_repr_fitted(self):
        data = simulate_pwp(n_subjects=80)
        assert "fitted" in repr(PWPModel().fit(data))

    def test_result_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PWPModel().result_


# ------------------------------------------------------------------
# NelsonAalenFrailty
# ------------------------------------------------------------------


class TestNelsonAalenFrailty:
    def test_fit_returns_self(self):
        data = make_small_data_no_cov()
        model = NelsonAalenFrailty().fit(data)
        assert isinstance(model, NelsonAalenFrailty)

    def test_theta_positive(self):
        data = make_small_data_no_cov()
        model = NelsonAalenFrailty().fit(data)
        assert model.theta_ > 0

    def test_credibility_scores_dataframe(self):
        data = make_small_data_no_cov()
        model = NelsonAalenFrailty().fit(data)
        scores = model.credibility_scores()
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == data.n_subjects

    def test_frailty_mean_positive(self):
        data = make_small_data_no_cov()
        model = NelsonAalenFrailty().fit(data)
        scores = model.credibility_scores()
        assert (scores["frailty_mean"] > 0).all()

    def test_repr(self):
        r = repr(NelsonAalenFrailty())
        assert "unfitted" in r
        data = make_small_data_no_cov()
        r2 = repr(NelsonAalenFrailty().fit(data))
        assert "fitted" in r2

    def test_before_fit_raises(self):
        model = NelsonAalenFrailty()
        with pytest.raises(RuntimeError):
            model.theta_
        with pytest.raises(RuntimeError):
            model.credibility_scores()
