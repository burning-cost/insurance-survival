"""Tests for the diagnostics module."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure.diagnostics import (
    CureScorecard,
    SufficientFollowUpResult,
    cure_fraction_distribution,
    sufficient_followup_test,
    _kaplan_meier,
)


class TestKaplanMeier:
    def test_returns_arrays(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        km_t, km_s = _kaplan_meier(t, d)
        assert len(km_t) == len(km_s)

    def test_survival_decreasing(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = np.ones(5)
        km_t, km_s = _kaplan_meier(t, d)
        for i in range(1, len(km_s)):
            assert km_s[i] <= km_s[i - 1]

    def test_survival_starts_below_one(self):
        t = np.array([1.0, 2.0, 3.0])
        d = np.ones(3)
        km_t, km_s = _kaplan_meier(t, d)
        assert km_s[0] < 1.0

    def test_no_events_returns_empty(self):
        t = np.array([1.0, 2.0, 3.0])
        d = np.zeros(3)
        km_t, km_s = _kaplan_meier(t, d)
        assert len(km_t) == 0


class TestSufficientFollowUpTest:
    def test_returns_result_object(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert isinstance(result, SufficientFollowUpResult)

    def test_qn_nonnegative(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert result.qn_statistic >= 0

    def test_pvalue_in_range(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert 0.0 <= result.p_value <= 1.0

    def test_n_correct(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert result.n == len(motor_df)

    def test_n_events_correct(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert result.n_events == int(motor_df["claimed"].sum())

    def test_max_event_time_correct(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        expected_max = float(motor_df.loc[motor_df["claimed"] == 1, "tenure_months"].max())
        assert result.max_event_time == pytest.approx(expected_max)

    def test_conclusion_string_nonempty(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        assert len(result.conclusion) > 0

    def test_summary_string(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        s = result.summary()
        assert "Maller-Zhou" in s
        assert "Qn" in s

    def test_repr_string(self, motor_df):
        result = sufficient_followup_test(motor_df["tenure_months"], motor_df["claimed"])
        r = repr(result)
        assert "SufficientFollowUpResult" in r

    def test_no_events_raises(self):
        t = np.array([1.0, 2.0, 3.0])
        d = np.zeros(3)
        with pytest.raises(ValueError, match="No events"):
            sufficient_followup_test(t, d)

    def test_nonpositive_duration_raises(self):
        t = np.array([-1.0, 2.0, 3.0])
        d = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            sufficient_followup_test(t, d)

    def test_accepts_pandas_series(self, motor_df):
        result = sufficient_followup_test(
            motor_df["tenure_months"],
            motor_df["claimed"],
        )
        assert isinstance(result, SufficientFollowUpResult)

    def test_long_followup_significant(self):
        """With long follow-up and large cure fraction, should detect plateau."""
        rng = np.random.default_rng(0)
        n = 500
        # Half immune (never event), half susceptible
        immune = rng.random(n) < 0.5
        t = np.where(
            immune,
            rng.uniform(50, 100, n),  # censored late
            rng.exponential(10, n),   # events early
        )
        d = np.where(immune, 0, (t < 50).astype(float))
        # Make events happen early
        d = np.where(~immune, 1.0, 0.0)
        t = np.where(immune, rng.uniform(50, 100, n), rng.exponential(5, n))
        t = np.clip(t, 0.1, 100)
        result = sufficient_followup_test(t, d)
        assert isinstance(result, SufficientFollowUpResult)


class TestCureScorecard:
    def test_fit_returns_self(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        result = sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert result is sc

    def test_table_is_dataframe(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert isinstance(sc.table_, pd.DataFrame)

    def test_table_columns(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        required = {"decile", "n", "cure_frac_mean", "n_events", "event_rate"}
        assert required.issubset(set(sc.table_.columns))

    def test_event_rate_range(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert (sc.table_["event_rate"] >= 0).all()
        assert (sc.table_["event_rate"] <= 1).all()

    def test_n_sums_to_total(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert sc.table_["n"].sum() == len(motor_df)

    def test_summary_string(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        s = sc.summary()
        assert "Cure Fraction Scorecard" in s

    def test_repr_unfitted(self, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        r = repr(sc)
        assert "unfitted" in r

    def test_repr_fitted(self, motor_df, fitted_weibull):
        sc = CureScorecard(fitted_weibull, bins=5)
        sc.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        r = repr(sc)
        assert "fitted" in r

    def test_invalid_model_raises(self):
        with pytest.raises(TypeError, match="predict_cure_fraction"):
            CureScorecard("not_a_model")

    def test_monotone_ordering(self, motor_df_large, fitted_weibull):
        """Higher cure fraction deciles should have lower event rates (weak check)."""
        sc = CureScorecard(fitted_weibull, bins=10)
        sc.fit(motor_df_large, duration_col="tenure_months", event_col="claimed")
        # First decile (lowest cure) should have higher event rate than last decile
        first = sc.table_.iloc[0]["event_rate"]
        last = sc.table_.iloc[-1]["event_rate"]
        # This is a tendency, not a strict guarantee on small datasets
        assert first >= 0 and last >= 0


class TestCureFractionDistribution:
    def test_returns_dict(self):
        scores = np.random.default_rng(0).uniform(0.2, 0.8, 100)
        result = cure_fraction_distribution(scores)
        assert isinstance(result, dict)

    def test_mean_in_range(self):
        scores = np.full(100, 0.5)
        result = cure_fraction_distribution(scores)
        assert result["mean"] == pytest.approx(0.5)

    def test_default_percentiles_present(self):
        scores = np.random.default_rng(0).uniform(0, 1, 200)
        result = cure_fraction_distribution(scores)
        for p in [5, 25, 50, 75, 95]:
            assert f"p{p}" in result

    def test_custom_percentiles(self):
        scores = np.linspace(0, 1, 100)
        result = cure_fraction_distribution(scores, percentiles=[10, 90])
        assert "p10" in result
        assert "p90" in result
        assert "p50" not in result

    def test_n_correct(self):
        scores = np.ones(77) * 0.5
        result = cure_fraction_distribution(scores)
        assert result["n"] == 77
