"""Tests for WeibullMixtureCure."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure import WeibullMixtureCure
from insurance_survival.cure._base import MCMResult


class TestWeibullMixtureCureInit:
    def test_default_params(self):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
        )
        assert m.n_em_starts == 5
        assert m.max_iter == 200
        assert m.tol == 1e-5
        assert not m.bootstrap_se

    def test_repr_unfitted(self):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
        )
        r = repr(m)
        assert "unfitted" in r
        assert "WeibullMixtureCure" in r

    def test_n_em_starts_minimum_one(self):
        m = WeibullMixtureCure(
            incidence_formula="a",
            latency_formula="a",
            n_em_starts=0,
        )
        assert m.n_em_starts == 1


class TestWeibullMixtureCureFit:
    def test_fit_returns_self(self, motor_df):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            random_state=1,
        )
        result = m.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert result is m

    def test_result_is_mcmresult(self, fitted_weibull):
        assert isinstance(fitted_weibull.result_, MCMResult)

    def test_fitted_flag(self, fitted_weibull):
        assert fitted_weibull._fitted

    def test_result_n_obs(self, motor_df, fitted_weibull):
        assert fitted_weibull.result_.n_obs == len(motor_df)

    def test_result_n_events(self, motor_df, fitted_weibull):
        expected = int(motor_df["claimed"].sum())
        assert fitted_weibull.result_.n_events == expected

    def test_cure_fraction_mean_in_range(self, fitted_weibull):
        cf = fitted_weibull.result_.cure_fraction_mean
        assert 0.0 < cf < 1.0

    def test_loglikelihood_finite(self, fitted_weibull):
        assert np.isfinite(fitted_weibull.result_.log_likelihood)

    def test_incidence_coef_keys(self, fitted_weibull):
        coef = fitted_weibull.result_.incidence_coef
        assert "ncb_years" in coef
        assert "age" in coef
        assert "vehicle_age" in coef

    def test_latency_param_keys(self, fitted_weibull):
        params = fitted_weibull.result_.latency_params
        assert "log_lambda" in params
        assert "log_rho" in params

    def test_repr_after_fit(self, fitted_weibull):
        r = repr(fitted_weibull)
        assert "fitted" in r

    def test_summary_string(self, fitted_weibull):
        s = fitted_weibull.result_.summary()
        assert "Mixture Cure Model Results" in s
        assert "Log-likelihood" in s

    def test_bad_duration_raises(self, motor_df):
        df_bad = motor_df.copy()
        df_bad["tenure_months"] = df_bad["tenure_months"] * -1
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=5,
        )
        with pytest.raises(ValueError, match="non-positive"):
            m.fit(df_bad, duration_col="tenure_months", event_col="claimed")

    def test_bad_event_raises(self, motor_df):
        df_bad = motor_df.copy()
        df_bad["claimed"] = 2
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=5,
        )
        with pytest.raises(ValueError, match="0 and 1"):
            m.fit(df_bad, duration_col="tenure_months", event_col="claimed")

    def test_missing_column_raises(self, motor_df):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years + nonexistent_col",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=5,
        )
        with pytest.raises(ValueError, match="not found"):
            m.fit(motor_df, duration_col="tenure_months", event_col="claimed")


class TestWeibullPrediction:
    def test_predict_cure_fraction_shape(self, motor_df, fitted_weibull):
        scores = fitted_weibull.predict_cure_fraction(motor_df)
        assert scores.shape == (len(motor_df),)

    def test_predict_cure_fraction_range(self, motor_df, fitted_weibull):
        scores = fitted_weibull.predict_cure_fraction(motor_df)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_predict_susceptibility_shape(self, motor_df, fitted_weibull):
        s = fitted_weibull.predict_susceptibility(motor_df)
        assert s.shape == (len(motor_df),)

    def test_cure_plus_suscept_equals_one(self, motor_df, fitted_weibull):
        cure = fitted_weibull.predict_cure_fraction(motor_df)
        suscept = fitted_weibull.predict_susceptibility(motor_df)
        assert np.allclose(cure + suscept, 1.0)

    def test_predict_population_survival_shape(self, motor_df, fitted_weibull):
        times = [12, 24, 36]
        result = fitted_weibull.predict_population_survival(motor_df, times)
        assert result.shape == (len(motor_df), 3)

    def test_predict_population_survival_columns(self, motor_df, fitted_weibull):
        times = [12.0, 36.0]
        result = fitted_weibull.predict_population_survival(motor_df, times)
        assert 12.0 in result.columns
        assert 36.0 in result.columns

    def test_predict_population_survival_range(self, motor_df, fitted_weibull):
        times = [12, 24]
        result = fitted_weibull.predict_population_survival(motor_df, times)
        assert (result.values >= 0).all()
        assert (result.values <= 1).all()

    def test_population_survival_decreasing_in_time(self, motor_df, fitted_weibull):
        """S_pop(t1) >= S_pop(t2) when t1 < t2."""
        times = [12, 24, 48]
        result = fitted_weibull.predict_population_survival(motor_df, times)
        assert (result[12] >= result[24]).all()
        assert (result[24] >= result[48]).all()

    def test_predict_susceptible_survival_shape(self, motor_df, fitted_weibull):
        times = [12, 24]
        result = fitted_weibull.predict_susceptible_survival(motor_df, times)
        assert result.shape == (len(motor_df), 2)

    def test_pop_surv_geq_suscept_surv_fraction(self, motor_df, fitted_weibull):
        """S_pop(t) >= (1 - pi) since non-claimers contribute."""
        times = [12]
        pop = fitted_weibull.predict_population_survival(motor_df, times)[12].values
        cure = fitted_weibull.predict_cure_fraction(motor_df)
        # S_pop >= cure fraction (population survival >= cure floor)
        assert np.all(pop >= cure - 1e-8)

    def test_predict_before_fit_raises(self, motor_df):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
        )
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict_cure_fraction(motor_df)

    def test_high_ncb_higher_cure(self, fitted_weibull):
        """Policyholders with high NCB should score higher cure probability."""
        from insurance_survival.cure.simulate import simulate_motor_panel
        df = simulate_motor_panel(n_policies=200, seed=99)
        cure = fitted_weibull.predict_cure_fraction(df)
        low = cure[df["ncb_years"] <= 1].mean()
        high = cure[df["ncb_years"] >= 8].mean()
        assert high > low


class TestWeibullBootstrap:
    def test_bootstrap_se_populated(self, motor_df):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            bootstrap_se=True,
            n_bootstrap=5,
            n_jobs=1,
            random_state=42,
        )
        m.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert m.result_.se_incidence_coef is not None
        assert "ncb_years" in m.result_.se_incidence_coef

    def test_bootstrap_se_nonnegative(self, motor_df):
        m = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            bootstrap_se=True,
            n_bootstrap=5,
            n_jobs=1,
            random_state=42,
        )
        m.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        for se in m.result_.se_incidence_coef.values():
            assert se >= 0
