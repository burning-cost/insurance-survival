"""Tests for LogNormalMixtureCure."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure import LogNormalMixtureCure


class TestLogNormalMixtureCureFit:
    def test_fit_returns_self(self, motor_df):
        m = LogNormalMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            random_state=1,
        )
        result = m.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert result is m

    def test_fitted_flag(self, fitted_lognormal):
        assert fitted_lognormal._fitted

    def test_result_n_obs(self, motor_df, fitted_lognormal):
        assert fitted_lognormal.result_.n_obs == len(motor_df)

    def test_latency_params_have_mu(self, fitted_lognormal):
        assert "mu" in fitted_lognormal.result_.latency_params

    def test_latency_params_have_log_sigma(self, fitted_lognormal):
        assert "log_sigma" in fitted_lognormal.result_.latency_params

    def test_cure_fraction_in_range(self, fitted_lognormal):
        cf = fitted_lognormal.result_.cure_fraction_mean
        assert 0.0 < cf < 1.0

    def test_loglikelihood_finite(self, fitted_lognormal):
        assert np.isfinite(fitted_lognormal.result_.log_likelihood)

    def test_summary_contains_model_name(self, fitted_lognormal):
        s = fitted_lognormal.result_.summary()
        assert "Mixture Cure Model" in s


class TestLogNormalPrediction:
    def test_predict_cure_fraction_shape(self, motor_df, fitted_lognormal):
        scores = fitted_lognormal.predict_cure_fraction(motor_df)
        assert scores.shape == (len(motor_df),)

    def test_predict_cure_fraction_range(self, motor_df, fitted_lognormal):
        scores = fitted_lognormal.predict_cure_fraction(motor_df)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_cure_plus_suscept_one(self, motor_df, fitted_lognormal):
        cure = fitted_lognormal.predict_cure_fraction(motor_df)
        suscept = fitted_lognormal.predict_susceptibility(motor_df)
        assert np.allclose(cure + suscept, 1.0)

    def test_population_survival_shape(self, motor_df, fitted_lognormal):
        result = fitted_lognormal.predict_population_survival(motor_df, [12, 24])
        assert result.shape == (len(motor_df), 2)

    def test_population_survival_range(self, motor_df, fitted_lognormal):
        result = fitted_lognormal.predict_population_survival(motor_df, [12, 24])
        assert (result.values >= 0).all() and (result.values <= 1).all()

    def test_population_survival_decreasing(self, motor_df, fitted_lognormal):
        result = fitted_lognormal.predict_population_survival(motor_df, [6, 24, 48])
        assert (result[6] >= result[24]).all()
        assert (result[24] >= result[48]).all()

    def test_susceptible_survival_shape(self, motor_df, fitted_lognormal):
        result = fitted_lognormal.predict_susceptible_survival(motor_df, [12])
        assert result.shape == (len(motor_df), 1)

    def test_repr(self, fitted_lognormal):
        r = repr(fitted_lognormal)
        assert "LogNormalMixtureCure" in r
        assert "fitted" in r

    def test_pet_data_fits(self, pet_df):
        m = LogNormalMixtureCure(
            incidence_formula="pet_age + breed_risk",
            latency_formula="pet_age",
            n_em_starts=1,
            max_iter=30,
            random_state=99,
        )
        m.fit(pet_df, duration_col="tenure_months", event_col="claimed")
        assert m._fitted
        cure = m.predict_cure_fraction(pet_df)
        assert np.all(cure >= 0) and np.all(cure <= 1)
