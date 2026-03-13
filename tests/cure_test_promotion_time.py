"""Tests for PromotionTimeCure."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure import PromotionTimeCure


class TestPromotionTimeCureFit:
    def test_fit_returns_self(self, motor_df):
        m = PromotionTimeCure(formula="ncb_years + age", random_state=0)
        result = m.fit(motor_df, duration_col="tenure_months", event_col="claimed")
        assert result is m

    def test_fitted_flag(self, fitted_promotion):
        assert fitted_promotion._fitted

    def test_result_dict_keys(self, fitted_promotion):
        r = fitted_promotion.result_
        assert "log_likelihood" in r
        assert "cure_fraction_mean" in r
        assert "n_obs" in r
        assert "n_events" in r

    def test_cure_fraction_mean_range(self, fitted_promotion):
        cf = fitted_promotion.result_["cure_fraction_mean"]
        assert 0.0 < cf < 1.0

    def test_loglik_finite(self, fitted_promotion):
        assert np.isfinite(fitted_promotion.result_["log_likelihood"])

    def test_rho_positive(self, fitted_promotion):
        assert fitted_promotion.result_["rho"] > 0

    def test_scale_positive(self, fitted_promotion):
        assert fitted_promotion.result_["scale"] > 0

    def test_bad_duration_raises(self, motor_df):
        df_bad = motor_df.copy()
        df_bad["tenure_months"] = -1.0
        m = PromotionTimeCure(formula="ncb_years")
        with pytest.raises(ValueError, match="non-positive"):
            m.fit(df_bad, duration_col="tenure_months", event_col="claimed")

    def test_repr_unfitted(self):
        m = PromotionTimeCure(formula="ncb_years")
        r = repr(m)
        assert "unfitted" in r

    def test_repr_fitted(self, fitted_promotion):
        r = repr(fitted_promotion)
        assert "fitted" in r


class TestPromotionTimePrediction:
    def test_cure_fraction_shape(self, motor_df, fitted_promotion):
        cure = fitted_promotion.predict_cure_fraction(motor_df)
        assert cure.shape == (len(motor_df),)

    def test_cure_fraction_range(self, motor_df, fitted_promotion):
        cure = fitted_promotion.predict_cure_fraction(motor_df)
        assert np.all(cure >= 0) and np.all(cure <= 1)

    def test_susceptibility_range(self, motor_df, fitted_promotion):
        s = fitted_promotion.predict_susceptibility(motor_df)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_cure_plus_suscept_one(self, motor_df, fitted_promotion):
        cure = fitted_promotion.predict_cure_fraction(motor_df)
        suscept = fitted_promotion.predict_susceptibility(motor_df)
        assert np.allclose(cure + suscept, 1.0)

    def test_population_survival_shape(self, motor_df, fitted_promotion):
        result = fitted_promotion.predict_population_survival(motor_df, [12, 24, 36])
        assert result.shape == (len(motor_df), 3)

    def test_population_survival_range(self, motor_df, fitted_promotion):
        result = fitted_promotion.predict_population_survival(motor_df, [12, 24])
        assert (result.values >= 0).all() and (result.values <= 1).all()

    def test_population_survival_at_zero_near_one(self, motor_df, fitted_promotion):
        """S_pop(t->0) should be close to 1."""
        result = fitted_promotion.predict_population_survival(motor_df, [0.01])
        assert np.all(result.values > 0.9)

    def test_population_survival_decreasing(self, motor_df, fitted_promotion):
        result = fitted_promotion.predict_population_survival(motor_df, [6, 24, 60])
        assert (result[6] >= result[24]).all()
        assert (result[24] >= result[60]).all()

    def test_predict_before_fit_raises(self, motor_df):
        m = PromotionTimeCure(formula="ncb_years")
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.predict_cure_fraction(motor_df)
