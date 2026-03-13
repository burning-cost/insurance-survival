"""Tests for CoxMixtureCure.

The Cox latency uses lifelines CoxPHFitter which is much slower than
parametric alternatives. Tests use tiny datasets and very few iterations
to keep runtime acceptable.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure import CoxMixtureCure


@pytest.fixture(scope="module")
def small_motor_df():
    from insurance_survival.cure.simulate import simulate_motor_panel
    return simulate_motor_panel(n_policies=150, cure_fraction=0.35, seed=42)


@pytest.fixture(scope="module")
def fitted_cox(small_motor_df):
    m = CoxMixtureCure(
        incidence_formula="ncb_years + age",
        latency_formula="ncb_years",
        n_em_starts=1,
        max_iter=5,
        random_state=42,
    )
    m.fit(small_motor_df, duration_col="tenure_months", event_col="claimed")
    return m


class TestCoxMixtureCureFit:
    def test_fit_returns_self(self, small_motor_df):
        m = CoxMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=3,
            random_state=1,
        )
        result = m.fit(small_motor_df, duration_col="tenure_months", event_col="claimed")
        assert result is m

    def test_fitted_flag(self, fitted_cox):
        assert fitted_cox._fitted

    def test_result_n_obs(self, small_motor_df, fitted_cox):
        assert fitted_cox.result_.n_obs == len(small_motor_df)

    def test_cure_fraction_in_range(self, fitted_cox):
        cf = fitted_cox.result_.cure_fraction_mean
        assert 0.0 <= cf <= 1.0

    def test_incidence_coef_keys(self, fitted_cox):
        coef = fitted_cox.result_.incidence_coef
        assert "ncb_years" in coef
        assert "age" in coef


class TestCoxMixtureCurePrediction:
    def test_cure_fraction_shape(self, small_motor_df, fitted_cox):
        cure = fitted_cox.predict_cure_fraction(small_motor_df)
        assert cure.shape == (len(small_motor_df),)

    def test_cure_fraction_range(self, small_motor_df, fitted_cox):
        cure = fitted_cox.predict_cure_fraction(small_motor_df)
        assert np.all(cure >= 0) and np.all(cure <= 1)

    def test_susceptibility_complementary(self, small_motor_df, fitted_cox):
        cure = fitted_cox.predict_cure_fraction(small_motor_df)
        suscept = fitted_cox.predict_susceptibility(small_motor_df)
        assert np.allclose(cure + suscept, 1.0)

    def test_population_survival_shape(self, small_motor_df, fitted_cox):
        result = fitted_cox.predict_population_survival(small_motor_df, [12, 24])
        assert result.shape == (len(small_motor_df), 2)

    def test_population_survival_range(self, small_motor_df, fitted_cox):
        result = fitted_cox.predict_population_survival(small_motor_df, [12])
        assert (result.values >= 0).all() and (result.values <= 1).all()

    def test_repr(self, fitted_cox):
        r = repr(fitted_cox)
        assert "CoxMixtureCure" in r
