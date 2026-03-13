"""Integration tests: end-to-end workflows across model types.

These tests validate the full pipeline that a pricing analyst would run:
simulate -> check follow-up -> fit -> score -> validate.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure import (
    LogNormalMixtureCure,
    PromotionTimeCure,
    WeibullMixtureCure,
)
from insurance_survival.cure.diagnostics import (
    CureScorecard,
    cure_fraction_distribution,
    sufficient_followup_test,
)
from insurance_survival.cure.simulate import simulate_motor_panel


class TestFullWorkflow:
    """Full pipeline from simulate to scorecard."""

    def test_weibull_pipeline(self):
        df = simulate_motor_panel(n_policies=400, cure_fraction=0.38, seed=123)

        # Step 1: sufficient follow-up
        qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
        assert isinstance(qn.qn_statistic, float)

        # Step 2: fit
        model = WeibullMixtureCure(
            incidence_formula="ncb_years + age + vehicle_age",
            latency_formula="ncb_years + age",
            n_em_starts=2,
            max_iter=30,
            random_state=42,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        assert model._fitted

        # Step 3: predict
        cure = model.predict_cure_fraction(df)
        assert cure.shape == (len(df),)
        assert np.all(cure >= 0) and np.all(cure <= 1)

        # Step 4: population survival
        pop_surv = model.predict_population_survival(df, [12, 24, 36, 60])
        assert pop_surv.shape == (len(df), 4)

        # Step 5: scorecard
        sc = CureScorecard(model, bins=5)
        sc.fit(df, duration_col="tenure_months", event_col="claimed")
        assert sc.table_ is not None

        # Step 6: distribution
        dist = cure_fraction_distribution(cure)
        assert dist["mean"] > 0

    def test_lognormal_pipeline(self):
        df = simulate_motor_panel(n_policies=400, cure_fraction=0.35, seed=200)

        model = LogNormalMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=2,
            max_iter=30,
            random_state=42,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        cure = model.predict_cure_fraction(df)
        assert np.all(cure >= 0) and np.all(cure <= 1)

    def test_promotion_time_pipeline(self):
        df = simulate_motor_panel(n_policies=400, cure_fraction=0.40, seed=300)

        model = PromotionTimeCure(
            formula="ncb_years + age + vehicle_age",
            random_state=42,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        cure = model.predict_cure_fraction(df)
        assert np.all(cure >= 0) and np.all(cure <= 1)


class TestCureFractionRecovery:
    """Validate that the model approximately recovers the true cure fraction."""

    def test_weibull_recovers_cure_fraction(self):
        """With enough data, Weibull MCM cure fraction should be close to true."""
        true_cf = 0.40
        df = simulate_motor_panel(n_policies=2000, cure_fraction=true_cf, seed=999)

        model = WeibullMixtureCure(
            incidence_formula="ncb_years + age + vehicle_age",
            latency_formula="ncb_years + age",
            n_em_starts=3,
            max_iter=100,
            random_state=42,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        estimated_cf = model.result_.cure_fraction_mean

        # Tolerance: within 15 percentage points is reasonable for n=2000
        assert abs(estimated_cf - true_cf) < 0.15, (
            f"Estimated cure fraction {estimated_cf:.3f} far from "
            f"true {true_cf:.3f}"
        )

    def test_weibull_high_ncb_higher_cure(self):
        """Higher NCB policies should score higher cure probability."""
        df = simulate_motor_panel(n_policies=1500, cure_fraction=0.40, seed=77)

        model = WeibullMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=2,
            max_iter=60,
            random_state=42,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        cure = model.predict_cure_fraction(df)

        low_ncb_cure = cure[df["ncb_years"] <= 1].mean()
        high_ncb_cure = cure[df["ncb_years"] >= 8].mean()
        assert high_ncb_cure > low_ncb_cure


class TestModelComparison:
    """Compare Weibull and log-normal on the same dataset."""

    def test_both_models_fit_and_predict(self):
        df = simulate_motor_panel(n_policies=400, cure_fraction=0.38, seed=55)

        weibull = WeibullMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=30,
            random_state=0,
        )
        weibull.fit(df, duration_col="tenure_months", event_col="claimed")

        lognormal = LogNormalMixtureCure(
            incidence_formula="ncb_years + age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=30,
            random_state=0,
        )
        lognormal.fit(df, duration_col="tenure_months", event_col="claimed")

        w_cure = weibull.predict_cure_fraction(df)
        l_cure = lognormal.predict_cure_fraction(df)

        # Both should be valid
        assert np.all(w_cure >= 0) and np.all(w_cure <= 1)
        assert np.all(l_cure >= 0) and np.all(l_cure <= 1)

        # Log-likelihoods are finite
        assert np.isfinite(weibull.result_.log_likelihood)
        assert np.isfinite(lognormal.result_.log_likelihood)


class TestEdgeCases:
    def test_single_covariate(self):
        df = simulate_motor_panel(n_policies=200, seed=1)
        model = WeibullMixtureCure(
            incidence_formula="ncb_years",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            random_state=1,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        assert model._fitted

    def test_different_incidence_latency_formulas(self):
        df = simulate_motor_panel(n_policies=300, seed=2)
        model = WeibullMixtureCure(
            incidence_formula="ncb_years + age + vehicle_age",
            latency_formula="ncb_years",
            n_em_starts=1,
            max_iter=20,
            random_state=2,
        )
        model.fit(df, duration_col="tenure_months", event_col="claimed")
        # Incidence has 3 covariates, latency has 1
        assert len(model.result_.incidence_coef) == 3
        assert "beta_ncb_years" in model.result_.latency_params

    def test_predict_on_subset(self, motor_df, fitted_weibull):
        """Predict on a subset of the original data."""
        subset = motor_df.iloc[:50].reset_index(drop=True)
        cure = fitted_weibull.predict_cure_fraction(subset)
        assert cure.shape == (50,)

    def test_predict_on_new_data(self, fitted_weibull):
        """Predict on unseen data (out-of-sample)."""
        new_df = simulate_motor_panel(n_policies=100, seed=888)
        cure = fitted_weibull.predict_cure_fraction(new_df)
        assert cure.shape == (100,)
        assert np.all(cure >= 0) and np.all(cure <= 1)
