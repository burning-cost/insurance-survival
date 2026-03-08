"""
Tests for SurvivalCLV.

Validates:
- Output schema and shape
- CLV is finite and non-negative for positive margins
- Discount sensitivity direction is correct
- NCD path marginalisation runs without error
- Analytical CLV comparison on known Weibull model
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_survival import SurvivalCLV


class TestSurvivalCLVOutputSchema:
    """Output schema and basic shape tests."""

    def test_predict_returns_polars_dataframe(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert isinstance(result, pl.DataFrame)

    def test_predict_n_rows(self, fitted_cure_fitter, small_cure_dgp):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert len(result) == len(small_cure_dgp)

    def test_predict_has_clv_column(self, fitted_cure_fitter, small_cure_dgp):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert "clv" in result.columns

    def test_predict_has_survival_integral(self, fitted_cure_fitter, small_cure_dgp):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert "survival_integral" in result.columns

    def test_predict_has_cure_prob_for_cure_model(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert "cure_prob" in result.columns

    def test_predict_has_annual_survival_columns(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        horizon = 4
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=horizon)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        for t in range(1, horizon + 1):
            assert f"s_yr{t}" in result.columns


class TestSurvivalCLVValues:
    """Numerical correctness tests."""

    def test_clv_finite(self, fitted_cure_fitter, small_cure_dgp):
        """All CLV values are finite."""
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        clv_vals = result["clv"].to_numpy()
        assert np.isfinite(clv_vals).all()

    def test_clv_positive_for_positive_margin(self, fitted_cure_fitter, small_cure_dgp):
        """Positive premium margin → positive CLV."""
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(600.0).alias("annual_premium"),
            pl.lit(100.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        clv_vals = result["clv"].to_numpy()
        assert (clv_vals > 0).all(), "CLV should be positive for positive margin"

    def test_clv_negative_for_negative_margin(self, fitted_cure_fitter, small_cure_dgp):
        """Negative premium margin → negative CLV."""
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(100.0).alias("annual_premium"),
            pl.lit(500.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        clv_vals = result["clv"].to_numpy()
        assert (clv_vals < 0).all(), "CLV should be negative for negative margin"

    def test_survival_integral_bounded(self, fitted_cure_fitter, small_cure_dgp):
        """Survival integral <= horizon (probability cannot exceed 1)."""
        horizon = 5
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=horizon)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        integral = result["survival_integral"].to_numpy()
        assert (integral <= horizon + 1e-6).all()
        assert (integral >= 0).all()

    def test_higher_discount_rate_reduces_clv(self, fitted_cure_fitter, small_cure_dgp):
        """Higher discount rate reduces CLV (time value of money)."""
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        clv_low = SurvivalCLV(
            survival_model=fitted_cure_fitter, horizon=5, discount_rate=0.02
        ).predict(df)["clv"].mean()
        clv_high = SurvivalCLV(
            survival_model=fitted_cure_fitter, horizon=5, discount_rate=0.15
        ).predict(df)["clv"].mean()
        assert clv_low > clv_high, (
            f"Higher discount rate should reduce CLV: low_r={clv_low:.2f}, high_r={clv_high:.2f}"
        )

    def test_cure_probs_in_range(self, fitted_cure_fitter, small_cure_dgp):
        """Cure probabilities are in [0, 1]."""
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        cure = result["cure_prob"].to_numpy()
        assert (cure >= 0).all() and (cure <= 1).all()

    def test_lifelines_model_clv(self, fitted_lifelines_fitter, small_cure_dgp):
        """CLV works with a standard lifelines WeibullAFTFitter."""
        clv = SurvivalCLV(survival_model=fitted_lifelines_fitter, horizon=3)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert len(result) == len(small_cure_dgp)
        assert np.isfinite(result["clv"].to_numpy()).all()

    def test_annual_survival_probabilities_decreasing(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        """S(year 1) >= S(year 2) >= ... for each policy."""
        horizon = 4
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=horizon)
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        for t in range(1, horizon):
            s_curr = result[f"s_yr{t}"].to_numpy()
            s_next = result[f"s_yr{t + 1}"].to_numpy()
            assert (s_curr >= s_next - 1e-9).all(), (
                f"Survival not monotone at year {t} vs {t+1}"
            )


class TestSurvivalCLVDiscountSensitivity:
    """Tests for discount_sensitivity()."""

    def test_discount_sensitivity_returns_dataframe(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.head(20).with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.discount_sensitivity(df, discount_amounts=[25.0, 50.0])
        assert isinstance(result, pl.DataFrame)

    def test_discount_sensitivity_n_rows(self, fitted_cure_fitter, small_cure_dgp):
        """One row per policy per discount amount."""
        n = 10
        n_discounts = 2
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.head(n).with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.discount_sensitivity(
            df, discount_amounts=[25.0, 50.0]
        )
        assert len(result) == n * n_discounts

    def test_discount_sensitivity_has_required_columns(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.head(10).with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.discount_sensitivity(df, discount_amounts=[25.0])
        expected = {
            "policy_id", "discount_amount", "clv_with_discount",
            "clv_without_discount", "incremental_clv", "discount_justified",
        }
        assert expected.issubset(set(result.columns))

    def test_larger_discount_reduces_clv(self, fitted_cure_fitter, small_cure_dgp):
        """Larger discount amount → lower CLV (premium reduction dominates)."""
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.head(10).with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.discount_sensitivity(
            df, discount_amounts=[10.0, 100.0]
        )
        clv_10 = result.filter(pl.col("discount_amount") == 10.0)["clv_with_discount"].mean()
        clv_100 = result.filter(pl.col("discount_amount") == 100.0)["clv_with_discount"].mean()
        assert clv_10 > clv_100

    def test_discount_justified_column_is_boolean(
        self, fitted_cure_fitter, small_cure_dgp
    ):
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        df = small_cure_dgp.head(10).with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.discount_sensitivity(df, discount_amounts=[25.0])
        assert result["discount_justified"].dtype == pl.Boolean


class TestSurvivalCLVNCDPath:
    """NCD transition matrix tests."""

    def test_custom_ncd_transitions_accepted(self, fitted_cure_fitter, small_cure_dgp):
        """SurvivalCLV accepts a custom NCD transition table."""
        from insurance_survival._utils import default_uk_ncd_transitions
        transitions = default_uk_ncd_transitions(max_ncd=5)
        clv = SurvivalCLV(
            survival_model=fitted_cure_fitter,
            horizon=3,
            ncd_transitions=transitions,
        )
        df = small_cure_dgp.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        result = clv.predict(df)
        assert len(result) == len(small_cure_dgp)
