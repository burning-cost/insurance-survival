"""
Regression tests for P0/P1 API bug fixes (batch 3 audit).

Covers:
- SurvivalCLV.predict(): premium_schedule is used year-by-year
- SurvivalCLV.discount_sensitivity(): uses year-by-year discount factors
- ExposureTransformer: written vs earned exposure differ correctly
- BaseMixtureCure.fit(): clear error when Polars DataFrame is passed
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from insurance_survival import ExposureTransformer, SurvivalCLV


# ---------------------------------------------------------------------------
# P0 Bug 1: SurvivalCLV.predict() premium_schedule ignored
# ---------------------------------------------------------------------------

class TestPremiumScheduleUsed:
    """premium_schedule must be consumed year-by-year, not ignored."""

    def _make_df(self, n: int = 5) -> pl.DataFrame:
        return pl.DataFrame({
            "policy_id": [f"P{i}" for i in range(n)],
            "annual_premium": [500.0] * n,
            "expected_loss": [200.0] * n,
        })

    def test_schedule_differs_from_flat(self, fitted_cure_fitter):
        """CLV with a schedule that drops premium in later years must differ from flat."""
        df = self._make_df(5)
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3, discount_rate=0.05)

        # Flat result (no schedule)
        flat = clv.predict(df)

        # Schedule: year 1 same as flat, year 2 premium drops to 300
        schedule_rows = []
        for pid in df["policy_id"].to_list():
            schedule_rows.append({"policy_id": pid, "policy_year": 1, "annual_premium": 500.0, "expected_loss": 200.0})
            schedule_rows.append({"policy_id": pid, "policy_year": 2, "annual_premium": 300.0, "expected_loss": 200.0})
            schedule_rows.append({"policy_id": pid, "policy_year": 3, "annual_premium": 300.0, "expected_loss": 200.0})
        schedule = pl.DataFrame(schedule_rows)

        sched_result = clv.predict(df, premium_schedule=schedule)

        # Lower premiums in years 2-3 → lower CLV
        assert sched_result["clv"].mean() < flat["clv"].mean(), (
            "CLV with lower scheduled premiums should be less than flat-rate CLV"
        )

    def test_schedule_year1_same_as_flat(self, fitted_cure_fitter):
        """If schedule year 1 == flat rate, year-1-only CLV should match."""
        df = self._make_df(3)
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=1, discount_rate=0.05)

        flat = clv.predict(df)

        # Schedule matches flat for year 1
        schedule_rows = []
        for pid in df["policy_id"].to_list():
            schedule_rows.append({"policy_id": pid, "policy_year": 1, "annual_premium": 500.0, "expected_loss": 200.0})
        schedule = pl.DataFrame(schedule_rows)

        sched_result = clv.predict(df, premium_schedule=schedule)

        np.testing.assert_allclose(
            flat["clv"].to_numpy(),
            sched_result["clv"].to_numpy(),
            rtol=1e-6,
            err_msg="Horizon=1 with matching schedule should equal flat CLV",
        )

    def test_schedule_partial_years_falls_back(self, fitted_cure_fitter):
        """Years missing from schedule fall back to flat rate."""
        df = self._make_df(3)
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3, discount_rate=0.05)

        flat = clv.predict(df)

        # Schedule only covers year 1 (years 2-3 fall back to flat)
        schedule_rows = []
        for pid in df["policy_id"].to_list():
            schedule_rows.append({"policy_id": pid, "policy_year": 1, "annual_premium": 500.0, "expected_loss": 200.0})
        schedule = pl.DataFrame(schedule_rows)

        sched_result = clv.predict(df, premium_schedule=schedule)
        # CLVs should be identical since schedule matches flat for year 1
        # and years 2-3 fall back to flat
        np.testing.assert_allclose(
            flat["clv"].to_numpy(),
            sched_result["clv"].to_numpy(),
            rtol=1e-6,
            err_msg="Partial schedule covering year 1 only should equal flat CLV",
        )


# ---------------------------------------------------------------------------
# P0 Bug 2: SurvivalCLV.discount_sensitivity() year-by-year discounting
# ---------------------------------------------------------------------------

class TestDiscountSensitivityDiscounting:
    """discount_sensitivity must use proper NPV discount factors."""

    def test_larger_discount_still_reduces_clv(self, fitted_cure_fitter, small_cure_dgp):
        """Sanity: larger discount → lower discounted CLV."""
        df = small_cure_dgp.head(10).with_columns([
            pl.lit(600.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        clv = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=3)
        result = clv.discount_sensitivity(df, discount_amounts=[10.0, 100.0], price_elasticity=0.0)
        clv_small = result.filter(pl.col("discount_amount") == 10.0)["clv_with_discount"].mean()
        clv_large = result.filter(pl.col("discount_amount") == 100.0)["clv_with_discount"].mean()
        assert clv_small > clv_large

    def test_higher_discount_rate_changes_sensitivity(self, fitted_cure_fitter, small_cure_dgp):
        """CLV at different discount rates should differ when horizon > 1."""
        df = small_cure_dgp.head(10).with_columns([
            pl.lit(600.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        clv_low_r = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=5, discount_rate=0.01)
        clv_high_r = SurvivalCLV(survival_model=fitted_cure_fitter, horizon=5, discount_rate=0.20)

        res_low = clv_low_r.discount_sensitivity(df, discount_amounts=[50.0], price_elasticity=0.0)
        res_high = clv_high_r.discount_sensitivity(df, discount_amounts=[50.0], price_elasticity=0.0)

        # Lower discount rate → higher CLV (closer to undiscounted)
        assert res_low["clv_with_discount"].mean() > res_high["clv_with_discount"].mean()


# ---------------------------------------------------------------------------
# P0 Bug 3: ExposureTransformer written vs earned
# ---------------------------------------------------------------------------

class TestExposureWrittenVsEarned:
    """Written and earned exposure must produce different results."""

    def _make_mid_year_policy(self) -> pl.DataFrame:
        """A policy incepted on 1 July, observed to 31 December (0.5 years earned)."""
        inception = date(2024, 7, 1)
        expiry = date(2025, 7, 1)
        return pl.DataFrame({
            "policy_id": ["P1"],
            "transaction_date": [inception],
            "transaction_type": ["inception"],
            "inception_date": [inception],
            "expiry_date": [expiry],
        })

    def test_written_ne_earned_for_partial_year(self):
        """A policy incepted mid-year should have written=1.0 and earned=0.5."""
        transactions = self._make_mid_year_policy()
        cutoff = date(2024, 12, 31)  # Mid-year observation

        earned_t = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="earned"
        )
        written_t = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="written"
        )

        earned_df = earned_t.fit_transform(transactions)
        written_df = written_t.fit_transform(transactions)

        earned_exp = earned_df["exposure_years"][0]
        written_exp = written_df["exposure_years"][0]

        # Earned should be ~0.5 (half-year elapsed)
        assert abs(earned_exp - 0.5) < 0.02, f"Earned expected ~0.5, got {earned_exp:.4f}"

        # Written should be 1.0 (full policy year written)
        assert abs(written_exp - 1.0) < 0.02, f"Written expected ~1.0, got {written_exp:.4f}"

        # They must differ
        assert written_exp != earned_exp, "Written and earned exposure must differ for partial years"

    def test_written_gt_earned_for_partial_year(self):
        """Written exposure >= earned exposure (written rounds up to whole years)."""
        inception = date(2022, 3, 15)
        expiry = date(2023, 3, 15)
        transactions = pl.DataFrame({
            "policy_id": ["P1"],
            "transaction_date": [inception],
            "transaction_type": ["inception"],
            "inception_date": [inception],
            "expiry_date": [expiry],
        })
        cutoff = date(2022, 9, 15)  # ~0.5 years elapsed

        earned_df = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="earned"
        ).fit_transform(transactions)
        written_df = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="written"
        ).fit_transform(transactions)

        assert written_df["exposure_years"][0] >= earned_df["exposure_years"][0]

    def test_full_year_written_equals_earned(self):
        """A complete annual policy has written == earned == 1.0."""
        inception = date(2023, 1, 1)
        expiry = date(2024, 1, 1)
        transactions = pl.DataFrame({
            "policy_id": ["P1"],
            "transaction_date": [inception],
            "transaction_type": ["inception"],
            "inception_date": [inception],
            "expiry_date": [expiry],
        })
        # Cutoff at exact expiry: full year
        cutoff = date(2023, 12, 31)

        earned_df = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="earned"
        ).fit_transform(transactions)
        written_df = ExposureTransformer(
            observation_cutoff=cutoff, exposure_basis="written"
        ).fit_transform(transactions)

        # Both should be 1 year (approximately)
        assert abs(earned_df["exposure_years"][0] - 1.0) < 0.01
        assert abs(written_df["exposure_years"][0] - 1.0) < 0.01


# ---------------------------------------------------------------------------
# P1 Bug (survival): BaseMixtureCure.fit() rejects Polars DataFrame
# ---------------------------------------------------------------------------

class TestCureModelPolarsRejection:
    """BaseMixtureCure.fit() must raise TypeError for Polars input."""

    def test_polars_input_raises_typeerror(self):
        from insurance_survival.cure import WeibullMixtureCure
        from insurance_survival.cure.simulate import simulate_motor_panel

        model = WeibullMixtureCure(
            incidence_formula="ncd_years",
            latency_formula="ncd_years",
            n_em_starts=1,
            max_iter=5,
        )
        polars_df = simulate_motor_panel(n_policies=50, seed=42)
        assert isinstance(polars_df, pl.DataFrame)

        with pytest.raises(TypeError, match="pandas DataFrame"):
            model.fit(polars_df, duration_col="tenure_months", event_col="claimed")

    def test_pandas_input_works(self):
        from insurance_survival.cure import WeibullMixtureCure
        from insurance_survival.cure.simulate import simulate_motor_panel

        model = WeibullMixtureCure(
            incidence_formula="ncd_years",
            latency_formula="ncd_years",
            n_em_starts=1,
            max_iter=10,
        )
        polars_df = simulate_motor_panel(n_policies=100, seed=42)
        pandas_df = polars_df.to_pandas()

        # Should not raise
        model.fit(pandas_df, duration_col="tenure_months", event_col="claimed")
        assert model._fitted
