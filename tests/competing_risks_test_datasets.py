"""Tests for the datasets module."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.datasets import (
    load_bone_marrow_transplant,
    simulate_competing_risks,
    simulate_insurance_retention,
)


class TestLoadBoneMarrowTransplant:
    def test_returns_dataframe(self):
        df = load_bone_marrow_transplant()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = load_bone_marrow_transplant()
        for col in ["T", "E", "group", "waiting_time", "FAB"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_sample_size(self):
        df = load_bone_marrow_transplant()
        # Should have at least 100 patients
        assert len(df) >= 100

    def test_event_codes(self):
        df = load_bone_marrow_transplant()
        # E should only contain 0, 1, 2
        assert set(df["E"].unique()).issubset({0, 1, 2})

    def test_times_positive(self):
        df = load_bone_marrow_transplant()
        assert (df["T"] > 0).all()

    def test_group_values(self):
        df = load_bone_marrow_transplant()
        assert set(df["group"].unique()).issubset({1, 2, 3})

    def test_has_events_of_each_type(self):
        df = load_bone_marrow_transplant()
        assert (df["E"] == 1).sum() > 0, "No cause-1 events"
        assert (df["E"] == 2).sum() > 0, "No cause-2 events"
        assert (df["E"] == 0).sum() > 0, "No censored subjects"


class TestSimulateCompetingRisks:
    def test_default_output(self):
        df = simulate_competing_risks(n=200, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_expected_columns(self):
        df = simulate_competing_risks(n=100)
        for col in ["T", "E", "x1", "x2"]:
            assert col in df.columns

    def test_event_codes(self):
        df = simulate_competing_risks(n=500, seed=1)
        assert set(df["E"].unique()).issubset({0, 1, 2})

    def test_times_positive(self):
        df = simulate_competing_risks(n=300)
        assert (df["T"] > 0).all()

    def test_reproduciblity(self):
        df1 = simulate_competing_risks(n=100, seed=42)
        df2 = simulate_competing_risks(n=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_competing_risks(n=100, seed=42)
        df2 = simulate_competing_risks(n=100, seed=99)
        assert not df1["T"].equals(df2["T"])

    def test_custom_betas(self):
        df = simulate_competing_risks(
            n=200, beta1=[1.0, 0.0], beta2=[0.0, -1.0], seed=7
        )
        assert len(df) == 200

    def test_both_causes_present(self):
        df = simulate_competing_risks(n=1000, seed=0)
        assert (df["E"] == 1).sum() > 10
        assert (df["E"] == 2).sum() > 10

    def test_mismatched_betas_raises(self):
        with pytest.raises(ValueError):
            simulate_competing_risks(beta1=[0.5], beta2=[0.3, 0.1])

    def test_censoring_present(self):
        df = simulate_competing_risks(n=500, censoring_scale=2.0, seed=0)
        assert (df["E"] == 0).sum() > 0

    def test_column_order(self):
        df = simulate_competing_risks(n=50)
        assert list(df.columns[:2]) == ["T", "E"]

    def test_three_covariates(self):
        df = simulate_competing_risks(n=100, beta1=[1.0, 0.0, -0.5], beta2=[0.0, 1.0, 0.3])
        assert "x3" in df.columns


class TestSimulateInsuranceRetention:
    def test_returns_dataframe(self):
        df = simulate_insurance_retention(n=200, seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_columns(self):
        df = simulate_insurance_retention(n=100)
        expected = ["T", "E", "premium_uplift", "tenure_years", "age_band", "ncd_years"]
        for col in expected:
            assert col in df.columns

    def test_event_codes(self):
        df = simulate_insurance_retention(n=500, seed=0)
        assert set(df["E"].unique()).issubset({0, 1, 2, 3})

    def test_multiple_causes_present(self):
        df = simulate_insurance_retention(n=2000, seed=0)
        assert (df["E"] == 1).sum() > 0  # lapse
        assert (df["E"] == 2).sum() > 0  # MTC

    def test_times_positive(self):
        df = simulate_insurance_retention(n=200)
        assert (df["T"] > 0).all()
