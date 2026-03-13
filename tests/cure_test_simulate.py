"""Tests for the synthetic data generators."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.cure.simulate import simulate_motor_panel, simulate_pet_panel


class TestSimulateMotorPanel:
    def test_returns_dataframe(self):
        df = simulate_motor_panel(n_policies=100, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = simulate_motor_panel(n_policies=200, seed=1)
        assert len(df) == 200

    def test_required_columns(self):
        df = simulate_motor_panel(n_policies=50, seed=2)
        required = {"policy_id", "ncb_years", "age", "vehicle_age",
                    "is_immune", "tenure_months", "claimed", "true_cure_prob"}
        assert required.issubset(set(df.columns))

    def test_duration_positive(self):
        df = simulate_motor_panel(n_policies=200, seed=3)
        assert (df["tenure_months"] > 0).all()

    def test_event_binary(self):
        df = simulate_motor_panel(n_policies=200, seed=4)
        assert set(df["claimed"].unique()).issubset({0, 1})

    def test_cure_fraction_close_to_target(self):
        df = simulate_motor_panel(n_policies=2000, cure_fraction=0.40, seed=5)
        # Immune policies can still be observed before end of window
        observed_immune_rate = df["is_immune"].mean()
        assert abs(observed_immune_rate - 0.40) < 0.05

    def test_cure_fraction_zero(self):
        """All policies are susceptible."""
        df = simulate_motor_panel(n_policies=300, cure_fraction=0.0, seed=6)
        # No immune policies
        assert df["is_immune"].sum() == 0

    def test_cure_fraction_one(self):
        """All policies are immune — no events."""
        df = simulate_motor_panel(n_policies=300, cure_fraction=1.0, seed=7)
        assert df["is_immune"].all()
        assert df["claimed"].sum() == 0

    def test_ncb_range(self):
        df = simulate_motor_panel(n_policies=200, seed=8)
        assert df["ncb_years"].min() >= 0
        assert df["ncb_years"].max() <= 9

    def test_age_range(self):
        df = simulate_motor_panel(n_policies=200, seed=9)
        assert df["age"].min() >= 18
        assert df["age"].max() <= 80

    def test_vehicle_age_range(self):
        df = simulate_motor_panel(n_policies=200, seed=10)
        assert df["vehicle_age"].min() >= 0
        assert df["vehicle_age"].max() <= 15

    def test_true_cure_prob_range(self):
        df = simulate_motor_panel(n_policies=200, seed=11)
        assert (df["true_cure_prob"] >= 0).all()
        assert (df["true_cure_prob"] <= 1).all()

    def test_seed_reproducibility(self):
        df1 = simulate_motor_panel(n_policies=100, seed=42)
        df2 = simulate_motor_panel(n_policies=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_motor_panel(n_policies=100, seed=1)
        df2 = simulate_motor_panel(n_policies=100, seed=2)
        assert not df1["ncb_years"].equals(df2["ncb_years"])

    def test_immune_policyholders_not_event(self):
        df = simulate_motor_panel(n_policies=500, cure_fraction=0.5, seed=13)
        # Immune policyholders must not have claimed
        immune_claims = df.loc[df["is_immune"], "claimed"].sum()
        assert immune_claims == 0

    def test_ncb_higher_means_more_immune(self):
        df = simulate_motor_panel(n_policies=2000, seed=14)
        low_ncb = df[df["ncb_years"] <= 2]["is_immune"].mean()
        high_ncb = df[df["ncb_years"] >= 7]["is_immune"].mean()
        # Higher NCB should yield higher immune rate
        assert high_ncb > low_ncb

    def test_n_years_affects_duration(self):
        df5 = simulate_motor_panel(n_policies=200, n_years=5, seed=15)
        df2 = simulate_motor_panel(n_policies=200, n_years=2, seed=15)
        assert df5["tenure_months"].max() <= 60.0 + 0.1
        assert df2["tenure_months"].max() <= 24.0 + 0.1


class TestSimulatePetPanel:
    def test_returns_dataframe(self):
        df = simulate_pet_panel(n_policies=100, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = simulate_pet_panel(n_policies=150, seed=1)
        assert len(df) == 150

    def test_required_columns(self):
        df = simulate_pet_panel(n_policies=50, seed=2)
        required = {"policy_id", "pet_age", "breed_risk", "indoor",
                    "tenure_months", "claimed", "true_cure_prob"}
        assert required.issubset(set(df.columns))

    def test_duration_positive(self):
        df = simulate_pet_panel(n_policies=200, seed=3)
        assert (df["tenure_months"] > 0).all()

    def test_event_binary(self):
        df = simulate_pet_panel(n_policies=200, seed=4)
        assert set(df["claimed"].unique()).issubset({0, 1})

    def test_breed_risk_range(self):
        df = simulate_pet_panel(n_policies=200, seed=5)
        assert (df["breed_risk"] >= 0).all()
        assert (df["breed_risk"] <= 1).all()

    def test_seed_reproducibility(self):
        df1 = simulate_pet_panel(n_policies=100, seed=42)
        df2 = simulate_pet_panel(n_policies=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)
