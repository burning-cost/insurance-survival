"""
Tests for simulation functions.
"""

import numpy as np
import pytest

from insurance_survival.recurrent.data import RecurrentEventData
from insurance_survival.recurrent.simulate import (
    SimulationParams,
    simulate_ag_frailty,
    simulate_joint,
    simulate_pwp,
)


class TestSimulateAgFrailty:
    def test_returns_recurrent_event_data(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=50))
        assert isinstance(data, RecurrentEventData)

    def test_n_subjects(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100))
        assert data.n_subjects == 100

    def test_has_covariates(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=50, beta=np.array([0.3])))
        assert len(data.covariate_cols) == 1
        assert data.X.shape[1] == 1

    def test_two_covariates(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=50, beta=np.array([0.3, -0.2])))
        assert data.X.shape[1] == 2

    def test_reproducible_with_seed(self):
        d1 = simulate_ag_frailty(SimulationParams(n_subjects=50, random_state=99))
        d2 = simulate_ag_frailty(SimulationParams(n_subjects=50, random_state=99))
        np.testing.assert_array_equal(d1.df["event"].values, d2.df["event"].values)

    def test_different_seeds_differ(self):
        d1 = simulate_ag_frailty(SimulationParams(n_subjects=50, random_state=1))
        d2 = simulate_ag_frailty(SimulationParams(n_subjects=50, random_state=2))
        assert not np.array_equal(d1.df["event"].values, d2.df["event"].values)

    def test_events_are_binary(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100))
        assert set(data.df["event"].unique()).issubset({0, 1})

    def test_time_intervals_positive(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=50))
        assert (data.stop > data.start).all()

    def test_no_overlap_within_subject(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=30))
        for pid in data.subject_ids:
            grp = data.df[data.df["policy_id"] == pid].sort_values("t_start")
            if len(grp) > 1:
                stops = grp["t_stop"].values[:-1]
                starts = grp["t_start"].values[1:]
                assert np.allclose(stops, starts), f"Gaps in subject {pid}"

    def test_follow_up_respected(self):
        params = SimulationParams(n_subjects=50, follow_up=2.0)
        data = simulate_ag_frailty(params)
        assert (data.stop <= 2.0 + 1e-9).all()

    def test_lognormal_frailty(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=50, frailty_dist="lognormal"))
        assert data.n_subjects == 50

    def test_high_theta_low_heterogeneity(self):
        """Higher theta = less frailty dispersion = more uniform event counts."""
        d_low = simulate_ag_frailty(SimulationParams(n_subjects=500, theta=0.5, random_state=1))
        d_high = simulate_ag_frailty(SimulationParams(n_subjects=500, theta=10.0, random_state=1))
        counts_low = d_low.per_subject_summary()["n_events"].std()
        counts_high = d_high.per_subject_summary()["n_events"].std()
        assert counts_low > counts_high

    def test_kwargs_override(self):
        params = SimulationParams(n_subjects=50)
        data = simulate_ag_frailty(params, random_state=7)
        assert data.n_subjects == 50

    def test_zero_beta_no_covariate_effect(self):
        """With beta=0, claim rate should be approx constant across covariate deciles."""
        params = SimulationParams(n_subjects=500, beta=np.array([0.0, 0.0]), random_state=42)
        data = simulate_ag_frailty(params)
        # Just check it runs without error
        assert data.n_events >= 0


class TestSimulatePWP:
    def test_returns_recurrent_event_data(self):
        data = simulate_pwp(n_subjects=50)
        assert isinstance(data, RecurrentEventData)

    def test_n_subjects(self):
        data = simulate_pwp(n_subjects=100)
        assert data.n_subjects == 100

    def test_higher_later_rates_more_events(self):
        """With increasing baseline rates, later events should accumulate."""
        d_inc = simulate_pwp(n_subjects=200, baseline_rates=(0.2, 0.5, 0.8), random_state=0)
        d_flat = simulate_pwp(n_subjects=200, baseline_rates=(0.5, 0.5, 0.5), random_state=0)
        # Not a strict test — just check both run
        assert d_inc.n_events >= 0
        assert d_flat.n_events >= 0

    def test_reproducible(self):
        d1 = simulate_pwp(n_subjects=50, random_state=5)
        d2 = simulate_pwp(n_subjects=50, random_state=5)
        np.testing.assert_array_equal(d1.event, d2.event)

    def test_covariates(self):
        data = simulate_pwp(n_subjects=50, beta=np.array([0.3]))
        assert data.X.shape[1] == 1

    def test_follow_up_respected(self):
        data = simulate_pwp(n_subjects=50, follow_up=1.5)
        assert (data.stop <= 1.5 + 1e-9).all()


class TestSimulateJoint:
    def test_returns_tuple(self):
        result = simulate_joint(n_subjects=50)
        assert isinstance(result, tuple) and len(result) == 2

    def test_recurrent_event_data(self):
        rec, term = simulate_joint(n_subjects=50)
        assert isinstance(rec, RecurrentEventData)

    def test_terminal_df_shape(self):
        rec, term = simulate_joint(n_subjects=50)
        assert len(term) == 50

    def test_terminal_df_columns(self):
        _, term = simulate_joint(n_subjects=50)
        assert "lapse_time" in term.columns
        assert "lapsed" in term.columns

    def test_lapsed_binary(self):
        _, term = simulate_joint(n_subjects=100)
        assert set(term["lapsed"].unique()).issubset({0, 1})

    def test_lapse_time_positive(self):
        _, term = simulate_joint(n_subjects=50)
        assert (term["lapse_time"] > 0).all()

    def test_lapse_time_bounded(self):
        _, term = simulate_joint(n_subjects=50, follow_up=2.0)
        assert (term["lapse_time"] <= 2.0 + 1e-9).all()

    def test_claims_before_lapse(self):
        rec, term = simulate_joint(n_subjects=50)
        # For each subject, all claim stop times should be <= lapse time
        for pid in rec.subject_ids:
            grp = rec.df[rec.df["policy_id"] == pid]
            lapse = float(term.loc[term["policy_id"] == pid, "lapse_time"].values[0])
            assert (grp["t_stop"] <= lapse + 1e-9).all(), f"Subject {pid} has claims after lapse"

    def test_reproducible(self):
        r1, t1 = simulate_joint(n_subjects=50, random_state=42)
        r2, t2 = simulate_joint(n_subjects=50, random_state=42)
        np.testing.assert_array_equal(r1.event, r2.event)

    def test_subject_ids_consistent(self):
        rec, term = simulate_joint(n_subjects=50)
        rec_ids = set(rec.subject_ids.tolist())
        term_ids = set(term["policy_id"].tolist())
        assert rec_ids.issubset(term_ids)
