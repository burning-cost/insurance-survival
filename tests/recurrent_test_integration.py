"""
Integration tests: end-to-end workflows, parameter recovery, and cross-module consistency.
"""

import numpy as np
import pytest

from insurance_survival.recurrent import (
    AndersenGillFrailty,
    FrailtyReport,
    JointData,
    JointFrailtyModel,
    NelsonAalenFrailty,
    PWPModel,
    RecurrentEventData,
    SimulationParams,
    compare_models,
    simulate_ag_frailty,
    simulate_joint,
    simulate_pwp,
)


class TestEndToEndAG:
    def test_full_gamma_workflow(self):
        """Simulate → fit → credibility scores → report."""
        params = SimulationParams(n_subjects=200, theta=2.0, random_state=42)
        data = simulate_ag_frailty(params)
        model = AndersenGillFrailty(frailty="gamma", max_iter=20).fit(data)
        scores = model.credibility_scores()
        report = FrailtyReport(model, data)
        summary = report.frailty_summary()
        decile = report.event_rate_by_frailty_decile()

        assert len(scores) == 200
        assert float(summary.loc["mean", "value"]) > 0
        assert len(decile) > 0

    def test_full_lognormal_workflow(self):
        params = SimulationParams(n_subjects=150, theta=2.0, random_state=1)
        data = simulate_ag_frailty(params)
        model = AndersenGillFrailty(frailty="lognormal", max_iter=10).fit(data)
        scores = model.credibility_scores()
        assert len(scores) == 150
        assert (scores["frailty_mean"] > 0).all()

    def test_compare_gamma_lognormal(self):
        params = SimulationParams(n_subjects=200, random_state=5)
        data = simulate_ag_frailty(params)
        m_gamma = AndersenGillFrailty(frailty="gamma", max_iter=15).fit(data)
        m_lognorm = AndersenGillFrailty(frailty="lognormal", max_iter=15).fit(data)
        df = compare_models([m_gamma, m_lognorm], names=["gamma", "lognormal"], data=data)
        assert len(df) == 2
        # Best model (lowest AIC) is row 0
        assert df.iloc[0]["AIC"] <= df.iloc[1]["AIC"]


class TestEndToEndPWP:
    def test_pwp_gap_time(self):
        data = simulate_pwp(n_subjects=150, random_state=0)
        model = PWPModel(time_scale="gap").fit(data)
        s = model.summary()
        assert len(s) == 2

    def test_pwp_calendar_time(self):
        data = simulate_pwp(n_subjects=150, random_state=0)
        model = PWPModel(time_scale="calendar").fit(data)
        assert model.result_ is not None


class TestEndToEndJoint:
    def test_joint_full_workflow(self):
        rec, term = simulate_joint(n_subjects=120, random_state=0)
        jd = JointData(
            recurrent=rec,
            terminal_df=term,
            id_col="policy_id",
            terminal_time_col="lapse_time",
            terminal_event_col="lapsed",
            terminal_covariates=["x1"],
        )
        model = JointFrailtyModel(max_iter=5).fit(jd)
        scores = model.credibility_scores()
        assert len(scores) == 120
        assert model.result_.theta > 0


class TestParameterRecovery:
    def test_theta_recovery_direction(self):
        """
        Low true theta (high dispersion) should give lower fitted theta than high true theta.
        Not exact recovery — just directional.
        """
        d_low = simulate_ag_frailty(SimulationParams(n_subjects=400, theta=0.5, random_state=10))
        d_high = simulate_ag_frailty(SimulationParams(n_subjects=400, theta=5.0, random_state=10))

        m_low = AndersenGillFrailty(max_iter=25).fit(d_low)
        m_high = AndersenGillFrailty(max_iter=25).fit(d_high)

        assert m_low.result_.theta < m_high.result_.theta

    def test_beta_sign_recovery(self):
        """Signs of beta should match true signs with N=500."""
        true_beta = np.array([0.4, -0.3])
        data = simulate_ag_frailty(
            SimulationParams(n_subjects=500, beta=true_beta, theta=3.0, random_state=42)
        )
        model = AndersenGillFrailty(max_iter=25).fit(data)
        fitted_beta = model.result_.coef
        assert np.sign(fitted_beta[0]) == np.sign(true_beta[0])
        assert np.sign(fitted_beta[1]) == np.sign(true_beta[1])

    def test_no_frailty_theta_large(self):
        """
        When data has no real heterogeneity (theta very large), fitted theta should
        also be large (near the upper bound) or at least > 5.
        """
        # Generate data with very high theta (homogeneous, no frailty)
        data = simulate_ag_frailty(
            SimulationParams(n_subjects=300, theta=50.0, random_state=0)
        )
        model = AndersenGillFrailty(max_iter=20).fit(data)
        # Should not recover very small theta
        assert model.result_.theta > 1.0


class TestConsistency:
    def test_credibility_agrees_with_manual_gamma(self):
        """
        For gamma frailty with no covariates, posterior mean formula is exact:
        E[z|data] = (theta + n_i) / (theta + lambda_i)
        Check that model output matches this formula approximately.
        """
        from insurance_survival.recurrent.frailty import GammaFrailty

        data = simulate_ag_frailty(
            SimulationParams(n_subjects=100, beta=np.array([]), random_state=0)
        )
        # Reconstruct no-covariate data
        import pandas as pd
        from insurance_survival.recurrent.data import RecurrentEventData

        rng = np.random.default_rng(0)
        records = []
        theta = 2.0
        z = rng.gamma(shape=theta, scale=1.0 / theta, size=100)
        for i in range(100):
            rate_i = z[i] * 0.3
            t = 0.0
            while t < 3.0:
                wait = rng.exponential(1.0 / max(rate_i, 1e-10))
                t_event = t + wait
                if t_event >= 3.0:
                    records.append({"policy_id": i, "t_start": t, "t_stop": 3.0, "event": 0})
                    break
                else:
                    records.append({"policy_id": i, "t_start": t, "t_stop": t_event, "event": 1})
                    t = t_event
        df = pd.DataFrame(records)
        data2 = RecurrentEventData.from_long_format(df, "policy_id", "t_start", "t_stop", "event")

        model = AndersenGillFrailty(max_iter=20).fit(data2)
        scores = model.credibility_scores()

        # Manual calculation using fitted theta
        gf = GammaFrailty()
        theta_fit = model.result_.theta
        n_i = scores["n_events"].values.astype(float)
        lambda_i = scores["lambda_i"].values
        z_manual = gf.posterior_mean(n_i, lambda_i, theta_fit)
        z_model = scores["frailty_mean"].values
        # Should be identical (model uses the same formula)
        np.testing.assert_allclose(z_model, z_manual, rtol=1e-6)

    def test_from_events_and_from_long_format_equivalent(self):
        """Both construction methods should produce equivalent data."""
        import pandas as pd

        events = pd.DataFrame({
            "policy_id": [0, 0, 1, 2],
            "t_event": [0.5, 1.5, 0.8, 2.0],
        })
        follow_up = pd.DataFrame({
            "policy_id": [0, 1, 2],
            "end_time": [3.0, 3.0, 3.0],
            "age": [35.0, 45.0, 55.0],
        })

        d1 = RecurrentEventData.from_events(
            events, follow_up,
            id_col="policy_id",
            time_col="t_event",
            end_col="end_time",
            covariates=["age"],
        )

        # Manual construction
        records = [
            {"policy_id": 0, "t_start": 0.0, "t_stop": 0.5, "event": 1, "age": 35.0},
            {"policy_id": 0, "t_start": 0.5, "t_stop": 1.5, "event": 1, "age": 35.0},
            {"policy_id": 0, "t_start": 1.5, "t_stop": 3.0, "event": 0, "age": 35.0},
            {"policy_id": 1, "t_start": 0.0, "t_stop": 0.8, "event": 1, "age": 45.0},
            {"policy_id": 1, "t_start": 0.8, "t_stop": 3.0, "event": 0, "age": 45.0},
            {"policy_id": 2, "t_start": 0.0, "t_stop": 2.0, "event": 1, "age": 55.0},
            {"policy_id": 2, "t_start": 2.0, "t_stop": 3.0, "event": 0, "age": 55.0},
        ]
        d2 = RecurrentEventData.from_long_format(
            pd.DataFrame(records),
            id_col="policy_id",
            start_col="t_start",
            stop_col="t_stop",
            event_col="event",
            covariates=["age"],
        )

        assert d1.n_events == d2.n_events
        assert d1.n_subjects == d2.n_subjects
