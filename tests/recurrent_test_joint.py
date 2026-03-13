"""
Tests for JointFrailtyModel.
"""

import numpy as np
import pytest

from insurance_survival.recurrent.joint import JointData, JointFrailtyModel, JointFrailtyResult
from insurance_survival.recurrent.simulate import simulate_joint


@pytest.fixture
def joint_data_small():
    rec, term = simulate_joint(
        n_subjects=100,
        random_state=42,
        follow_up=3.0,
    )
    return JointData(
        recurrent=rec,
        terminal_df=term,
        id_col="policy_id",
        terminal_time_col="lapse_time",
        terminal_event_col="lapsed",
        terminal_covariates=["x1", "x2"],
    )


@pytest.fixture
def joint_data_no_term_cov():
    rec, term = simulate_joint(n_subjects=80, random_state=1)
    return JointData(
        recurrent=rec,
        terminal_df=term,
        id_col="policy_id",
        terminal_time_col="lapse_time",
        terminal_event_col="lapsed",
        terminal_covariates=[],
    )


class TestJointData:
    def test_construction(self, joint_data_small):
        assert joint_data_small.n_subjects == 100

    def test_n_subjects(self, joint_data_small):
        assert joint_data_small.n_subjects == 100

    def test_missing_subjects_raises(self):
        rec, term = simulate_joint(n_subjects=50)
        # Drop some subjects from terminal_df
        term_partial = term.iloc[:40].copy()
        with pytest.raises(ValueError, match="not in terminal_df"):
            JointData(
                recurrent=rec,
                terminal_df=term_partial,
                id_col="policy_id",
                terminal_time_col="lapse_time",
                terminal_event_col="lapsed",
                terminal_covariates=[],
            )


class TestJointFrailtyModelFit:
    def test_fit_returns_self(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5)
        result = model.fit(joint_data_small)
        assert result is model

    def test_result_type(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        assert isinstance(model.result_, JointFrailtyResult)

    def test_theta_positive(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        assert model.result_.theta > 0

    def test_log_likelihood_finite(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        assert np.isfinite(model.result_.log_likelihood)

    def test_coef_recurrent_shape(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        # Recurrent has x1, x2 covariates
        assert model.result_.coef_recurrent.shape == (2,)

    def test_coef_terminal_shape(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        assert model.result_.coef_terminal.shape == (2,)

    def test_no_terminal_covariates(self, joint_data_no_term_cov):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_no_term_cov)
        assert model.result_.coef_terminal.shape == (0,)

    def test_summary_recurrent_dataframe(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        s = model.result_.summary_recurrent()
        assert len(s) == 2
        assert "HR" in s.columns

    def test_summary_terminal_dataframe(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        s = model.result_.summary_terminal()
        assert len(s) == 2

    def test_credibility_scores(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        scores = model.credibility_scores()
        assert len(scores) == 100
        assert (scores["frailty_mean"] > 0).all()

    def test_repr_unfitted(self):
        r = repr(JointFrailtyModel())
        assert "unfitted" in r

    def test_repr_fitted(self, joint_data_small):
        model = JointFrailtyModel(max_iter=3).fit(joint_data_small)
        r = repr(model)
        assert "fitted" in r

    def test_result_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            JointFrailtyModel().result_

    def test_verbose(self, joint_data_small, capsys):
        JointFrailtyModel(max_iter=2, verbose=True).fit(joint_data_small)
        captured = capsys.readouterr()
        assert "Iter" in captured.out

    def test_custom_frailty_powers(self, joint_data_small):
        model = JointFrailtyModel(
            frailty_power_recurrent=0.8,
            frailty_power_terminal=1.2,
            max_iter=5,
        ).fit(joint_data_small)
        assert model.alpha == 0.8
        assert model.gamma == 1.2
        assert model.result_.alpha == 0.8
        assert model.result_.gamma == 1.2

    def test_n_iter_recorded(self, joint_data_small):
        model = JointFrailtyModel(max_iter=5).fit(joint_data_small)
        assert 1 <= model.result_.n_iter <= 5
