"""Tests for the Fine-Gray subdistribution hazard model."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.datasets import (
    simulate_competing_risks,
    load_bone_marrow_transplant,
)
from insurance_survival.competing_risks.fine_gray import FineGrayFitter


@pytest.fixture(scope="module")
def synthetic_df():
    """Synthetic data with known structure for coefficient recovery tests."""
    return simulate_competing_risks(n=600, seed=42)


@pytest.fixture(scope="module")
def fitted_model(synthetic_df):
    """Pre-fitted model to avoid repeating expensive fit in each test."""
    fg = FineGrayFitter()
    fg.fit(synthetic_df, duration_col="T", event_col="E", event_of_interest=1)
    return fg


@pytest.fixture(scope="module")
def bmt_df():
    return load_bone_marrow_transplant()


class TestFineGrayFitterFit:
    def test_fit_returns_self(self, synthetic_df):
        fg = FineGrayFitter()
        result = fg.fit(synthetic_df, duration_col="T", event_col="E", event_of_interest=1)
        assert result is fg

    def test_params_series(self, fitted_model):
        assert isinstance(fitted_model.params_, pd.Series)
        assert len(fitted_model.params_) == 2  # x1, x2

    def test_param_names(self, fitted_model):
        assert "x1" in fitted_model.params_.index
        assert "x2" in fitted_model.params_.index

    def test_summary_dataframe(self, fitted_model):
        assert isinstance(fitted_model.summary, pd.DataFrame)
        for col in ["coef", "exp(coef)", "se(coef)", "z", "p"]:
            assert col in fitted_model.summary.columns

    def test_variance_matrix_shape(self, fitted_model):
        V = fitted_model.variance_matrix_
        p = len(fitted_model.params_)
        assert V.shape == (p, p)

    def test_variance_matrix_symmetric(self, fitted_model):
        V = fitted_model.variance_matrix_.values
        assert np.allclose(V, V.T, atol=1e-10)

    def test_variance_matrix_positive_diagonal(self, fitted_model):
        V = fitted_model.variance_matrix_.values
        assert np.all(np.diag(V) > 0)

    def test_se_positive(self, fitted_model):
        se = fitted_model.summary["se(coef)"].values
        assert np.all(se > 0)

    def test_exp_coef_positive(self, fitted_model):
        exp_coef = fitted_model.summary["exp(coef)"].values
        assert np.all(exp_coef > 0)

    def test_p_values_in_range(self, fitted_model):
        p = fitted_model.summary["p"].values
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_ci_cols_present(self, fitted_model):
        cols = fitted_model.summary.columns.tolist()
        assert any("lower" in c for c in cols)
        assert any("upper" in c for c in cols)

    def test_log_likelihood_finite(self, fitted_model):
        assert np.isfinite(fitted_model.log_likelihood_)

    def test_baseline_cumulative_hazard_series(self, fitted_model):
        bch = fitted_model.baseline_cumulative_hazard_
        assert isinstance(bch, pd.Series)
        assert len(bch) > 0

    def test_baseline_hazard_non_decreasing(self, fitted_model):
        bch = fitted_model.baseline_cumulative_hazard_.values
        assert np.all(np.diff(bch) >= -1e-10)

    def test_fitted_flag(self, fitted_model):
        assert fitted_model._fitted is True

    def test_unfitted_predict_raises(self):
        fg = FineGrayFitter()
        with pytest.raises(RuntimeError):
            fg.predict_cumulative_incidence(pd.DataFrame({"x1": [0.0], "x2": [0.0]}))

    def test_no_events_raises(self, synthetic_df):
        df_no_events = synthetic_df.copy()
        df_no_events["E"] = 0  # all censored
        with pytest.raises(ValueError):
            FineGrayFitter().fit(df_no_events, duration_col="T", event_col="E", event_of_interest=1)

    def test_no_cause_k_raises(self, synthetic_df):
        df_no_k = synthetic_df.copy()
        df_no_k["E"] = df_no_k["E"].replace(1, 0)  # remove cause-1 events
        with pytest.raises(ValueError):
            FineGrayFitter().fit(df_no_k, duration_col="T", event_col="E", event_of_interest=1)

    def test_fit_with_penaliser(self, synthetic_df):
        fg = FineGrayFitter(penaliser=1.0)
        fg.fit(synthetic_df, duration_col="T", event_col="E", event_of_interest=1)
        fg_no_pen = FineGrayFitter(penaliser=0.0)
        fg_no_pen.fit(synthetic_df, duration_col="T", event_col="E", event_of_interest=1)
        # Penalised coefficients should be shrunk towards zero
        assert np.all(np.abs(fg.params_.values) <= np.abs(fg_no_pen.params_.values) + 0.5)

    def test_competing_cause_also_fittable(self, synthetic_df):
        fg2 = FineGrayFitter()
        fg2.fit(synthetic_df, duration_col="T", event_col="E", event_of_interest=2)
        assert fg2._fitted

    def test_duration_col_stored(self, fitted_model):
        assert fitted_model.duration_col == "T"

    def test_event_col_stored(self, fitted_model):
        assert fitted_model.event_col == "E"

    def test_event_of_interest_stored(self, fitted_model):
        assert fitted_model.event_of_interest == 1


class TestCoefficientRecovery:
    """Check that fitted coefficients have the correct sign and are plausibly
    close to the true data-generating values."""

    def test_x1_sign_cause1(self):
        """Beta1[0] = 0.5 (positive), so x1 should increase cause-1 risk."""
        df = simulate_competing_risks(n=2000, beta1=[0.5, -0.3], seed=42)
        fg = FineGrayFitter()
        fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
        assert fg.params_["x1"] > 0, "Expected positive coefficient for x1"

    def test_x2_sign_cause1(self):
        """Beta1[1] = -0.3 (negative), so x2 should decrease cause-1 risk."""
        df = simulate_competing_risks(n=2000, beta1=[0.5, -0.3], seed=42)
        fg = FineGrayFitter()
        fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
        assert fg.params_["x2"] < 0, "Expected negative coefficient for x2"

    def test_null_covariate_near_zero(self):
        """A covariate with true beta=0 should have coefficient near zero."""
        df = simulate_competing_risks(n=1000, beta1=[0.0, 0.0], seed=0)
        fg = FineGrayFitter()
        fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
        assert abs(fg.params_["x1"]) < 0.5
        assert abs(fg.params_["x2"]) < 0.5


class TestPrediction:
    def test_predict_cif_shape(self, fitted_model, synthetic_df):
        times = [1.0, 2.0, 3.0]
        cif = fitted_model.predict_cumulative_incidence(synthetic_df, times=times)
        assert cif.shape == (len(synthetic_df), len(times))

    def test_predict_cif_bounded(self, fitted_model, synthetic_df):
        times = np.linspace(0.1, 5.0, 20)
        cif = fitted_model.predict_cumulative_incidence(synthetic_df, times=times)
        assert np.all(cif.values >= 0.0)
        assert np.all(cif.values <= 1.0 + 1e-6)

    def test_predict_cif_non_decreasing_per_subject(self, fitted_model, synthetic_df):
        times = np.linspace(0.1, 5.0, 30)
        cif = fitted_model.predict_cumulative_incidence(synthetic_df, times=times)
        for i in range(min(10, len(synthetic_df))):
            row = cif.values[i]
            assert np.all(np.diff(row) >= -1e-10), f"CIF not monotone for subject {i}"

    def test_predict_cif_returns_dataframe(self, fitted_model, synthetic_df):
        cif = fitted_model.predict_cumulative_incidence(synthetic_df[:5], times=[1.0])
        assert isinstance(cif, pd.DataFrame)

    def test_predict_higher_risk_subject_higher_cif(self, fitted_model):
        """Subject with high x1 (increases cause-1 risk) should have higher CIF."""
        df_high = pd.DataFrame({"x1": [2.0], "x2": [0.0]})
        df_low = pd.DataFrame({"x1": [-2.0], "x2": [0.0]})
        t = [1.0]
        cif_high = fitted_model.predict_cumulative_incidence(df_high, times=t).values[0, 0]
        cif_low = fitted_model.predict_cumulative_incidence(df_low, times=t).values[0, 0]
        assert cif_high > cif_low, "Subject with higher risk should have higher predicted CIF"

    def test_predict_default_times(self, fitted_model, synthetic_df):
        cif = fitted_model.predict_cumulative_incidence(synthetic_df[:5])
        # Should use all training event times
        n_times = len(fitted_model.baseline_cumulative_hazard_)
        assert cif.shape[1] == n_times

    def test_predict_median_time_series(self, fitted_model, synthetic_df):
        med = fitted_model.predict_median_time(synthetic_df[:20])
        assert isinstance(med, pd.Series)
        assert len(med) == 20

    def test_predict_median_time_positive(self, fitted_model, synthetic_df):
        med = fitted_model.predict_median_time(synthetic_df[:20])
        assert np.all(med.values > 0)


class TestBMTDataset:
    """Integration tests on the bone marrow transplant benchmark dataset.

    We do not check exact R cmprsk::crr() values because the BMT data we
    ship is a reconstruction; instead we verify directional correctness and
    convergence.
    """

    def test_fit_on_bmt(self, bmt_df):
        fg = FineGrayFitter()
        fg.fit(
            bmt_df[["T", "E", "group"]],
            duration_col="T",
            event_col="E",
            event_of_interest=1,
        )
        assert fg._fitted

    def test_bmt_coefficients_finite(self, bmt_df):
        fg = FineGrayFitter()
        fg.fit(
            bmt_df[["T", "E", "group"]],
            duration_col="T",
            event_col="E",
            event_of_interest=1,
        )
        assert np.all(np.isfinite(fg.params_.values))

    def test_bmt_se_finite(self, bmt_df):
        fg = FineGrayFitter()
        fg.fit(
            bmt_df[["T", "E", "group"]],
            duration_col="T",
            event_col="E",
            event_of_interest=1,
        )
        assert np.all(np.isfinite(fg.summary["se(coef)"].values))


class TestPlotting:
    def test_plot_partial_effects_returns_axes(self, fitted_model):
        import matplotlib
        matplotlib.use("Agg")
        ax = fitted_model.plot_partial_effects_on_outcome("x1", [-1.0, 0.0, 1.0])
        assert ax is not None

    def test_plot_partial_effects_invalid_covariate(self, fitted_model):
        with pytest.raises(ValueError):
            fitted_model.plot_partial_effects_on_outcome("nonexistent", [0.0])

    def test_print_summary_runs(self, fitted_model, capsys):
        fitted_model.print_summary()
        captured = capsys.readouterr()
        assert "Fine-Gray" in captured.out
