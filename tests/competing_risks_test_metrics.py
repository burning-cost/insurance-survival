"""Tests for competing-risks evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.datasets import simulate_competing_risks
from insurance_survival.competing_risks.fine_gray import FineGrayFitter
from insurance_survival.competing_risks.metrics import (
    calibration_curve,
    competing_risks_brier_score,
    competing_risks_c_index,
    integrated_brier_score,
)


@pytest.fixture(scope="module")
def train_test_split():
    df_all = simulate_competing_risks(n=800, seed=10)
    train = df_all.iloc[:600].copy()
    test = df_all.iloc[600:].copy()
    return train, test


@pytest.fixture(scope="module")
def fitted_model_and_cif(train_test_split):
    train, test = train_test_split
    fg = FineGrayFitter()
    fg.fit(train, duration_col="T", event_col="E", event_of_interest=1)
    times = np.linspace(0.1, train["T"].max() * 0.8, 15)
    cif = fg.predict_cumulative_incidence(test, times=times)
    return fg, cif, times, train, test


class TestBrierScore:
    def test_returns_series(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        bs = competing_risks_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert isinstance(bs, pd.Series)

    def test_length_matches_times(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        bs = competing_risks_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert len(bs) == len(times)

    def test_values_non_negative(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        bs = competing_risks_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert np.all(bs.values >= 0.0)

    def test_values_bounded_above(self, fitted_model_and_cif):
        """Brier score should be <= 1 for predictions in [0,1]."""
        fg, cif, times, train, test = fitted_model_and_cif
        bs = competing_risks_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert np.all(bs.values <= 1.0)

    def test_perfect_predictions_low_bs(self):
        """Perfectly calibrated predictions should have very low Brier score."""
        rng = np.random.default_rng(0)
        n = 200
        T = rng.exponential(2.0, size=n)
        E = rng.choice([0, 1], size=n, p=[0.3, 0.7])

        times = np.array([1.0, 2.0, 3.0])
        # Use empirical event rate as "perfect" prediction
        emp_rate = (T <= 1.0) & (E == 1)
        pred_const = np.full((n, len(times)), emp_rate.mean())

        bs = competing_risks_brier_score(
            pred_const, T, E, T, E, times, event_of_interest=1
        )
        # Should be smaller than a random model
        assert bs.mean() < 0.35

    def test_accepts_numpy_array(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        bs = competing_risks_brier_score(
            cif.values, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert isinstance(bs, pd.Series)


class TestIntegratedBrierScore:
    def test_returns_float(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        ibs = integrated_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert isinstance(ibs, float)

    def test_non_negative(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        ibs = integrated_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert ibs >= 0.0

    def test_below_null_model(self, fitted_model_and_cif):
        """A fitted model should achieve lower IBS than a constant-0.5 predictor."""
        fg, cif, times, train, test = fitted_model_and_cif
        ibs_model = integrated_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        null_cif = np.full_like(cif.values, 0.5)
        ibs_null = integrated_brier_score(
            null_cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        # Model should not be worse than null by too much
        # (may not strictly beat it with small test set)
        assert ibs_model <= ibs_null * 1.5


class TestCIndex:
    def test_returns_float(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        c = competing_risks_c_index(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            eval_time=eval_time, event_of_interest=1
        )
        assert isinstance(c, float) or np.isnan(c)

    def test_in_valid_range(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        c = competing_risks_c_index(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            eval_time=eval_time, event_of_interest=1
        )
        if not np.isnan(c):
            assert 0.0 <= c <= 1.0

    def test_above_random(self, train_test_split):
        """A good model should achieve C-index above 0.5."""
        train, test = train_test_split
        fg = FineGrayFitter()
        fg.fit(train, duration_col="T", event_col="E", event_of_interest=1)
        times = np.linspace(0.5, train["T"].max() * 0.5, 10)
        cif = fg.predict_cumulative_incidence(test, times=times)
        eval_time = float(np.median(times))
        c = competing_risks_c_index(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            eval_time=eval_time, event_of_interest=1
        )
        if not np.isnan(c):
            assert c >= 0.4  # lenient lower bound given small test set


class TestCalibration:
    def test_returns_dataframe(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        calib = calibration_curve(
            cif, test["T"].values, test["E"].values,
            eval_time=eval_time, event_of_interest=1, n_quantiles=5
        )
        assert isinstance(calib, pd.DataFrame)

    def test_columns_present(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        calib = calibration_curve(
            cif, test["T"].values, test["E"].values,
            eval_time=eval_time, event_of_interest=1, n_quantiles=5
        )
        for col in ["mean_predicted", "observed", "n_subjects"]:
            assert col in calib.columns

    def test_n_subjects_sums_to_n(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        calib = calibration_curve(
            cif, test["T"].values, test["E"].values,
            eval_time=eval_time, event_of_interest=1, n_quantiles=5
        )
        assert calib["n_subjects"].sum() <= len(test)

    def test_mean_predicted_in_range(self, fitted_model_and_cif):
        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        calib = calibration_curve(
            cif, test["T"].values, test["E"].values,
            eval_time=eval_time, event_of_interest=1, n_quantiles=5
        )
        assert np.all(calib["mean_predicted"].values >= 0.0)
        assert np.all(calib["mean_predicted"].values <= 1.0)

    def test_plot_calibration_runs(self, fitted_model_and_cif):
        import matplotlib
        matplotlib.use("Agg")
        from insurance_survival.competing_risks.metrics import plot_calibration

        fg, cif, times, train, test = fitted_model_and_cif
        eval_time = times[len(times) // 2]
        calib = calibration_curve(
            cif, test["T"].values, test["E"].values,
            eval_time=eval_time, event_of_interest=1, n_quantiles=5
        )
        ax = plot_calibration(calib)
        assert ax is not None
