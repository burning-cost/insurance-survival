"""Tests for the Aalen-Johansen CIF estimator."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.cif import AalenJohansenFitter, plot_stacked_cif
from insurance_survival.competing_risks.datasets import simulate_competing_risks


@pytest.fixture
def simple_data():
    """Minimal deterministic dataset for unit tests."""
    # 10 subjects: clear competing risks structure
    T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    E = np.array([1, 2, 1, 0, 2, 1, 0, 2, 1, 0])
    return T, E


@pytest.fixture
def synthetic_data():
    return simulate_competing_risks(n=400, seed=0)


class TestAalenJohansenFitter:
    def test_fit_returns_self(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        result = fitter.fit(T, E, event_of_interest=1)
        assert result is fitter

    def test_fit_produces_cif(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        assert hasattr(fitter, "cumulative_incidence_")
        assert isinstance(fitter.cumulative_incidence_, pd.Series)

    def test_cif_starts_at_zero(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        assert fitter.cumulative_incidence_.iloc[0] == pytest.approx(0.0)

    def test_cif_is_non_decreasing(self, synthetic_data):
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        cif = fitter.cumulative_incidence_.values
        diffs = np.diff(cif)
        assert np.all(diffs >= -1e-10), "CIF should be non-decreasing"

    def test_cif_bounded_zero_one(self, synthetic_data):
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        cif = fitter.cumulative_incidence_.values
        assert np.all(cif >= 0.0)
        assert np.all(cif <= 1.0)

    def test_confidence_intervals_structure(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        ci = fitter.confidence_intervals_
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    def test_ci_lower_le_cif(self, synthetic_data):
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        cif = fitter.cumulative_incidence_.values
        lower = fitter.confidence_intervals_["lower"].values
        assert np.all(lower <= cif + 1e-10)

    def test_ci_upper_ge_cif(self, synthetic_data):
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        cif = fitter.cumulative_incidence_.values
        upper = fitter.confidence_intervals_["upper"].values
        assert np.all(upper >= cif - 1e-10)

    def test_times_start_at_zero(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        assert fitter.times_[0] == pytest.approx(0.0)

    def test_predict_at_zero(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        pred = fitter.predict(np.array([0.0]))
        assert pred[0] == pytest.approx(0.0)

    def test_predict_interpolates(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        # Prediction at a fine time grid should be non-decreasing
        times = np.linspace(0, 10, 50)
        preds = fitter.predict(times)
        assert np.all(np.diff(preds) >= -1e-10)

    def test_predict_extrapolation_right(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        pred_far = fitter.predict(np.array([1000.0]))
        pred_end = fitter.cumulative_incidence_.values[-1]
        assert pred_far[0] == pytest.approx(pred_end)

    def test_different_cause_codes(self, simple_data):
        T, E = simple_data
        f1 = AalenJohansenFitter().fit(T, E, event_of_interest=1)
        f2 = AalenJohansenFitter().fit(T, E, event_of_interest=2)
        # CIFs for different causes should generally differ
        assert not np.allclose(
            f1.cumulative_incidence_.values,
            f2.cumulative_incidence_.values
        )

    def test_cif_sum_le_one(self, synthetic_data):
        """Sum of CIFs across causes should not exceed 1.0."""
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        f1 = AalenJohansenFitter().fit(T, E, event_of_interest=1)
        f2 = AalenJohansenFitter().fit(T, E, event_of_interest=2)
        times_common = np.linspace(0, T.max() * 0.9, 20)
        cif1 = f1.predict(times_common)
        cif2 = f2.predict(times_common)
        assert np.all(cif1 + cif2 <= 1.0 + 1e-6)

    def test_unfitted_predict_raises(self):
        fitter = AalenJohansenFitter()
        with pytest.raises(RuntimeError):
            fitter.predict(np.array([1.0]))

    def test_label_stored(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1, label="My Cause")
        assert fitter.label_ == "My Cause"

    def test_default_label(self, simple_data):
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        assert "1" in fitter.label_

    def test_plot_returns_axes(self, simple_data):
        import matplotlib
        matplotlib.use("Agg")
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        ax = fitter.plot()
        assert ax is not None

    def test_plot_no_ci(self, simple_data):
        import matplotlib
        matplotlib.use("Agg")
        T, E = simple_data
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1)
        ax = fitter.plot(ci=False)
        assert ax is not None

    def test_alpha_affects_ci_width(self, synthetic_data):
        T = synthetic_data["T"].values
        E = synthetic_data["E"].values
        f90 = AalenJohansenFitter(alpha=0.10).fit(T, E, event_of_interest=1)
        f95 = AalenJohansenFitter(alpha=0.05).fit(T, E, event_of_interest=1)
        w90 = f90.confidence_intervals_["upper"] - f90.confidence_intervals_["lower"]
        w95 = f95.confidence_intervals_["upper"] - f95.confidence_intervals_["lower"]
        assert w95.mean() >= w90.mean()

    def test_weighted_fit(self, simple_data):
        T, E = simple_data
        weights = np.ones(len(T))
        fitter = AalenJohansenFitter()
        fitter.fit(T, E, event_of_interest=1, weights=weights)
        assert hasattr(fitter, "cumulative_incidence_")


class TestPlotStackedCif:
    def test_returns_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        df = simulate_competing_risks(n=200, seed=0)
        ax = plot_stacked_cif(df["T"].values, df["E"].values)
        assert ax is not None

    def test_custom_causes(self):
        import matplotlib
        matplotlib.use("Agg")
        df = simulate_competing_risks(n=200, seed=0)
        ax = plot_stacked_cif(
            df["T"].values, df["E"].values, causes=[1, 2]
        )
        assert ax is not None

    def test_cause_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        df = simulate_competing_risks(n=200, seed=0)
        ax = plot_stacked_cif(
            df["T"].values, df["E"].values,
            cause_labels={1: "Relapse", 2: "Death"},
        )
        assert ax is not None
