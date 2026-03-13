"""Tests for the plots module."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.cif import AalenJohansenFitter
from insurance_survival.competing_risks.datasets import simulate_competing_risks
from insurance_survival.competing_risks.fine_gray import FineGrayFitter
from insurance_survival.competing_risks.plots import (
    plot_brier_score,
    plot_cif_comparison,
    plot_cumulative_hazard,
    plot_forest,
)


@pytest.fixture(scope="module")
def sample_data():
    return simulate_competing_risks(n=300, seed=5)


@pytest.fixture(scope="module")
def fitted_fg(sample_data):
    fg = FineGrayFitter()
    fg.fit(sample_data, duration_col="T", event_col="E", event_of_interest=1)
    return fg


@pytest.fixture(scope="module")
def fitted_aj(sample_data):
    fitter = AalenJohansenFitter()
    fitter.fit(sample_data["T"].values, sample_data["E"].values, event_of_interest=1)
    return fitter


class TestPlotCifComparison:
    def test_returns_axes(self, fitted_aj):
        ax = plot_cif_comparison({"Group": fitted_aj})
        assert ax is not None

    def test_multiple_fitters(self, sample_data):
        aj1 = AalenJohansenFitter().fit(
            sample_data["T"].values, sample_data["E"].values, event_of_interest=1
        )
        aj2 = AalenJohansenFitter().fit(
            sample_data["T"].values, sample_data["E"].values, event_of_interest=2
        )
        ax = plot_cif_comparison({"Cause 1": aj1, "Cause 2": aj2})
        assert ax is not None

    def test_accepts_custom_ax(self, fitted_aj):
        _, custom_ax = plt.subplots()
        result_ax = plot_cif_comparison({"Test": fitted_aj}, ax=custom_ax)
        assert result_ax is custom_ax

    def test_title_set(self, fitted_aj):
        ax = plot_cif_comparison({"G": fitted_aj}, title="My Plot")
        assert ax.get_title() == "My Plot"


class TestPlotForest:
    def test_returns_axes(self, fitted_fg):
        ax = plot_forest(fitted_fg)
        assert ax is not None

    def test_exponentiate_false(self, fitted_fg):
        ax = plot_forest(fitted_fg, exponentiate=False)
        assert ax is not None

    def test_custom_title(self, fitted_fg):
        ax = plot_forest(fitted_fg, title="My Forest")
        assert ax.get_title() == "My Forest"

    def test_unfitted_model_raises(self):
        fg = FineGrayFitter()
        with pytest.raises(RuntimeError):
            plot_forest(fg)


class TestPlotCumulativeHazard:
    def test_returns_axes(self, fitted_fg):
        ax = plot_cumulative_hazard(fitted_fg)
        assert ax is not None

    def test_custom_title(self, fitted_fg):
        ax = plot_cumulative_hazard(fitted_fg, title="Baseline CHF")
        assert ax.get_title() == "Baseline CHF"

    def test_unfitted_raises(self):
        fg = FineGrayFitter()
        with pytest.raises(RuntimeError):
            plot_cumulative_hazard(fg)


class TestPlotBrierScore:
    def test_returns_axes(self):
        times = np.linspace(0.5, 5.0, 10)
        bs = np.random.default_rng(0).uniform(0.05, 0.2, size=10)
        ax = plot_brier_score(times, bs)
        assert ax is not None

    def test_custom_title(self):
        times = np.array([1.0, 2.0, 3.0])
        bs = np.array([0.1, 0.12, 0.15])
        ax = plot_brier_score(times, bs, title="Custom Brier")
        assert ax.get_title() == "Custom Brier"

    def test_accepts_pandas_series(self):
        times = np.array([1.0, 2.0, 3.0])
        bs = pd.Series([0.1, 0.12, 0.15], index=times)
        ax = plot_brier_score(times, bs)
        assert ax is not None
