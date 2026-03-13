"""
Tests for FrailtyReport and compare_models.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.recurrent.models import AndersenGillFrailty, NelsonAalenFrailty
from insurance_survival.recurrent.report import FrailtyReport, compare_models
from insurance_survival.recurrent.simulate import SimulationParams, simulate_ag_frailty


@pytest.fixture
def fitted_ag_model():
    data = simulate_ag_frailty(SimulationParams(n_subjects=150, random_state=0))
    model = AndersenGillFrailty(frailty="gamma", max_iter=15).fit(data)
    return model, data


@pytest.fixture
def fitted_report(fitted_ag_model):
    model, data = fitted_ag_model
    return FrailtyReport(model=model, data=data)


class TestFrailtyReport:
    def test_frailty_summary_type(self, fitted_report):
        s = fitted_report.frailty_summary()
        assert isinstance(s, pd.DataFrame)

    def test_frailty_summary_index(self, fitted_report):
        s = fitted_report.frailty_summary()
        assert "mean" in s.index
        assert "std" in s.index
        assert "median" in s.index

    def test_frailty_mean_near_one(self, fitted_report):
        """Population mean of frailty should be close to 1."""
        s = fitted_report.frailty_summary()
        mean = float(s.loc["mean", "value"])
        assert 0.4 < mean < 2.5

    def test_frailty_std_positive(self, fitted_report):
        s = fitted_report.frailty_summary()
        std = float(s.loc["std", "value"])
        assert std > 0

    def test_event_rate_by_decile_type(self, fitted_report):
        d = fitted_report.event_rate_by_frailty_decile()
        assert isinstance(d, pd.DataFrame)

    def test_event_rate_by_decile_has_columns(self, fitted_report):
        d = fitted_report.event_rate_by_frailty_decile()
        assert "mean_events_per_subject" in d.columns
        assert "mean_frailty" in d.columns

    def test_event_rate_monotone_in_frailty(self, fitted_report):
        """Higher frailty deciles should have higher event rates (roughly)."""
        d = fitted_report.event_rate_by_frailty_decile()
        # Not strictly monotone due to noise, but top half > bottom half
        n = len(d)
        bottom = d.iloc[:n // 2]["mean_events_per_subject"].mean()
        top = d.iloc[n // 2:]["mean_events_per_subject"].mean()
        assert top > bottom

    def test_model_aic_finite(self, fitted_report):
        aic = fitted_report.model_aic()
        assert np.isfinite(aic)

    def test_model_bic_finite(self, fitted_report):
        bic = fitted_report.model_bic()
        assert np.isfinite(bic)

    def test_aic_bic_ordering(self, fitted_report):
        """BIC >= AIC for reasonable sample sizes."""
        aic = fitted_report.model_aic()
        bic = fitted_report.model_bic()
        # BIC penalises more heavily for n > 7 subjects
        assert bic >= aic or abs(bic - aic) < 10  # loose check

    def test_convergence_summary_type(self, fitted_report):
        s = fitted_report.convergence_summary()
        assert isinstance(s, dict)

    def test_convergence_summary_keys(self, fitted_report):
        s = fitted_report.convergence_summary()
        assert "theta" in s
        assert "log_likelihood" in s

    def test_credibility_by_group(self, fitted_ag_model):
        model, data = fitted_ag_model
        # Add a group column to data
        import pandas as pd
        data.df["region"] = (data.df["x1"] > 0).map({True: "north", False: "south"})
        report = FrailtyReport(model=model, data=data)
        result = report.credibility_by_group("region")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "n_subjects" in result.columns


class TestCompareModels:
    def test_returns_dataframe(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100, random_state=0))
        m1 = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        m2 = AndersenGillFrailty(frailty="lognormal", max_iter=10).fit(data)
        result = compare_models([m1, m2], names=["gamma", "lognormal"], data=data)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_aic(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100, random_state=0))
        m1 = AndersenGillFrailty(frailty="gamma", max_iter=10).fit(data)
        m2 = AndersenGillFrailty(frailty="lognormal", max_iter=10).fit(data)
        result = compare_models([m1, m2], data=data)
        aics = result["AIC"].values
        assert aics[0] <= aics[-1]

    def test_columns(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100, random_state=0))
        m1 = AndersenGillFrailty(max_iter=5).fit(data)
        result = compare_models([m1], data=data)
        for col in ["model", "log_likelihood", "AIC", "BIC", "theta"]:
            assert col in result.columns

    def test_no_data_gives_nan_bic(self):
        data = simulate_ag_frailty(SimulationParams(n_subjects=100, random_state=0))
        m1 = AndersenGillFrailty(max_iter=5).fit(data)
        result = compare_models([m1])
        assert np.isnan(result.iloc[0]["BIC"])
