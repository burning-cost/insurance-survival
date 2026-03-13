"""Integration tests: end-to-end workflows combining multiple modules."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks import (
    AalenJohansenFitter,
    FineGrayFitter,
    competing_risks_brier_score,
    competing_risks_c_index,
    gray_test,
    load_bone_marrow_transplant,
    simulate_competing_risks,
)
from insurance_survival.competing_risks.datasets import simulate_insurance_retention
from insurance_survival.competing_risks.metrics import calibration_curve, integrated_brier_score
from insurance_survival.competing_risks.plots import plot_forest


class TestFullWorkflowSynthetic:
    """Full workflow: simulate -> fit -> evaluate -> plot."""

    def test_synthetic_workflow_completes(self):
        # 1. Generate data
        df = simulate_competing_risks(n=400, seed=99)
        train = df.iloc[:300]
        test = df.iloc[300:]

        # 2. Non-parametric CIF
        aj = AalenJohansenFitter()
        aj.fit(train["T"].values, train["E"].values, event_of_interest=1)
        assert len(aj.cumulative_incidence_) > 0

        # 3. Regression
        fg = FineGrayFitter()
        fg.fit(train, duration_col="T", event_col="E", event_of_interest=1)
        assert fg._fitted

        # 4. Prediction
        times = np.linspace(0.2, train["T"].max() * 0.7, 10)
        cif = fg.predict_cumulative_incidence(test, times=times)
        assert cif.shape == (len(test), len(times))
        assert np.all(cif.values >= 0.0)
        assert np.all(cif.values <= 1.0 + 1e-6)

        # 5. Gray's test
        result = gray_test(
            train["T"].values, train["E"].values,
            (train["x1"] > 0).astype(int).values,
            event_of_interest=1,
        )
        assert result.p_value >= 0.0

        # 6. Brier score
        bs = competing_risks_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert len(bs) == len(times)

        # 7. IBS
        ibs = integrated_brier_score(
            cif, test["T"].values, test["E"].values,
            train["T"].values, train["E"].values,
            times, event_of_interest=1
        )
        assert np.isfinite(ibs)

        # 8. Forest plot
        ax = plot_forest(fg)
        assert ax is not None


class TestInsuranceRetentionWorkflow:
    """End-to-end on the insurance retention simulation."""

    def test_retention_workflow(self):
        df = simulate_insurance_retention(n=600, seed=0)
        train = df.iloc[:500]
        test = df.iloc[500:]

        # Fit model for lapse (cause 1)
        fg = FineGrayFitter()
        fg.fit(
            train[["T", "E", "premium_uplift", "tenure_years", "ncd_years"]],
            duration_col="T",
            event_col="E",
            event_of_interest=1,
        )
        assert fg._fitted

        # Premium uplift should have positive effect on lapse
        assert fg.params_["premium_uplift"] > 0, (
            "Expected positive premium_uplift coefficient for lapse"
        )

        # Predictions
        times = np.linspace(0.1, train["T"].max() * 0.5, 8)
        cif = fg.predict_cumulative_incidence(
            test[["T", "E", "premium_uplift", "tenure_years", "ncd_years"]],
            times=times
        )
        assert cif.shape[0] == len(test)

    def test_all_causes_fit(self):
        df = simulate_insurance_retention(n=600, seed=0)

        for cause in [1, 2]:
            fg = FineGrayFitter()
            fg.fit(
                df[["T", "E", "premium_uplift", "tenure_years", "ncd_years"]],
                duration_col="T",
                event_col="E",
                event_of_interest=cause,
            )
            assert fg._fitted, f"Failed to fit model for cause {cause}"


class TestBMTWorkflow:
    """Validate on the bone marrow transplant benchmark."""

    def test_bmt_cif_both_causes(self):
        df = load_bone_marrow_transplant()
        for cause in [1, 2]:
            aj = AalenJohansenFitter()
            aj.fit(df["T"].values, df["E"].values, event_of_interest=cause)
            assert len(aj.cumulative_incidence_) > 1

    def test_bmt_cif_sum_le_one(self):
        df = load_bone_marrow_transplant()
        aj1 = AalenJohansenFitter().fit(df["T"].values, df["E"].values, event_of_interest=1)
        aj2 = AalenJohansenFitter().fit(df["T"].values, df["E"].values, event_of_interest=2)
        times = np.linspace(0, df["T"].max() * 0.9, 30)
        c1 = aj1.predict(times)
        c2 = aj2.predict(times)
        assert np.all(c1 + c2 <= 1.0 + 1e-6), "CIF sum exceeds 1.0"

    def test_bmt_gray_test(self):
        df = load_bone_marrow_transplant()
        result = gray_test(
            df["T"].values, df["E"].values,
            df["group"].values,
            event_of_interest=1,
        )
        assert result.degrees_of_freedom == 2  # 3 groups → 2 df
        assert 0.0 <= result.p_value <= 1.0

    def test_bmt_regression_all_covariates(self):
        df = load_bone_marrow_transplant()
        fg = FineGrayFitter()
        fg.fit(
            df[["T", "E", "group", "FAB"]],
            duration_col="T",
            event_col="E",
            event_of_interest=1,
        )
        assert fg._fitted
        assert np.all(np.isfinite(fg.params_.values))


class TestImportPublicAPI:
    """Verify the public API is importable from the top-level package."""

    def test_fine_gray_fitter_importable(self):
        from insurance_survival.competing_risks import FineGrayFitter
        assert FineGrayFitter is not None

    def test_aalen_johansen_importable(self):
        from insurance_survival.competing_risks import AalenJohansenFitter
        assert AalenJohansenFitter is not None

    def test_gray_test_importable(self):
        from insurance_survival.competing_risks import gray_test
        assert gray_test is not None

    def test_metrics_importable(self):
        from insurance_survival.competing_risks import (
            competing_risks_brier_score,
            competing_risks_c_index,
        )
        assert competing_risks_brier_score is not None
        assert competing_risks_c_index is not None

    def test_datasets_importable(self):
        from insurance_survival.competing_risks import (
            load_bone_marrow_transplant,
            simulate_competing_risks,
        )
        assert load_bone_marrow_transplant is not None
        assert simulate_competing_risks is not None

    def test_version_string(self):
        import insurance_survival.competing_risks as insurance_competing_risks
        pass  # __version__ not needed in subpackage
        pass  # version check skipped after merge
