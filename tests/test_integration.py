"""
Integration tests: full pipeline from transactions to CLV.

These tests exercise the whole stack:
    ExposureTransformer → WeibullMixtureCureFitter → SurvivalCLV → LapseTable
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from insurance_survival import (
    ExposureTransformer,
    LapseTable,
    SurvivalCLV,
    WeibullMixtureCureFitter,
)
from conftest import make_transaction_dgp


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Run the full pipeline once and return intermediate results."""
        cutoff = date(2025, 12, 31)
        transactions = make_transaction_dgp(
            n_policies=100, seed=7, cutoff=cutoff
        )

        # Step 1: Transform
        transformer = ExposureTransformer(observation_cutoff=cutoff)
        survival_df = transformer.fit_transform(transactions)

        # Step 2: Fit cure model
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=0.1,
            max_iter=100,
        )
        # Use single-interval data (last interval per policy for simplicity)
        single_interval = survival_df.group_by("policy_id").agg([
            pl.col("stop").max(),
            pl.col("event").max(),
            pl.col("ncd_level").last(),
        ])
        fitter.fit(
            single_interval,
            duration_col="stop",
            event_col="event",
        )

        # Step 3: CLV
        policy_profiles = single_interval.with_columns([
            pl.lit(500.0).alias("annual_premium"),
            pl.lit(200.0).alias("expected_loss"),
        ])
        clv_model = SurvivalCLV(survival_model=fitter, horizon=3)
        clv_result = clv_model.predict(policy_profiles)

        # Step 4: Lapse table
        lapse_table = LapseTable(survival_model=fitter, time_points=[1, 2, 3])
        table_result = lapse_table.generate({"ncd_level": 3})

        return {
            "survival_df": survival_df,
            "fitter": fitter,
            "clv_result": clv_result,
            "lapse_table": table_result,
            "transformer_summary": transformer.summary(),
        }

    def test_pipeline_runs_without_error(self, pipeline_result):
        """The full pipeline completes without raising."""
        assert pipeline_result is not None

    def test_transform_produces_intervals(self, pipeline_result):
        """ExposureTransformer produces a non-empty survival DataFrame."""
        assert len(pipeline_result["survival_df"]) > 0

    def test_fitter_converges(self, pipeline_result):
        """WeibullMixtureCureFitter converges."""
        fitter = pipeline_result["fitter"]
        assert fitter._gamma is not None
        assert fitter._beta is not None

    def test_clv_result_has_expected_shape(self, pipeline_result):
        """CLV result has the correct number of rows and columns."""
        clv_result = pipeline_result["clv_result"]
        assert len(clv_result) > 0
        assert "clv" in clv_result.columns
        assert "cure_prob" in clv_result.columns

    def test_lapse_table_valid_output(self, pipeline_result):
        """Lapse table has valid qx values."""
        table = pipeline_result["lapse_table"]
        qx = table["qx"].to_numpy()
        assert (qx >= 0).all() and (qx <= 1).all()

    def test_transformer_summary_consistent(self, pipeline_result):
        """Transformer summary n_intervals matches survival_df length."""
        summary = pipeline_result["transformer_summary"]
        survival_df = pipeline_result["survival_df"]
        assert summary["n_intervals"] == len(survival_df)
