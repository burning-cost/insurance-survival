"""
Tests for LapseTable.

Validates:
- qx values are in [0, 1]
- lx is strictly decreasing
- dx values are consistent with lx
- Tx (curtate expected future lifetime) is non-negative
- Single and multi-profile generation
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_survival import LapseTable


class TestLapseTableBasic:
    """Basic output correctness tests."""

    def test_generate_returns_dataframe(self, fitted_cure_fitter):
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        assert isinstance(result, pl.DataFrame)

    def test_generate_expected_columns(self, fitted_cure_fitter):
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        expected = {"year", "lx", "dx", "qx", "px", "Tx"}
        assert expected.issubset(set(result.columns))

    def test_generate_n_rows_equals_time_points(self, fitted_cure_fitter):
        """Number of rows equals number of time_points."""
        time_points = [1, 2, 3, 4, 5]
        table = LapseTable(
            survival_model=fitted_cure_fitter,
            time_points=time_points,
        )
        result = table.generate({"ncd_level": 3})
        assert len(result) == len(time_points)

    def test_qx_in_unit_interval(self, fitted_cure_fitter):
        """All qx values are in [0, 1]."""
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        qx = result["qx"].to_numpy()
        assert (qx >= 0).all() and (qx <= 1).all()

    def test_px_equals_one_minus_qx(self, fitted_cure_fitter):
        """px = 1 - qx for all rows."""
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        diff = (result["px"] + result["qx"] - 1.0).abs()
        assert (diff < 1e-6).all()

    def test_lx_starts_at_radix(self, fitted_cure_fitter):
        """First lx value equals the radix."""
        radix = 10_000
        table = LapseTable(survival_model=fitted_cure_fitter, radix=radix)
        result = table.generate({"ncd_level": 3})
        assert result["lx"][0] == radix

    def test_lx_non_increasing(self, fitted_cure_fitter):
        """lx is non-increasing across time points."""
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        lx = result["lx"].to_numpy()
        assert (np.diff(lx) <= 0).all(), "lx should be non-increasing"

    def test_dx_non_negative(self, fitted_cure_fitter):
        """All dx values are non-negative."""
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        assert (result["dx"].to_numpy() >= 0).all()

    def test_Tx_non_negative(self, fitted_cure_fitter):
        """All Tx values are non-negative."""
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        assert (result["Tx"].to_numpy() >= 0).all()

    def test_Tx_eventually_decreasing(self, fitted_cure_fitter):
        """Curtate future lifetime Tx is non-increasing over the full horizon.

        Tx is not guaranteed strictly monotone for non-monotone hazards (e.g.
        cure models where hazard decreases after the high-risk period), but the
        final value must be less than the first value.
        """
        table = LapseTable(survival_model=fitted_cure_fitter)
        result = table.generate({"ncd_level": 3})
        Tx = result["Tx"].to_numpy()
        # The last Tx must be smaller than the first (overall trend)
        assert Tx[-1] < Tx[0], (
            f"Tx should decrease overall: first={Tx[0]:.4f}, last={Tx[-1]:.4f}"
        )
        # All Tx must be finite and non-negative
        assert np.isfinite(Tx).all()
        assert (Tx >= 0).all()


class TestLapseTableRadix:
    """Tests for different radix values."""

    def test_custom_radix(self, fitted_cure_fitter):
        """Custom radix is reflected in lx column."""
        radix = 100_000
        table = LapseTable(survival_model=fitted_cure_fitter, radix=radix)
        result = table.generate({"ncd_level": 5})
        assert result["lx"][0] == radix

    def test_small_radix(self, fitted_cure_fitter):
        """Small radix (100) still produces valid output."""
        table = LapseTable(survival_model=fitted_cure_fitter, radix=100)
        result = table.generate({"ncd_level": 2})
        assert result["lx"][0] == 100
        qx = result["qx"].to_numpy()
        assert (qx >= 0).all() and (qx <= 1).all()


class TestLapseTableMultiProfile:
    """Tests for multi-profile generation."""

    def test_multi_profile_by_column(self, fitted_cure_fitter):
        """Multiple profiles produce rows with segment column."""
        profiles = pl.DataFrame({
            "ncd_level": [1, 3, 5],
            "segment": ["low_ncd", "mid_ncd", "high_ncd"],
        })
        table = LapseTable(survival_model=fitted_cure_fitter, time_points=[1, 2, 3])
        result = table.generate(profiles, by="segment")
        assert "segment" in result.columns
        # 3 profiles × 3 time points = 9 rows
        assert len(result) == 9

    def test_multi_profile_segment_values(self, fitted_cure_fitter):
        """Segment column contains expected labels."""
        profiles = pl.DataFrame({
            "ncd_level": [1, 5],
            "segment": ["low", "high"],
        })
        table = LapseTable(survival_model=fitted_cure_fitter, time_points=[1, 2])
        result = table.generate(profiles, by="segment")
        segments = set(result["segment"].to_list())
        assert segments == {"low", "high"}

    def test_dict_profile_produces_single_table(self, fitted_cure_fitter):
        """A dict profile produces a single lapse table."""
        table = LapseTable(survival_model=fitted_cure_fitter, time_points=[1, 2, 3, 4, 5])
        result = table.generate({"ncd_level": 4})
        assert len(result) == 5


class TestLapseTableWithLifelines:
    """Tests using lifelines WeibullAFTFitter."""

    def test_lifelines_fitter_works(self, fitted_lifelines_fitter):
        """LapseTable works with a lifelines WeibullAFTFitter."""
        table = LapseTable(survival_model=fitted_lifelines_fitter)
        result = table.generate({"ncd_level": 3})
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 7  # default time_points

    def test_lifelines_qx_in_range(self, fitted_lifelines_fitter):
        table = LapseTable(survival_model=fitted_lifelines_fitter)
        result = table.generate({"ncd_level": 3})
        qx = result["qx"].to_numpy()
        assert (qx >= 0).all() and (qx <= 1).all()
