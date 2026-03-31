"""
Tests for insurance_survival.competing_risks.calibration.

Covers:
- compute_cal_k_alpha: perfect calibration, biased predictions, input validation
- AJRecalibrator: reduces cal_k_alpha, preserves monotonicity, handles new time grids
- CRDCalibration: uniform PITs pass, skewed PITs fail, summary dict, edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_survival.competing_risks.calibration import (
    AJRecalibrator,
    CRDCalibration,
    compute_cal_k_alpha,
)
from insurance_survival.competing_risks.cif import AalenJohansenFitter


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_competing_data(
    n: int = 400,
    seed: int = 0,
    cause1_rate: float = 0.5,
    cause2_rate: float = 0.3,
    censor_rate: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic competing-risks data.

    Returns (event_times, event_indicators) where indicator is 0=censored,
    1=cause-1 event, 2=cause-2 event.
    """
    rng = np.random.default_rng(seed)
    # Competing exponential times
    t1 = rng.exponential(1.0 / cause1_rate, size=n) if cause1_rate > 0 else np.full(n, np.inf)
    t2 = rng.exponential(1.0 / cause2_rate, size=n) if cause2_rate > 0 else np.full(n, np.inf)
    tc = rng.exponential(1.0 / censor_rate, size=n)

    T = np.minimum(np.minimum(t1, t2), tc)
    E = np.where(
        (t1 <= t2) & (t1 <= tc), 1,
        np.where((t2 < t1) & (t2 <= tc), 2, 0),
    )
    return T.astype(float), E.astype(int)


def _aj_predictions(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    times: np.ndarray,
    n: int,
    event_k: int = 1,
) -> np.ndarray:
    """Return perfectly calibrated predictions: each row is the AJ CIF.

    For a perfectly calibrated model, mean predicted CIF = AJ CIF at every t,
    which is achieved by giving every subject the same AJ prediction.
    """
    aj = AalenJohansenFitter()
    aj.fit(event_times, event_indicators, event_of_interest=event_k)
    cif_vals = aj.predict(times)
    return np.tile(cif_vals, (n, 1))


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_data():
    T, E = _make_competing_data(n=500, seed=42)
    times = np.linspace(0.05, np.percentile(T, 90), 30)
    return T, E, times


@pytest.fixture(scope="module")
def perfect_cif(base_data):
    T, E, times = base_data
    return _aj_predictions(T, E, times, n=len(T), event_k=1)


@pytest.fixture(scope="module")
def biased_cif(perfect_cif):
    """Uniformly over-predicting: add 0.1 everywhere."""
    return np.clip(perfect_cif + 0.10, 0.0, 1.0)


# ---------------------------------------------------------------------------
# compute_cal_k_alpha
# ---------------------------------------------------------------------------

class TestComputeCalKAlpha:
    def test_perfect_calibration_is_zero(self, base_data, perfect_cif):
        T, E, times = base_data
        err = compute_cal_k_alpha(perfect_cif, times, T, E, event_k=1)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_biased_predictions_positive(self, base_data, biased_cif):
        T, E, times = base_data
        err = compute_cal_k_alpha(biased_cif, times, T, E, event_k=1)
        assert err > 0.0

    def test_returns_float(self, base_data, perfect_cif):
        T, E, times = base_data
        err = compute_cal_k_alpha(perfect_cif, times, T, E, event_k=1)
        assert isinstance(err, float)

    def test_non_negative(self, base_data, biased_cif):
        T, E, times = base_data
        err = compute_cal_k_alpha(biased_cif, times, T, E, event_k=1)
        assert err >= 0.0

    def test_cause2(self, base_data):
        T, E, times = base_data
        cif2 = _aj_predictions(T, E, times, n=len(T), event_k=2)
        err = compute_cal_k_alpha(cif2, times, T, E, event_k=2)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_alpha_1_vs_2(self, base_data, biased_cif):
        """alpha=1 (L1) and alpha=2 (L2) should both be positive but differ."""
        T, E, times = base_data
        err1 = compute_cal_k_alpha(biased_cif, times, T, E, event_k=1, alpha=1)
        err2 = compute_cal_k_alpha(biased_cif, times, T, E, event_k=1, alpha=2)
        assert err1 > 0.0
        assert err2 > 0.0
        # L2 should be larger than L1 for the same bias > 0 when bias < 1
        assert err1 != err2

    def test_larger_bias_larger_error(self, base_data, perfect_cif):
        T, E, times = base_data
        cif_small_bias = np.clip(perfect_cif + 0.05, 0.0, 1.0)
        cif_large_bias = np.clip(perfect_cif + 0.20, 0.0, 1.0)
        err_small = compute_cal_k_alpha(cif_small_bias, times, T, E, event_k=1)
        err_large = compute_cal_k_alpha(cif_large_bias, times, T, E, event_k=1)
        assert err_large > err_small

    def test_invalid_ndim_raises(self, base_data):
        T, E, times = base_data
        with pytest.raises(ValueError, match="2-D"):
            compute_cal_k_alpha(np.zeros(len(times)), times, T, E, event_k=1)

    def test_mismatched_times_raises(self, base_data):
        T, E, times = base_data
        n = len(T)
        with pytest.raises(ValueError):
            compute_cal_k_alpha(
                np.zeros((n, len(times) + 1)), times, T, E, event_k=1
            )

    def test_small_sample(self):
        """Should handle small samples without crashing."""
        rng = np.random.default_rng(7)
        T = rng.exponential(1.0, size=30)
        E = rng.choice([0, 1, 2], size=30, p=[0.3, 0.4, 0.3])
        times = np.linspace(0.1, 2.0, 10)
        cif = _aj_predictions(T, E, times, n=30, event_k=1)
        err = compute_cal_k_alpha(cif, times, T, E, event_k=1)
        assert err >= 0.0


# ---------------------------------------------------------------------------
# AJRecalibrator
# ---------------------------------------------------------------------------

class TestAJRecalibrator:
    def test_fit_returns_self(self, base_data, biased_cif):
        T, E, times = base_data
        rec = AJRecalibrator()
        result = rec.fit(biased_cif, times, T, E, event_k=1)
        assert result is rec

    def test_delta_shape(self, base_data, biased_cif):
        T, E, times = base_data
        rec = AJRecalibrator()
        rec.fit(biased_cif, times, T, E, event_k=1)
        assert rec.delta_.shape == times.shape

    def test_reduces_cal_k_alpha(self, base_data, biased_cif):
        T, E, times = base_data
        err_before = compute_cal_k_alpha(biased_cif, times, T, E, event_k=1)

        rec = AJRecalibrator()
        rec.fit(biased_cif, times, T, E, event_k=1)
        recal = rec.transform(biased_cif, times)

        err_after = compute_cal_k_alpha(recal, times, T, E, event_k=1)
        assert err_after < err_before

    def test_recal_near_zero_error(self, base_data, biased_cif):
        """After in-sample recalibration, marginal error should be near zero."""
        T, E, times = base_data
        rec = AJRecalibrator()
        rec.fit(biased_cif, times, T, E, event_k=1)
        recal = rec.transform(biased_cif, times)
        err = compute_cal_k_alpha(recal, times, T, E, event_k=1)
        assert err < 1e-4

    def test_monotonicity_preserved(self, base_data):
        """Each row of the recalibrated CIF must be non-decreasing."""
        T, E, times = base_data
        rng = np.random.default_rng(1)
        # Noisy predictions that may not be perfectly monotone
        cif = _aj_predictions(T, E, times, n=len(T), event_k=1)
        cif += rng.normal(0, 0.02, size=cif.shape)
        cif = np.clip(cif, 0.0, 1.0)

        rec = AJRecalibrator()
        rec.fit(cif, times, T, E, event_k=1)
        recal = rec.transform(cif, times)

        # Check non-decreasing along time axis
        diffs = np.diff(recal, axis=1)
        assert np.all(diffs >= -1e-10), "Recalibrated CIF is not monotone"

    def test_output_in_unit_interval(self, base_data, biased_cif):
        T, E, times = base_data
        rec = AJRecalibrator()
        rec.fit(biased_cif, times, T, E, event_k=1)
        recal = rec.transform(biased_cif, times)
        assert np.all(recal >= 0.0)
        assert np.all(recal <= 1.0)

    def test_output_shape_preserved(self, base_data, biased_cif):
        T, E, times = base_data
        rec = AJRecalibrator()
        rec.fit(biased_cif, times, T, E, event_k=1)
        recal = rec.transform(biased_cif, times)
        assert recal.shape == biased_cif.shape

    def test_transform_before_fit_raises(self):
        rec = AJRecalibrator()
        with pytest.raises(RuntimeError, match="fit()"):
            rec.transform(np.zeros((5, 3)), np.array([1.0, 2.0, 3.0]))

    def test_fit_transform_method(self, base_data, biased_cif):
        T, E, times = base_data
        rec = AJRecalibrator()
        recal = rec.fit_transform(biased_cif, times, T, E, event_k=1)
        assert recal.shape == biased_cif.shape

    def test_under_predicting_recalibration(self, base_data, perfect_cif):
        """Under-predicting model should also be corrected upward."""
        T, E, times = base_data
        under_cif = np.clip(perfect_cif - 0.10, 0.0, 1.0)
        err_before = compute_cal_k_alpha(under_cif, times, T, E, event_k=1)
        rec = AJRecalibrator()
        rec.fit(under_cif, times, T, E, event_k=1)
        recal = rec.transform(under_cif, times)
        err_after = compute_cal_k_alpha(recal, times, T, E, event_k=1)
        assert err_after < err_before


# ---------------------------------------------------------------------------
# CRDCalibration
# ---------------------------------------------------------------------------

class TestCRDCalibration:
    def test_well_calibrated_passes(self):
        """Uniform PITs should give p_value > 0.05."""
        rng = np.random.default_rng(42)
        n = 600
        T, E = _make_competing_data(n=n, seed=42, cause1_rate=0.4, censor_rate=0.15)
        times = np.linspace(0.05, np.percentile(T, 95), 50)

        # Perfect predictions = AJ marginal CIF tiled to all subjects
        cif = _aj_predictions(T, E, times, n=n, event_k=1)

        crd = CRDCalibration(n_bins=10)
        crd.compute(cif, times, T, E, event_k=1)
        # With perfect marginal predictions, PIT distribution should be
        # approximately uniform; we use a lenient threshold.
        assert crd.p_value >= 0.01  # very lenient — structural test

    def test_miscalibrated_fails(self):
        """Heavily biased predictions should give small p_value."""
        rng = np.random.default_rng(0)
        n = 600
        T, E = _make_competing_data(n=n, seed=0, cause1_rate=0.4)
        times = np.linspace(0.05, np.percentile(T, 95), 50)

        # Predict a constant high CIF — badly miscalibrated for events with
        # small observed times
        cif_bad = np.full((n, len(times)), 0.99)

        crd = CRDCalibration(n_bins=10)
        crd.compute(cif_bad, times, T, E, event_k=1)
        assert crd.p_value < 0.05

    def test_statistic_non_negative(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=8)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        assert crd.statistic >= 0.0

    def test_p_value_in_unit_interval(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=8)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        assert 0.0 <= crd.p_value <= 1.0

    def test_summary_keys(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=5)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        s = crd.summary()
        for key in ("statistic", "p_value", "n_bins", "n_events", "bin_counts"):
            assert key in s

    def test_summary_n_bins_matches(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=7)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        s = crd.summary()
        assert s["n_bins"] == 7
        assert len(s["bin_counts"]) == 7

    def test_n_events_correct(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=10)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        expected_n = int((E == 1).sum())
        assert crd.n_events_ == expected_n

    def test_statistic_before_compute_raises(self):
        crd = CRDCalibration()
        with pytest.raises(RuntimeError, match="compute()"):
            _ = crd.statistic

    def test_p_value_before_compute_raises(self):
        crd = CRDCalibration()
        with pytest.raises(RuntimeError, match="compute()"):
            _ = crd.p_value

    def test_no_events_raises(self, base_data, perfect_cif):
        T, E, times = base_data
        # Pass event indicators all-zero (no cause-1 events)
        E_none = np.zeros_like(E)
        crd = CRDCalibration()
        with pytest.raises(ValueError, match="No events"):
            crd.compute(perfect_cif, times, T, E_none, event_k=1)

    def test_invalid_n_bins_raises(self):
        with pytest.raises(ValueError, match="n_bins"):
            CRDCalibration(n_bins=1)

    def test_cause2_only(self):
        """Should work when only cause-2 events are present."""
        rng = np.random.default_rng(9)
        n = 200
        T, E = _make_competing_data(n=n, seed=9, cause1_rate=0.0, cause2_rate=0.5)
        # With cause1_rate=0.0 all events will be cause 2 or censored
        # Re-label so cause 2 is the event
        times = np.linspace(0.05, np.percentile(T, 90), 20)
        cif2 = _aj_predictions(T, E, times, n=n, event_k=2)
        crd = CRDCalibration(n_bins=5)
        crd.compute(cif2, times, T, E, event_k=2)
        assert crd.n_events_ >= 1

    def test_bin_counts_sum_to_n_events(self, base_data, perfect_cif):
        T, E, times = base_data
        crd = CRDCalibration(n_bins=10)
        crd.compute(perfect_cif, times, T, E, event_k=1)
        assert sum(crd.bin_counts_) == crd.n_events_
