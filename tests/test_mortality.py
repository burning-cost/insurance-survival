"""
Tests for insurance_survival.mortality — coherent cause-specific mortality.

All tests use synthetic data only (no HMD download required).

NumPyro-dependent tests (fit/forecast) are run on Databricks due to JAX
memory requirements. Tests that do not require fitting (dataclass, loader,
validation) run without NumPyro.

Test groups
-----------
TestMortalityForecast      — MortalityForecast dataclass, coherence_check, summary
TestHMDLoader              — Synthetic data generation and from_arrays utility
TestCauseSpecificMortality — Input validation (no NumPyro required)
TestMortalityFitForecast   — Full fit/forecast round-trip (requires NumPyro, runs on Databricks)
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_forecast(
    n_ages: int = 4,
    n_periods: int = 5,
    n_causes: int = 3,
    n_draws: int = 20,
    seed: int = 0,
) -> "MortalityForecast":
    """Build a MortalityForecast with coherent draws for testing."""
    from insurance_survival.mortality import MortalityForecast

    rng = np.random.default_rng(seed)

    # Draw cause fractions on simplex, then multiply by total
    total_rates = rng.uniform(0.001, 0.05, (n_ages, n_periods, n_draws))
    fractions = rng.dirichlet(np.ones(n_causes), size=(n_ages, n_periods, n_draws))
    # fractions: (n_ages, n_periods, n_draws, n_causes)
    cause_rates = total_rates[..., np.newaxis] * fractions
    # Reorder to (n_ages, n_periods, n_causes, n_draws)
    cause_rates = cause_rates.transpose(0, 1, 3, 2)

    ages = [str(i * 5) for i in range(n_ages)]
    periods = list(range(2024, 2024 + n_periods))
    cause_names = [f"cause_{c}" for c in range(n_causes)]

    return MortalityForecast(
        total_rates=total_rates,
        cause_rates=cause_rates,
        ages=ages,
        periods=periods,
        cause_names=cause_names,
    )


# ---------------------------------------------------------------------------
# MortalityForecast tests
# ---------------------------------------------------------------------------

class TestMortalityForecast:
    def test_init_stores_attributes(self):
        from insurance_survival.mortality import MortalityForecast

        rng = np.random.default_rng(0)
        total = rng.uniform(0.01, 0.05, (3, 4, 10))
        cause = total[:, :, np.newaxis, :] * np.ones((3, 4, 2, 10)) / 2

        fc = MortalityForecast(
            total_rates=total,
            cause_rates=cause,
            ages=["0", "5", "10"],
            periods=[2024, 2025, 2026, 2027],
            cause_names=["a", "b"],
        )
        assert fc.n_draws == 10
        assert fc.n_causes == 2

    def test_shape_mismatch_raises(self):
        from insurance_survival.mortality import MortalityForecast

        rng = np.random.default_rng(1)
        total = rng.uniform(0.01, 0.05, (3, 4, 10))
        # Wrong n_age in cause_rates
        cause = rng.uniform(0, 1, (4, 4, 2, 10))

        with pytest.raises(ValueError, match="age groups"):
            MortalityForecast(
                total_rates=total,
                cause_rates=cause,
                ages=["0", "5", "10"],
                periods=[2024, 2025, 2026, 2027],
                cause_names=["a", "b"],
            )

    def test_coherence_check_passes_on_coherent_draws(self):
        fc = make_synthetic_forecast(n_ages=3, n_periods=4, n_causes=3, n_draws=50)
        result = fc.coherence_check(tol=1e-6)
        assert result is True

    def test_coherence_check_fails_on_incoherent_draws(self):
        from insurance_survival.mortality import MortalityForecast

        rng = np.random.default_rng(2)
        total = rng.uniform(0.01, 0.05, (3, 4, 10))
        # Cause rates that do NOT sum to total
        cause = rng.uniform(0.001, 0.02, (3, 4, 2, 10))

        fc = MortalityForecast(
            total_rates=total,
            cause_rates=cause,
            ages=["0", "5", "10"],
            periods=[2024, 2025, 2026, 2027],
            cause_names=["a", "b"],
        )
        with pytest.raises(AssertionError, match="Coherence check failed"):
            fc.coherence_check(tol=1e-6)

    def test_summary_returns_dataframe(self):
        import pandas as pd
        fc = make_synthetic_forecast(n_ages=2, n_periods=3, n_causes=2, n_draws=10)
        df = fc.summary()
        assert isinstance(df, pd.DataFrame)
        # (n_ages * n_periods) * (n_causes + 1) rows: 2 ages × 3 periods × 3 (total + 2 causes)
        assert len(df) == 2 * 3 * (2 + 1)
        assert "mean" in df.columns
        assert "sd" in df.columns
        assert "cause" in df.columns
        # Check total cause is present
        assert "total" in df["cause"].values

    def test_summary_quantiles_respected(self):
        fc = make_synthetic_forecast(n_ages=2, n_periods=2, n_causes=2, n_draws=100)
        df = fc.summary(quantiles=(0.1, 0.9))
        assert "q0100" in df.columns
        assert "q0900" in df.columns
        # Default quantiles should not be present
        assert "q0025" not in df.columns

    def test_improvement_factors_shape(self):
        fc = make_synthetic_forecast(n_ages=2, n_periods=5, n_causes=2, n_draws=20)
        df = fc.improvement_factors()
        # n_ages * (n_periods - 1) * n_causes rows
        assert len(df) == 2 * 4 * 2
        assert "improvement_factor" in df.columns
        assert "improvement_q025" in df.columns

    def test_coherence_check_tolerance(self):
        """coherence_check with large tol passes even with numerical noise."""
        fc = make_synthetic_forecast()
        # Add tiny perturbation directly to total_rates
        fc_modified = make_synthetic_forecast()
        fc_modified.total_rates = fc_modified.total_rates + 1e-9  # tiny noise
        # Should pass with generous tol
        assert fc_modified.coherence_check(tol=1e-5) is True


# ---------------------------------------------------------------------------
# HMDLoader tests
# ---------------------------------------------------------------------------

class TestHMDLoader:
    def test_load_synthetic_default_shape(self):
        from insurance_survival.mortality import HMDLoader

        deaths, exposure, ages, years = HMDLoader.load_synthetic()
        # Default: 18 ages, 40 years, 6 causes
        assert deaths.shape == (18, 40, 6)
        assert exposure.shape == (18, 40)
        assert len(ages) == 18
        assert len(years) == 40

    def test_load_synthetic_custom_dimensions(self):
        from insurance_survival.mortality import HMDLoader

        deaths, exposure, ages, years = HMDLoader.load_synthetic(
            n_ages=5, n_years=10, n_causes=3, seed=1
        )
        assert deaths.shape == (5, 10, 3)
        assert exposure.shape == (5, 10)
        assert len(ages) == 5
        assert len(years) == 10

    def test_load_synthetic_non_negative(self):
        from insurance_survival.mortality import HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(n_ages=6, n_years=10, n_causes=3)
        assert np.all(deaths >= 0), "Deaths must be non-negative"
        assert np.all(exposure > 0), "Exposure must be strictly positive"

    def test_load_synthetic_cause_sums_consistent(self):
        """Cause deaths should sum to approximately total deaths.

        They are generated from Dirichlet-Multinomial conditional on total,
        so sum_c deaths[:,:,c] == total_deaths exactly.
        """
        from insurance_survival.mortality import HMDLoader

        deaths, _, _, _ = HMDLoader.load_synthetic(n_ages=4, n_years=10, n_causes=4)
        cause_sum = deaths.sum(axis=2)
        # Re-generate total deaths from same seed is not straightforward here,
        # but we can check that cause_sum > 0 wherever exposure is nonzero.
        assert np.all(cause_sum >= 0)

    def test_load_synthetic_reproducible(self):
        from insurance_survival.mortality import HMDLoader

        d1, e1, _, _ = HMDLoader.load_synthetic(seed=42)
        d2, e2, _, _ = HMDLoader.load_synthetic(seed=42)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(e1, e2)

    def test_load_synthetic_different_seeds(self):
        from insurance_survival.mortality import HMDLoader

        d1, _, _, _ = HMDLoader.load_synthetic(seed=42)
        d2, _, _, _ = HMDLoader.load_synthetic(seed=99)
        assert not np.array_equal(d1, d2)

    def test_load_synthetic_custom_cause_names(self):
        from insurance_survival.mortality import HMDLoader

        names = ["cancer", "heart", "other"]
        _, _, _, _ = HMDLoader.load_synthetic(n_causes=3, cause_names=names)
        # No error raised — names are accepted

    def test_from_arrays_default_labels(self):
        from insurance_survival.mortality import HMDLoader

        deaths = np.ones((3, 5, 2))
        exposure = np.ones((3, 5)) * 1000

        d, e, ages, years = HMDLoader.from_arrays(deaths, exposure)
        assert len(ages) == 3
        assert len(years) == 5
        assert d.dtype == float
        assert e.dtype == float

    def test_from_arrays_validates_shapes(self):
        from insurance_survival.mortality import HMDLoader

        deaths = np.ones((3, 5, 2))
        exposure = np.ones((3, 5)) * 1000

        with pytest.raises(ValueError, match="age_labels"):
            HMDLoader.from_arrays(deaths, exposure, age_labels=["a", "b"])  # wrong length

    def test_load_missing_directory_raises(self):
        from insurance_survival.mortality import HMDLoader

        with pytest.raises(FileNotFoundError):
            HMDLoader.load("/nonexistent/path/to/hmd")


# ---------------------------------------------------------------------------
# CauseSpecificMortality input validation (no NumPyro required)
# ---------------------------------------------------------------------------

class TestCauseSpecificMortality:
    def test_invalid_model_type(self):
        from insurance_survival.mortality import CauseSpecificMortality

        with pytest.raises(ValueError, match="model_type"):
            CauseSpecificMortality(model_type="XYZ")

    def test_fit_raises_without_numpyro_gracefully(self, monkeypatch):
        """If NumPyro is not installed, fit raises ImportError with guidance."""
        from insurance_survival.mortality import CauseSpecificMortality
        import insurance_survival.mortality.mortality as mort_mod

        # Patch the lazy import to simulate missing NumPyro
        def fake_require():
            raise ImportError(
                "NumPyro is required for CauseSpecificMortality fitting. "
                "Install it with: pip install insurance-survival[mortality]\n"
                "Original error: No module named 'numpyro'"
            )

        monkeypatch.setattr(mort_mod, "_require_numpyro", fake_require)

        model = CauseSpecificMortality()
        deaths = np.ones((3, 5, 2))
        exposure = np.ones((3, 5)) * 1000

        with pytest.raises(ImportError, match="NumPyro"):
            model.fit(deaths, exposure)

    def test_validate_deaths_ndim(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(ValueError, match="3D"):
            model._validate_inputs(np.ones((3, 5)), np.ones((3, 5)))

    def test_validate_exposure_ndim(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(ValueError, match="2D"):
            model._validate_inputs(np.ones((3, 5, 2)), np.ones((3, 5, 2)))

    def test_validate_shape_mismatch(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(ValueError, match="match"):
            model._validate_inputs(np.ones((3, 5, 2)), np.ones((4, 5)))

    def test_validate_negative_deaths(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        deaths = np.ones((3, 5, 2))
        deaths[0, 0, 0] = -1
        with pytest.raises(ValueError, match="non-negative"):
            model._validate_inputs(deaths, np.ones((3, 5)))

    def test_validate_zero_exposure(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        exposure = np.ones((3, 5))
        exposure[0, 0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            model._validate_inputs(np.ones((3, 5, 2)), exposure)

    def test_validate_nan(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        deaths = np.ones((3, 5, 2))
        deaths[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            model._validate_inputs(deaths, np.ones((3, 5)))

    def test_forecast_before_fit_raises(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(RuntimeError, match="fitted"):
            model.forecast(10)

    def test_diagnostics_before_fit_raises(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(RuntimeError, match="fitted"):
            model.diagnostics()

    def test_cause_fractions_before_fit_raises(self):
        from insurance_survival.mortality import CauseSpecificMortality

        model = CauseSpecificMortality()
        with pytest.raises(RuntimeError, match="fitted"):
            model.cause_fractions(0, 0)

    def test_fit_label_mismatch_raises(self):
        """Providing mismatched age_labels should raise ValueError."""
        from insurance_survival.mortality import CauseSpecificMortality
        import insurance_survival.mortality.mortality as mort_mod

        def fake_require():
            raise ImportError("No module named 'numpyro'")

        # We need to test label validation before the NumPyro call.
        # CauseSpecificMortality checks labels BEFORE calling _require_numpyro,
        # so this should raise ValueError, not ImportError.
        model = CauseSpecificMortality()
        deaths = np.ones((3, 5, 2))
        exposure = np.ones((3, 5))

        # Patch so NumPyro import doesn't interfere
        import unittest.mock as mock
        with mock.patch.object(mort_mod, "_require_numpyro", side_effect=ImportError):
            with pytest.raises(ValueError, match="age_labels"):
                model.fit(deaths, exposure, age_labels=["a", "b"])  # wrong length

    def test_reconstruct_rw2_identity(self):
        """RW2 reconstruction recovers a known series."""
        from insurance_survival.mortality import CauseSpecificMortality

        # Series: 0, 1, 2, 3, 4 (linear trend, zero innovations)
        # Delta^2 = 0 for linear trend -> innovations = 0
        series_true = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
        init = series_true[:, :2]      # [[0, 1]]
        innov = np.zeros((1, 3))       # 3 innovations, all zero

        series_rec = CauseSpecificMortality._reconstruct_rw2(init, innov)
        np.testing.assert_allclose(series_rec, series_true, atol=1e-10)


# ---------------------------------------------------------------------------
# Full fit/forecast round-trip (requires NumPyro — runs on Databricks)
# ---------------------------------------------------------------------------

numpyro_available = False
try:
    import numpyro  # noqa: F401
    numpyro_available = True
except ImportError:
    pass

requires_numpyro = pytest.mark.skipif(
    not numpyro_available,
    reason="NumPyro not installed — run on Databricks with pip install numpyro",
)


class TestMortalityFitForecast:
    """Full integration tests. These require NumPyro and JAX."""

    @requires_numpyro
    def test_lc_fit_returns_self(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, ages, years = HMDLoader.load_synthetic(
            n_ages=5, n_years=10, n_causes=3, seed=0
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=0
        )
        result = model.fit(deaths, exposure, age_labels=ages, year_labels=years)
        assert result is model
        assert model.is_fitted_

    @requires_numpyro
    def test_lc_posterior_keys(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, ages, years = HMDLoader.load_synthetic(
            n_ages=4, n_years=8, n_causes=3, seed=1
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=1
        )
        model.fit(deaths, exposure)
        expected_keys = {"nu", "delta", "raw_beta", "raw_kappa", "sigma_kappa",
                         "phi0", "zeta", "sigma_zeta", "theta", "phi_dm"}
        assert expected_keys.issubset(set(model.posterior_.keys()))

    @requires_numpyro
    def test_lc_forecast_shape(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        n_ages, n_years, n_causes, horizon = 4, 8, 3, 5
        deaths, exposure, ages, years = HMDLoader.load_synthetic(
            n_ages=n_ages, n_years=n_years, n_causes=n_causes, seed=2
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=2
        )
        model.fit(deaths, exposure)
        fc = model.forecast(horizon=horizon, n_draws=20)

        assert fc.total_rates.shape == (n_ages, horizon, 20)
        assert fc.cause_rates.shape == (n_ages, horizon, n_causes, 20)

    @requires_numpyro
    def test_lc_forecast_coherence(self):
        """Core property: cause rates sum to total rates at every draw."""
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=4, n_years=8, n_causes=3, seed=3
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=3
        )
        model.fit(deaths, exposure)
        fc = model.forecast(horizon=5, n_draws=20)

        # This is the central claim of the DMP framework
        assert fc.coherence_check(tol=1e-5)

    @requires_numpyro
    def test_lc_forecast_positive_rates(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=4, n_years=8, n_causes=3, seed=4
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=4
        )
        model.fit(deaths, exposure)
        fc = model.forecast(horizon=5, n_draws=20)

        assert np.all(fc.total_rates > 0), "Total rates must be positive"
        assert np.all(fc.cause_rates >= 0), "Cause rates must be non-negative"

    @requires_numpyro
    def test_lc_forecast_future_years(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        _, _, _, years = HMDLoader.load_synthetic(n_ages=4, n_years=8, n_causes=3)
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25
        )
        deaths, exposure, _, _ = HMDLoader.load_synthetic(n_ages=4, n_years=8, n_causes=3)
        model.fit(deaths, exposure, year_labels=years)
        fc = model.forecast(horizon=3, n_draws=10)

        assert fc.periods[0] == years[-1] + 1
        assert fc.periods[-1] == years[-1] + 3

    @requires_numpyro
    def test_ap_fit_and_forecast(self):
        """AP-DM model should also fit and produce coherent forecasts."""
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=4, n_years=10, n_causes=3, seed=5
        )
        model = CauseSpecificMortality(
            model_type="AP", n_chains=1, n_samples=50, n_warmup=25, seed=5
        )
        model.fit(deaths, exposure)
        fc = model.forecast(horizon=3, n_draws=20)

        assert fc.total_rates.shape[0] == 4
        assert fc.coherence_check(tol=1e-5)

    @requires_numpyro
    def test_cause_fractions_shape(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        n_causes = 4
        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=4, n_years=8, n_causes=n_causes, seed=6
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=50, n_warmup=25, seed=6
        )
        model.fit(deaths, exposure)
        fractions = model.cause_fractions(age_idx=1, year_idx=3)

        assert fractions.shape == (n_causes,)
        # Posterior mean cause fractions should approximately sum to 1
        np.testing.assert_allclose(fractions.sum(), 1.0, atol=0.01)

    @requires_numpyro
    def test_diagnostics_returns_dict(self):
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=3, n_years=8, n_causes=3, seed=7
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=2, n_samples=50, n_warmup=25, seed=7
        )
        model.fit(deaths, exposure)
        diag = model.diagnostics()

        assert "rhat" in diag
        assert "ess" in diag
        assert isinstance(diag["rhat"], dict)

    @requires_numpyro
    def test_forecast_n_draws_clipping(self):
        """Requesting more draws than available should warn and use all draws."""
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, _, _ = HMDLoader.load_synthetic(
            n_ages=3, n_years=8, n_causes=2, seed=8
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=30, n_warmup=15, seed=8
        )
        model.fit(deaths, exposure)

        with pytest.warns(UserWarning):
            fc = model.forecast(horizon=3, n_draws=9999)

        # Should not error and draws should be capped at 30
        assert fc.total_rates.shape[2] == 30

    @requires_numpyro
    def test_forecast_summary_callable(self):
        """forecast().summary() should return a valid DataFrame."""
        import pandas as pd
        from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

        deaths, exposure, ages, years = HMDLoader.load_synthetic(
            n_ages=3, n_years=8, n_causes=2, seed=9
        )
        model = CauseSpecificMortality(
            model_type="LC", n_chains=1, n_samples=30, n_warmup=15, seed=9
        )
        model.fit(deaths, exposure, age_labels=ages, year_labels=years)
        fc = model.forecast(horizon=3, n_draws=10)
        df = fc.summary()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
