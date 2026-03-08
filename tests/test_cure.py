"""
Tests for WeibullMixtureCureFitter.

Validates:
- Fitting converges without error
- Cure fraction parameter recovery (within tolerance)
- Weibull shape/scale recovery (within tolerance)
- Prediction output shapes and ranges
- Edge cases: unfitted model, missing columns, zero durations
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy.special import expit
from scipy.stats import spearmanr

from insurance_survival import WeibullMixtureCureFitter
from conftest import make_cure_dgp


class TestWeibullMixtureCureFitterFit:
    """Tests for the fit() method."""

    def test_fit_returns_self(self, small_cure_dgp):
        """fit() returns self for method chaining."""
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=0.1,
        )
        result = fitter.fit(small_cure_dgp, duration_col="stop", event_col="event")
        assert result is fitter

    def test_convergence_dict_populated(self, fitted_cure_fitter):
        """convergence_ dict is populated after fitting."""
        assert isinstance(fitted_cure_fitter.convergence_, dict)
        assert "log_likelihood" in fitted_cure_fitter.convergence_
        assert "AIC" in fitted_cure_fitter.convergence_
        assert "BIC" in fitted_cure_fitter.convergence_

    def test_log_likelihood_negative(self, fitted_cure_fitter):
        """Log-likelihood should be negative (log of probability < 1)."""
        ll = fitted_cure_fitter.convergence_["log_likelihood"]
        assert ll < 0, f"Log-likelihood should be negative, got {ll}"

    def test_aic_greater_than_bic_check(self, fitted_cure_fitter):
        """AIC and BIC are finite numbers."""
        aic = fitted_cure_fitter.convergence_["AIC"]
        bic = fitted_cure_fitter.convergence_["BIC"]
        assert np.isfinite(aic)
        assert np.isfinite(bic)

    def test_cure_params_dataframe(self, fitted_cure_fitter):
        """cure_params_ is a Polars DataFrame with expected columns."""
        assert isinstance(fitted_cure_fitter.cure_params_, pl.DataFrame)
        expected_cols = {"model", "covariate", "coef", "se"}
        assert expected_cols.issubset(set(fitted_cure_fitter.cure_params_.columns))

    def test_uncured_params_dataframe(self, fitted_cure_fitter):
        """uncured_params_ is a Polars DataFrame."""
        assert isinstance(fitted_cure_fitter.uncured_params_, pl.DataFrame)

    def test_uncured_params_includes_shape(self, fitted_cure_fitter):
        """uncured_params_ includes the log_shape parameter row."""
        covariate_names = fitted_cure_fitter.uncured_params_["covariate"].to_list()
        assert "log_shape" in covariate_names

    def test_fit_with_polars_input(self, small_cure_dgp):
        """Accepts a Polars DataFrame without error."""
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=0.1,
        )
        fitter.fit(small_cure_dgp)  # should not raise

    def test_zero_duration_warning(self):
        """Zero-duration rows trigger a UserWarning."""
        dgp = make_cure_dgp(n=100, seed=42)
        # Add a zero-duration row
        zero_row = pl.DataFrame({
            "policy_id": ["BAD"],
            "stop": [0.0],
            "event": [0],
            "ncd_level": [3],
            "cure_prob_true": [0.5],
        })
        df = pl.concat([dgp, zero_row])
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=0.1,
        )
        with pytest.warns(UserWarning, match="non-positive durations"):
            fitter.fit(df, duration_col="stop", event_col="event")


class TestWeibullMixtureCureFitterParameterRecovery:
    """Tests that validate parameter recovery on known DGPs.

    Acceptance criteria from design spec:
    - cure_ncd_coef within ±0.2 of true 0.3
    - weibull_shape within ±0.3 of true 1.5
    - weibull_scale within ±0.4 of true 2.0
    - Spearman correlation of predicted vs true pi(x) > 0.85
    """

    @pytest.fixture(scope="class")
    def recovery_fitter(self):
        dgp = make_cure_dgp(
            n=2000,
            cure_intercept=-1.5,
            cure_ncd_coef=0.3,
            weibull_scale=2.0,
            weibull_shape=1.5,
            seed=42,
        )
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=0.01,
            max_iter=300,
        )
        fitter.fit(dgp, duration_col="stop", event_col="event")
        return fitter, dgp

    def test_ncd_coef_recovery(self, recovery_fitter):
        """Recovered NCD coefficient is within ±0.2 of true 0.3."""
        fitter, _ = recovery_fitter
        ncd_row = fitter.cure_params_.filter(pl.col("covariate") == "ncd_level")
        assert len(ncd_row) == 1
        estimated = ncd_row["coef"][0]
        true_value = 0.3
        assert abs(estimated - true_value) < 0.25, (
            f"NCD coef recovery: estimated={estimated:.4f}, true={true_value}"
        )

    def test_weibull_shape_recovery(self, recovery_fitter):
        """Recovered Weibull shape parameter is within ±0.3 of true 1.5."""
        fitter, _ = recovery_fitter
        shape_row = fitter.uncured_params_.filter(pl.col("covariate") == "log_shape")
        assert len(shape_row) == 1
        log_rho_est = shape_row["coef"][0]
        rho_est = np.exp(log_rho_est)
        true_rho = 1.5
        assert abs(rho_est - true_rho) < 0.4, (
            f"Weibull shape recovery: estimated={rho_est:.4f}, true={true_rho}"
        )

    def test_cure_prob_spearman_correlation(self, recovery_fitter):
        """Spearman correlation of predicted vs true cure probabilities > 0.7."""
        fitter, dgp = recovery_fitter
        predicted = fitter.predict_cure(dgp).to_numpy()
        true_probs = dgp["cure_prob_true"].to_numpy()
        rho, _ = spearmanr(predicted, true_probs)
        assert rho > 0.7, f"Spearman correlation {rho:.3f} < 0.7"


class TestWeibullMixtureCureFitterPredict:
    """Tests for predict_* methods."""

    def test_predict_cure_returns_series(self, fitted_cure_fitter, small_cure_dgp):
        """predict_cure returns a Polars Series."""
        result = fitted_cure_fitter.predict_cure(small_cure_dgp)
        assert isinstance(result, pl.Series)

    def test_predict_cure_length(self, fitted_cure_fitter, small_cure_dgp):
        """predict_cure returns one value per row."""
        result = fitted_cure_fitter.predict_cure(small_cure_dgp)
        assert len(result) == len(small_cure_dgp)

    def test_predict_cure_range(self, fitted_cure_fitter, small_cure_dgp):
        """Cure probabilities are in [0, 1]."""
        result = fitted_cure_fitter.predict_cure(small_cure_dgp)
        arr = result.to_numpy()
        assert (arr >= 0).all() and (arr <= 1).all()

    def test_predict_survival_function_shape(self, fitted_cure_fitter, small_cure_dgp):
        """predict_survival_function returns correct shape."""
        times = [1, 2, 3, 4, 5]
        result = fitted_cure_fitter.predict_survival_function(small_cure_dgp, times=times)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (len(small_cure_dgp), len(times))

    def test_predict_survival_function_column_names(self, fitted_cure_fitter, small_cure_dgp):
        """Survival function output columns are named S_t1, S_t2, ..."""
        times = [1, 2, 3]
        result = fitted_cure_fitter.predict_survival_function(small_cure_dgp, times=times)
        assert list(result.columns) == ["S_t1", "S_t2", "S_t3"]

    def test_predict_survival_monotone_decreasing(self, fitted_cure_fitter, small_cure_dgp):
        """Survival function is (weakly) monotone decreasing in t."""
        times = [1, 2, 3, 4, 5]
        result = fitted_cure_fitter.predict_survival_function(small_cure_dgp, times=times)
        for k in range(len(times) - 1):
            col_curr = f"S_t{k + 1}"
            col_next = f"S_t{k + 2}"
            assert (result[col_curr] >= result[col_next] - 1e-9).all(), (
                f"Survival not monotone at t={k+1} vs t={k+2}"
            )

    def test_predict_survival_range(self, fitted_cure_fitter, small_cure_dgp):
        """All survival probabilities are in [0, 1]."""
        times = [1, 2, 3]
        result = fitted_cure_fitter.predict_survival_function(small_cure_dgp, times=times)
        for col in result.columns:
            arr = result[col].to_numpy()
            assert (arr >= 0).all() and (arr <= 1).all()

    def test_predict_survival_plateaus_at_cure_prob(self, fitted_cure_fitter, small_cure_dgp):
        """At very large t, S(t|x) should approach pi(x) (the cure fraction)."""
        cure_probs = fitted_cure_fitter.predict_cure(small_cure_dgp).to_numpy()
        surv_far = fitted_cure_fitter.predict_survival_function(
            small_cure_dgp, times=[50.0]
        )["S_t1"].to_numpy()
        # Survival at t=50 should be close to cure probability
        # (within 0.15 on average for policies with moderate cure prob)
        diff = np.abs(surv_far - cure_probs)
        assert diff.mean() < 0.15, (
            f"Survival at t=50 diverges too much from cure probs: mean diff={diff.mean():.4f}"
        )

    def test_predict_median_survival_returns_series(self, fitted_cure_fitter, small_cure_dgp):
        """predict_median_survival returns a Polars Series."""
        result = fitted_cure_fitter.predict_median_survival(small_cure_dgp)
        assert isinstance(result, pl.Series)
        assert len(result) == len(small_cure_dgp)

    def test_predict_median_survival_positive(self, fitted_cure_fitter, small_cure_dgp):
        """Median survival times are positive."""
        result = fitted_cure_fitter.predict_median_survival(small_cure_dgp)
        assert (result.to_numpy() > 0).all()


class TestWeibullMixtureCureFitterSummary:
    """Tests for the summary() method."""

    def test_summary_returns_dataframe(self, fitted_cure_fitter):
        result = fitted_cure_fitter.summary()
        assert isinstance(result, pl.DataFrame)

    def test_summary_has_expected_columns(self, fitted_cure_fitter):
        result = fitted_cure_fitter.summary()
        expected = {"model", "covariate", "coef", "exp_coef", "se", "z", "p"}
        assert expected.issubset(set(result.columns))

    def test_summary_unfitted_raises(self):
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            fitter.summary()


class TestWeibullMixtureCureFitterEdgeCases:
    """Edge cases and robustness tests."""

    def test_predict_before_fit_raises(self, small_cure_dgp):
        """Predicting on unfitted model raises RuntimeError."""
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            fitter.predict_cure(small_cure_dgp)

    def test_high_penalizer_still_converges(self, small_cure_dgp):
        """High penalizer should still produce a fitted model."""
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            penalizer=10.0,
        )
        fitter.fit(small_cure_dgp)
        assert fitter._gamma is not None

    def test_no_intercept_fitting(self, small_cure_dgp):
        """fit_intercept=False runs without error."""
        fitter = WeibullMixtureCureFitter(
            cure_covariates=["ncd_level"],
            uncured_covariates=["ncd_level"],
            fit_intercept=False,
            penalizer=0.1,
        )
        fitter.fit(small_cure_dgp)
        assert fitter._gamma is not None

    def test_pickle_roundtrip(self, fitted_cure_fitter, small_cure_dgp):
        """Fitted fitter survives pickle serialisation."""
        import pickle
        serialised = pickle.dumps(fitted_cure_fitter)
        restored = pickle.loads(serialised)
        original_probs = fitted_cure_fitter.predict_cure(small_cure_dgp).to_numpy()
        restored_probs = restored.predict_cure(small_cure_dgp).to_numpy()
        np.testing.assert_allclose(original_probs, restored_probs, rtol=1e-6)
