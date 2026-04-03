"""
Tests for insurance_survival._utils — internal mathematical helpers.

These functions are not part of the public API but are used throughout the
library. Testing them directly catches regressions before they propagate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_survival._utils import (
    build_design_matrix,
    build_ncd_transition_matrix,
    coef_names,
    default_uk_ncd_transitions,
    expected_ncd_path,
    sigmoid,
    to_polars,
    weibull_median,
    weibull_pdf,
    weibull_sf,
)

# numpy<2.0 compat: trapezoid was added in 2.0
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero_input(self):
        """sigmoid(0) == 0.5."""
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-12

    def test_large_positive(self):
        """sigmoid(large) -> 1."""
        result = sigmoid(np.array([100.0]))
        assert result[0] > 0.999

    def test_large_negative(self):
        """sigmoid(large negative) -> 0."""
        result = sigmoid(np.array([-100.0]))
        assert result[0] < 0.001

    def test_monotone_increasing(self):
        """sigmoid should be strictly increasing."""
        x = np.linspace(-5, 5, 100)
        s = sigmoid(x)
        assert np.all(np.diff(s) > 0)

    def test_output_in_unit_interval(self):
        """All outputs are in (0, 1) for moderate x values."""
        # Use a moderate range to avoid float saturation at ±36+ where
        # sigmoid rounds to exactly 0.0 or 1.0 in float64.
        x = np.random.default_rng(0).uniform(-10, 10, 500)
        s = sigmoid(x)
        assert np.all(s > 0) and np.all(s < 1)

    def test_symmetry(self):
        """sigmoid(x) + sigmoid(-x) == 1."""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)

    def test_numerically_stable_negative(self):
        """Should not underflow for very negative inputs."""
        result = sigmoid(np.array([-500.0]))
        assert np.isfinite(result[0])
        assert result[0] >= 0.0


# ---------------------------------------------------------------------------
# weibull_sf
# ---------------------------------------------------------------------------

class TestWeibullSF:
    def test_at_zero(self):
        """S(0) = 1 for all shape/scale."""
        t = np.array([0.0])
        assert weibull_sf(t, scale=np.array([2.0]), shape=1.5)[0] == pytest.approx(1.0)

    def test_at_infinity_like(self):
        """S(very large t) -> 0."""
        t = np.array([1e10])
        s = weibull_sf(t, scale=np.array([2.0]), shape=1.5)[0]
        assert s < 1e-10

    def test_monotone_decreasing(self):
        """Survival function decreases with time."""
        t = np.linspace(0.01, 10.0, 200)
        s = weibull_sf(t, scale=np.array([3.0]), shape=2.0)
        assert np.all(np.diff(s) < 0)

    def test_shape_one_is_exponential(self):
        """With shape=1, Weibull is Exponential: S(t) = exp(-t/scale)."""
        t = np.array([1.0, 2.0, 3.0])
        scale = 2.0
        s = weibull_sf(t, scale=np.array([scale]), shape=1.0)
        expected = np.exp(-t / scale)
        np.testing.assert_allclose(s, expected, rtol=1e-10)

    def test_output_in_unit_interval(self):
        """S(t) always in [0, 1]."""
        t = np.linspace(0.0, 20.0, 100)
        s = weibull_sf(t, scale=np.array([5.0]), shape=2.0)
        assert np.all(s >= 0.0) and np.all(s <= 1.0)

    def test_vectorised_scale(self):
        """Accepts an array of scales matching t."""
        t = np.array([1.0, 2.0, 3.0])
        scale = np.array([1.0, 2.0, 3.0])
        s = weibull_sf(t, scale=scale, shape=2.0)
        # Each observation: S(t_i) = exp(-(t_i/scale_i)^2) = exp(-1) for all
        expected = np.exp(-np.ones(3))
        np.testing.assert_allclose(s, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# weibull_pdf
# ---------------------------------------------------------------------------

class TestWeibullPDF:
    def test_non_negative(self):
        """PDF is non-negative for t > 0."""
        t = np.linspace(0.01, 10.0, 200)
        f = weibull_pdf(t, scale=np.array([3.0]), shape=2.0)
        assert np.all(f >= 0.0)

    def test_integrates_to_one(self):
        """Numerical integral of PDF over large interval ~ 1."""
        t = np.linspace(0.001, 50.0, 5000)
        f = weibull_pdf(t, scale=np.array([5.0]), shape=2.0)
        integral = _trapz(f, t)
        assert abs(integral - 1.0) < 0.01

    def test_consistent_with_sf_derivative(self):
        """PDF(t) ~ -dS/dt numerically."""
        t = np.linspace(0.5, 5.0, 500)
        s = weibull_sf(t, scale=np.array([3.0]), shape=1.5)
        dS_dt = -np.gradient(s, t)
        f = weibull_pdf(t, scale=np.array([3.0]), shape=1.5)
        # Should agree in the middle (boundary effects at edges)
        np.testing.assert_allclose(f[10:-10], dS_dt[10:-10], rtol=0.02)


# ---------------------------------------------------------------------------
# weibull_median
# ---------------------------------------------------------------------------

class TestWeibullMedian:
    def test_definition(self):
        """Median satisfies S(median) = 0.5."""
        scale = np.array([3.0])
        shape = 2.0
        median = weibull_median(scale, shape)
        s_at_median = weibull_sf(median, scale=scale, shape=shape)
        np.testing.assert_allclose(s_at_median, np.array([0.5]), atol=1e-10)

    def test_increases_with_scale(self):
        """Larger scale -> larger median."""
        shape = 2.0
        scales = np.array([1.0, 2.0, 5.0])
        medians = weibull_median(scales, shape)
        assert np.all(np.diff(medians) > 0)

    def test_positive_and_finite_across_shapes(self):
        """Median is positive and finite for a range of shape parameters."""
        scale = np.array([1.0])
        shapes = [0.5, 1.0, 2.0, 5.0]
        # weibull_median returns array; access [0] for scalar comparison
        medians = [float(weibull_median(scale, sh)[0]) for sh in shapes]
        assert all(m > 0 and np.isfinite(m) for m in medians)

    def test_scalar_output(self):
        """Returns array matching input shape."""
        scale = np.array([2.0, 4.0])
        shape = 1.5
        result = weibull_median(scale, shape)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# to_polars
# ---------------------------------------------------------------------------

class TestToPolars:
    def test_polars_passthrough(self):
        """Polars DataFrame returned unchanged."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = to_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result is df

    def test_pandas_converted(self):
        """pandas DataFrame is converted to Polars."""
        import pandas as pd
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
        result = to_polars(df)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_unsupported_type_raises(self):
        """Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Expected"):
            to_polars([1, 2, 3])

    def test_dict_raises(self):
        """Dict raises TypeError (not a DataFrame)."""
        with pytest.raises(TypeError):
            to_polars({"a": [1, 2]})


# ---------------------------------------------------------------------------
# default_uk_ncd_transitions
# ---------------------------------------------------------------------------

class TestDefaultUKNCDTransitions:
    def test_returns_dataframe(self):
        result = default_uk_ncd_transitions()
        assert isinstance(result, pl.DataFrame)

    def test_default_columns(self):
        result = default_uk_ncd_transitions()
        assert set(result.columns) == {
            "from_ncd", "to_ncd_no_claim", "to_ncd_one_claim", "claim_probability"
        }

    def test_default_max_ncd(self):
        """Default max_ncd=9 gives 10 rows (NCD 0 through 9)."""
        result = default_uk_ncd_transitions()
        assert len(result) == 10

    def test_custom_max_ncd(self):
        result = default_uk_ncd_transitions(max_ncd=5)
        assert len(result) == 6
        assert result["from_ncd"].to_list() == list(range(6))

    def test_no_claim_increases_ncd(self):
        """No-claim transition should move NCD up by 1 (capped at max)."""
        result = default_uk_ncd_transitions(max_ncd=9)
        # NCD=3: no claim -> 4
        row = result.filter(pl.col("from_ncd") == 3).row(0, named=True)
        assert row["to_ncd_no_claim"] == 4

    def test_no_claim_at_max_stays_at_max(self):
        """At max NCD, no claim should stay at max."""
        result = default_uk_ncd_transitions(max_ncd=9)
        row = result.filter(pl.col("from_ncd") == 9).row(0, named=True)
        assert row["to_ncd_no_claim"] == 9

    def test_claim_drops_ncd_by_two(self):
        """One claim drops NCD by 2."""
        result = default_uk_ncd_transitions(max_ncd=9)
        row = result.filter(pl.col("from_ncd") == 5).row(0, named=True)
        assert row["to_ncd_one_claim"] == 3

    def test_claim_at_low_ncd_floors_at_zero(self):
        """NCD can't go below 0."""
        result = default_uk_ncd_transitions(max_ncd=9)
        row = result.filter(pl.col("from_ncd") == 1).row(0, named=True)
        assert row["to_ncd_one_claim"] == 0

    def test_claim_probability_default(self):
        """Default claim probability is 0.10."""
        result = default_uk_ncd_transitions()
        assert all(p == pytest.approx(0.10) for p in result["claim_probability"].to_list())

    def test_to_ncd_values_non_negative(self):
        result = default_uk_ncd_transitions()
        assert result["to_ncd_no_claim"].min() >= 0
        assert result["to_ncd_one_claim"].min() >= 0

    def test_to_ncd_values_bounded_by_max(self):
        max_ncd = 9
        result = default_uk_ncd_transitions(max_ncd=max_ncd)
        assert result["to_ncd_no_claim"].max() <= max_ncd
        assert result["to_ncd_one_claim"].max() <= max_ncd


# ---------------------------------------------------------------------------
# build_ncd_transition_matrix
# ---------------------------------------------------------------------------

class TestBuildNCDTransitionMatrix:
    def test_shape(self):
        """Matrix has shape (max_ncd+1, max_ncd+1)."""
        transitions = default_uk_ncd_transitions(max_ncd=9)
        M = build_ncd_transition_matrix(transitions, max_ncd=9)
        assert M.shape == (10, 10)

    def test_rows_sum_to_one(self):
        """Each row of the transition matrix sums to 1.0."""
        transitions = default_uk_ncd_transitions()
        M = build_ncd_transition_matrix(transitions, max_ncd=9)
        row_sums = M.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-12)

    def test_non_negative(self):
        """All entries non-negative."""
        transitions = default_uk_ncd_transitions()
        M = build_ncd_transition_matrix(transitions, max_ncd=9)
        assert np.all(M >= 0.0)

    def test_custom_claim_probability(self):
        """Custom claim probability should be reflected in matrix."""
        transitions = default_uk_ncd_transitions()
        # Override claim probability to 0.2 for all rows
        transitions = transitions.with_columns(pl.lit(0.2).alias("claim_probability"))
        M = build_ncd_transition_matrix(transitions, max_ncd=9)
        # Row 5 (NCD=5): no claim -> 6 (p=0.8), claim -> 3 (p=0.2)
        assert M[5, 6] == pytest.approx(0.8)
        assert M[5, 3] == pytest.approx(0.2)

    def test_known_entry_ncd5(self):
        """NCD=5, no claim -> 6 (p=0.9), claim -> 3 (p=0.1) by default."""
        transitions = default_uk_ncd_transitions()
        M = build_ncd_transition_matrix(transitions, max_ncd=9)
        assert M[5, 6] == pytest.approx(0.9)
        assert M[5, 3] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# expected_ncd_path
# ---------------------------------------------------------------------------

class TestExpectedNCDPath:
    @pytest.fixture
    def transition_matrix(self):
        transitions = default_uk_ncd_transitions()
        return build_ncd_transition_matrix(transitions, max_ncd=9)

    def test_output_shape(self, transition_matrix):
        """Returns array of length horizon."""
        path = expected_ncd_path(ncd_0=5, horizon=10, transition_matrix=transition_matrix)
        assert path.shape == (10,)

    def test_values_in_valid_range(self, transition_matrix):
        """Expected NCD should stay within [0, max_ncd]."""
        path = expected_ncd_path(ncd_0=0, horizon=20, transition_matrix=transition_matrix)
        assert np.all(path >= 0.0)
        assert np.all(path <= 9.0)

    def test_starting_from_zero_increases(self, transition_matrix):
        """NCD=0 with 10% claim probability: expected NCD should increase over time."""
        path = expected_ncd_path(ncd_0=0, horizon=5, transition_matrix=transition_matrix)
        # With p_claim=0.1, most transitions are upward, so expected NCD increases
        assert path[-1] > path[0]

    def test_starting_from_max_stays_near_max(self, transition_matrix):
        """Starting at max NCD, expected NCD stays high."""
        path = expected_ncd_path(ncd_0=9, horizon=5, transition_matrix=transition_matrix)
        # With only 10% claim probability, NCD should stay near 9
        assert path[-1] > 7.0

    def test_horizon_zero(self, transition_matrix):
        """Horizon=0 returns empty array."""
        path = expected_ncd_path(ncd_0=5, horizon=0, transition_matrix=transition_matrix)
        assert len(path) == 0

    def test_clamped_ncd_0_exceeds_max(self, transition_matrix):
        """ncd_0 > max_ncd is clamped to max_ncd."""
        path = expected_ncd_path(ncd_0=100, horizon=1, transition_matrix=transition_matrix)
        assert path.shape == (1,)
        assert np.isfinite(path[0])


# ---------------------------------------------------------------------------
# build_design_matrix
# ---------------------------------------------------------------------------

class TestBuildDesignMatrix:
    @pytest.fixture
    def sample_df(self):
        return pl.DataFrame({
            "age": [25.0, 35.0, 45.0],
            "ncd": [0.0, 5.0, 9.0],
            "tenure": [1.0, 3.0, 7.0],
        })

    def test_with_intercept_shape(self, sample_df):
        """With intercept: output has n_rows x (n_covs + 1) shape."""
        X = build_design_matrix(sample_df, ["age", "ncd"], fit_intercept=True)
        assert X.shape == (3, 3)

    def test_without_intercept_shape(self, sample_df):
        """Without intercept: output has n_rows x n_covs shape."""
        X = build_design_matrix(sample_df, ["age", "ncd"], fit_intercept=False)
        assert X.shape == (3, 2)

    def test_intercept_column_is_ones(self, sample_df):
        """First column is all 1s when fit_intercept=True."""
        X = build_design_matrix(sample_df, ["age"], fit_intercept=True)
        np.testing.assert_array_equal(X[:, 0], np.ones(3))

    def test_covariate_values_correct(self, sample_df):
        """Covariate values match source DataFrame."""
        X = build_design_matrix(sample_df, ["age", "ncd"], fit_intercept=False)
        expected_age = np.array([25.0, 35.0, 45.0])
        np.testing.assert_array_equal(X[:, 0], expected_age)

    def test_single_covariate(self, sample_df):
        X = build_design_matrix(sample_df, ["tenure"], fit_intercept=True)
        assert X.shape == (3, 2)
        expected_tenure = np.array([1.0, 3.0, 7.0])
        np.testing.assert_array_equal(X[:, 1], expected_tenure)

    def test_output_dtype_float(self, sample_df):
        """Output is always float array."""
        X = build_design_matrix(sample_df, ["age"], fit_intercept=True)
        assert X.dtype == float

    def test_three_covariates(self, sample_df):
        X = build_design_matrix(sample_df, ["age", "ncd", "tenure"], fit_intercept=True)
        assert X.shape == (3, 4)


# ---------------------------------------------------------------------------
# coef_names
# ---------------------------------------------------------------------------

class TestCoefNames:
    def test_with_intercept(self):
        names = coef_names(["age", "ncd"], fit_intercept=True)
        assert names == ["Intercept", "age", "ncd"]

    def test_without_intercept(self):
        names = coef_names(["age", "ncd"], fit_intercept=False)
        assert names == ["age", "ncd"]

    def test_empty_covariates_with_intercept(self):
        names = coef_names([], fit_intercept=True)
        assert names == ["Intercept"]

    def test_empty_covariates_without_intercept(self):
        names = coef_names([], fit_intercept=False)
        assert names == []

    def test_single_covariate(self):
        names = coef_names(["x"], fit_intercept=True)
        assert names == ["Intercept", "x"]

    def test_returns_list(self):
        names = coef_names(["a", "b", "c"], fit_intercept=False)
        assert isinstance(names, list)
