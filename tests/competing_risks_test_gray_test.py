"""Tests for Gray's K-sample CIF test."""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.competing_risks.gray_test import GrayTestResult, gray_test
from insurance_survival.competing_risks.datasets import simulate_competing_risks


@pytest.fixture(scope="module")
def identical_groups_data():
    """Two groups drawn from the same distribution — should not reject H0."""
    rng = np.random.default_rng(0)
    n = 300
    T = rng.exponential(2.0, size=n)
    E = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
    groups = rng.choice(["A", "B"], size=n)
    return T, E, groups


@pytest.fixture(scope="module")
def different_groups_data():
    """Two groups with clearly different CIFs — should reject H0 at 5%."""
    rng = np.random.default_rng(7)
    n_per = 400
    # Group A: higher cause-1 rate
    T_a = rng.exponential(1.0, size=n_per)
    E_a = rng.choice([0, 1, 2], size=n_per, p=[0.2, 0.6, 0.2])
    # Group B: lower cause-1 rate
    T_b = rng.exponential(4.0, size=n_per)
    E_b = rng.choice([0, 1, 2], size=n_per, p=[0.4, 0.2, 0.4])
    T = np.concatenate([T_a, T_b])
    E = np.concatenate([E_a, E_b])
    groups = np.array(["A"] * n_per + ["B"] * n_per)
    return T, E, groups


class TestGrayTestResult:
    def test_repr_contains_statistic(self):
        result = GrayTestResult(statistic=5.0, p_value=0.025, degrees_of_freedom=1)
        assert "5.0" in repr(result) or "5.00" in repr(result)

    def test_significant_flag_true(self):
        result = GrayTestResult(statistic=10.0, p_value=0.01, degrees_of_freedom=1)
        assert result.significant is True

    def test_significant_flag_false(self):
        result = GrayTestResult(statistic=1.0, p_value=0.4, degrees_of_freedom=1)
        assert result.significant is False

    def test_significant_boundary(self):
        result = GrayTestResult(statistic=3.84, p_value=0.05, degrees_of_freedom=1)
        # Boundary: exactly 0.05 is not significant (strict <)
        assert result.significant is False


class TestGrayTest:
    def test_returns_gray_test_result(self, identical_groups_data):
        T, E, groups = identical_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert isinstance(result, GrayTestResult)

    def test_statistic_positive(self, different_groups_data):
        T, E, groups = different_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert result.statistic >= 0.0

    def test_p_value_in_range(self, identical_groups_data):
        T, E, groups = identical_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert 0.0 <= result.p_value <= 1.0

    def test_degrees_of_freedom_two_groups(self, identical_groups_data):
        T, E, groups = identical_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert result.degrees_of_freedom == 1

    def test_degrees_of_freedom_three_groups(self):
        rng = np.random.default_rng(1)
        n = 300
        T = rng.exponential(2.0, size=n)
        E = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        groups = rng.choice(["A", "B", "C"], size=n)
        result = gray_test(T, E, groups, event_of_interest=1)
        assert result.degrees_of_freedom == 2

    def test_high_power_with_different_groups(self, different_groups_data):
        """With strongly different groups, test should reject at 5%."""
        T, E, groups = different_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert result.p_value < 0.05, (
            f"Expected to reject H0 with different groups; "
            f"got p={result.p_value:.4f}"
        )

    def test_same_distribution_does_not_reject_reliably(self, identical_groups_data):
        """With identical groups, test should not reject at 1% significance."""
        T, E, groups = identical_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert result.p_value > 0.01, (
            f"Incorrectly rejected H0 for identical groups; "
            f"got p={result.p_value:.4f}"
        )

    def test_works_with_integer_groups(self):
        rng = np.random.default_rng(2)
        n = 200
        T = rng.exponential(2.0, size=n)
        E = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        groups = rng.choice([1, 2], size=n)
        result = gray_test(T, E, groups, event_of_interest=1)
        assert isinstance(result, GrayTestResult)

    def test_works_with_pandas_input(self):
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame({
            "T": rng.exponential(2.0, size=n),
            "E": rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3]),
            "group": rng.choice(["X", "Y"], size=n),
        })
        result = gray_test(df["T"], df["E"], df["group"], event_of_interest=1)
        assert isinstance(result, GrayTestResult)

    def test_single_group_raises(self):
        rng = np.random.default_rng(4)
        n = 100
        T = rng.exponential(2.0, size=n)
        E = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        groups = np.array(["A"] * n)
        with pytest.raises(ValueError):
            gray_test(T, E, groups, event_of_interest=1)

    def test_no_cause_k_events_raises(self):
        rng = np.random.default_rng(5)
        n = 100
        T = rng.exponential(2.0, size=n)
        E = np.zeros(n, dtype=int)  # all censored
        groups = rng.choice(["A", "B"], size=n)
        with pytest.raises(ValueError):
            gray_test(T, E, groups, event_of_interest=1)

    def test_cause_2_result(self, different_groups_data):
        T, E, groups = different_groups_data
        result = gray_test(T, E, groups, event_of_interest=2)
        assert isinstance(result, GrayTestResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_repr_contains_test_name(self, identical_groups_data):
        T, E, groups = identical_groups_data
        result = gray_test(T, E, groups, event_of_interest=1)
        assert "Gray" in repr(result)

    def test_test_name_includes_n_groups(self):
        rng = np.random.default_rng(6)
        n = 200
        T = rng.exponential(2.0, size=n)
        E = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        groups = rng.choice(["A", "B", "C"], size=n)
        result = gray_test(T, E, groups, event_of_interest=1)
        assert "3" in result.test_name
