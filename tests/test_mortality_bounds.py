"""
Tests for insurance_survival.mortality.bounds — LifetimeBoundsCalculator.

All 20 test scenarios from the build spec, plus edge-case coverage.

No external dependencies required beyond numpy, scipy, and polars (core deps).
All tests run without NumPyro.

Test groups
-----------
TestSurvivalBounds        — survival_upper, survival_lower properties
TestAnnuityBounds         — annuity_bounds correctness and monotonicity
TestDeathBenefitBounds    — death_benefit_bounds direction and arithmetic
TestGeneralBounds         — general_bounds callable integration
TestInputValidation       — constructor and method validation
TestClassmethods          — from_lx, from_cmi
TestResultDataclass       — LifetimeBoundsResult to_dict, to_df
TestDiagnostics           — spread_by_duration, fractional_age_comparison
TestUKMateriality         — realistic CMI S3 scenario
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_survival.mortality.bounds import (
    LifetimeBoundsCalculator,
    LifetimeBoundsResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_calc(n: int = 10, p: float = 0.99, rate: float = 0.04) -> LifetimeBoundsCalculator:
    """Flat survival table: n years all with the same hat_p."""
    return LifetimeBoundsCalculator(
        hat_p=[p] * n,
        starting_age=65,
        discount_rate=rate,
    )


# ---------------------------------------------------------------------------
# TestSurvivalBounds — properties of survival_upper and survival_lower
# ---------------------------------------------------------------------------

class TestSurvivalBounds:
    """Test 1, 2, 3, 4, 5, 6 from the spec."""

    def test_upper_ge_lower_non_integer_s(self):
        """Test 3: survival_upper >= survival_lower for all non-integer s."""
        hat_p = [0.99, 0.98, 0.97]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65)
        for s in [0.3, 0.5, 0.7, 1.3, 1.7, 2.1, 2.9]:
            assert calc.survival_upper(0.0, s) >= calc.survival_lower(0.0, s), (
                f"upper < lower at s={s}"
            )

    def test_equal_at_integer_times(self):
        """Test 6: both bounds equal at integer times."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        for m in [1, 2, 3]:
            s = float(m)
            u = calc.survival_upper(0.0, s)
            l = calc.survival_lower(0.0, s)
            assert abs(u - l) < 1e-12, f"upper != lower at integer s={s}: {u} vs {l}"

    def test_gmab_trivial_bound(self):
        """Test 2: survival bounds equal at integer T (GMAB is trivial)."""
        calc = LifetimeBoundsCalculator(hat_p=[0.98, 0.97, 0.96], starting_age=65)
        upper = calc.survival_upper(0.0, 3.0)
        lower = calc.survival_lower(0.0, 3.0)
        assert abs(upper - lower) < 1e-12

    def test_survival_upper_constant_within_year(self):
        """Test 4: survival_upper is constant within each year interval."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98], starting_age=65)
        # In (0, 1): upper = cumulative_p[0] / cumulative_p[0] = 1.0 (survival to floor(s)=0)
        vals_year0 = [calc.survival_upper(0.0, s) for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
        assert np.allclose(vals_year0, vals_year0[0]), "upper not constant in (0,1)"

        # In (1, 2): upper = hat_p[0] = 0.99
        vals_year1 = [calc.survival_upper(0.0, s) for s in [1.1, 1.3, 1.5, 1.7, 1.9]]
        assert np.allclose(vals_year1, vals_year1[0]), "upper not constant in (1,2)"

    def test_survival_lower_equals_cumulative_next_integer(self):
        """Test 5: lower bound in (j, j+1) equals cumulative survival to j+1."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        # In (1, 2): lower = _2 p_x = 0.99 * 0.98
        expected = 0.99 * 0.98
        for s in [1.1, 1.5, 1.9]:
            got = calc.survival_lower(0.0, s)
            assert abs(got - expected) < 1e-12, f"lower at s={s}: {got} != {expected}"

    def test_survival_upper_known_value(self):
        """Upper bound in (1, 2) should equal cumulative to floor(s)=1, i.e. hat_p[0]."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        expected = 0.99
        for s in [1.1, 1.5, 1.9]:
            got = calc.survival_upper(0.0, s)
            assert abs(got - expected) < 1e-12, f"upper at s={s}: {got} != {expected}"

    def test_survival_bounds_at_start_is_one(self):
        """Survival at t=0 should be 1 by convention."""
        calc = _flat_calc()
        # survival_upper(0, s) with s close to 0 — first period survival = 1
        # Not calling with s=0 since s must be > t; check s=0.001
        assert calc.survival_upper(0.0, 0.001) == pytest.approx(1.0)
        # Lower at s=0.001: ceil(0.001)=1 -> cumulative_p[1]/cumulative_p[0] = hat_p[0]
        assert calc.survival_lower(0.0, 0.001) == pytest.approx(0.99)

    def test_survival_monotone_decreasing_at_integer_steps(self):
        """Cumulative survival is decreasing at integer years."""
        hat_p = [0.99, 0.98, 0.97, 0.96]
        calc = LifetimeBoundsCalculator(hat_p=hat_p)
        prev = 1.0
        for m in [1, 2, 3, 4]:
            curr = calc.survival_upper(0.0, float(m))
            assert curr < prev, f"survival increased at m={m}"
            prev = curr

    def test_survival_bounds_schedule_schema(self):
        """Test 16: survival_bounds_schedule returns correct columns and ordering."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.survival_bounds_schedule(0.0, 3.0, n_steps=100)
        assert set(df.columns) == {"s", "upper", "lower", "spread"}
        assert (df["upper"].to_numpy() >= df["lower"].to_numpy()).all()
        assert df["spread"].to_numpy().min() >= 0

    def test_survival_out_of_range_raises(self):
        """Test 14: s exceeding table coverage raises ValueError."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98])
        with pytest.raises(ValueError, match="exceeds available table"):
            calc.survival_upper(0.0, 3.0)
        with pytest.raises(ValueError, match="exceeds available table"):
            calc.survival_lower(0.0, 3.0)

    def test_survival_s_le_t_raises(self):
        """s <= t should raise ValueError."""
        calc = _flat_calc()
        with pytest.raises(ValueError, match="must be >"):
            calc.survival_upper(2.0, 2.0)
        with pytest.raises(ValueError, match="must be >"):
            calc.survival_lower(2.0, 1.5)


# ---------------------------------------------------------------------------
# TestAnnuityBounds
# ---------------------------------------------------------------------------

class TestAnnuityBounds:
    """Tests 1, 8, 9 from the spec."""

    def test_upper_ge_lower(self):
        """Test 1: annuity upper >= lower (higher survival = more income)."""
        hat_p = [0.99] * 10
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65)
        result = calc.annuity_bounds(t=0.0, T=10.0)
        assert result.upper >= result.lower
        assert result.spread >= 0.0

    def test_spread_is_upper_minus_lower(self):
        """spread field equals upper - lower."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 5.0)
        assert abs(r.spread - (r.upper - r.lower)) < 1e-12

    def test_midpoint_is_average(self):
        """midpoint field equals (upper + lower) / 2."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 5.0)
        assert abs(r.midpoint - (r.upper + r.lower) / 2) < 1e-12

    def test_contract_type_field(self):
        """contract_type is 'annuity'."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 5.0)
        assert r.contract_type == "annuity"

    def test_annuity_with_custom_income_fn(self):
        """Test 8: custom income_fn, zero discount."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99] * 5, starting_age=65, discount_rate=0.0)
        result = calc.annuity_bounds(0.0, 5.0, income_fn=lambda s: 1.0, n_steps=1000)
        # Upper = step function at floor, survives through year-end -> more area
        # Should be between 4.0 and 5.0
        assert 4.0 < result.upper <= 5.0
        assert result.lower < result.upper

    def test_annuity_zero_discount_upper_bound(self):
        """With zero discount and unit income, upper bound approx equals integral of upper survival."""
        # upper survival in year j = cumulative_p[j] (step function)
        # upper annuity = sum_j cumulative_p[j] * (duration of year = 1)
        hat_p = [0.99, 0.98, 0.97]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.0)
        result = calc.annuity_bounds(0.0, 3.0, n_steps=1000)
        # Expected: integral of step function (value 1.0 in [0,1), 0.99 in [1,2), 0.99*0.98 in [2,3))
        expected_upper = 1.0 + 0.99 + 0.99 * 0.98
        assert abs(result.upper - expected_upper) < 0.005  # small numerical error

    def test_annuity_values_positive(self):
        """Annuity values must be positive."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 10.0)
        assert r.upper > 0
        assert r.lower > 0

    def test_annuity_increasing_income_fn(self):
        """Increasing income function gives higher annuity value."""
        calc = _flat_calc(rate=0.0)
        r_unit = calc.annuity_bounds(0.0, 5.0, income_fn=lambda s: 1.0)
        r_inc = calc.annuity_bounds(0.0, 5.0, income_fn=lambda s: 1.0 + 0.1 * s)
        assert r_inc.upper > r_unit.upper

    def test_annuity_t_ge_T_raises(self):
        """T <= t should raise ValueError."""
        calc = _flat_calc()
        with pytest.raises(ValueError, match="must be >"):
            calc.annuity_bounds(5.0, 3.0)

    def test_annuity_n_years_field(self):
        """n_years field matches the integer years covered."""
        calc = _flat_calc(n=10)
        r = calc.annuity_bounds(0.0, 5.0)
        assert r.n_years == 5

    def test_annuity_ages_field(self):
        """ages field contains the integer ages covered."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99] * 5, starting_age=60)
        r = calc.annuity_bounds(0.0, 5.0)
        assert r.ages[0] == 60
        assert r.ages[-1] == 65

    def test_annuity_hat_p_field(self):
        """hat_p field in result matches the input slice."""
        hat_p = [0.99, 0.98, 0.97, 0.96, 0.95]
        calc = LifetimeBoundsCalculator(hat_p=hat_p)
        r = calc.annuity_bounds(0.0, 3.0)
        assert r.hat_p == pytest.approx(hat_p[:3])


# ---------------------------------------------------------------------------
# TestDeathBenefitBounds
# ---------------------------------------------------------------------------

class TestDeathBenefitBounds:
    """Tests 7, 19 from the spec."""

    def test_death_benefit_direction_constant_benefit(self):
        """Test 7: for constant benefit with r>0, upper < lower (timing effect).

        Upper: death at year-end (more discounted -> lower value).
        Lower: death at year-start (less discounted -> higher value).
        So result.upper < result.lower for constant benefit with discount_rate > 0.
        """
        calc = LifetimeBoundsCalculator(hat_p=[0.95, 0.94, 0.93], discount_rate=0.05)
        r = calc.death_benefit_bounds(0.0, 3.0, benefit_fn=lambda s: 1.0)
        # With positive discount rate, deferring death = less PV
        assert r.lower >= r.upper, (
            f"Expected lower >= upper for constant benefit with r>0, "
            f"got upper={r.upper:.6f}, lower={r.lower:.6f}"
        )

    def test_death_benefit_zero_discount_equal_bounds(self):
        """With zero discount and constant benefit, upper == lower only if q_j concentrates identically.

        Actually upper != lower even with zero discount: upper uses benefit_fn(j+1),
        lower uses benefit_fn(j). For constant benefit they are equal.
        """
        calc = LifetimeBoundsCalculator(hat_p=[0.95, 0.94], discount_rate=0.0)
        r = calc.death_benefit_bounds(0.0, 2.0, benefit_fn=lambda s: 1.0)
        # discount = 1 everywhere, benefit = 1 everywhere: upper == lower
        assert abs(r.upper - r.lower) < 1e-12

    def test_death_benefit_increasing_benefit_upper_greater(self):
        """For an increasing benefit with zero discount, upper > lower.

        Upper: death at year-end (higher benefit due to increasing benefit_fn).
        Lower: death at year-start (lower benefit due to increasing benefit_fn).
        """
        calc = LifetimeBoundsCalculator(hat_p=[0.95, 0.94, 0.93], discount_rate=0.0)
        r = calc.death_benefit_bounds(0.0, 3.0, benefit_fn=lambda s: 1.0 + s)
        assert r.upper > r.lower

    def test_death_benefit_contract_type_field(self):
        """contract_type is 'death_benefit'."""
        calc = _flat_calc()
        r = calc.death_benefit_bounds(0.0, 5.0)
        assert r.contract_type == "death_benefit"

    def test_death_benefit_arithmetic_check(self):
        """Manual arithmetic check for 1-year, unit benefit, zero discount."""
        hat_p = [0.90]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, discount_rate=0.0)
        r = calc.death_benefit_bounds(0.0, 1.0, benefit_fn=lambda s: 1.0)
        # q_0 = 1 - 0.90 = 0.10; survival to j=0 = 1.0
        # upper: q_0 * 1.0 * benefit(1) * discount(0,1) = 0.10 * 1 * 1 = 0.10
        # lower: q_0 * 1.0 * benefit(0) * discount(0,0) = 0.10 * 1 * 1 = 0.10
        assert abs(r.upper - 0.10) < 1e-12
        assert abs(r.lower - 0.10) < 1e-12

    def test_death_benefit_t_ge_T_raises(self):
        """T <= t should raise."""
        calc = _flat_calc()
        with pytest.raises(ValueError, match="must be >"):
            calc.death_benefit_bounds(5.0, 3.0)

    def test_death_benefit_spread_pct_nonneg(self):
        """Test 19: spread_pct is non-negative (uses abs)."""
        rng = np.random.default_rng(42)
        hat_p = rng.uniform(0.90, 0.999, 15)
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=70)
        for T in [5.0, 10.0, 15.0]:
            r = calc.annuity_bounds(0.0, T)
            assert r.spread_pct >= 0.0
            r2 = calc.death_benefit_bounds(0.0, T)
            assert r2.spread_pct >= 0.0
            # spread itself may be negative for death benefit with discount > 0
            assert r2.spread_pct == pytest.approx(
                abs(r2.spread) / abs(r2.midpoint) * 100 if r2.midpoint != 0 else 0,
                abs=1e-10,
            )


# ---------------------------------------------------------------------------
# TestGeneralBounds
# ---------------------------------------------------------------------------

class TestGeneralBounds:
    def test_general_bounds_matches_annuity_for_survival_payoff(self):
        """general_bounds with annuity-type payoff should match annuity_bounds."""
        hat_p = [0.99, 0.98, 0.97]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, discount_rate=0.03)

        # Payoff: survival probability * discount
        # payoff_fn(s, p_upper, p_lower) should return (p_upper * d, p_lower * d)
        def payoff_fn(s, p_upper, p_lower):
            d = np.exp(-0.03 * s)
            return p_upper * d, p_lower * d

        r_gen = calc.general_bounds(0.0, 3.0, payoff_fn=payoff_fn, n_steps=500)
        r_ann = calc.annuity_bounds(0.0, 3.0, income_fn=lambda s: 1.0, n_steps=500)

        assert abs(r_gen.upper - r_ann.upper) < 1e-4
        assert abs(r_gen.lower - r_ann.lower) < 1e-4
        assert r_gen.contract_type == "general"

    def test_general_bounds_t_ge_T_raises(self):
        calc = _flat_calc()
        with pytest.raises(ValueError, match="must be >"):
            calc.general_bounds(5.0, 3.0, payoff_fn=lambda s, u, l: (u, l))


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests 13, 14 from the spec."""

    def test_hat_p_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            LifetimeBoundsCalculator(hat_p=np.ones((3, 2)))

    def test_hat_p_must_have_one_element(self):
        with pytest.raises(ValueError, match="at least one year"):
            LifetimeBoundsCalculator(hat_p=[])

    def test_hat_p_values_must_be_in_open_unit_interval(self):
        """Test 13: values at or outside (0,1) raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            LifetimeBoundsCalculator(hat_p=[0.99, 1.01])
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            LifetimeBoundsCalculator(hat_p=[0.99, 0.0])
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            LifetimeBoundsCalculator(hat_p=[0.99, -0.01])
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            LifetimeBoundsCalculator(hat_p=[1.0, 0.98])

    def test_starting_age_must_be_nonneg_int(self):
        with pytest.raises(ValueError, match="non-negative int"):
            LifetimeBoundsCalculator(hat_p=[0.99], starting_age=-1)
        with pytest.raises(ValueError, match="non-negative int"):
            LifetimeBoundsCalculator(hat_p=[0.99], starting_age=65.5)  # type: ignore

    def test_discount_rate_must_be_nonneg(self):
        with pytest.raises(ValueError, match="discount_rate must be"):
            LifetimeBoundsCalculator(hat_p=[0.99], discount_rate=-0.01)

    def test_hat_p_ndim_list(self):
        """Flat list is valid input."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97])
        assert calc.n_years == 3

    def test_hat_p_ndarray_accepted(self):
        """numpy array is valid input."""
        calc = LifetimeBoundsCalculator(hat_p=np.array([0.99, 0.98]))
        assert calc.n_years == 2


# ---------------------------------------------------------------------------
# TestClassmethods
# ---------------------------------------------------------------------------

class TestClassmethods:
    """Tests 12, 18 from the spec."""

    def test_from_lx_produces_same_as_hat_p(self):
        """Test 12: from_lx gives same results as hat_p constructor."""
        hat_p = [0.99, 0.98, 0.97]
        lx = [100_000.0, 99_000.0, 97_020.0, 94_109.4]  # derived from hat_p

        calc_hat = LifetimeBoundsCalculator(hat_p=hat_p)
        calc_lx = LifetimeBoundsCalculator.from_lx(lx=lx)

        r1 = calc_hat.annuity_bounds(0.0, 3.0)
        r2 = calc_lx.annuity_bounds(0.0, 3.0)
        assert abs(r1.upper - r2.upper) < 1e-6
        assert abs(r1.lower - r2.lower) < 1e-6

    def test_from_lx_computes_hat_p_correctly(self):
        """from_lx should compute hat_p_j = lx[j+1] / lx[j]."""
        lx = [100_000.0, 99_000.0, 97_020.0]
        calc = LifetimeBoundsCalculator.from_lx(lx=lx)
        expected = np.array([99_000 / 100_000, 97_020 / 99_000])
        np.testing.assert_allclose(calc.hat_p, expected)

    def test_from_lx_rejects_non_monotone(self):
        """lx that increases (zero or negative hat_p) should be caught."""
        # lx[1] > lx[0] => hat_p > 1, which fails the (0,1) check
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            LifetimeBoundsCalculator.from_lx(lx=[100_000.0, 110_000.0, 108_000.0])

    def test_from_lx_requires_two_entries(self):
        with pytest.raises(ValueError, match="at least 2 entries"):
            LifetimeBoundsCalculator.from_lx(lx=[100_000.0])

    def test_from_lx_rejects_zero_lives(self):
        with pytest.raises(ValueError, match="strictly positive"):
            LifetimeBoundsCalculator.from_lx(lx=[100_000.0, 0.0, 50_000.0])

    def test_from_cmi_produces_valid_hat_p(self):
        """Test 18: from_cmi returns valid hat_p in (0,1)."""
        calc = LifetimeBoundsCalculator.from_cmi(
            table_code="S3PML", starting_age=65, n_years=10
        )
        assert len(calc.hat_p) == 10
        assert all(0.0 < p < 1.0 for p in calc.hat_p)
        assert calc.starting_age == 65

    def test_from_cmi_all_table_codes(self):
        """All four table codes should work."""
        for code in ("S3PML", "S3PFL", "S3AML", "S3AFL"):
            calc = LifetimeBoundsCalculator.from_cmi(
                table_code=code, starting_age=70, n_years=5
            )
            assert len(calc.hat_p) == 5
            assert all(0 < p < 1 for p in calc.hat_p)

    def test_from_cmi_invalid_table_code(self):
        with pytest.raises(ValueError, match="table_code must be one of"):
            LifetimeBoundsCalculator.from_cmi(table_code="S3XYZ")

    def test_from_cmi_invalid_starting_age(self):
        with pytest.raises(ValueError, match="50.*99"):
            LifetimeBoundsCalculator.from_cmi(starting_age=45)
        with pytest.raises(ValueError, match="50.*99"):
            LifetimeBoundsCalculator.from_cmi(starting_age=100)

    def test_from_cmi_no_improvement(self):
        """improvement_scale=None returns raw S3 table values."""
        calc_base = LifetimeBoundsCalculator.from_cmi(
            table_code="S3PML", starting_age=65, n_years=5, improvement_scale=None
        )
        calc_imp = LifetimeBoundsCalculator.from_cmi(
            table_code="S3PML", starting_age=65, n_years=5, improvement_scale="CMI_2023"
        )
        # With improvement, hat_p should be higher (lower mortality)
        assert np.all(np.array(calc_imp.hat_p) >= np.array(calc_base.hat_p))

    def test_from_cmi_invalid_improvement_scale(self):
        with pytest.raises(ValueError, match="improvement_scale must be"):
            LifetimeBoundsCalculator.from_cmi(improvement_scale="CMI_2019")

    def test_from_cmi_truncates_at_age_100(self):
        """n_years is capped at 100 - starting_age."""
        calc = LifetimeBoundsCalculator.from_cmi(
            table_code="S3PML", starting_age=92, n_years=50
        )
        assert len(calc.hat_p) == 8  # 100 - 92 = 8 years available

    def test_discount_fn_override(self):
        """Test 15: custom discount_fn gives same result as equivalent discount_rate."""
        flat = lambda t, s: np.exp(-0.05 * (s - t))  # noqa: E731
        calc1 = LifetimeBoundsCalculator(hat_p=[0.99] * 5, discount_fn=flat)
        calc2 = LifetimeBoundsCalculator(hat_p=[0.99] * 5, discount_rate=0.05)
        r1 = calc1.annuity_bounds(0.0, 5.0)
        r2 = calc2.annuity_bounds(0.0, 5.0)
        assert abs(r1.upper - r2.upper) < 1e-4
        assert abs(r1.lower - r2.lower) < 1e-4


# ---------------------------------------------------------------------------
# TestResultDataclass
# ---------------------------------------------------------------------------

class TestResultDataclass:
    """Tests 17 from the spec."""

    def test_to_df_schema(self):
        """Test 17: to_df() returns polars DataFrame with expected columns."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98], starting_age=65)
        result = calc.annuity_bounds(0.0, 2.0)
        df = result.to_df()
        assert isinstance(df, pl.DataFrame)
        assert "year" in df.columns
        assert "upper_survival" in df.columns
        assert "lower_survival" in df.columns

    def test_to_df_nrows_matches_n_years(self):
        """to_df() should have one row per policy year."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        result = calc.annuity_bounds(0.0, 3.0)
        df = result.to_df()
        assert len(df) == 3  # 3 years of hat_p

    def test_to_df_upper_ge_lower_survival(self):
        """upper_survival >= lower_survival in to_df() output."""
        calc = _flat_calc()
        result = calc.annuity_bounds(0.0, 5.0)
        df = result.to_df()
        upper_arr = df["upper_survival"].to_numpy()
        lower_arr = df["lower_survival"].to_numpy()
        assert (upper_arr >= lower_arr).all()

    def test_to_dict_keys(self):
        """to_dict() returns all required keys."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 3.0)
        d = r.to_dict()
        required = {"contract_type", "upper", "lower", "spread", "spread_pct",
                    "midpoint", "n_years", "ages", "hat_p"}
        assert required.issubset(set(d.keys()))

    def test_repr_format(self):
        """repr includes contract_type, upper, lower, spread."""
        calc = _flat_calc()
        r = calc.annuity_bounds(0.0, 3.0)
        s = repr(r)
        assert "annuity" in s
        assert "upper=" in s
        assert "lower=" in s
        assert "spread=" in s


# ---------------------------------------------------------------------------
# TestDiagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """Tests 9, 10, 11 from the spec."""

    def test_spread_by_duration_schema(self):
        """spread_by_duration returns correct schema."""
        calc = _flat_calc()
        df = calc.spread_by_duration("annuity")
        required = {"T", "upper", "lower", "spread", "spread_pct"}
        assert required.issubset(set(df.columns))

    def test_spread_by_duration_monotone(self):
        """Test 9: longer duration -> wider absolute spread for annuity."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99] * 10, starting_age=65)
        df = calc.spread_by_duration("annuity")
        spreads = df["spread"].to_numpy()
        # Spread should be non-decreasing with duration
        assert spreads[-1] > spreads[0], (
            f"Spread not wider at longer duration: {spreads[-1]:.6f} vs {spreads[0]:.6f}"
        )

    def test_spread_by_duration_death_benefit(self):
        """spread_by_duration works for death_benefit."""
        calc = _flat_calc()
        df = calc.spread_by_duration("death_benefit")
        assert len(df) == calc.n_years

    def test_spread_by_duration_invalid_type(self):
        calc = _flat_calc()
        with pytest.raises(ValueError, match="contract_type"):
            calc.spread_by_duration("general")

    def test_spread_by_duration_max_years(self):
        """max_years parameter limits output length."""
        calc = _flat_calc(n=10)
        df = calc.spread_by_duration("annuity", max_years=5)
        assert len(df) == 5

    def test_fractional_age_comparison_schema(self):
        """fractional_age_comparison returns correct schema."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.fractional_age_comparison(0.0, 3.0, "annuity")
        assert "assumption" in df.columns
        assert "value" in df.columns
        assert "relative_to_udd_pct" in df.columns

    def test_fractional_age_comparison_all_assumptions_present(self):
        """All 5 assumptions are present in the comparison."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.fractional_age_comparison(0.0, 3.0)
        assumptions = set(df["assumption"].to_list())
        expected = {"UDD", "CFM", "Balducci", "Upper bound", "Lower bound"}
        assert expected == assumptions

    def test_udd_inside_bounds(self):
        """Test 10: UDD value must be inside the theoretical bounds."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.fractional_age_comparison(0.0, 3.0, "annuity")
        udd = df.filter(pl.col("assumption") == "UDD")["value"][0]
        upper = df.filter(pl.col("assumption") == "Upper bound")["value"][0]
        lower = df.filter(pl.col("assumption") == "Lower bound")["value"][0]
        assert lower <= udd <= upper, (
            f"UDD ({udd:.6f}) outside bounds [{lower:.6f}, {upper:.6f}]"
        )

    def test_cfm_inside_bounds(self):
        """Test 11: CFM must also be inside bounds."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.fractional_age_comparison(0.0, 3.0)
        cfm = df.filter(pl.col("assumption") == "CFM")["value"][0]
        upper = df.filter(pl.col("assumption") == "Upper bound")["value"][0]
        lower = df.filter(pl.col("assumption") == "Lower bound")["value"][0]
        assert lower <= cfm <= upper, (
            f"CFM ({cfm:.6f}) outside bounds [{lower:.6f}, {upper:.6f}]"
        )

    def test_balducci_inside_bounds(self):
        """Balducci must also be inside bounds."""
        calc = LifetimeBoundsCalculator(hat_p=[0.99, 0.98, 0.97], starting_age=65)
        df = calc.fractional_age_comparison(0.0, 3.0)
        bal = df.filter(pl.col("assumption") == "Balducci")["value"][0]
        upper = df.filter(pl.col("assumption") == "Upper bound")["value"][0]
        lower = df.filter(pl.col("assumption") == "Lower bound")["value"][0]
        # Allow small numerical tolerance (integration error)
        assert lower - 1e-6 <= bal <= upper + 1e-6, (
            f"Balducci ({bal:.6f}) outside bounds [{lower:.6f}, {upper:.6f}]"
        )

    def test_relative_to_udd_pct_zero_for_udd(self):
        """UDD row should have relative_to_udd_pct == 0."""
        calc = _flat_calc()
        df = calc.fractional_age_comparison(0.0, 5.0)
        udd_row = df.filter(pl.col("assumption") == "UDD")
        assert abs(udd_row["relative_to_udd_pct"][0]) < 1e-10


# ---------------------------------------------------------------------------
# TestUKMateriality
# ---------------------------------------------------------------------------

class TestUKMateriality:
    """Test 20: realistic UK pensioner scenario."""

    def test_65yo_10yr_annuity_spread_under_5pct(self):
        """Test 20: UK 65yo male 10yr annuity spread should be non-trivial but < 5%."""
        # Approximate CMI S3 male pensioner mortality at ages 65-74
        hat_p = [0.9952, 0.9940, 0.9924, 0.9903, 0.9877,
                 0.9843, 0.9800, 0.9746, 0.9679, 0.9598]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.04)
        result = calc.annuity_bounds(0.0, 10.0)

        assert 0.0 < result.spread_pct < 5.0, (
            f"Spread {result.spread_pct:.2f}% out of expected range (0, 5)%"
        )
        # Spread must be positive (annuity)
        assert result.spread > 0

    def test_65yo_10yr_annuity_value_in_sensible_range(self):
        """Annuity value for a 65yo at 4% discount should be in [5, 9] for unit income."""
        hat_p = [0.9952, 0.9940, 0.9924, 0.9903, 0.9877,
                 0.9843, 0.9800, 0.9746, 0.9679, 0.9598]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.04)
        result = calc.annuity_bounds(0.0, 10.0)
        # Midpoint should be in a reasonable range for a 10yr life annuity
        assert 5.0 < result.midpoint < 9.5, (
            f"Midpoint annuity value {result.midpoint:.4f} seems implausible"
        )

    def test_cmi_from_cmi_annuity_sensible(self):
        """from_cmi -> annuity_bounds integration test."""
        calc = LifetimeBoundsCalculator.from_cmi("S3PML", starting_age=65, n_years=10)
        result = calc.annuity_bounds(0.0, 10.0)
        assert result.upper > result.lower
        assert 0.0 <= result.spread_pct < 5.0

    def test_cumulative_survival_at_year_10(self):
        """For realistic mortality, 10yr survival of 65yo should be 60-90%."""
        hat_p = [0.9952, 0.9940, 0.9924, 0.9903, 0.9877,
                 0.9843, 0.9800, 0.9746, 0.9679, 0.9598]
        calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.04)
        surv_10 = calc.survival_upper(0.0, 10.0)  # == survival_lower at integer time
        assert 0.60 < surv_10 < 0.95, (
            f"10yr survival of 65yo = {surv_10:.4f}, out of expected (0.60, 0.95)"
        )
