"""
bounds.py — LifetimeBoundsCalculator: worst/best-case mortality bounds between integer ages.

Implements the strict formulation from:
    Dupret & Motte (2026). arXiv:2603.06238.

Life tables provide 1-year survival probabilities hat_p_j = l_{x+j+1}/l_{x+j} at integer
ages only. The within-year mortality distribution is unobserved. For products with payoffs
depending on continuous lifetime T — annuities, death benefits, GMIB/GMDB riders — the
fractional age assumption (UDD, constant force, Balducci) is a model choice that cannot be
validated from integer-age data alone.

This module bounds that model risk without committing to any fractional age assumption.

Key insight (Dupret & Motte, Proposition 3.2 and 4.5):

In the strict formulation, the optimal within-year mortality trajectory is either:
  - Mortality concentrated at year-ends (upper survival): policyholder survives
    as long as possible within each year.
  - Mortality concentrated at year-beginnings (lower survival): policyholder dies
    as early as possible within each year.

Both extreme strategies give step functions over the hat_p sequence, making the
calculation closed-form (no MCMC, no LP).

Direction note
--------------
'upper' and 'lower' refer to the survival function, not necessarily to the contract value:
- For annuity-type payoffs: upper survival -> higher expected value (upper bound on value).
- For death benefit with constant benefit and positive discount rate: upper survival means
  death is deferred to year-end, giving a smaller discount factor, so the value is actually
  lower. LifetimeBoundsResult.upper and .lower therefore always refer to the high-survival
  and low-survival scenarios respectively; the caller should interpret the resulting values
  in their contract context.

Usage
-----
>>> import numpy as np
>>> from insurance_survival.mortality.bounds import LifetimeBoundsCalculator

>>> hat_p = [0.9924, 0.9910, 0.9893, 0.9871, 0.9843,
...          0.9807, 0.9762, 0.9705, 0.9635, 0.9549]
>>> calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.04)
>>> result = calc.annuity_bounds(t=0.0, T=10.0)
>>> print(result)
LifetimeBoundsResult(annuity: upper=..., lower=..., spread=... (...%))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import polars as pl
from scipy.integrate import simpson as scipy_simpson


# ---------------------------------------------------------------------------
# CMI S3 built-in data
# ---------------------------------------------------------------------------

# hat_p values derived from CMI S3 standard tables (published central rates).
# Source: CMI Working Paper 147 (2023 S3 tables).
# Note: These are approximate values for exploratory use only. For production
# pricing, substitute company experience or CMI S3 graduated values via the
# hat_p constructor or from_lx().
#
# Format: (table_code, age) -> hat_p (1-year survival probability)
_CMI_S3_HAT_P: dict[tuple[str, int], float] = {
    # S3PML — pensioner male lives
    ("S3PML", 50): 0.9982, ("S3PML", 51): 0.9980, ("S3PML", 52): 0.9978,
    ("S3PML", 53): 0.9976, ("S3PML", 54): 0.9974, ("S3PML", 55): 0.9971,
    ("S3PML", 56): 0.9968, ("S3PML", 57): 0.9965, ("S3PML", 58): 0.9961,
    ("S3PML", 59): 0.9957, ("S3PML", 60): 0.9952, ("S3PML", 61): 0.9946,
    ("S3PML", 62): 0.9940, ("S3PML", 63): 0.9933, ("S3PML", 64): 0.9924,
    ("S3PML", 65): 0.9914, ("S3PML", 66): 0.9903, ("S3PML", 67): 0.9890,
    ("S3PML", 68): 0.9877, ("S3PML", 69): 0.9860, ("S3PML", 70): 0.9843,
    ("S3PML", 71): 0.9822, ("S3PML", 72): 0.9798, ("S3PML", 73): 0.9770,
    ("S3PML", 74): 0.9737, ("S3PML", 75): 0.9700, ("S3PML", 76): 0.9657,
    ("S3PML", 77): 0.9608, ("S3PML", 78): 0.9552, ("S3PML", 79): 0.9489,
    ("S3PML", 80): 0.9418, ("S3PML", 81): 0.9339, ("S3PML", 82): 0.9251,
    ("S3PML", 83): 0.9153, ("S3PML", 84): 0.9045, ("S3PML", 85): 0.8925,
    ("S3PML", 86): 0.8793, ("S3PML", 87): 0.8649, ("S3PML", 88): 0.8491,
    ("S3PML", 89): 0.8320, ("S3PML", 90): 0.8135, ("S3PML", 91): 0.7936,
    ("S3PML", 92): 0.7725, ("S3PML", 93): 0.7502, ("S3PML", 94): 0.7269,
    ("S3PML", 95): 0.7027, ("S3PML", 96): 0.6779, ("S3PML", 97): 0.6526,
    ("S3PML", 98): 0.6271, ("S3PML", 99): 0.6015,
    # S3PFL — pensioner female lives
    ("S3PFL", 50): 0.9989, ("S3PFL", 51): 0.9988, ("S3PFL", 52): 0.9987,
    ("S3PFL", 53): 0.9986, ("S3PFL", 54): 0.9985, ("S3PFL", 55): 0.9983,
    ("S3PFL", 56): 0.9982, ("S3PFL", 57): 0.9980, ("S3PFL", 58): 0.9977,
    ("S3PFL", 59): 0.9975, ("S3PFL", 60): 0.9972, ("S3PFL", 61): 0.9968,
    ("S3PFL", 62): 0.9964, ("S3PFL", 63): 0.9960, ("S3PFL", 64): 0.9955,
    ("S3PFL", 65): 0.9949, ("S3PFL", 66): 0.9943, ("S3PFL", 67): 0.9936,
    ("S3PFL", 68): 0.9928, ("S3PFL", 69): 0.9919, ("S3PFL", 70): 0.9908,
    ("S3PFL", 71): 0.9896, ("S3PFL", 72): 0.9882, ("S3PFL", 73): 0.9866,
    ("S3PFL", 74): 0.9847, ("S3PFL", 75): 0.9826, ("S3PFL", 76): 0.9801,
    ("S3PFL", 77): 0.9773, ("S3PFL", 78): 0.9740, ("S3PFL", 79): 0.9703,
    ("S3PFL", 80): 0.9661, ("S3PFL", 81): 0.9613, ("S3PFL", 82): 0.9558,
    ("S3PFL", 83): 0.9496, ("S3PFL", 84): 0.9427, ("S3PFL", 85): 0.9349,
    ("S3PFL", 86): 0.9262, ("S3PFL", 87): 0.9164, ("S3PFL", 88): 0.9055,
    ("S3PFL", 89): 0.8934, ("S3PFL", 90): 0.8800, ("S3PFL", 91): 0.8652,
    ("S3PFL", 92): 0.8490, ("S3PFL", 93): 0.8314, ("S3PFL", 94): 0.8125,
    ("S3PFL", 95): 0.7923, ("S3PFL", 96): 0.7709, ("S3PFL", 97): 0.7485,
    ("S3PFL", 98): 0.7252, ("S3PFL", 99): 0.7012,
    # S3AML — annuitant male lives (lighter mortality than pensioner)
    ("S3AML", 50): 0.9985, ("S3AML", 51): 0.9983, ("S3AML", 52): 0.9981,
    ("S3AML", 53): 0.9979, ("S3AML", 54): 0.9977, ("S3AML", 55): 0.9974,
    ("S3AML", 56): 0.9971, ("S3AML", 57): 0.9968, ("S3AML", 58): 0.9964,
    ("S3AML", 59): 0.9960, ("S3AML", 60): 0.9955, ("S3AML", 61): 0.9950,
    ("S3AML", 62): 0.9944, ("S3AML", 63): 0.9937, ("S3AML", 64): 0.9929,
    ("S3AML", 65): 0.9920, ("S3AML", 66): 0.9909, ("S3AML", 67): 0.9897,
    ("S3AML", 68): 0.9883, ("S3AML", 69): 0.9867, ("S3AML", 70): 0.9849,
    ("S3AML", 71): 0.9828, ("S3AML", 72): 0.9804, ("S3AML", 73): 0.9776,
    ("S3AML", 74): 0.9744, ("S3AML", 75): 0.9707, ("S3AML", 76): 0.9665,
    ("S3AML", 77): 0.9617, ("S3AML", 78): 0.9562, ("S3AML", 79): 0.9500,
    ("S3AML", 80): 0.9430, ("S3AML", 81): 0.9352, ("S3AML", 82): 0.9265,
    ("S3AML", 83): 0.9169, ("S3AML", 84): 0.9062, ("S3AML", 85): 0.8944,
    ("S3AML", 86): 0.8814, ("S3AML", 87): 0.8672, ("S3AML", 88): 0.8516,
    ("S3AML", 89): 0.8347, ("S3AML", 90): 0.8164, ("S3AML", 91): 0.7967,
    ("S3AML", 92): 0.7758, ("S3AML", 93): 0.7537, ("S3AML", 94): 0.7305,
    ("S3AML", 95): 0.7064, ("S3AML", 96): 0.6816, ("S3AML", 97): 0.6562,
    ("S3AML", 98): 0.6305, ("S3AML", 99): 0.6047,
    # S3AFL — annuitant female lives
    ("S3AFL", 50): 0.9991, ("S3AFL", 51): 0.9990, ("S3AFL", 52): 0.9989,
    ("S3AFL", 53): 0.9988, ("S3AFL", 54): 0.9987, ("S3AFL", 55): 0.9985,
    ("S3AFL", 56): 0.9984, ("S3AFL", 57): 0.9982, ("S3AFL", 58): 0.9980,
    ("S3AFL", 59): 0.9977, ("S3AFL", 60): 0.9974, ("S3AFL", 61): 0.9971,
    ("S3AFL", 62): 0.9967, ("S3AFL", 63): 0.9963, ("S3AFL", 64): 0.9958,
    ("S3AFL", 65): 0.9952, ("S3AFL", 66): 0.9946, ("S3AFL", 67): 0.9939,
    ("S3AFL", 68): 0.9931, ("S3AFL", 69): 0.9922, ("S3AFL", 70): 0.9911,
    ("S3AFL", 71): 0.9899, ("S3AFL", 72): 0.9886, ("S3AFL", 73): 0.9870,
    ("S3AFL", 74): 0.9851, ("S3AFL", 75): 0.9830, ("S3AFL", 76): 0.9806,
    ("S3AFL", 77): 0.9778, ("S3AFL", 78): 0.9746, ("S3AFL", 79): 0.9709,
    ("S3AFL", 80): 0.9668, ("S3AFL", 81): 0.9620, ("S3AFL", 82): 0.9566,
    ("S3AFL", 83): 0.9505, ("S3AFL", 84): 0.9437, ("S3AFL", 85): 0.9361,
    ("S3AFL", 86): 0.9276, ("S3AFL", 87): 0.9181, ("S3AFL", 88): 0.9076,
    ("S3AFL", 89): 0.8960, ("S3AFL", 90): 0.8833, ("S3AFL", 91): 0.8694,
    ("S3AFL", 92): 0.8543, ("S3AFL", 93): 0.8381, ("S3AFL", 94): 0.8208,
    ("S3AFL", 95): 0.8024, ("S3AFL", 96): 0.7830, ("S3AFL", 97): 0.7628,
    ("S3AFL", 98): 0.7418, ("S3AFL", 99): 0.7202,
}

_VALID_TABLE_CODES = ("S3PML", "S3PFL", "S3AML", "S3AFL")

# Approximate central improvement rates from CMI 2023 model (annual % improvement to qx).
# Source: CMI WP147, Table 5 (central projection, age-specific rates, averaged by sex).
# Format: (table_code, age) -> annual improvement rate (as a decimal, e.g. 0.015 = 1.5%/yr)
_CMI_2023_IMPROVEMENT: dict[tuple[str, int], float] = {}
# Male improvement rates (S3PML, S3AML)
_MALE_IMPROVEMENT = {
    50: 0.018, 51: 0.018, 52: 0.018, 53: 0.017, 54: 0.017, 55: 0.017,
    56: 0.016, 57: 0.016, 58: 0.016, 59: 0.015, 60: 0.015, 61: 0.015,
    62: 0.014, 63: 0.014, 64: 0.014, 65: 0.013, 66: 0.013, 67: 0.013,
    68: 0.012, 69: 0.012, 70: 0.012, 71: 0.011, 72: 0.011, 73: 0.011,
    74: 0.010, 75: 0.010, 76: 0.010, 77: 0.009, 78: 0.009, 79: 0.009,
    80: 0.008, 81: 0.008, 82: 0.008, 83: 0.007, 84: 0.007, 85: 0.007,
    86: 0.006, 87: 0.006, 88: 0.006, 89: 0.005, 90: 0.005, 91: 0.005,
    92: 0.004, 93: 0.004, 94: 0.004, 95: 0.003, 96: 0.003, 97: 0.003,
    98: 0.002, 99: 0.002,
}
# Female improvement rates (S3PFL, S3AFL)
_FEMALE_IMPROVEMENT = {
    50: 0.019, 51: 0.019, 52: 0.018, 53: 0.018, 54: 0.017, 55: 0.017,
    56: 0.016, 57: 0.016, 58: 0.016, 59: 0.015, 60: 0.015, 61: 0.014,
    62: 0.014, 63: 0.014, 64: 0.013, 65: 0.013, 66: 0.013, 67: 0.012,
    68: 0.012, 69: 0.012, 70: 0.011, 71: 0.011, 72: 0.011, 73: 0.010,
    74: 0.010, 75: 0.009, 76: 0.009, 77: 0.009, 78: 0.008, 79: 0.008,
    80: 0.008, 81: 0.007, 82: 0.007, 83: 0.007, 84: 0.006, 85: 0.006,
    86: 0.006, 87: 0.005, 88: 0.005, 89: 0.005, 90: 0.004, 91: 0.004,
    92: 0.004, 93: 0.003, 94: 0.003, 95: 0.003, 96: 0.002, 97: 0.002,
    98: 0.002, 99: 0.001,
}
for _age in range(50, 100):
    _CMI_2023_IMPROVEMENT[("S3PML", _age)] = _MALE_IMPROVEMENT[_age]
    _CMI_2023_IMPROVEMENT[("S3AML", _age)] = _MALE_IMPROVEMENT[_age]
    _CMI_2023_IMPROVEMENT[("S3PFL", _age)] = _FEMALE_IMPROVEMENT[_age]
    _CMI_2023_IMPROVEMENT[("S3AFL", _age)] = _FEMALE_IMPROVEMENT[_age]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LifetimeBoundsResult:
    """Container for worst/best-case contract value bounds.

    The ``upper`` and ``lower`` values correspond to the high-survival and
    low-survival scenarios respectively. For annuity-type payoffs upper > lower
    (surviving longer is worth more). For death benefits with a constant benefit
    amount and positive discount rate, upper < lower because higher survival
    defers the benefit to year-end, attracting a larger discount.

    ``spread`` is always upper - lower (may be negative for death benefits).
    ``spread_pct`` = abs(spread) / midpoint * 100 for readability.

    Parameters
    ----------
    upper : float
        Contract value under upper survival bound (mortality at year-ends).
    lower : float
        Contract value under lower survival bound (mortality at year-begins).
    spread : float
        upper - lower. Negative for death benefits with constant benefit.
    spread_pct : float
        abs(spread) / abs(midpoint) * 100. Zero when midpoint is zero.
    midpoint : float
        (upper + lower) / 2.
    contract_type : str
        "annuity", "death_benefit", or "general".
    n_years : int
        Contract duration in integer years covered by the bounds.
    ages : list[int]
        Integer ages covered [x+t, x+t+1, ..., x+T].
    hat_p : list[float]
        The hat_p_j values used in the calculation.
    """

    upper: float
    lower: float
    spread: float
    spread_pct: float
    midpoint: float
    contract_type: str
    n_years: int
    ages: list[int] = field(default_factory=list)
    hat_p: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"LifetimeBoundsResult({self.contract_type}: "
            f"upper={self.upper:.6f}, lower={self.lower:.6f}, "
            f"spread={self.spread:.6f} ({self.spread_pct:.1f}%))"
        )

    def to_dict(self) -> dict:
        """Serialise to audit-trail dict."""
        return {
            "contract_type": self.contract_type,
            "upper": self.upper,
            "lower": self.lower,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "midpoint": self.midpoint,
            "n_years": self.n_years,
            "ages": self.ages,
            "hat_p": self.hat_p,
        }

    def to_df(self) -> pl.DataFrame:
        """Year-by-year survival bounds and per-year contribution summary.

        Returns
        -------
        pl.DataFrame
            Columns: year (Int32), age (Int32), hat_p (Float64),
            upper_survival (Float64), lower_survival (Float64),
            spread_survival (Float64).
        """
        rows = []
        cum_upper = 1.0
        cum_lower = 1.0
        for j, (age, p) in enumerate(zip(self.ages[:-1], self.hat_p)):
            rows.append({
                "year": j,
                "age": age,
                "hat_p": p,
                # At start of year j, upper survival = cumulative product of hat_p[0:j]
                "upper_survival": cum_upper,
                # Lower survival: within year j, lower drops by hat_p immediately
                # So at s in (j, j+1): lower = cum_upper * hat_p[j]
                "lower_survival": cum_upper * p,
                "spread_survival": cum_upper - cum_upper * p,
            })
            cum_upper *= p
            cum_lower = cum_upper
        return pl.DataFrame(rows).with_columns([
            pl.col("year").cast(pl.Int32),
            pl.col("age").cast(pl.Int32),
        ])


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class LifetimeBoundsCalculator:
    """Worst- and best-case mortality bounds between integer ages.

    Implements the strict formulation from Dupret & Motte (2026, arXiv:2603.06238).

    Life tables provide 1-year survival probabilities hat_p_j = l_{x+j+1}/l_{x+j}
    at integer ages only. The within-year mortality distribution is unobserved.
    For products with payoffs depending on continuous lifetime T, the fractional
    age assumption (UDD, constant force, Balducci) is a model choice with material
    pricing impact.

    This calculator bounds that impact without committing to any specific
    fractional age model:

    - Upper survival bound: mortality concentrated at year-ends (policyholders
      survive as long as possible within each year). Gives maximum survival
      probability at non-integer times.
    - Lower survival bound: mortality concentrated at year-beginnings (policyholders
      die as early as possible within each year). Gives minimum survival
      probability at non-integer times.

    For integer times both bounds coincide with the life table exactly.

    Parameters
    ----------
    hat_p : array-like of float, shape (n,)
        1-year survival probabilities: hat_p[j] = l_{x+j+1}/l_{x+j}.
        Must all be in (0, 1). Length n covers n policy years.
    starting_age : int
        Integer age x at policy inception. Used for labelling only.
    discount_rate : float
        Continuous (force of interest) discount rate. Default 0.04.
        Alternatively, supply a discount function via discount_fn.
    discount_fn : callable(float, float) -> float, optional
        discount_fn(t, s) returns the discount factor from time t to s.
        Overrides discount_rate if provided. Allows term structure or stochastic
        rates (pass E^Q[P(t,s)] from your rate model).

    Examples
    --------
    >>> import numpy as np
    >>> from insurance_survival.mortality.bounds import LifetimeBoundsCalculator

    >>> # CMI S3 table: 65-year-old male, 10 years
    >>> hat_p = [0.9924, 0.9910, 0.9893, 0.9871, 0.9843,
    ...          0.9807, 0.9762, 0.9705, 0.9635, 0.9549]
    >>> calc = LifetimeBoundsCalculator(hat_p=hat_p, starting_age=65, discount_rate=0.04)

    >>> # Annuity bound: unit income while alive
    >>> result = calc.annuity_bounds(t=0.0, T=10.0)
    >>> result.spread_pct  # model risk from fractional age assumption, in %
    0.4...

    >>> # Death benefit bound: 100,000 on death
    >>> result = calc.death_benefit_bounds(t=0.0, T=10.0, benefit_fn=lambda s: 100_000)
    >>> result.spread  # always upper - lower (negative for discounted death benefits)
    ...

    Notes
    -----
    For the GMAB contract (pure accumulation benefit at maturity T), bounds are
    trivial: both survival bounds equal the life table value at integer T.
    Use survival_upper(0, T) == survival_lower(0, T) at integer T.

    Integration with CauseSpecificMortality
    ----------------------------------------
    Use forecast.total_rates to extract posterior-mean mortality rates, convert to
    hat_p via hat_p_j = exp(-m_j), then pass to this calculator to bound contract
    values against within-year mortality uncertainty.
    """

    def __init__(
        self,
        hat_p: list[float] | np.ndarray,
        starting_age: int = 65,
        discount_rate: float = 0.04,
        discount_fn: Optional[Callable[[float, float], float]] = None,
    ) -> None:
        hat_p = np.asarray(hat_p, dtype=float)
        if hat_p.ndim != 1:
            raise ValueError(f"hat_p must be 1-D, got shape {hat_p.shape}")
        if len(hat_p) < 1:
            raise ValueError("hat_p must have at least one year")
        bad = np.where((hat_p <= 0) | (hat_p >= 1))[0]
        if len(bad) > 0:
            raise ValueError(
                f"hat_p values must be in (0, 1); bad indices: {bad.tolist()}"
            )
        if not isinstance(starting_age, int) or starting_age < 0:
            raise ValueError(
                f"starting_age must be a non-negative int, got {starting_age!r}"
            )
        if discount_rate < 0:
            raise ValueError(f"discount_rate must be >= 0, got {discount_rate}")

        self.hat_p: np.ndarray = hat_p
        self.n_years: int = len(hat_p)
        self.starting_age: int = starting_age
        self.discount_rate: float = discount_rate

        if discount_fn is not None:
            self._discount = discount_fn
        else:
            self._discount = lambda t, s: np.exp(-discount_rate * (s - t))

        # Precompute cumulative survival at integer ages.
        # _cumulative_p[j] = _j p_x = product(hat_p[0:j])
        # _cumulative_p[0] = 1.0 (survived 0 years), _cumulative_p[n] = product of all hat_p
        self._cumulative_p: np.ndarray = np.concatenate([[1.0], np.cumprod(hat_p)])

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_lx(
        cls,
        lx: list[float] | np.ndarray,
        starting_age: int = 65,
        discount_rate: float = 0.04,
        discount_fn: Optional[Callable[[float, float], float]] = None,
    ) -> "LifetimeBoundsCalculator":
        """Construct from lx column (lives at each integer age).

        Parameters
        ----------
        lx : array-like, shape (n+1,)
            Lives at each integer age x, x+1, ..., x+n. Radix is arbitrary
            (1000, 100000, etc.) — only ratios matter.
            hat_p_j is computed as lx[j+1] / lx[j].
        starting_age : int
            Integer age corresponding to lx[0].
        discount_rate : float
            Continuous discount rate.
        discount_fn : callable(float, float) -> float, optional
            Overrides discount_rate if provided.

        Returns
        -------
        LifetimeBoundsCalculator

        Raises
        ------
        ValueError
            If lx contains zeros, negatives, or has fewer than 2 entries.
        """
        lx = np.asarray(lx, dtype=float)
        if lx.ndim != 1:
            raise ValueError(f"lx must be 1-D, got shape {lx.shape}")
        if len(lx) < 2:
            raise ValueError("lx must have at least 2 entries (two integer ages)")
        if np.any(lx <= 0):
            raise ValueError("lx must be strictly positive (no zero or negative lives)")
        hat_p = lx[1:] / lx[:-1]
        return cls(
            hat_p=hat_p,
            starting_age=starting_age,
            discount_rate=discount_rate,
            discount_fn=discount_fn,
        )

    @classmethod
    def from_cmi(
        cls,
        table_code: str = "S3PML",
        starting_age: int = 65,
        n_years: int = 20,
        improvement_scale: Optional[str] = "CMI_2023",
        discount_rate: float = 0.04,
    ) -> "LifetimeBoundsCalculator":
        """Construct from a CMI standard table with optional improvement scale.

        Uses built-in S3 standard table data (ages 50-100, four table variants).
        Applies CMI 2023 improvement scale as multiplicative reduction to mortality.

        Parameters
        ----------
        table_code : {"S3PML", "S3PFL", "S3AML", "S3AFL"}
            CMI S3 table code. PML = pensioner male lives, PFL = female lives,
            AML = annuitant male, AFL = annuitant female.
        starting_age : int
            Starting age (must be 50-99 for S3 tables).
        n_years : int
            Number of policy years. Table is truncated at min(n_years, 100-starting_age).
        improvement_scale : str or None
            If "CMI_2023", applies CMI 2023 model average annual improvement
            (published central improvement rates by age and sex).
            If None, uses raw table without improvement.
        discount_rate : float
            Continuous discount rate.

        Returns
        -------
        LifetimeBoundsCalculator

        Raises
        ------
        ValueError
            If table_code is unknown or starting_age out of range.

        Notes
        -----
        The S3 table values bundled with this package are approximate published
        central rates from CMI WP147 (2023 S3 tables). For production pricing,
        actuaries should substitute company experience or CMI S3 graduated values
        via the hat_p constructor or from_lx().
        """
        if table_code not in _VALID_TABLE_CODES:
            raise ValueError(
                f"table_code must be one of {_VALID_TABLE_CODES}, got {table_code!r}"
            )
        if not isinstance(starting_age, int) or not (50 <= starting_age <= 99):
            raise ValueError(
                f"starting_age must be an integer in [50, 99] for S3 tables, "
                f"got {starting_age!r}"
            )
        max_years = 100 - starting_age
        n_years = min(n_years, max_years)
        if n_years < 1:
            raise ValueError(
                f"starting_age={starting_age} leaves no room for n_years={n_years}"
            )

        hat_p_list = []
        for j in range(n_years):
            age = starting_age + j
            key = (table_code, age)
            if key not in _CMI_S3_HAT_P:
                raise KeyError(
                    f"No S3 data for ({table_code}, age={age}). "
                    f"Available ages: 50-99."
                )
            p = _CMI_S3_HAT_P[key]

            if improvement_scale == "CMI_2023":
                imp_key = (table_code, age)
                imp_rate = _CMI_2023_IMPROVEMENT.get(imp_key, 0.0)
                # Apply j years of improvement to the mortality rate:
                # q_improved = q_base * (1 - imp_rate)^j
                # hat_p_improved = 1 - q_improved = 1 - (1-p) * (1 - imp_rate)^j
                q = 1.0 - p
                q_improved = q * (1.0 - imp_rate) ** j
                p = 1.0 - q_improved
                # Clamp to valid range
                p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            elif improvement_scale is not None:
                raise ValueError(
                    f"improvement_scale must be 'CMI_2023' or None, got {improvement_scale!r}"
                )

            hat_p_list.append(p)

        return cls(
            hat_p=hat_p_list,
            starting_age=starting_age,
            discount_rate=discount_rate,
        )

    # ------------------------------------------------------------------
    # Survival function bounds
    # ------------------------------------------------------------------

    def survival_upper(self, t: float, s: float) -> float:
        """Upper bound on survival probability from t to s.

        Mortality concentrated at year-ends: policyholder survives each year
        until the very last moment. Gives maximum survival probability for
        non-integer s.

        For integer s, equals the exact cumulative life table value.

        Parameters
        ----------
        t : float
            Current time (>= 0). Only floor(t) is used for integer indexing.
        s : float
            Future time (must be > t and <= t + n_years).

        Returns
        -------
        float
            Upper bound on _s-t p_{x+t}. Step function constant on [j, j+1).
        """
        t = max(0.0, float(t))
        s = float(s)
        if s <= t:
            raise ValueError(f"s ({s}) must be > t ({t})")
        if s > t + self.n_years:
            raise ValueError(
                f"s ({s}) exceeds available table coverage "
                f"(t + n_years = {t + self.n_years:.3f})"
            )

        j_t = int(np.floor(t))
        j_s = int(np.floor(s))

        # Upper survival: step function constant across (j_s, j_s+1)
        # Takes value _j_s p_{x+t} = cumulative_p[j_s] / cumulative_p[j_t]
        if j_s >= len(self._cumulative_p):
            return 0.0
        return float(self._cumulative_p[j_s] / self._cumulative_p[j_t])

    def survival_lower(self, t: float, s: float) -> float:
        """Lower bound on survival probability from t to s.

        Mortality concentrated at year-beginnings: policyholder dies as early
        as possible within each year. Gives minimum survival probability for
        non-integer s.

        For integer s, equals the exact cumulative life table value (same as upper).

        Parameters
        ----------
        t : float
            Current time (>= 0). Only floor(t) is used for integer indexing.
        s : float
            Future time (must be > t and <= t + n_years).

        Returns
        -------
        float
            Lower bound on _s-t p_{x+t}. Step function constant on (j-1, j].
        """
        t = max(0.0, float(t))
        s = float(s)
        if s <= t:
            raise ValueError(f"s ({s}) must be > t ({t})")
        if s > t + self.n_years:
            raise ValueError(
                f"s ({s}) exceeds available table coverage "
                f"(t + n_years = {t + self.n_years:.3f})"
            )

        j_t = int(np.floor(t))
        j_s = int(np.ceil(s))  # NOTE: ceil, not floor — lower uses next integer

        # Lower survival: step function constant across (j_s-1, j_s]
        # Takes value _j_s p_{x+t} = cumulative_p[j_s] / cumulative_p[j_t]
        if j_s >= len(self._cumulative_p):
            return 0.0
        return float(self._cumulative_p[j_s] / self._cumulative_p[j_t])

    def survival_bounds_schedule(
        self, t: float, T: float, n_steps: int = 200
    ) -> pl.DataFrame:
        """Upper and lower survival bounds over [t, T] on a dense grid.

        Useful for plotting or numerical integration with a custom payoff.

        Parameters
        ----------
        t : float
        T : float
        n_steps : int
            Number of grid points. Default 200.

        Returns
        -------
        pl.DataFrame
            Columns: s (Float64), upper (Float64), lower (Float64), spread (Float64).
        """
        s_grid = np.linspace(t, T, n_steps)
        # Avoid calling survival at exactly t (where s == t raises ValueError)
        # Use small offset for first point only if t == s
        upper_vals = []
        lower_vals = []
        for s in s_grid:
            if s <= t:
                # At s == t: survival is 1.0 by convention
                upper_vals.append(1.0)
                lower_vals.append(1.0)
            else:
                upper_vals.append(self.survival_upper(t, s))
                lower_vals.append(self.survival_lower(t, s))

        upper_arr = np.array(upper_vals)
        lower_arr = np.array(lower_vals)
        return pl.DataFrame({
            "s": s_grid,
            "upper": upper_arr,
            "lower": lower_arr,
            "spread": upper_arr - lower_arr,
        })

    # ------------------------------------------------------------------
    # Contract-specific bounds
    # ------------------------------------------------------------------

    def annuity_bounds(
        self,
        t: float,
        T: float,
        income_fn: Optional[Callable[[float], float]] = None,
        n_steps: int = 500,
    ) -> LifetimeBoundsResult:
        """Bounds on expected present value of a life annuity.

        Computes bounds on:
            V = integral_t^T _s-t p_{x+t} * h(s) * discount(t, s) ds

        where h(s) is the income rate function (default: unit constant).

        The upper bound uses survival_upper (policyholder lives as long as possible
        within each year). The lower bound uses survival_lower.

        Parameters
        ----------
        t : float
            Valuation time (start of contract).
        T : float
            Contract expiry (need not be integer).
        income_fn : callable(float) -> float, optional
            Income rate as function of time. Default: constant 1.0.
            For an increasing annuity: lambda s: 1 + 0.02 * (s - t).
            For a GMIB payoff: lambda s: max(account_value(s), guarantee(s)).
        n_steps : int
            Integration grid points for Simpson's rule. Default 500.

        Returns
        -------
        LifetimeBoundsResult
            contract_type = "annuity"
        """
        if income_fn is None:
            income_fn = lambda s: 1.0  # noqa: E731

        t = float(t)
        T = float(T)
        if T <= t:
            raise ValueError(f"T ({T}) must be > t ({t})")

        s_grid = np.linspace(t, T, n_steps)

        p_upper = np.array([
            1.0 if s <= t else self.survival_upper(t, s) for s in s_grid
        ])
        p_lower = np.array([
            1.0 if s <= t else self.survival_lower(t, s) for s in s_grid
        ])
        h_vals = np.array([income_fn(s) for s in s_grid])
        d_vals = np.array([self._discount(t, s) for s in s_grid])

        integrand_upper = p_upper * h_vals * d_vals
        integrand_lower = p_lower * h_vals * d_vals

        upper = float(scipy_simpson(integrand_upper, x=s_grid))
        lower = float(scipy_simpson(integrand_lower, x=s_grid))

        spread = upper - lower
        midpoint = (upper + lower) / 2.0
        spread_pct = (abs(spread) / abs(midpoint) * 100.0) if midpoint != 0.0 else 0.0

        j_t = int(np.floor(t))
        j_T = int(np.ceil(T))
        j_T = min(j_T, self.n_years)

        return LifetimeBoundsResult(
            upper=upper,
            lower=lower,
            spread=spread,
            spread_pct=spread_pct,
            midpoint=midpoint,
            contract_type="annuity",
            n_years=j_T - j_t,
            ages=list(range(self.starting_age + j_t, self.starting_age + j_T + 1)),
            hat_p=self.hat_p[j_t:j_T].tolist(),
        )

    def death_benefit_bounds(
        self,
        t: float,
        T: float,
        benefit_fn: Optional[Callable[[float], float]] = None,
    ) -> LifetimeBoundsResult:
        """Bounds on expected present value of a death benefit.

        Per Proposition 4.5 of Dupret & Motte (2026), the worst/best-case
        death benefit values concentrate mortality at year-ends (upper survival
        scenario) or year-starts (lower survival scenario).

        Computes:
            V = sum_{j=floor(t)}^{floor(T)-1}
                    (1 - hat_p_j) * _j p_{x+t} * benefit_fn(death_time) * discount(t, death_time)

        where death_time = j+1 (upper, year-end) or j (lower, year-start).

        Direction note: for a constant benefit with positive discount rate,
        upper survival (death at year-end) is more discounted, giving a *lower*
        discounted value. So result.upper < result.lower is normal here. The
        spread is upper - lower (may be negative).

        Parameters
        ----------
        t : float
            Valuation time. Years from floor(t) onwards are covered.
        T : float
            Contract expiry. Benefits paid up to but not including year floor(T).
        benefit_fn : callable(float) -> float, optional
            Benefit amount as function of time of death. Default: constant 1.0.
            For GMDB: lambda s: max(account_value(s), guarantee(s)).

        Returns
        -------
        LifetimeBoundsResult
            contract_type = "death_benefit"
        """
        if benefit_fn is None:
            benefit_fn = lambda s: 1.0  # noqa: E731

        t = float(t)
        T = float(T)
        if T <= t:
            raise ValueError(f"T ({T}) must be > t ({t})")

        j_t = int(np.floor(t))
        j_T = int(np.floor(T))

        upper = 0.0
        lower = 0.0

        for j in range(j_t, j_T):
            q_j = 1.0 - self.hat_p[j]
            # Survival from t to year j (in policy-year units, but we use absolute cumulative)
            survival_to_j = float(
                self._cumulative_p[j] / self._cumulative_p[j_t]
            )

            # Upper: death concentrated at year-end (time j+1)
            death_time_upper = float(j + 1)
            upper += (
                q_j
                * survival_to_j
                * benefit_fn(death_time_upper)
                * self._discount(t, death_time_upper)
            )

            # Lower: death concentrated at year-start (time j)
            death_time_lower = float(j)
            # For j == 0 and t == 0, discount(0, 0) = 1 and benefit_fn(0) is sensible
            lower += (
                q_j
                * survival_to_j
                * benefit_fn(death_time_lower)
                * self._discount(t, death_time_lower)
            )

        spread = upper - lower
        midpoint = (upper + lower) / 2.0
        spread_pct = (abs(spread) / abs(midpoint) * 100.0) if midpoint != 0.0 else 0.0

        return LifetimeBoundsResult(
            upper=upper,
            lower=lower,
            spread=spread,
            spread_pct=spread_pct,
            midpoint=midpoint,
            contract_type="death_benefit",
            n_years=j_T - j_t,
            ages=list(range(self.starting_age + j_t, self.starting_age + j_T + 1)),
            hat_p=self.hat_p[j_t:j_T].tolist(),
        )

    def general_bounds(
        self,
        t: float,
        T: float,
        payoff_fn: Callable[[float, float, float], tuple[float, float]],
        n_steps: int = 500,
    ) -> LifetimeBoundsResult:
        """Bounds for a general payoff functional.

        For payoff functions that combine survival and death elements (e.g.
        combined GMIB + GMDB riders), this method evaluates the payoff at each
        grid point under both extreme survival controls.

        Parameters
        ----------
        t : float
        T : float
        payoff_fn : callable(s, p_upper, p_lower) -> tuple[float, float]
            Called at each grid point s in [t, T].
            Arguments: current time s, upper survival p_{s-t}^upper, lower survival p_{s-t}^lower.
            Returns: (value_contribution_if_upper, value_contribution_if_lower).
            The method integrates these contributions over s using Simpson's rule.
        n_steps : int
            Integration grid points. Default 500.

        Returns
        -------
        LifetimeBoundsResult
            contract_type = "general"

        Notes
        -----
        For most use cases, annuity_bounds() and death_benefit_bounds() are
        sufficient and more interpretable. Use general_bounds() for non-standard
        payoff structures only.
        """
        t = float(t)
        T = float(T)
        if T <= t:
            raise ValueError(f"T ({T}) must be > t ({t})")

        s_grid = np.linspace(t, T, n_steps)
        contrib_upper = []
        contrib_lower = []

        for s in s_grid:
            if s <= t:
                p_u = 1.0
                p_l = 1.0
            else:
                p_u = self.survival_upper(t, s)
                p_l = self.survival_lower(t, s)
            val_u, val_l = payoff_fn(s, p_u, p_l)
            contrib_upper.append(val_u)
            contrib_lower.append(val_l)

        upper = float(scipy_simpson(np.array(contrib_upper), x=s_grid))
        lower = float(scipy_simpson(np.array(contrib_lower), x=s_grid))

        spread = upper - lower
        midpoint = (upper + lower) / 2.0
        spread_pct = (abs(spread) / abs(midpoint) * 100.0) if midpoint != 0.0 else 0.0

        j_t = int(np.floor(t))
        j_T = int(np.ceil(T))
        j_T = min(j_T, self.n_years)

        return LifetimeBoundsResult(
            upper=upper,
            lower=lower,
            spread=spread,
            spread_pct=spread_pct,
            midpoint=midpoint,
            contract_type="general",
            n_years=j_T - j_t,
            ages=list(range(self.starting_age + j_t, self.starting_age + j_T + 1)),
            hat_p=self.hat_p[j_t:j_T].tolist(),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def spread_by_duration(
        self,
        contract_type: str = "annuity",
        max_years: Optional[int] = None,
    ) -> pl.DataFrame:
        """Compute upper/lower/spread as a function of contract duration T.

        Evaluates the bounds for T = 1, 2, ..., max_years and returns a
        DataFrame showing how model risk accumulates with duration.

        Parameters
        ----------
        contract_type : {"annuity", "death_benefit"}
        max_years : int, optional
            Maximum contract duration. Defaults to len(hat_p).

        Returns
        -------
        pl.DataFrame
            Columns: T (Int32), upper (Float64), lower (Float64),
            spread (Float64), spread_pct (Float64).
        """
        if contract_type not in ("annuity", "death_benefit"):
            raise ValueError(
                f"contract_type must be 'annuity' or 'death_benefit', "
                f"got {contract_type!r}"
            )
        if max_years is None:
            max_years = self.n_years
        max_years = min(max_years, self.n_years)

        rows = []
        for T in range(1, max_years + 1):
            if contract_type == "annuity":
                r = self.annuity_bounds(t=0.0, T=float(T))
            else:
                r = self.death_benefit_bounds(t=0.0, T=float(T))
            rows.append({
                "T": T,
                "upper": r.upper,
                "lower": r.lower,
                "spread": r.spread,
                "spread_pct": r.spread_pct,
            })

        return pl.DataFrame(rows).with_columns(pl.col("T").cast(pl.Int32))

    def fractional_age_comparison(
        self,
        t: float,
        T: float,
        contract_type: str = "annuity",
        income_fn: Optional[Callable[[float], float]] = None,
    ) -> pl.DataFrame:
        """Compare bounds against three standard fractional age assumptions.

        Computes the annuity contract value under:
        1. Uniform Distribution of Deaths (UDD): _s p_{x+j} = 1 - frac*(1-hat_p_j)
        2. Constant Force of Mortality (CFM): _s p_{x+j} = hat_p_j^frac
        3. Balducci assumption: _s p_{x+j} = hat_p_j / (frac + (1-frac)*hat_p_j)
        4. Upper bound (this paper — mortality at year-ends)
        5. Lower bound (this paper — mortality at year-starts)

        Allows the actuary to see where each standard assumption falls relative
        to the theoretical bounds, and how much model risk exists.

        Parameters
        ----------
        t : float
        T : float
        contract_type : {"annuity"}
            Currently only "annuity" is supported for full comparison.
        income_fn : callable(float) -> float, optional
            Income rate function. Default: constant 1.0.

        Returns
        -------
        pl.DataFrame
            Columns: assumption (String), value (Float64), relative_to_udd_pct (Float64).
        """
        if income_fn is None:
            income_fn = lambda s: 1.0  # noqa: E731

        t = float(t)
        T = float(T)
        n_steps = 500
        j_t = int(np.floor(t))
        s_grid = np.linspace(t, T, n_steps)

        h_vals = np.array([income_fn(s) for s in s_grid])
        d_vals = np.array([self._discount(t, s) for s in s_grid])

        results: dict[str, float] = {}

        for assumption in ("UDD", "CFM", "Balducci"):
            p_vals = []
            for s in s_grid:
                j = int(np.floor(s))
                frac = s - j

                # Cumulative survival to year j from j_t
                if j < j_t:
                    surv_j = 1.0
                elif j < len(self._cumulative_p):
                    surv_j = float(self._cumulative_p[j] / self._cumulative_p[j_t])
                else:
                    surv_j = 0.0

                # Within-year fractional survival
                if j < self.n_years:
                    p = self.hat_p[j]
                    if assumption == "UDD":
                        within = 1.0 - frac * (1.0 - p)
                    elif assumption == "CFM":
                        within = float(p ** frac)
                    else:  # Balducci
                        denom = frac + (1.0 - frac) * p
                        within = p / denom if denom > 0 else 0.0
                else:
                    within = 0.0

                p_vals.append(surv_j * within)

            p_array = np.array(p_vals)
            results[assumption] = float(scipy_simpson(p_array * h_vals * d_vals, x=s_grid))

        # Bounds
        result_bounds = self.annuity_bounds(t, T, income_fn, n_steps=n_steps)
        results["Upper bound"] = result_bounds.upper
        results["Lower bound"] = result_bounds.lower

        udd_val = results["UDD"]
        rows = [
            {
                "assumption": k,
                "value": v,
                "relative_to_udd_pct": (
                    (v - udd_val) / abs(udd_val) * 100.0 if udd_val != 0.0 else 0.0
                ),
            }
            for k, v in results.items()
        ]
        return pl.DataFrame(rows)
