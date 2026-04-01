"""
forecast.py — MortalityForecast: posterior forecast container for DMP models.

Holds posterior draws of total and cause-specific mortality rates, provides
coherence validation, and generates credible-interval summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class MortalityForecast:
    """Posterior forecast from a fitted CauseSpecificMortality model.

    All rate arrays hold posterior draws (not summaries), so downstream
    uncertainty propagation is straightforward. Coherence holds at every
    individual draw by model construction.

    Attributes
    ----------
    total_rates : np.ndarray, shape (n_age, n_period, n_draw)
        Posterior draws of central death rate m_{a,t}.
    cause_rates : np.ndarray, shape (n_age, n_period, n_cause, n_draw)
        Posterior draws of cause-specific rates m_{a,t,c}.
        By construction, sum over axis=2 equals total_rates.
    ages : list[str]
        Age group labels, e.g. ["0", "1-4", "5-9", ...].
    periods : list[int]
        Calendar years covered by the forecast, e.g. [2024, 2025, ...].
    cause_names : list[str]
        Names of the cause groups, e.g. ["neoplasms", "CVD", ...].
    """

    total_rates: np.ndarray
    cause_rates: np.ndarray
    ages: list[str]
    periods: list[int]
    cause_names: list[str]

    def __post_init__(self) -> None:
        """Validate array shapes are mutually consistent."""
        n_age, n_period, n_draw = self.total_rates.shape
        n_age2, n_period2, n_cause, n_draw2 = self.cause_rates.shape

        if n_age != n_age2:
            raise ValueError(
                f"total_rates has {n_age} age groups but cause_rates has {n_age2}"
            )
        if n_period != n_period2:
            raise ValueError(
                f"total_rates has {n_period} periods but cause_rates has {n_period2}"
            )
        if n_draw != n_draw2:
            raise ValueError(
                f"total_rates has {n_draw} draws but cause_rates has {n_draw2}"
            )
        if n_cause != len(self.cause_names):
            raise ValueError(
                f"cause_rates has {n_cause} causes but cause_names has {len(self.cause_names)} entries"
            )
        if len(self.ages) != n_age:
            raise ValueError(
                f"ages has {len(self.ages)} entries but arrays have {n_age} age groups"
            )
        if len(self.periods) != n_period:
            raise ValueError(
                f"periods has {len(self.periods)} entries but arrays have {n_period} periods"
            )

    @property
    def n_draws(self) -> int:
        """Number of posterior draws."""
        return self.total_rates.shape[2]

    @property
    def n_causes(self) -> int:
        """Number of cause groups."""
        return self.cause_rates.shape[2]

    def coherence_check(self, tol: float = 1e-6) -> bool:
        """Assert that cause-specific rates sum to total rates at every draw.

        This should hold by construction for any forecast produced by
        CauseSpecificMortality. If it fails, something went wrong in the
        forecast computation.

        Parameters
        ----------
        tol : float
            Absolute tolerance for the sum check. Default 1e-6 is loose enough
            to tolerate float32 accumulation errors from JAX/NumPyro.

        Returns
        -------
        bool
            True if coherence holds everywhere.

        Raises
        ------
        AssertionError
            If any draw violates the coherence condition beyond tolerance.
        """
        cause_sum = self.cause_rates.sum(axis=2)  # (n_age, n_period, n_draw)
        max_error = np.abs(cause_sum - self.total_rates).max()
        if max_error > tol:
            raise AssertionError(
                f"Coherence check failed: max |sum_c cause_rates - total_rates| = "
                f"{max_error:.3e} exceeds tolerance {tol:.3e}. "
                "This indicates a bug in forecast construction."
            )
        return True

    def summary(
        self,
        quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
    ) -> pd.DataFrame:
        """Credible-interval summary table over age groups and periods.

        Returns one row per (age, period, cause) combination, with columns
        for each requested quantile of the posterior draw distribution.

        Parameters
        ----------
        quantiles : tuple[float, ...]
            Quantiles to compute over the draw axis.

        Returns
        -------
        pd.DataFrame
            Columns: age, period, cause, q{pct} for each quantile.
            Also includes mean and sd columns.
        """
        rows = []
        q_labels = [f"q{int(q * 1000):04d}" for q in quantiles]

        for ai, age in enumerate(self.ages):
            for ti, period in enumerate(self.periods):
                # Total mortality row
                draws = self.total_rates[ai, ti, :]
                row: dict = {
                    "age": age,
                    "period": period,
                    "cause": "total",
                    "mean": float(draws.mean()),
                    "sd": float(draws.std()),
                }
                for q, label in zip(quantiles, q_labels):
                    row[label] = float(np.quantile(draws, q))
                rows.append(row)

                # Per-cause rows
                for ci, cause in enumerate(self.cause_names):
                    draws = self.cause_rates[ai, ti, ci, :]
                    row = {
                        "age": age,
                        "period": period,
                        "cause": cause,
                        "mean": float(draws.mean()),
                        "sd": float(draws.std()),
                    }
                    for q, label in zip(quantiles, q_labels):
                        row[label] = float(np.quantile(draws, q))
                    rows.append(row)

        return pd.DataFrame(rows)

    def improvement_factors(self, base_year_idx: int = 0) -> pd.DataFrame:
        """Compute posterior mean annual improvement factors by age and cause.

        Improvement factor F_{a,t,c} = 1 - m_{a,t,c} / m_{a,t-1,c}.
        This is the CMI-compatible format UK reserving actuaries need.

        Parameters
        ----------
        base_year_idx : int
            Index of the first year in ``periods``. Improvement factors are
            computed relative to each year's predecessor, so the output starts
            from periods[base_year_idx + 1].

        Returns
        -------
        pd.DataFrame
            Columns: age, period, cause, improvement_factor (posterior mean).
        """
        rows = []
        for ai, age in enumerate(self.ages):
            for ci, cause in enumerate(self.cause_names):
                rates = self.cause_rates[ai, :, ci, :]  # (n_period, n_draw)
                for ti in range(1, len(self.periods)):
                    prev = rates[ti - 1, :]
                    curr = rates[ti, :]
                    improvement = 1.0 - curr / np.where(prev > 0, prev, np.nan)
                    rows.append({
                        "age": age,
                        "period": self.periods[ti],
                        "cause": cause,
                        "improvement_factor": float(np.nanmean(improvement)),
                        "improvement_q025": float(np.nanquantile(improvement, 0.025)),
                        "improvement_q975": float(np.nanquantile(improvement, 0.975)),
                    })
        return pd.DataFrame(rows)
