"""
loader.py — HMDLoader: Human Mortality Database data fetching and parsing.

The Human Mortality Database (mortality.org) is the canonical source for
national life tables. For UK insurance work, GBRTENW (England and Wales) is
the primary population series.

HMD files are space-delimited text with two header lines. Each row is:
    Year  Age  Female  Male  Total

This loader handles the Deaths1x1 (single-year age and calendar year) and
Exposures-CentralDeath1x1 formats. Cause-specific deaths are not available
from HMD itself — those come from ONS/WHO data — but HMD provides the
all-cause exposure denominator.

Notes on cause-of-death data
-----------------------------
The DMP model requires cause-specific deaths arrays. These are *not* in HMD.
Sources for UK cause-specific counts:
- ONS cause of death bulletins (annual, available via NOMIS API)
- WHO Mortality Database (ICD-coded, older vintages)
- CMI cause-of-death investigation (insured lives, restricted access)

For development and testing, use HMDLoader.load_synthetic() which generates
plausible synthetic data matching the expected array format.
"""

from __future__ import annotations

import io
import os
import pathlib
import urllib.request
import urllib.error
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd


# Standard HMD five-year age group labels (0-100+)
_AGE_GROUPS_5YR = [
    "0", "1-4",
    "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
    "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
    "65-69", "70-74", "75-79", "80-84", "85-89", "90-94",
    "95-99", "100+",
]

# Default cause group names matching ICD chapter aggregation
_DEFAULT_CAUSES = [
    "neoplasms",
    "cardiovascular",
    "respiratory",
    "infectious",
    "external",
    "other",
]

# HMD base URL — requires registered account for live download
_HMD_BASE_URL = "https://www.mortality.org/File/GetDocument/hmd.v6/zip/by_statistic"


class HMDLoader:
    """Load and parse Human Mortality Database mortality data.

    This class handles two data sources:
    1. **Local HMD files**: downloaded from mortality.org and cached locally
    2. **Synthetic data**: realistic fake data for development and testing

    For production use you need an HMD account (free registration at
    mortality.org). Download the Deaths1x1.txt and Exposures-CentralDeath1x1.txt
    files for your country and pass the directory path to ``load()``.

    Note on cause-specific deaths
    ------------------------------
    HMD provides all-cause counts only. For the full DMP model you need
    cause-specific deaths from ONS/WHO. The ``load()`` method therefore
    returns synthetic cause splits unless you supply cause_deaths separately
    via ``from_arrays()``.

    Examples
    --------
    >>> # Synthetic data for development
    >>> deaths, exposure = HMDLoader.load_synthetic(n_ages=18, n_years=30, n_causes=6)

    >>> # From local HMD files (downloaded from mortality.org)
    >>> deaths, exposure = HMDLoader.load(
    ...     data_dir="/data/hmd/GBRTENW",
    ...     sex="total",
    ...     years=(1979, 2020),
    ... )
    """

    @staticmethod
    def load(
        data_dir: str | os.PathLike,
        sex: str = "total",
        years: Optional[tuple[int, int]] = None,
        age_max: int = 100,
        cause_deaths: Optional[np.ndarray] = None,
        cause_names: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """Load mortality data from local HMD-format files.

        Expects Deaths1x1.txt and Exposures-CentralDeath1x1.txt in
        ``data_dir``. These are the standard single-year age × calendar-year
        HMD formats, downloadable from mortality.org after free registration.

        Parameters
        ----------
        data_dir : str or Path
            Directory containing the HMD text files.
        sex : {"total", "female", "male"}
            Which sex column to extract.
        years : tuple[int, int], optional
            (start_year, end_year) inclusive. If None, all available years
            are returned.
        age_max : int
            Ages above this value are collapsed into an open age group.
            Default 100 gives "100+" as the top group.
        cause_deaths : np.ndarray, optional
            Pre-aggregated cause-specific death counts, shape
            (n_age, n_year, n_cause). If None, an all-in-one "total" cause
            is used (no cause split). Supply this from ONS/WHO data.
        cause_names : list[str], optional
            Names for the cause groups. Required if cause_deaths is supplied.

        Returns
        -------
        deaths : np.ndarray, shape (n_age, n_year, n_cause)
            Death counts. If cause_deaths is None, shape is (n_age, n_year, 1).
        exposure : np.ndarray, shape (n_age, n_year)
            Central exposed-to-risk (person-years).
        age_labels : list[str]
            Age group labels matching the first axis.
        year_labels : list[int]
            Calendar years matching the second axis.

        Raises
        ------
        FileNotFoundError
            If the expected HMD files are not found in data_dir.
        ValueError
            If sex is not one of the accepted values, or if arrays are
            inconsistent.
        """
        if sex not in ("total", "female", "male"):
            raise ValueError(f"sex must be 'total', 'female', or 'male', got {sex!r}")

        data_dir = pathlib.Path(data_dir)
        deaths_path = data_dir / "Deaths_1x1.txt"
        exposure_path = data_dir / "Exposures_1x1.txt"

        for path in (deaths_path, exposure_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"HMD file not found: {path}. Download Deaths_1x1.txt and "
                    "Exposures_1x1.txt from mortality.org for your country and "
                    f"place them in {data_dir}."
                )

        deaths_df = HMDLoader._parse_hmd_file(deaths_path, sex)
        exposure_df = HMDLoader._parse_hmd_file(exposure_path, sex)

        if years is not None:
            start, end = years
            deaths_df = deaths_df[
                (deaths_df["year"] >= start) & (deaths_df["year"] <= end)
            ]
            exposure_df = exposure_df[
                (exposure_df["year"] >= start) & (exposure_df["year"] <= end)
            ]

        # Collapse ages above age_max into open group
        deaths_df = HMDLoader._collapse_ages(deaths_df, age_max)
        exposure_df = HMDLoader._collapse_ages(exposure_df, age_max)

        year_labels = sorted(deaths_df["year"].unique().tolist())
        age_labels = sorted(
            deaths_df["age"].unique().tolist(),
            key=lambda a: int(str(a).split("-")[0].replace("+", "")),
        )

        # Pivot to (n_age, n_year) arrays
        deaths_pivot = deaths_df.pivot(index="age", columns="year", values="value")
        exposure_pivot = exposure_df.pivot(index="age", columns="year", values="value")

        deaths_pivot = deaths_pivot.reindex(index=age_labels, columns=year_labels)
        exposure_pivot = exposure_pivot.reindex(index=age_labels, columns=year_labels)

        deaths_2d = deaths_pivot.values.astype(float)
        exposure_2d = exposure_pivot.values.astype(float)

        if cause_deaths is not None:
            if cause_deaths.shape[:2] != deaths_2d.shape:
                raise ValueError(
                    f"cause_deaths shape {cause_deaths.shape[:2]} does not match "
                    f"all-cause array shape {deaths_2d.shape}"
                )
            if cause_names is None:
                raise ValueError(
                    "cause_names must be provided when cause_deaths is supplied"
                )
            deaths_3d = cause_deaths.astype(float)
        else:
            # Single all-cause "cause" — useful for testing total-mortality model
            deaths_3d = deaths_2d[:, :, np.newaxis]

        return deaths_3d, exposure_2d, age_labels, year_labels

    @staticmethod
    def _parse_hmd_file(path: pathlib.Path, sex: str) -> pd.DataFrame:
        """Parse a single HMD text file into a long-format DataFrame.

        HMD format: two header lines, then space-delimited columns:
            Year  Age  Female  Male  Total

        The Age column uses "110+" as the open interval for very old ages.
        """
        col_map = {"female": "Female", "male": "Male", "total": "Total"}
        col = col_map[sex]

        rows = []
        with open(path, encoding="utf-8") as fh:
            # Skip header lines (lines starting with non-digit or containing text)
            for line in fh:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                year_str, age_str = parts[0], parts[1]
                # Map column index: Female=2, Male=3, Total=4
                idx = {"Female": 2, "Male": 3, "Total": 4}[col]
                value_str = parts[idx]

                try:
                    year = int(year_str)
                    age_val = int(age_str.replace("+", ""))
                    value = float(value_str.replace(".", "0") if value_str == "." else value_str)
                except (ValueError, IndexError):
                    continue

                rows.append({"year": year, "age": age_val, "value": value})

        return pd.DataFrame(rows)

    @staticmethod
    def _collapse_ages(df: pd.DataFrame, age_max: int) -> pd.DataFrame:
        """Collapse single-year ages into 5-year groups up to age_max+."""
        def to_group(age: int) -> str:
            if age >= age_max:
                return f"{age_max}+"
            if age == 0:
                return "0"
            if age < 5:
                return "1-4"
            lo = (age // 5) * 5
            return f"{lo}-{lo + 4}"

        df = df.copy()
        df["age"] = df["age"].apply(to_group)
        return df.groupby(["year", "age"], as_index=False)["value"].sum()

    @staticmethod
    def load_synthetic(
        n_ages: int = 18,
        n_years: int = 40,
        n_causes: int = 6,
        seed: int = 42,
        cause_names: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """Generate synthetic mortality data for development and testing.

        The data-generating process mimics realistic UK mortality patterns:
        - Exponential age gradient (Makeham-Gompertz-ish)
        - Declining period trend (mortality improvement ~ 1% per year)
        - Cause mix dominated by CVD and neoplasms in older ages

        Parameters
        ----------
        n_ages : int
            Number of age groups. Default 18 gives 5-year groups 0 to 85+.
        n_years : int
            Number of calendar years. Starts at 1980.
        n_causes : int
            Number of cause groups. Default 6 matches the paper's structure.
        seed : int
            Random seed for reproducibility.
        cause_names : list[str], optional
            Override the default cause names.

        Returns
        -------
        deaths : np.ndarray, shape (n_ages, n_years, n_causes)
        exposure : np.ndarray, shape (n_ages, n_years)
        age_labels : list[str]
        year_labels : list[int]
        """
        rng = np.random.default_rng(seed)

        if cause_names is None:
            cause_names = _DEFAULT_CAUSES[:n_causes]
            if len(cause_names) < n_causes:
                cause_names += [f"cause_{i}" for i in range(len(cause_names), n_causes)]

        # Age groups: 0, 1-4, 5-9, ..., then aggregate to n_ages groups
        age_labels = _AGE_GROUPS_5YR[:n_ages]
        year_labels = list(range(1980, 1980 + n_years))

        # Exposure: roughly 50,000 person-years per age/year cell, age-varying
        age_idx = np.arange(n_ages)
        exposure_age_factor = np.exp(-0.02 * age_idx)  # older ages smaller cohort
        exposure = (
            50_000
            * exposure_age_factor[:, np.newaxis]
            * np.ones((n_ages, n_years))
            * rng.lognormal(0, 0.05, (n_ages, n_years))  # small noise
        )

        # Base mortality rate: Gompertz-like age gradient
        # m(a) ~ exp(-8 + 0.09 * a) — roughly 0.003/yr at age 30, 0.05/yr at age 70
        log_m_base = -8.0 + 0.09 * age_idx  # (n_ages,)

        # Period trend: kappa_t ~ RW1 with negative drift (improvement)
        drift = -0.015
        sigma_kappa = 0.02
        kappa = np.zeros(n_years)
        kappa[0] = 0.0
        for t in range(1, n_years):
            kappa[t] = kappa[t - 1] + drift + rng.normal(0, sigma_kappa)

        # Identifiability: centre kappa
        kappa -= kappa.mean()

        # Lee-Carter: log m_{a,t} = nu + delta_a + beta_a * kappa_t
        beta = np.ones(n_ages) / n_ages  # uniform loading for synthetic data
        log_m = log_m_base[:, np.newaxis] + beta[:, np.newaxis] * kappa[np.newaxis, :]

        total_rate = np.exp(log_m)  # (n_ages, n_years)
        total_deaths = rng.poisson(exposure * total_rate).astype(float)

        # Cause mix: softmax of age-varying linear predictor
        # Older ages: more CVD and neoplasms; younger ages: more external causes
        eta_base = rng.normal(0, 1, (n_ages, n_causes))  # random cause intercepts

        # Make CVD/neoplasms dominant at older ages
        if n_causes >= 2:
            eta_base[:, 0] += 0.5 * (age_idx / n_ages)  # neoplasms
            eta_base[:, 1] += 1.0 * (age_idx / n_ages)  # CVD
        if n_causes >= 5:
            eta_base[:, 4] -= 0.5 * (age_idx / n_ages)  # external (younger)

        # Softmax -> cause probabilities
        eta_exp = np.exp(eta_base - eta_base.max(axis=1, keepdims=True))
        gamma = eta_exp / eta_exp.sum(axis=1, keepdims=True)  # (n_ages, n_causes)

        # Cause deaths via Dirichlet-Multinomial (phi=50 gives moderate overdispersion)
        phi = 50.0
        cause_deaths = np.zeros((n_ages, n_years, n_causes))
        for a in range(n_ages):
            for t in range(n_years):
                n_total = int(total_deaths[a, t])
                if n_total == 0:
                    continue
                alpha = phi * gamma[a, :]
                # Sample Dirichlet then Multinomial
                p = rng.dirichlet(alpha)
                cause_deaths[a, t, :] = rng.multinomial(n_total, p).astype(float)

        return cause_deaths, exposure, age_labels, year_labels

    @staticmethod
    def from_arrays(
        deaths: np.ndarray,
        exposure: np.ndarray,
        age_labels: Optional[list[str]] = None,
        year_labels: Optional[list[int]] = None,
        cause_names: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """Wrap pre-built arrays with default labels.

        Convenience method for passing already-constructed numpy arrays
        into CauseSpecificMortality with consistent labelling.

        Parameters
        ----------
        deaths : np.ndarray, shape (n_age, n_year, n_cause)
        exposure : np.ndarray, shape (n_age, n_year)
        age_labels : list[str], optional
        year_labels : list[int], optional
        cause_names : list[str], optional

        Returns
        -------
        Same arrays plus default labels if not provided.
        """
        n_age, n_year, n_cause = deaths.shape
        if age_labels is None:
            age_labels = [str(i * 5) for i in range(n_age)]
        if year_labels is None:
            year_labels = list(range(2000, 2000 + n_year))
        if cause_names is None:
            cause_names = [f"cause_{i}" for i in range(n_cause)]

        if len(age_labels) != n_age:
            raise ValueError(
                f"age_labels has {len(age_labels)} entries but deaths has {n_age} age groups"
            )
        if len(year_labels) != n_year:
            raise ValueError(
                f"year_labels has {len(year_labels)} entries but deaths has {n_year} years"
            )
        if len(cause_names) != n_cause:
            raise ValueError(
                f"cause_names has {len(cause_names)} entries but deaths has {n_cause} causes"
            )

        return deaths.astype(float), exposure.astype(float), age_labels, year_labels
