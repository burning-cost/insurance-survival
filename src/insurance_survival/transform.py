"""
ExposureTransformer: convert policy transaction tables to survival format.

The mechanics of insurance survival analysis differ from clinical survival in
several ways that bite you if you ignore them:

- Renewal cliff: the hazard is highest at integer policy years (renewal decision
  point), not uniformly distributed.
- MTAs update covariates within a policy year without constituting a lapse.
- Fractional first-year exposure: a policy incepted on 1 July contributes 0.5
  years to the first observation period.
- Left truncation: policies already in force at the study start date are
  conditionally observed, not inception-to-observation.

This module handles all of these. The output is a Polars DataFrame in start/stop
format ready for lifelines fitters.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from ._utils import to_polars

# Transaction types that constitute a lapse (policy exit) event
_EXIT_TRANSACTION_TYPES = {"cancellation", "nonrenewal"}

# Transaction types that update covariates without constituting an exit
_MTA_TRANSACTION_TYPES = {"mta"}

# The full set of valid transaction types
_VALID_TRANSACTION_TYPES = {
    "inception",
    "renewal",
    "mta",
    "cancellation",
    "nonrenewal",
}


class ExposureTransformer:
    """Convert policy transaction tables to survival format for lifelines.

    Handles the mechanics that make insurance survival analysis different
    from standard medical survival:

    - Fractional first-year earned exposure (policies written mid-year)
    - Mid-term adjustment (MTA) transactions that update covariates without
      constituting a lapse event
    - Distinguishing non-renewal (voluntary lapse at policy anniversary) from
      mid-term cancellation (involuntary or MTD exit)
    - Left truncation: policies already in force at observation start date
    - Interval-censored: the exact date of nonrenewal is known (policy expiry
      date) even when the decision was made earlier

    The output is a start/stop Polars DataFrame compatible with lifelines
    CoxTimeVaryingFitter and all insurance_survival fitters.

    Parameters
    ----------
    observation_cutoff : date
        Policies still active at this date are right-censored. Usually today
        or end of the modelling period.
    time_scale : str
        "policy_year" (default, recommended) or "calendar". Policy year is
        the natural scale for UK personal lines (renewal cliff at integer years).
    exposure_basis : str
        "earned" (default) or "written". Earned exposure adjusts for fractional
        first/last periods. Written assigns full-year exposure to inception year.
    min_duration : float
        Minimum observed duration in policy years to include. Default 0.0.
        Set to 0.5 to exclude very short policies (common artefact of MTA
        records).

    Examples
    --------
    >>> import polars as pl
    >>> from datetime import date
    >>> from insurance_survival import ExposureTransformer
    >>>
    >>> transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
    >>> survival_df = transformer.fit_transform(transactions)
    >>> survival_df.head()
    """

    def __init__(
        self,
        observation_cutoff: date,
        time_scale: str = "policy_year",
        exposure_basis: str = "earned",
        min_duration: float = 0.0,
    ) -> None:
        if time_scale not in ("policy_year", "calendar"):
            raise ValueError(
                f"time_scale must be 'policy_year' or 'calendar', got {time_scale!r}"
            )
        if exposure_basis not in ("earned", "written"):
            raise ValueError(
                f"exposure_basis must be 'earned' or 'written', got {exposure_basis!r}"
            )
        self.observation_cutoff = observation_cutoff
        self.time_scale = time_scale
        self.exposure_basis = exposure_basis
        self.min_duration = min_duration

        # Populated after fit_transform
        self._summary: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, transactions: pl.DataFrame) -> pl.DataFrame:
        """Transform transaction table to start/stop survival DataFrame.

        Parameters
        ----------
        transactions : pl.DataFrame
            Policy transaction table. Required columns:
            - policy_id (Utf8)
            - transaction_date (Date)
            - transaction_type (Utf8): one of inception, renewal, mta,
              cancellation, nonrenewal
            - inception_date (Date)
            - expiry_date (Date)

            Optional columns passed through to output: ncd_level,
            annual_premium, vehicle_age, policyholder_age, channel.

        Returns
        -------
        pl.DataFrame
            Start/stop format (see Section 4.2 of the design spec).
        """
        df = to_polars(transactions)
        self._validate_input(df)

        # Build intervals per policy
        intervals = self._build_intervals(df)

        # Filter by min_duration
        if self.min_duration > 0.0:
            intervals = intervals.filter(pl.col("stop") >= self.min_duration)

        self._compute_summary(df, intervals)
        return intervals

    def summary(self) -> dict[str, Any]:
        """Return summary statistics from the last transform call.

        Returns
        -------
        dict
            Keys: n_policies, n_intervals, event_rate, median_duration,
            censoring_rate, left_truncated_count, mta_count.
        """
        if not self._summary:
            raise RuntimeError("Call fit_transform() before summary().")
        return self._summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pl.DataFrame) -> None:
        required = {"policy_id", "transaction_date", "transaction_type",
                    "inception_date", "expiry_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        bad_types = (
            df.filter(
                ~pl.col("transaction_type").is_in(list(_VALID_TRANSACTION_TYPES))
            )
            .get_column("transaction_type")
            .unique()
            .to_list()
        )
        if bad_types:
            raise ValueError(
                f"Unknown transaction_type values: {bad_types}. "
                f"Valid: {_VALID_TRANSACTION_TYPES}"
            )

    def _build_intervals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build start/stop intervals for every policy."""
        # Optional covariate columns to carry through
        optional_cols = [
            c for c in ("ncd_level", "annual_premium", "vehicle_age",
                        "policyholder_age", "channel", "event_type")
            if c in df.columns
        ]

        all_intervals: list[dict[str, Any]] = []
        cutoff = self.observation_cutoff
        mta_count = 0
        left_trunc_count = 0

        # Group by policy_id and process each policy's transaction history
        grouped = df.sort(["policy_id", "transaction_date"]).partition_by(
            "policy_id", maintain_order=True
        )

        for policy_df in grouped:
            policy_id = policy_df["policy_id"][0]
            inception_date = policy_df["inception_date"][0]

            # Count MTAs for summary
            mta_count += (
                policy_df.filter(pl.col("transaction_type") == "mta").height
            )

            rows = policy_df.iter_rows(named=True)
            row_list = list(rows)

            # Determine if left-truncated: if inception is before observation
            # window but first transaction is not inception
            first_txn_type = row_list[0]["transaction_type"]
            is_left_truncated = (
                first_txn_type not in ("inception",)
                and inception_date < cutoff
            )
            if is_left_truncated:
                left_trunc_count += 1

            intervals = self._process_policy(
                policy_id=policy_id,
                inception_date=inception_date,
                transactions=row_list,
                optional_cols=optional_cols,
                cutoff=cutoff,
            )
            all_intervals.extend(intervals)

        self._mta_count = mta_count
        self._left_trunc_count = left_trunc_count

        if not all_intervals:
            # Return empty DataFrame with correct schema
            return self._empty_schema(optional_cols)

        result = pl.DataFrame(all_intervals)
        # Ensure correct types
        result = result.with_columns([
            pl.col("start").cast(pl.Float64),
            pl.col("stop").cast(pl.Float64),
            pl.col("event").cast(pl.Int32),
            pl.col("exposure_years").cast(pl.Float64),
        ])
        return result

    def _process_policy(
        self,
        policy_id: str,
        inception_date: date,
        transactions: list[dict[str, Any]],
        optional_cols: list[str],
        cutoff: date,
    ) -> list[dict[str, Any]]:
        """Build intervals for a single policy's transaction list.

        Returns a list of interval dicts.
        """
        intervals: list[dict[str, Any]] = []

        def days_to_years(d: date) -> float:
            return (d - inception_date).days / 365.25

        # Determine the effective observation end for this policy
        # Last transaction determines exit type
        last_txn = transactions[-1]
        last_type = last_txn["transaction_type"]

        is_exit = last_type in _EXIT_TRANSACTION_TYPES

        # Build segment boundaries: each non-exit transaction starts a segment.
        # Exit transactions close the final segment.
        # We walk through transactions building intervals.
        #
        # Segment logic:
        # - inception → start of interval 0
        # - renewal → starts a new full-year interval
        # - mta → splits the current year interval, updates covariates
        # - cancellation / nonrenewal → closes the last interval with event=1

        segments: list[dict[str, Any]] = []

        for i, txn in enumerate(transactions):
            txn_type = txn["transaction_type"]

            if txn_type in _EXIT_TRANSACTION_TYPES:
                # Close the last open segment
                if segments:
                    segments[-1]["end_date"] = txn["transaction_date"]
                    segments[-1]["event"] = 1
                    segments[-1]["event_type"] = txn.get("event_type") or txn_type
                break
            else:
                # Open a new segment starting from this transaction
                seg: dict[str, Any] = {
                    "start_date": txn["transaction_date"],
                    "end_date": None,
                    "event": 0,
                    "event_type": "censored",
                }
                # Copy optional covariates from this transaction
                for col in optional_cols:
                    if col != "event_type":
                        seg[col] = txn.get(col)

                # If there's a next transaction, its date is our end
                if i + 1 < len(transactions):
                    next_txn = transactions[i + 1]
                    next_type = next_txn["transaction_type"]
                    if next_type in _EXIT_TRANSACTION_TYPES:
                        seg["end_date"] = next_txn["transaction_date"]
                        seg["event"] = 1
                        seg["event_type"] = (
                            next_txn.get("event_type") or next_type
                        )
                    else:
                        seg["end_date"] = next_txn["transaction_date"]
                else:
                    # Last non-exit transaction: censor at cutoff or expiry
                    expiry = txn.get("expiry_date")
                    if expiry is not None and expiry <= cutoff:
                        # Policy lapsed by non-renewal (not recorded explicitly)
                        seg["end_date"] = expiry
                    else:
                        seg["end_date"] = cutoff

                segments.append(seg)

        # Convert each segment to start/stop in policy years
        for seg in segments:
            start_date = seg["start_date"]
            end_date = seg["end_date"]

            if end_date is None or end_date <= start_date:
                continue

            start_yr = days_to_years(start_date)
            end_yr = days_to_years(end_date)

            if end_yr <= start_yr:
                continue

            exposure = self._compute_exposure(start_yr, end_yr)

            interval: dict[str, Any] = {
                "policy_id": policy_id,
                "start": round(start_yr, 6),
                "stop": round(end_yr, 6),
                "event": seg["event"],
                "event_type": seg["event_type"],
                "exposure_years": round(exposure, 6),
            }
            for col in optional_cols:
                if col != "event_type":
                    interval[col] = seg.get(col)

            intervals.append(interval)

        return intervals

    def _compute_exposure(self, start: float, stop: float) -> float:
        """Compute earned or written exposure for an interval."""
        if self.exposure_basis == "written":
            return stop - start
        # Earned: the same as duration for intervals within a policy year;
        # for multi-year intervals this is just stop - start (full years)
        return stop - start

    def _empty_schema(self, optional_cols: list[str]) -> pl.DataFrame:
        """Return empty DataFrame with correct schema."""
        schema: dict[str, pl.DataType] = {
            "policy_id": pl.Utf8,
            "start": pl.Float64,
            "stop": pl.Float64,
            "event": pl.Int32,
            "event_type": pl.Utf8,
            "exposure_years": pl.Float64,
        }
        for col in optional_cols:
            if col != "event_type":
                schema[col] = pl.Float64
        return pl.DataFrame(schema=schema)

    def _compute_summary(
        self,
        original: pl.DataFrame,
        intervals: pl.DataFrame,
    ) -> None:
        """Populate the _summary dict after transformation."""
        n_policies = original["policy_id"].n_unique()
        n_intervals = len(intervals)

        if n_intervals == 0:
            self._summary = {
                "n_policies": n_policies,
                "n_intervals": 0,
                "event_rate": 0.0,
                "median_duration": 0.0,
                "censoring_rate": 1.0,
                "left_truncated_count": self._left_trunc_count,
                "mta_count": self._mta_count,
            }
            return

        # Per-policy summary (last interval per policy)
        per_policy = (
            intervals.group_by("policy_id")
            .agg([
                pl.col("stop").max().alias("duration"),
                pl.col("event").max().alias("had_event"),
            ])
        )

        event_rate = per_policy["had_event"].mean()
        median_duration = per_policy["duration"].median()

        self._summary = {
            "n_policies": n_policies,
            "n_intervals": n_intervals,
            "event_rate": float(event_rate) if event_rate is not None else 0.0,
            "median_duration": float(median_duration) if median_duration is not None else 0.0,
            "censoring_rate": 1.0 - (float(event_rate) if event_rate is not None else 0.0),
            "left_truncated_count": self._left_trunc_count,
            "mta_count": self._mta_count,
        }
