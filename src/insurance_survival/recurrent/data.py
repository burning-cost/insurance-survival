"""
RecurrentEventData — counting process representation for recurrent insurance claims.

The counting process formulation (start, stop] is the standard approach for
recurrent events following Andersen & Gill (1982). Each row is an interval of
risk time; event=1 marks the end-of-interval event occurrence.

This is deliberately a thin data container — validation and sorting logic live
here, but model logic stays in models.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class RecurrentEventData:
    """
    Counting process representation of recurrent insurance claims.

    Stores data in (start, stop] interval format — the standard input for
    Andersen-Gill models. Each policy appears as multiple rows, one per
    risk interval.

    Attributes
    ----------
    df : pd.DataFrame
        Long-format data with columns: id, start, stop, event, plus covariates.
    id_col : str
        Column identifying the policy/subject.
    start_col : str
        Left endpoint of the risk interval (inclusive boundary excluded).
    stop_col : str
        Right endpoint of the risk interval (inclusive).
    event_col : str
        Binary indicator: 1 = event occurred at stop, 0 = censored.
    covariate_cols : list[str]
        Columns used as covariates in model fitting.
    """

    df: pd.DataFrame
    id_col: str
    start_col: str
    stop_col: str
    event_col: str
    covariate_cols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        required = [self.id_col, self.start_col, self.stop_col, self.event_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        for c in self.covariate_cols:
            if c not in self.df.columns:
                raise ValueError(f"Covariate column not found: {c!r}")
        if (self.df[self.stop_col] <= self.df[self.start_col]).any():
            raise ValueError("All stop times must be strictly greater than start times.")
        if not self.df[self.event_col].isin([0, 1]).all():
            raise ValueError("event_col must be binary (0 or 1).")

    @classmethod
    def from_long_format(
        cls,
        df: pd.DataFrame,
        id_col: str,
        start_col: str,
        stop_col: str,
        event_col: str,
        covariates: Sequence[str] = (),
    ) -> "RecurrentEventData":
        """
        Construct from a long-format counting process data frame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per risk interval per subject.
        id_col : str
            Policy/subject identifier column.
        start_col : str
            Start of risk interval (usually 0 or time of previous event).
        stop_col : str
            End of risk interval.
        event_col : str
            1 if a claim occurred at stop, else 0.
        covariates : sequence of str
            Covariate columns to include.

        Returns
        -------
        RecurrentEventData

        Examples
        --------
        >>> data = RecurrentEventData.from_long_format(
        ...     df, id_col="policy_id", start_col="t_start",
        ...     stop_col="t_stop", event_col="claim",
        ...     covariates=["age", "vehicle_age", "region"],
        ... )
        """
        df = df.copy()
        df = df.sort_values([id_col, start_col]).reset_index(drop=True)
        return cls(
            df=df,
            id_col=id_col,
            start_col=start_col,
            stop_col=stop_col,
            event_col=event_col,
            covariate_cols=list(covariates),
        )

    @classmethod
    def from_events(
        cls,
        events_df: pd.DataFrame,
        follow_up_df: pd.DataFrame,
        id_col: str,
        time_col: str,
        end_col: str,
        covariates: Sequence[str] = (),
    ) -> "RecurrentEventData":
        """
        Construct from a separate events table and a follow-up/exposure table.

        This is the more natural format when you have a claims table and a
        policies table. Converts to counting process (start, stop] internally.

        Parameters
        ----------
        events_df : pd.DataFrame
            One row per claim. Must have id_col and time_col.
        follow_up_df : pd.DataFrame
            One row per policy. Must have id_col and end_col (study end /
            lapse date / policy expiry).
        id_col : str
            Shared policy identifier.
        time_col : str
            Time of each claim (relative to policy start = 0).
        end_col : str
            End of follow-up for each policy.
        covariates : sequence of str
            Covariate columns present in follow_up_df.

        Returns
        -------
        RecurrentEventData
        """
        records = []
        for pid, grp in follow_up_df.set_index(id_col).iterrows():
            end_time = grp[end_col]
            claims = events_df.loc[events_df[id_col] == pid, time_col].sort_values().tolist()
            cov_vals = {c: grp[c] for c in covariates if c in grp.index}

            times = [0.0] + claims + [end_time]
            for i in range(len(times) - 1):
                t_start = times[i]
                t_stop = times[i + 1]
                is_claim = i < len(claims)
                if t_stop <= t_start:
                    continue
                row = {id_col: pid, "t_start": t_start, "t_stop": t_stop, "event": int(is_claim)}
                row.update(cov_vals)
                records.append(row)

        df = pd.DataFrame(records)
        return cls.from_long_format(
            df,
            id_col=id_col,
            start_col="t_start",
            stop_col="t_stop",
            event_col="event",
            covariates=list(covariates),
        )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_subjects(self) -> int:
        """Number of unique policies/subjects."""
        return self.df[self.id_col].nunique()

    @property
    def n_events(self) -> int:
        """Total number of observed events (claims)."""
        return int(self.df[self.event_col].sum())

    @property
    def subject_ids(self) -> np.ndarray:
        """Sorted array of unique subject IDs."""
        return np.sort(self.df[self.id_col].unique())

    @property
    def X(self) -> np.ndarray:
        """Covariate matrix (n_intervals, n_covariates). Float64."""
        if not self.covariate_cols:
            return np.zeros((len(self.df), 0))
        return self.df[self.covariate_cols].to_numpy(dtype=float)

    @property
    def start(self) -> np.ndarray:
        return self.df[self.start_col].to_numpy(dtype=float)

    @property
    def stop(self) -> np.ndarray:
        return self.df[self.stop_col].to_numpy(dtype=float)

    @property
    def event(self) -> np.ndarray:
        return self.df[self.event_col].to_numpy(dtype=int)

    def per_subject_summary(self) -> pd.DataFrame:
        """
        Return a summary table with one row per subject.

        Columns: id, n_events, total_time, mean_gap (between events).
        """
        rows = []
        for pid, grp in self.df.groupby(self.id_col):
            n_ev = int(grp[self.event_col].sum())
            total_time = float((grp[self.stop_col] - grp[self.start_col]).sum())
            event_times = grp.loc[grp[self.event_col] == 1, self.stop_col].tolist()
            if len(event_times) > 1:
                mean_gap = float(np.mean(np.diff(sorted(event_times))))
            else:
                mean_gap = float("nan")
            rows.append(
                {
                    self.id_col: pid,
                    "n_events": n_ev,
                    "total_time": total_time,
                    "mean_gap": mean_gap,
                }
            )
        return pd.DataFrame(rows)

    def event_counts(self) -> pd.Series:
        """Distribution of event counts per subject."""
        return (
            self.df.groupby(self.id_col)[self.event_col]
            .sum()
            .astype(int)
            .value_counts()
            .sort_index()
        )

    def __repr__(self) -> str:
        return (
            f"RecurrentEventData("
            f"n_subjects={self.n_subjects}, "
            f"n_events={self.n_events}, "
            f"n_intervals={len(self.df)}, "
            f"covariates={self.covariate_cols})"
        )
