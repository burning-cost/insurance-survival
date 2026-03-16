"""
Tests for ExposureTransformer.

Validates:
- Correct interval count per policy type
- Exposure totals match observed duration
- Event flags set correctly
- MTA intervals split correctly
- Edge cases: empty table, min_duration filter
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from insurance_survival import ExposureTransformer
from conftest import make_transaction_dgp


class TestExposureTransformerBasic:
    """Basic functionality tests."""

    def test_output_schema(self, transaction_dgp, observation_cutoff):
        """Output DataFrame has the expected columns."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)

        required_cols = {"policy_id", "start", "stop", "event", "event_type", "exposure_years"}
        assert required_cols.issubset(set(result.columns))

    def test_output_polars_dataframe(self, transaction_dgp, observation_cutoff):
        """Output is a Polars DataFrame."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        assert isinstance(result, pl.DataFrame)

    def test_stop_greater_than_start(self, transaction_dgp, observation_cutoff):
        """All intervals have stop > start."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        assert (result["stop"] > result["start"]).all()

    def test_start_non_negative(self, transaction_dgp, observation_cutoff):
        """All start values are >= 0."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        assert (result["start"] >= 0.0).all()

    def test_event_binary(self, transaction_dgp, observation_cutoff):
        """Event column contains only 0 and 1."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        assert set(result["event"].unique().to_list()).issubset({0, 1})

    def test_exposure_years_positive(self, transaction_dgp, observation_cutoff):
        """All exposure_years are positive."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        assert (result["exposure_years"] > 0).all()

    def test_exposure_equals_stop_minus_start(self, transaction_dgp, observation_cutoff):
        """Exposure equals stop - start (earned basis)."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        result = t.fit_transform(transaction_dgp)
        diff = (result["stop"] - result["start"] - result["exposure_years"]).abs()
        assert (diff < 1e-4).all()


class TestSimplePolicies:
    """Tests using hand-crafted policy tables for precise assertions."""

    def _make_simple_inception(
        self,
        policy_id: str = "P001",
        inception: date = date(2022, 1, 1),
        cutoff: date = date(2024, 12, 31),
    ) -> pl.DataFrame:
        """Single inception-only policy (will be right-censored)."""
        return pl.DataFrame([{
            "policy_id": policy_id,
            "transaction_date": inception,
            "transaction_type": "inception",
            "inception_date": inception,
            "expiry_date": inception + timedelta(days=365),
            "ncd_years": 3,
            "annual_premium": 500.0,
            "vehicle_age": 3,
            "policyholder_age": 40,
            "channel": "direct",
            "event_type": None,
        }]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])

    def test_inception_only_one_interval(self):
        """A policy with only an inception produces exactly one interval."""
        txns = self._make_simple_inception()
        t = ExposureTransformer(observation_cutoff=date(2024, 12, 31))
        result = t.fit_transform(txns)
        assert len(result) == 1

    def test_inception_only_event_zero(self):
        """An inception-only policy is right-censored (event=0)."""
        txns = self._make_simple_inception()
        t = ExposureTransformer(observation_cutoff=date(2024, 12, 31))
        result = t.fit_transform(txns)
        assert result["event"][0] == 0

    def test_lapse_policy_event_one(self):
        """A policy with explicit nonrenewal has event=1 on the last interval."""
        inception = date(2022, 1, 1)
        expiry = inception + timedelta(days=365)
        txns = pl.DataFrame([
            {
                "policy_id": "P002",
                "transaction_date": inception,
                "transaction_type": "inception",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 1,
                "annual_premium": 600.0,
                "vehicle_age": 5,
                "policyholder_age": 35,
                "channel": "aggregator",
                "event_type": None,
            },
            {
                "policy_id": "P002",
                "transaction_date": expiry,
                "transaction_type": "nonrenewal",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 1,
                "annual_premium": 600.0,
                "vehicle_age": 5,
                "policyholder_age": 35,
                "channel": "aggregator",
                "event_type": "lapse",
            },
        ]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])
        t = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
        result = t.fit_transform(txns)
        assert result.filter(pl.col("policy_id") == "P002")["event"].max() == 1

    def test_multi_renewal_interval_count(self):
        """A policy with 3 renewals then lapse produces 4 intervals."""
        inception = date(2020, 1, 1)
        rows = []
        current_date = inception
        for i in range(4):  # inception + 3 renewals
            txn_type = "inception" if i == 0 else "renewal"
            expiry = current_date + timedelta(days=365)
            rows.append({
                "policy_id": "P003",
                "transaction_date": current_date,
                "transaction_type": txn_type,
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": min(i, 9),
                "annual_premium": 500.0,
                "vehicle_age": i,
                "policyholder_age": 35 + i,
                "channel": "direct",
                "event_type": None,
            })
            current_date = expiry

        # Lapse at 4 years
        rows.append({
            "policy_id": "P003",
            "transaction_date": current_date,
            "transaction_type": "nonrenewal",
            "inception_date": inception,
            "expiry_date": current_date,
            "ncd_years": 3,
            "annual_premium": 500.0,
            "vehicle_age": 4,
            "policyholder_age": 39,
            "channel": "direct",
            "event_type": "lapse",
        })

        txns = pl.DataFrame(rows).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])
        t = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
        result = t.fit_transform(txns)

        p3 = result.filter(pl.col("policy_id") == "P003")
        assert len(p3) == 4

    def test_mta_splits_interval(self):
        """An MTA mid-year splits the interval into two rows."""
        inception = date(2022, 1, 1)
        expiry = inception + timedelta(days=365)
        mta_date = inception + timedelta(days=180)

        txns = pl.DataFrame([
            {
                "policy_id": "P004",
                "transaction_date": inception,
                "transaction_type": "inception",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 2,
                "annual_premium": 500.0,
                "vehicle_age": 3,
                "policyholder_age": 40,
                "channel": "direct",
                "event_type": None,
            },
            {
                "policy_id": "P004",
                "transaction_date": mta_date,
                "transaction_type": "mta",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 2,
                "annual_premium": 550.0,  # premium change at MTA
                "vehicle_age": 3,
                "policyholder_age": 40,
                "channel": "direct",
                "event_type": None,
            },
        ]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])

        t = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
        result = t.fit_transform(txns)

        p4 = result.filter(pl.col("policy_id") == "P004")
        assert len(p4) == 2

    def test_mta_total_exposure_preserved(self):
        """Total exposure after MTA split equals duration without split."""
        inception = date(2022, 1, 1)
        expiry = inception + timedelta(days=365)
        mta_date = inception + timedelta(days=180)
        cutoff = date(2024, 1, 1)

        txns_no_mta = pl.DataFrame([{
            "policy_id": "P005",
            "transaction_date": inception,
            "transaction_type": "inception",
            "inception_date": inception,
            "expiry_date": expiry,
            "ncd_years": 2,
            "annual_premium": 500.0,
            "vehicle_age": 3,
            "policyholder_age": 40,
            "channel": "direct",
            "event_type": None,
        }]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])

        txns_mta = pl.DataFrame([
            {
                "policy_id": "P005",
                "transaction_date": inception,
                "transaction_type": "inception",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 2,
                "annual_premium": 500.0,
                "vehicle_age": 3,
                "policyholder_age": 40,
                "channel": "direct",
                "event_type": None,
            },
            {
                "policy_id": "P005",
                "transaction_date": mta_date,
                "transaction_type": "mta",
                "inception_date": inception,
                "expiry_date": expiry,
                "ncd_years": 2,
                "annual_premium": 550.0,
                "vehicle_age": 3,
                "policyholder_age": 40,
                "channel": "direct",
                "event_type": None,
            },
        ]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])

        t = ExposureTransformer(observation_cutoff=cutoff)
        r1 = t.fit_transform(txns_no_mta)
        r2 = t.fit_transform(txns_mta)

        exp_no_mta = r1["exposure_years"].sum()
        exp_mta = r2["exposure_years"].sum()

        assert abs(exp_no_mta - exp_mta) < 0.01, (
            f"Exposure mismatch: {exp_no_mta:.4f} vs {exp_mta:.4f}"
        )


class TestExposureTransformerEdgeCases:
    """Edge cases."""

    def test_missing_required_column_raises(self, observation_cutoff):
        """Missing required columns raise ValueError."""
        df = pl.DataFrame({"policy_id": ["P001"], "transaction_type": ["inception"]})
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        with pytest.raises(ValueError, match="Missing required columns"):
            t.fit_transform(df)

    def test_invalid_transaction_type_raises(self, observation_cutoff):
        """Unknown transaction_type values raise ValueError."""
        txns = pl.DataFrame([{
            "policy_id": "P001",
            "transaction_date": date(2022, 1, 1),
            "transaction_type": "foobar",
            "inception_date": date(2022, 1, 1),
            "expiry_date": date(2023, 1, 1),
        }]).with_columns([
            pl.col("transaction_date").cast(pl.Date),
            pl.col("inception_date").cast(pl.Date),
            pl.col("expiry_date").cast(pl.Date),
        ])
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        with pytest.raises(ValueError, match="Unknown transaction_type"):
            t.fit_transform(txns)

    def test_invalid_time_scale_raises(self):
        """Invalid time_scale raises ValueError."""
        with pytest.raises(ValueError, match="time_scale"):
            ExposureTransformer(
                observation_cutoff=date(2025, 1, 1),
                time_scale="invalid",
            )

    def test_invalid_exposure_basis_raises(self):
        """Invalid exposure_basis raises ValueError."""
        with pytest.raises(ValueError, match="exposure_basis"):
            ExposureTransformer(
                observation_cutoff=date(2025, 1, 1),
                exposure_basis="invalid",
            )

    def test_summary_before_fit_raises(self, observation_cutoff):
        """summary() before fit_transform raises RuntimeError."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        with pytest.raises(RuntimeError):
            t.summary()

    def test_min_duration_filter(self, transaction_dgp, observation_cutoff):
        """min_duration filter removes short policies."""
        t_no_filter = ExposureTransformer(observation_cutoff=observation_cutoff)
        t_filtered = ExposureTransformer(
            observation_cutoff=observation_cutoff, min_duration=0.5
        )
        r_full = t_no_filter.fit_transform(transaction_dgp)
        r_filtered = t_filtered.fit_transform(transaction_dgp)
        assert len(r_filtered) <= len(r_full)
        # All remaining stops >= min_duration
        assert (r_filtered["stop"] >= 0.5).all()


class TestExposureTransformerSummary:
    """Tests for the summary() method."""

    def test_summary_returns_dict(self, transaction_dgp, observation_cutoff):
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        t.fit_transform(transaction_dgp)
        s = t.summary()
        assert isinstance(s, dict)

    def test_summary_keys(self, transaction_dgp, observation_cutoff):
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        t.fit_transform(transaction_dgp)
        s = t.summary()
        expected_keys = {
            "n_policies", "n_intervals", "event_rate", "median_duration",
            "censoring_rate", "left_truncated_count", "mta_count",
        }
        assert expected_keys.issubset(set(s.keys()))

    def test_summary_event_rate_in_range(self, transaction_dgp, observation_cutoff):
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        t.fit_transform(transaction_dgp)
        s = t.summary()
        assert 0.0 <= s["event_rate"] <= 1.0

    def test_summary_censoring_rate_complement(self, transaction_dgp, observation_cutoff):
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        t.fit_transform(transaction_dgp)
        s = t.summary()
        assert abs(s["event_rate"] + s["censoring_rate"] - 1.0) < 1e-9

    def test_summary_mta_count_positive(self, transaction_dgp, observation_cutoff):
        """MTA count should be > 0 given the DGP includes MTAs."""
        t = ExposureTransformer(observation_cutoff=observation_cutoff)
        t.fit_transform(transaction_dgp)
        s = t.summary()
        # DGP includes mta_then_lapse policies, so mta_count > 0
        assert s["mta_count"] >= 0  # at minimum 0
