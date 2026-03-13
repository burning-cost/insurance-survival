"""
Tests for RecurrentEventData.
"""

import numpy as np
import pandas as pd
import pytest

from insurance_survival.recurrent.data import RecurrentEventData


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def make_simple_df(n_subjects=5, max_events=3, seed=0):
    """Create a simple counting process dataframe."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_subjects):
        t = 0.0
        n_ev = rng.integers(0, max_events + 1)
        for _ in range(n_ev):
            wait = rng.uniform(0.1, 0.5)
            records.append({
                "policy_id": i,
                "t_start": t,
                "t_stop": t + wait,
                "event": 1,
                "age": rng.uniform(20, 70),
                "region": rng.choice(["north", "south"]),
            })
            t = t + wait
        # Final censored interval
        records.append({
            "policy_id": i,
            "t_start": t,
            "t_stop": t + rng.uniform(0.1, 1.0),
            "event": 0,
            "age": rng.uniform(20, 70),
            "region": rng.choice(["north", "south"]),
        })
    return pd.DataFrame(records)


@pytest.fixture
def simple_df():
    return make_simple_df()


@pytest.fixture
def simple_data(simple_df):
    return RecurrentEventData.from_long_format(
        simple_df,
        id_col="policy_id",
        start_col="t_start",
        stop_col="t_stop",
        event_col="event",
        covariates=["age"],
    )


# ------------------------------------------------------------------
# Construction tests
# ------------------------------------------------------------------


class TestFromLongFormat:
    def test_basic_construction(self, simple_df):
        data = RecurrentEventData.from_long_format(
            simple_df,
            id_col="policy_id",
            start_col="t_start",
            stop_col="t_stop",
            event_col="event",
        )
        assert data.n_subjects == 5

    def test_with_covariates(self, simple_df):
        data = RecurrentEventData.from_long_format(
            simple_df, "policy_id", "t_start", "t_stop", "event", covariates=["age"]
        )
        assert "age" in data.covariate_cols
        assert data.X.shape[1] == 1

    def test_sorted_by_id_and_time(self, simple_df):
        shuffled = simple_df.sample(frac=1, random_state=42)
        data = RecurrentEventData.from_long_format(
            shuffled, "policy_id", "t_start", "t_stop", "event"
        )
        # Should be sorted
        df = data.df
        for pid in df["policy_id"].unique():
            grp = df[df["policy_id"] == pid]
            assert list(grp["t_start"]) == sorted(grp["t_start"].tolist())

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="Missing columns"):
            RecurrentEventData.from_long_format(
                simple_df, "policy_id", "t_start", "NONEXISTENT", "event"
            )

    def test_missing_covariate_raises(self, simple_df):
        with pytest.raises(ValueError, match="Covariate column not found"):
            RecurrentEventData.from_long_format(
                simple_df, "policy_id", "t_start", "t_stop", "event",
                covariates=["nonexistent_col"]
            )

    def test_invalid_event_raises(self, simple_df):
        df_bad = simple_df.copy()
        df_bad["event"] = df_bad["event"] * 2  # put 2 in there
        with pytest.raises(ValueError, match="binary"):
            RecurrentEventData.from_long_format(
                df_bad, "policy_id", "t_start", "t_stop", "event"
            )

    def test_invalid_time_order_raises(self):
        df = pd.DataFrame({
            "id": [0, 0],
            "t_start": [1.0, 0.0],
            "t_stop": [0.5, 0.0],  # stop < start
            "event": [0, 0],
        })
        # First row has stop < start, second has stop == start
        with pytest.raises(ValueError):
            RecurrentEventData.from_long_format(df, "id", "t_start", "t_stop", "event")


class TestFromEvents:
    def test_basic(self):
        events_df = pd.DataFrame({
            "policy_id": [0, 0, 1],
            "claim_time": [0.5, 1.2, 0.8],
        })
        follow_up_df = pd.DataFrame({
            "policy_id": [0, 1, 2],
            "end_time": [3.0, 3.0, 3.0],
            "age": [35.0, 45.0, 55.0],
        })
        data = RecurrentEventData.from_events(
            events_df, follow_up_df,
            id_col="policy_id",
            time_col="claim_time",
            end_col="end_time",
            covariates=["age"],
        )
        assert data.n_subjects == 3
        assert data.n_events == 3
        assert data.covariate_cols == ["age"]

    def test_policy_with_no_claims(self):
        events_df = pd.DataFrame({"policy_id": [], "claim_time": []})
        follow_up_df = pd.DataFrame({
            "policy_id": [0, 1],
            "end_time": [2.0, 2.0],
        })
        data = RecurrentEventData.from_events(
            events_df, follow_up_df,
            id_col="policy_id",
            time_col="claim_time",
            end_col="end_time",
        )
        assert data.n_events == 0
        assert data.n_subjects == 2


# ------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------


class TestProperties:
    def test_n_subjects(self, simple_data):
        assert simple_data.n_subjects == 5

    def test_n_events(self, simple_data):
        assert simple_data.n_events == int(simple_data.df["event"].sum())

    def test_subject_ids_sorted(self, simple_data):
        ids = simple_data.subject_ids
        assert list(ids) == sorted(ids.tolist())

    def test_X_shape(self, simple_data):
        assert simple_data.X.shape == (len(simple_data.df), 1)

    def test_X_dtype(self, simple_data):
        assert simple_data.X.dtype == float

    def test_start_stop_event_arrays(self, simple_data):
        assert len(simple_data.start) == len(simple_data.df)
        assert len(simple_data.stop) == len(simple_data.df)
        assert len(simple_data.event) == len(simple_data.df)

    def test_no_covariates_X_empty(self, simple_df):
        data = RecurrentEventData.from_long_format(
            simple_df, "policy_id", "t_start", "t_stop", "event"
        )
        assert data.X.shape == (len(simple_df), 0)

    def test_repr(self, simple_data):
        r = repr(simple_data)
        assert "n_subjects" in r
        assert "n_events" in r


# ------------------------------------------------------------------
# Derived computations
# ------------------------------------------------------------------


class TestDerivedComputations:
    def test_per_subject_summary_shape(self, simple_data):
        summary = simple_data.per_subject_summary()
        assert len(summary) == simple_data.n_subjects
        assert "n_events" in summary.columns
        assert "total_time" in summary.columns

    def test_per_subject_summary_nonneg_time(self, simple_data):
        summary = simple_data.per_subject_summary()
        assert (summary["total_time"] > 0).all()

    def test_event_counts_distribution(self, simple_data):
        counts = simple_data.event_counts()
        assert isinstance(counts, pd.Series)
        assert counts.sum() == simple_data.n_subjects

    def test_event_counts_nonneg_indices(self, simple_data):
        counts = simple_data.event_counts()
        assert (counts.index >= 0).all()

    def test_large_dataset(self):
        """Construct data with 1000 subjects."""
        df = make_simple_df(n_subjects=1000, max_events=5, seed=1)
        data = RecurrentEventData.from_long_format(
            df, "policy_id", "t_start", "t_stop", "event", covariates=["age"]
        )
        assert data.n_subjects == 1000
        assert data.X.shape[1] == 1
