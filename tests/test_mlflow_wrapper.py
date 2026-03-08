"""
Tests for LifelinesMLflowWrapper.

Validates:
- Serialise/deserialise roundtrip (pickle)
- predict() output shape and column names
- Works with both WeibullMixtureCureFitter and lifelines fitters
- register_survival_model() requires mlflow (skipped if not installed)
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from insurance_survival import LifelinesMLflowWrapper


class TestLifelinesMLflowWrapperPredict:
    """Tests for the predict() method (no MLflow required)."""

    def test_predict_returns_dataframe(self, fitted_cure_fitter, small_cure_dgp):
        """predict() returns a pandas DataFrame."""
        import pandas as pd
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=[1, 2, 3],
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        assert isinstance(result, pd.DataFrame)

    def test_predict_n_rows(self, fitted_cure_fitter, small_cure_dgp):
        """predict() returns one row per input policy."""
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=[1, 2, 3],
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        assert len(result) == len(small_cure_dgp)

    def test_predict_column_names(self, fitted_cure_fitter, small_cure_dgp):
        """Survival columns are named S_t1, S_t2, ..."""
        predict_times = [1, 2, 3]
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=predict_times,
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        for k in range(len(predict_times)):
            assert f"S_t{k + 1}" in result.columns

    def test_predict_cure_prob_for_cure_model(self, fitted_cure_fitter, small_cure_dgp):
        """cure_prob column present for WeibullMixtureCureFitter."""
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=[1, 2],
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        assert "cure_prob" in result.columns

    def test_predict_survival_in_range(self, fitted_cure_fitter, small_cure_dgp):
        """All survival probabilities are in [0, 1]."""
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=[1, 2, 3],
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        for col in ["S_t1", "S_t2", "S_t3"]:
            arr = result[col].values
            assert (arr >= 0).all() and (arr <= 1).all()

    def test_predict_with_lifelines_fitter(self, fitted_lifelines_fitter, small_cure_dgp):
        """predict() works with a lifelines WeibullAFTFitter."""
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_lifelines_fitter,
            predict_times=[1, 2, 3],
        )
        pdf = small_cure_dgp.to_pandas()
        result = wrapper.predict(context=None, model_input=pdf)
        assert "S_t1" in result.columns
        assert len(result) == len(small_cure_dgp)

    def test_predict_no_fitter_raises(self, small_cure_dgp):
        """Predicting without a fitter raises RuntimeError."""
        wrapper = LifelinesMLflowWrapper(predict_times=[1, 2, 3])
        pdf = small_cure_dgp.to_pandas()
        with pytest.raises(RuntimeError, match="No fitter loaded"):
            wrapper.predict(context=None, model_input=pdf)


class TestLifelinesMLflowWrapperSerialisation:
    """Serialisation tests."""

    def test_pickle_roundtrip(self, fitted_cure_fitter, small_cure_dgp):
        """Wrapper survives pickle roundtrip."""
        wrapper = LifelinesMLflowWrapper(
            fitter=fitted_cure_fitter,
            predict_times=[1, 2, 3],
        )
        serialised = pickle.dumps(wrapper)
        restored = pickle.loads(serialised)

        pdf = small_cure_dgp.to_pandas()
        original_result = wrapper.predict(context=None, model_input=pdf)
        restored_result = restored.predict(context=None, model_input=pdf)

        np.testing.assert_allclose(
            original_result["S_t1"].values,
            restored_result["S_t1"].values,
            rtol=1e-6,
        )

    def test_save_fitter_creates_file(self, fitted_cure_fitter):
        """save_fitter() writes a pickle file."""
        wrapper = LifelinesMLflowWrapper(fitter=fitted_cure_fitter)
        with tempfile.TemporaryDirectory() as tmp:
            path = wrapper.save_fitter(tmp)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_load_context_restores_fitter(self, fitted_cure_fitter, small_cure_dgp):
        """load_context() restores the fitter from saved file."""
        wrapper = LifelinesMLflowWrapper(fitter=fitted_cure_fitter, predict_times=[1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            fitter_path = wrapper.save_fitter(tmp)

            # Simulate MLflow context
            class FakeContext:
                artifacts = {LifelinesMLflowWrapper._FITTER_FILENAME: fitter_path}

            # New wrapper without fitter
            wrapper2 = LifelinesMLflowWrapper(predict_times=[1, 2])
            wrapper2.load_context(FakeContext())

            # Should now predict correctly
            pdf = small_cure_dgp.to_pandas()
            result = wrapper2.predict(context=None, model_input=pdf)
            assert "S_t1" in result.columns
