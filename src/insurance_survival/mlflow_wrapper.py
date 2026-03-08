"""
LifelinesMLflowWrapper: MLflow pyfunc wrapper for lifelines survival models.

lifelines has no MLflow native flavour. This wrapper serialises any lifelines
fitter or WeibullMixtureCureFitter as a pyfunc model, enabling:

- mlflow.pyfunc.log_model() registration
- MLflow Model Serving (REST endpoint)
- Model Registry versioning
- Databricks Model Serving deployment

The predict() method follows the pyfunc contract: pandas DataFrame in,
pandas DataFrame out.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

# MLflow is an optional dependency. We do not import at module level to avoid
# making it a hard requirement for users who only need the fitting classes.


def _require_mlflow() -> Any:
    try:
        import mlflow
        return mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required for LifelinesMLflowWrapper. "
            "Install with: pip install insurance-survival[mlflow]"
        )


class LifelinesMLflowWrapper:
    """MLflow pyfunc wrapper making lifelines survival models serveable.

    lifelines models have no mlflow flavour (unlike sklearn, xgboost, etc.).
    This wrapper serialises any lifelines fitter or WeibullMixtureCureFitter
    as a pyfunc model.

    The predict() method accepts a pandas DataFrame (MLflow requirement) and
    returns survival probabilities at the configured time points.

    Parameters
    ----------
    fitter : Any
        Fitted lifelines fitter or WeibullMixtureCureFitter.
    predict_times : list[float] | None
        Time points (policy years) at which to predict survival probability.
        Default [1, 2, 3, 4, 5].
    predict_clv : bool
        If True, also returns CLV using default SurvivalCLV settings.
        Default False.

    Examples
    --------
    >>> import mlflow
    >>> from insurance_survival import LifelinesMLflowWrapper
    >>>
    >>> wrapper = LifelinesMLflowWrapper(fitter=aft, predict_times=[1, 2, 3, 4, 5])
    >>>
    >>> with mlflow.start_run():
    ...     mlflow.pyfunc.log_model(
    ...         artifact_path="survival_model",
    ...         python_model=wrapper,
    ...         input_example=sample_df.to_pandas(),
    ...     )
    """

    # Filename for the pickled fitter inside MLflow artifacts
    _FITTER_FILENAME = "fitter.pkl"

    def __init__(
        self,
        fitter: Any | None = None,
        predict_times: list[float] | None = None,
        predict_clv: bool = False,
    ) -> None:
        self.fitter = fitter
        self.predict_times = predict_times if predict_times is not None else [1, 2, 3, 4, 5]
        self.predict_clv = predict_clv

    def load_context(self, context: Any) -> None:
        """Called by MLflow when loading the model from artifacts.

        Restores the fitter from the pickled artifact.
        """
        fitter_path = context.artifacts[self._FITTER_FILENAME]
        with open(fitter_path, "rb") as f:
            self.fitter = pickle.load(f)

    def predict(self, context: Any, model_input: Any) -> Any:
        """Predict survival probabilities for input policies.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext
            MLflow context (may be None in testing).
        model_input : pd.DataFrame
            Policy covariates. Column names must match those used at fit time.

        Returns
        -------
        pd.DataFrame
            Columns: S_t1, S_t2, ... S_t{T}.
            Plus cure_prob if WeibullMixtureCureFitter was used.
            Plus clv if predict_clv=True.
        """
        import pandas as pd
        import polars as pl

        from .cure import WeibullMixtureCureFitter

        if self.fitter is None:
            raise RuntimeError(
                "No fitter loaded. Either pass fitter= at construction or "
                "load via MLflow load_model()."
            )

        is_cure = isinstance(self.fitter, WeibullMixtureCureFitter)

        if is_cure:
            df_pl = pl.from_pandas(model_input)
            surv_df = self.fitter.predict_survival_function(
                df_pl, times=self.predict_times
            )
            result = surv_df.to_pandas()

            cure_probs = self.fitter.predict_cure(df_pl)
            result["cure_prob"] = cure_probs.to_numpy()
        else:
            # lifelines fitter
            # Drop non-numeric columns that lifelines cannot use as covariates
            input_for_lifelines = model_input.copy()
            object_cols = [c for c in input_for_lifelines.columns
                           if input_for_lifelines[c].dtype == object]
            if object_cols:
                input_for_lifelines = input_for_lifelines.drop(columns=object_cols)
            sf = self.fitter.predict_survival_function(
                input_for_lifelines, times=self.predict_times
            )
            # lifelines returns (n_times, n_policies), transpose
            sf_t = sf.T.reset_index(drop=True)
            sf_t.columns = [f"S_t{k + 1}" for k in range(len(self.predict_times))]
            result = sf_t

        if self.predict_clv:
            from .clv import SurvivalCLV
            df_pl = pl.from_pandas(model_input) if not is_cure else df_pl
            clv_model = SurvivalCLV(
                survival_model=self.fitter,
                horizon=len(self.predict_times),
            )
            if "annual_premium" in model_input.columns and "expected_loss" in model_input.columns:
                clv_result = clv_model.predict(df_pl)
                result["clv"] = clv_result["clv"].to_numpy()

        return result

    def save_fitter(self, directory: str) -> str:
        """Pickle the fitter to a directory and return the file path.

        Called internally by log_model() via the artifacts dict.
        """
        path = Path(directory) / self._FITTER_FILENAME
        with open(path, "wb") as f:
            pickle.dump(self.fitter, f)
        return str(path)

    def log_model(
        self,
        artifact_path: str,
        input_example: Any | None = None,
        registered_model_name: str | None = None,
    ) -> Any:
        """Log this wrapper as an MLflow pyfunc model.

        Convenience method that handles artifact packaging.

        Parameters
        ----------
        artifact_path : str
            MLflow artifact path (e.g. "survival_model").
        input_example : pd.DataFrame | None
            Example input for schema inference.
        registered_model_name : str | None
            If provided, register the model in the Model Registry.

        Returns
        -------
        mlflow.models.model.ModelInfo
        """
        mlflow = _require_mlflow()

        with tempfile.TemporaryDirectory() as tmp_dir:
            fitter_path = self.save_fitter(tmp_dir)
            artifacts = {self._FITTER_FILENAME: fitter_path}

            return mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=self,
                artifacts=artifacts,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )


def register_survival_model(
    fitter: Any,
    run_id: str,
    model_name: str,
    predict_times: list[float] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Register a fitted lifelines model in the MLflow Model Registry.

    Convenience wrapper around LifelinesMLflowWrapper that also logs standard
    survival model metrics and tags.

    Parameters
    ----------
    fitter : Any
        Fitted lifelines fitter or WeibullMixtureCureFitter.
    run_id : str
        MLflow run ID to log the model under.
    model_name : str
        Model Registry name (e.g. "motor_lapse_survival_v2").
    predict_times : list[float] | None
        Time points for survival prediction. Default [1, 2, 3, 4, 5].
    metrics : dict[str, float] | None
        Additional metrics to log.
    tags : dict[str, str] | None
        Additional tags.

    Returns
    -------
    str
        Model version URI (e.g. "models:/motor_lapse_survival_v2/1").
    """
    mlflow = _require_mlflow()

    wrapper = LifelinesMLflowWrapper(
        fitter=fitter,
        predict_times=predict_times or [1, 2, 3, 4, 5],
    )

    with mlflow.start_run(run_id=run_id):
        # Log standard survival metrics if available
        _log_standard_metrics(fitter, mlflow)

        # Log user-supplied metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Log tags
        default_tags = {
            "library": "insurance-survival",
            "fitter_class": type(fitter).__name__,
        }
        if tags:
            default_tags.update(tags)
        mlflow.set_tags(default_tags)

        with tempfile.TemporaryDirectory() as tmp_dir:
            fitter_path = wrapper.save_fitter(tmp_dir)
            artifacts = {wrapper._FITTER_FILENAME: fitter_path}

            model_info = mlflow.pyfunc.log_model(
                artifact_path="survival_model",
                python_model=wrapper,
                artifacts=artifacts,
                registered_model_name=model_name,
            )

    # Return the model version URI
    if model_info.registered_model_version:
        version = model_info.registered_model_version
        return f"models:/{model_name}/{version}"
    return model_info.model_uri


def _log_standard_metrics(fitter: Any, mlflow: Any) -> None:
    """Log standard survival model metrics available on the fitter."""
    metric_attrs = [
        ("concordance_index_", "c_index"),
        ("AIC_", "aic"),
        ("BIC_", "bic"),
        ("log_likelihood_", "log_likelihood"),
    ]
    for attr, metric_name in metric_attrs:
        if hasattr(fitter, attr):
            val = getattr(fitter, attr)
            if val is not None:
                try:
                    mlflow.log_metric(metric_name, float(val))
                except Exception:
                    pass

    # WeibullMixtureCureFitter convergence metrics
    if hasattr(fitter, "convergence_") and fitter.convergence_:
        conv = fitter.convergence_
        for key in ("log_likelihood", "AIC", "BIC"):
            if key in conv and conv[key] is not None:
                mlflow.log_metric(key.lower(), float(conv[key]))
