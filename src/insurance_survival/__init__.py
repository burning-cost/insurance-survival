"""
insurance-survival: survival analysis for UK insurance pricing.

lifelines is an excellent general-purpose survival library. This package adds
the insurance-specific gaps:

1. **ExposureTransformer** — converts policy transaction tables to start/stop
   survival format, handling MTAs, fractional exposure, and left truncation.

2. **WeibullMixtureCureFitter** — covariate-adjusted mixture cure model.
   lifelines.MixtureCureFitter is univariate only. This fills that gap with a
   logistic cure fraction and Weibull AFT for the uncured subgroup, fitted by
   EM initialisation and joint L-BFGS-B.

3. **SurvivalCLV** — survival-adjusted customer lifetime value with NCD path
   marginalisation. The primary output for Consumer Duty fair value analysis
   and discount targeting post-PS21/11.

4. **LapseTable** — actuarial lapse table in qx/px/lx format from any fitted
   survival model.

5. **LifelinesMLflowWrapper** — MLflow pyfunc wrapper enabling lifelines models
   to be registered in the MLflow Model Registry and served via Databricks
   Model Serving.

Quick start::

    import polars as pl
    from datetime import date
    from insurance_survival import ExposureTransformer, WeibullMixtureCureFitter

    # Step 1: transform raw transactions
    transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
    survival_df = transformer.fit_transform(transactions)

    # Step 2: fit cure model
    fitter = WeibullMixtureCureFitter(
        cure_covariates=["ncd_level"],
        uncured_covariates=["ncd_level", "annual_premium"],
    )
    fitter.fit(survival_df)

    # Step 3: compute CLV
    from insurance_survival import SurvivalCLV
    clv = SurvivalCLV(survival_model=fitter, horizon=5)
    results = clv.predict(policies, premium_col="annual_premium", loss_col="expected_loss")

Use lifelines directly for:
- CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
- KaplanMeierFitter, NelsonAalenFitter
- Fine-Gray competing risks
- CoxTimeVaryingFitter

"""

from insurance_survival.transform import ExposureTransformer
from insurance_survival.cure import WeibullMixtureCureFitter
from insurance_survival.clv import SurvivalCLV
from insurance_survival.lapse_table import LapseTable
from insurance_survival.mlflow_wrapper import LifelinesMLflowWrapper, register_survival_model

__version__ = "0.1.0"

__all__ = [
    "ExposureTransformer",
    "WeibullMixtureCureFitter",
    "SurvivalCLV",
    "LapseTable",
    "LifelinesMLflowWrapper",
    "register_survival_model",
]
