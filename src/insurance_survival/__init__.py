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

v0.2.0 adds:
- insurance_survival.cure subpackage: Full mixture cure model suite.
  Covariate-aware logistic incidence with Weibull, log-normal, or Cox latency.
  Multiple EM restarts, bootstrap SEs, non-claimer scoring for UK motor/pet/home.
  See insurance_survival.cure for WeibullMixtureCure, LogNormalMixtureCure,
  CoxMixtureCure, PromotionTimeCure, and supporting diagnostics/simulation.

- insurance_survival.competing_risks subpackage: Fine-Gray subdistribution
  hazard regression and Aalen-Johansen CIF estimation. The only pip-installable
  Fine-Gray implementation with proper IPCW weighting and a lifelines-compatible
  API. See insurance_survival.competing_risks for FineGrayFitter,
  AalenJohansenFitter, gray_test, and evaluation metrics.

- insurance_survival.recurrent subpackage: Shared frailty models for recurrent
  insurance claims. Fills the Python gap (lifelines/scikit-survival are single-
  event only). Connects frailty models to Bühlmann-Straub credibility theory.
  See insurance_survival.recurrent for AndersenGillFrailty, PWPModel,
  JointFrailtyModel, and simulation utilities.

v0.4.0 adds:
- insurance_survival.mortality subpackage: Coherent cause-specific mortality
  forecasting using the Dirichlet-Multinomial-Poisson (DMP) framework from
  Nigri, Shang & Ungolo (2026, arXiv:2603.00973).
  Enforces sum_c m_{a,t,c} = m_{a,t} by construction at every posterior draw —
  no post-hoc reconciliation. Two variants: LC-DM (Lee-Carter, recommended) and
  AP-DM (additive age-period with RW2 priors). HMC via NumPyro (optional dep).
  See insurance_survival.mortality for CauseSpecificMortality, MortalityForecast,
  HMDLoader. Requires: pip install insurance-survival[mortality]

v0.6.0 adds:
- insurance_survival.evaluation subpackage: Proper scoring rules for right-censored
  time-to-event forecasts. Based on Taggart, Loveday & Louis (2026, arXiv:2603.14835).
  Implements threshold-weighted CRPS (twCRPS), quantile score, interval score, Murphy
  diagrams, and reliability diagrams under provisional strict propriety. The confirmed
  gap: no existing library (pycox, scikit-survival, lifelines) provides proper scoring
  rules for predictive survival distributions under censoring.
  See insurance_survival.evaluation for CensoredForecastEvaluator, from_matrix.

Quick start::

    import polars as pl
    from datetime import date
    from insurance_survival import ExposureTransformer, WeibullMixtureCureFitter

    # Step 1: transform raw transactions
    transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
    survival_df = transformer.fit_transform(transactions)

    # Step 2: fit cure model
    fitter = WeibullMixtureCureFitter(
        cure_covariates=["ncd_years"],
        uncured_covariates=["ncd_years", "annual_premium"],
    )
    fitter.fit(survival_df)

    # Step 3: compute CLV
    from insurance_survival import SurvivalCLV
    clv = SurvivalCLV(survival_model=fitter, horizon=5)
    results = clv.predict(policies, premium_col="annual_premium", loss_col="expected_loss")

For mixture cure models (full suite)::

    from insurance_survival.cure import WeibullMixtureCure
    from insurance_survival.cure.simulate import simulate_motor_panel

    df = simulate_motor_panel(n_policies=3000, cure_fraction=0.40, seed=42)
    model = WeibullMixtureCure(
        incidence_formula="ncd_years + age + vehicle_age",
        latency_formula="ncd_years + age",
    )
    model.fit(df, duration_col="tenure_months", event_col="claimed")
    cure_scores = model.predict_cure_fraction(df)

For competing risks::

    from insurance_survival.competing_risks import FineGrayFitter, AalenJohansenFitter

    fg = FineGrayFitter()
    fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
    cif = fg.predict_cumulative_incidence(df_new, times=[1, 2, 3])

For recurrent events with frailty::

    from insurance_survival.recurrent import simulate_ag_frailty, AndersenGillFrailty

    data = simulate_ag_frailty()
    model = AndersenGillFrailty(frailty="gamma").fit(data)
    scores = model.credibility_scores()

v0.5.0 adds:
- insurance_survival.mortality.LifetimeBoundsCalculator: worst/best-case
  contract bounds from life table data without fractional age assumption.
  Implements Dupret & Motte (2026, arXiv:2603.06238). Supports annuity,
  death benefit, and general payoff functionals. CMI S3 table built-in for
  exploratory use. Quantifies model risk from within-year mortality timing.
  See insurance_survival.mortality for LifetimeBoundsCalculator, LifetimeBoundsResult.
  Pure numpy/scipy — no extra install required.

For coherent cause-specific mortality forecasting (CI/LTC/annuity pricing)::

    from insurance_survival.mortality import CauseSpecificMortality, HMDLoader

    deaths, exposure, ages, years = HMDLoader.load_synthetic(
        n_ages=18, n_years=40, n_causes=6
    )
    model = CauseSpecificMortality(model_type="LC")
    model.fit(deaths, exposure, age_labels=ages, year_labels=years)
    forecast = model.forecast(horizon=20)
    forecast.coherence_check()  # True — by construction

For lifetime bounds (within-year mortality model risk)::

    from insurance_survival.mortality import LifetimeBoundsCalculator

    calc = LifetimeBoundsCalculator.from_cmi("S3PML", starting_age=65, n_years=10)
    result = calc.annuity_bounds(t=0.0, T=10.0)
    print(f"Annuity spread: {result.spread_pct:.2f}%")
    df = calc.fractional_age_comparison(0.0, 10.0)
    print(df)  # UDD/CFM/Balducci vs theoretical bounds

Use lifelines directly for:
- CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
- KaplanMeierFitter, NelsonAalenFitter
- CoxTimeVaryingFitter

"""

from insurance_survival.transform import ExposureTransformer
from insurance_survival._cure_legacy import WeibullMixtureCureFitter
from insurance_survival.clv import SurvivalCLV
from insurance_survival.lapse_table import LapseTable
from insurance_survival.mlflow_wrapper import LifelinesMLflowWrapper, register_survival_model

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-survival")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed

__all__ = [
    # Core (v0.1)
    "ExposureTransformer",
    "WeibullMixtureCureFitter",
    "SurvivalCLV",
    "LapseTable",
    "LifelinesMLflowWrapper",
    "register_survival_model",
    # v0.2: subpackages (import from insurance_survival.cure etc.)
    # "cure", "competing_risks", "recurrent"
    # v0.4: mortality subpackage (import from insurance_survival.mortality)
    # "mortality"
]
