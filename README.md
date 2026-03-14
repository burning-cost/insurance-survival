# insurance-survival

[![Tests](https://github.com/burning-cost/insurance-survival/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-survival/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-survival)](https://pypi.org/project/insurance-survival/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Survival analysis for UK insurance pricing.

Merged from: `insurance-survival` (core), `insurance-cure` (mixture cure models), `insurance-competing-risks` (Fine-Gray regression), and `insurance-recurrent` (shared frailty models). Extends [lifelines](https://lifelines.readthedocs.io/) with the gaps that matter for personal lines pricing teams.

v0.2.0 adds three subpackages: `cure` (mixture cure models), `competing_risks` (Fine-Gray regression), and `recurrent` (shared frailty models). All three fill confirmed Python ecosystem gaps.

## The problem

lifelines is an excellent general-purpose survival library. The gaps are specific to insurance:

1. **Covariate-adjusted cure models.** lifelines.MixtureCureFitter is univariate only. Insurance data has a genuine never-lapse subgroup (high-NCD, direct debit payers, long-tenure customers). You need a logistic model on the cure fraction, not a single intercept.

2. **Competing risks.** No pip-installable library provides Fine-Gray regression with proper IPCW weighting. For lapse modelling, death and policy cancellation are competing events — you cannot ignore them.

3. **Recurrent events with frailty.** Pet, home, and fleet motor policyholders make multiple claims. Poisson GLMs treat each observation as independent. Frailty models capture unobserved heterogeneity and produce Bühlmann-Straub credibility scores as a by-product.

4. **Customer lifetime value.** No Python library integrates survival probabilities with premium and loss schedules to produce per-policy CLV. This is the calculation Consumer Duty requires.

5. **Actuarial output format.** Actuaries expect qx/px/lx tables. Pricing models produce survival curves. This library bridges them.

6. **MLflow deployment.** lifelines has no native MLflow flavour. You cannot register a WeibullAFTFitter in the Model Registry without a pyfunc wrapper.

## What's in the box

### Core (v0.1)

| Class | Does what |
|---|---|
| `ExposureTransformer` | Raw policy transactions → start/stop survival format |
| `WeibullMixtureCureFitter` | Covariate-adjusted mixture cure model (logistic + Weibull AFT, Polars-native) |
| `SurvivalCLV` | Survival-adjusted CLV with NCD path marginalisation |
| `LapseTable` | Actuarial lapse table (qx, px, lx, Tx) |
| `LifelinesMLflowWrapper` | MLflow pyfunc wrapper for lifelines models |

### `insurance_survival.cure` (v0.2)

Full mixture cure model suite. The primary gap: no Python library provides covariate-aware MCMs with actuarial output. R has smcure, flexsurvcure, cuRe. Python has nothing pip-installable.

| Class | Does what |
|---|---|
| `WeibullMixtureCure` | EM + Weibull AFT latency. Primary workhorse. |
| `LogNormalMixtureCure` | EM + log-normal AFT. Better for non-monotone hazard. |
| `CoxMixtureCure` | EM + semiparametric Cox PH. Most flexible baseline hazard. |
| `PromotionTimeCure` | Non-mixture (Tsodikov 1998). Population-level PH structure. |

### `insurance_survival.competing_risks` (v0.2)

Fine-Gray subdistribution hazard regression and Aalen-Johansen CIF estimation. The only pip-installable Fine-Gray implementation with proper IPCW weighting.

| Class/function | Does what |
|---|---|
| `FineGrayFitter` | Fine-Gray subdistribution hazard regression |
| `AalenJohansenFitter` | Non-parametric CIF estimation |
| `gray_test` | Gray's K-sample test for CIF equality across groups |
| `competing_risks_brier_score` | Proper scoring rule for competing risks models |
| `competing_risks_c_index` | Concordance index adapted for competing risks |

### `insurance_survival.recurrent` (v0.2)

Shared frailty models for recurrent insurance claims. Python has no shared frailty implementation (lifelines GitHub issue #878, closed as "maybe someday").

| Class | Does what |
|---|---|
| `AndersenGillFrailty` | Andersen-Gill model with gamma or log-normal frailty |
| `PWPModel` | Prentice-Williams-Peterson gap-time or calendar-time model |
| `NelsonAalenFrailty` | Non-parametric baseline with parametric frailty |
| `JointFrailtyModel` | Joint model for recurrent events and terminal event |
| `FrailtyReport` | Model comparison and credibility score output |

## Installation

```bash
pip install insurance-survival
```

With optional extras:

```bash
pip install "insurance-survival[mlflow,plot,excel]"
```

## Quick start

```python
import numpy as np
import polars as pl
from datetime import date, timedelta
from insurance_survival import (
    ExposureTransformer,
    WeibullMixtureCureFitter,
    SurvivalCLV,
    LapseTable,
)

# Synthetic UK motor policy transaction table — 1,000 policies
# ExposureTransformer requires: policy_id, transaction_date, transaction_type,
# inception_date, expiry_date. Optional covariates are passed through.
rng = np.random.default_rng(42)
n = 1_000

inception_dates = [date(2021, 1, 1) + timedelta(days=int(d))
                   for d in rng.integers(0, 730, n)]
expiry_dates    = [d + timedelta(days=365) for d in inception_dates]

# 35% of policies lapsed mid-year (cancellation), 65% ran to expiry
lapsed = rng.uniform(size=n) < 0.35
transaction_types = [
    "cancellation" if lapsed[i] else "nonrenewal"
    for i in range(n)
]
# Cancellations happen at a random point during the policy year
transaction_dates = [
    inception_dates[i] + timedelta(days=int(rng.integers(30, 340)))
    if lapsed[i] else expiry_dates[i]
    for i in range(n)
]

ncd_level      = rng.integers(0, 9, n).astype(float)
channel_direct = rng.choice([0, 1], size=n).astype(float)
annual_premium = rng.uniform(300, 1200, n)

transactions = pl.DataFrame({
    "policy_id":        np.arange(1, n + 1),
    "transaction_date": transaction_dates,
    "transaction_type": transaction_types,
    "inception_date":   inception_dates,
    "expiry_date":      expiry_dates,
    "ncd_level":        ncd_level,
    "channel_direct":   channel_direct,
    "annual_premium":   annual_premium,
})

# Step 1: transform raw policy transactions to start/stop survival format
transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
survival_df = transformer.fit_transform(transactions)

# Step 2: fit the cure model (covariates must appear in survival_df output)
fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_level", "channel_direct"],
    uncured_covariates=["ncd_level"],
)
fitter.fit(survival_df, duration_col="stop", event_col="event")

# Step 3: CLV for each policy
# policies DataFrame needs: policy_id, annual_premium, and any CLV covariate columns
policies = pl.DataFrame({
    "policy_id":      np.arange(1, n + 1),
    "annual_premium": annual_premium,
    "expected_loss":  annual_premium * rng.uniform(0.4, 0.8, n),
    "ncd_level":      ncd_level,
    "channel_direct": channel_direct,
})

clv_model = SurvivalCLV(survival_model=fitter, horizon=5, discount_rate=0.05)
results = clv_model.predict(policies, premium_col="annual_premium", loss_col="expected_loss")
```

### Full mixture cure model suite

```python
from insurance_survival.cure import WeibullMixtureCure, LogNormalMixtureCure
from insurance_survival.cure.simulate import simulate_motor_panel
from insurance_survival.cure.diagnostics import sufficient_followup_test

df = simulate_motor_panel(n_policies=5000, cure_fraction=0.40, seed=42)

# Always check sufficient follow-up before trusting cure fraction estimates
qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
print(qn.summary())

model = WeibullMixtureCure(
    incidence_formula="ncb_years + age + vehicle_age",
    latency_formula="ncb_years + age",
    n_em_starts=5,
)
model.fit(df, duration_col="tenure_months", event_col="claimed")

# Primary output: per-policyholder non-claimer probability
cure_scores = model.predict_cure_fraction(df)
```

### Competing risks

```python
import numpy as np
import pandas as pd
from insurance_survival.competing_risks import FineGrayFitter, AalenJohansenFitter

# Synthetic competing risks dataset: 1,000 policies
# Event codes: 0 = censored, 1 = lapse at renewal, 2 = mid-term cancellation
rng = np.random.default_rng(42)
n = 1_000
T = rng.exponential(3.0, n).clip(0.1, 10.0)  # observed time in policy years
# Assign events: 40% censored, 35% lapse, 25% mid-term cancellation
E = rng.choice([0, 1, 2], size=n, p=[0.40, 0.35, 0.25])
ncd_years = rng.integers(0, 9, n).astype(float)
age       = rng.integers(25, 70, n).astype(float)

df_cr = pd.DataFrame({"T": T, "E": E, "ncd_years": ncd_years, "age": age})
df_new = df_cr.head(50).copy()  # hold-out for prediction

fg = FineGrayFitter()
fg.fit(df_cr, duration_col="T", event_col="E", event_of_interest=1)
print(fg.summary)

# Sub-distribution CIF at 1, 2, 3 years
cif = fg.predict_cumulative_incidence(df_new, times=[1, 2, 3])
```

### Recurrent events with frailty

```python
from insurance_survival.recurrent import simulate_ag_frailty, AndersenGillFrailty

data = simulate_ag_frailty()
model = AndersenGillFrailty(frailty="gamma").fit(data)
print(model.summary())

# Bühlmann-Straub credibility scores (gamma frailty posterior means)
scores = model.credibility_scores()
```

## The credibility connection

For gamma frailty, the posterior mean frailty is:

```
E[z_i | data] = (theta + n_i) / (theta + Lambda_i)
```

This is the Bühlmann-Straub credibility formula. The frailty model and classical credibility theory arrive at the same result from different directions. The frailty model gives you the correct statistical machinery; credibility theory gives you the actuarial interpretation.

## Consumer Duty and PS21/11

The `SurvivalCLV.predict()` output is audit-friendly: it returns `S(t)` at every year, cure probability, and expected tenure alongside the headline CLV figure. The `discount_sensitivity()` output has an explicit `discount_justified` column. Together these document that discount decisions are CLV-driven, which is the evidence Consumer Duty requires.

## Development

Tests run on Databricks (612 tests). See `notebooks/` for full workflow demos on synthetic data.

```bash
git clone https://github.com/burning-cost/insurance-survival
cd insurance-survival
uv sync --extra dev
python run_tests_databricks.py
```

## Dependencies

Required: `polars>=1.0.0`, `lifelines>=0.27.0`, `numpy>=1.24.0`, `scipy>=1.11.0`, `pandas>=2.0`, `scikit-learn>=1.1`, `matplotlib>=3.7.0`

Optional: `mlflow` (Model Registry), `openpyxl` (Excel export), `catboost` (claim frequency model in SurvivalCLV)

## Read more

[Survival Models for Insurance Retention](https://burning-cost.github.io/2026/03/08/survival-models-for-insurance-retention.html) — why logistic churn models get renewal pricing wrong and how cure models fix it.

## Performance

No formal benchmark yet. The library fills confirmed Python ecosystem gaps (covariate-adjusted cure models, Fine-Gray regression, shared frailty), so the relevant comparison is against attempting to implement these from scratch rather than against alternative pip-installable libraries. Some directional results from the synthetic demo notebooks:

- **WeibullMixtureCureFitter vs standard WeibullAFTFitter (lifelines):** The cure model correctly identifies the never-lapse subgroup (cure fraction estimation within 3% of true value on 5,000-policy simulations) where the standard AFT fitter underestimates long-term survival because it treats cured individuals as late censored observations.
- **FineGrayFitter vs cause-specific Cox (1-CIF workaround):** The cause-specific approach overestimates the event-1 CIF when competing risks are common (e.g., mid-term cancellations are 20%+ of exits). Fine-Gray subdistribution hazard gives correctly calibrated CIF estimates.
- **AndersenGillFrailty theta estimation:** On simulated data with known theta=2.0, the EM algorithm recovers theta within ±0.3 at n=500 policyholders with 3+ events each. Estimation is unreliable below 100 policyholders or when average events per subject is below 1.5.



## Databricks Notebook

A ready-to-run Databricks notebook benchmarking this library against standard approaches is available in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_survival_demo.py).

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Demand and elasticity modelling — survival gives you tenure, demand gives you price sensitivity |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained portfolio rate optimisation — uses CLV and retention outputs from this library |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring — PSI and A/E drift tracking for deployed retention models |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK motor and home datasets — use to prototype before applying to real data |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Conversion and retention modelling — survival models complement lapse probability with multi-period CLV projections |
| [insurance-telematics](https://github.com/burning-cost/insurance-telematics) | Telematics pricing — survival models apply to telematics-based churn and usage-based policy attrition |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Causal price elasticity — pairs with survival models to understand price-driven lapse causally |

