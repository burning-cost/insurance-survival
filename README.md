# insurance-survival
[![Tests](https://github.com/burning-cost/insurance-survival/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-survival/actions/workflows/tests.yml)

Survival analysis for UK insurance pricing. Extends [lifelines](https://lifelines.readthedocs.io/) with the gaps that matter for personal lines pricing teams.

## The problem

lifelines is an excellent general-purpose survival library. The gaps are specific to insurance:

1. **Covariate-adjusted cure models.** lifelines.MixtureCureFitter is univariate only. Insurance data has a genuine never-lapse subgroup (high-NCD, direct debit payers, long-tenure customers). You need a logistic model on the cure fraction, not a single intercept.

2. **Customer lifetime value.** No Python library integrates survival probabilities with premium and loss schedules to produce per-policy CLV. This is the calculation Consumer Duty requires: can you demonstrate that a loyalty discount is CLV-justified?

3. **Actuarial output format.** Actuaries expect qx/px/lx tables. Pricing models produce survival curves. This library bridges them.

4. **MLflow deployment.** lifelines has no native MLflow flavour. You cannot register a WeibullAFTFitter in the Model Registry without a pyfunc wrapper.

This library handles all four. It does not replace lifelines — it calls lifelines for the standard models and adds the insurance-specific layer on top.

## What's in the box

| Class | File | Does what |
|---|---|---|
| `ExposureTransformer` | `transform.py` | Raw policy transactions → start/stop survival format |
| `WeibullMixtureCureFitter` | `cure.py` | Covariate-adjusted mixture cure model (logistic + Weibull AFT) |
| `SurvivalCLV` | `clv.py` | Survival-adjusted CLV with NCD path marginalisation |
| `LapseTable` | `lapse_table.py` | Actuarial lapse table (qx, px, lx, Tx) |
| `LifelinesMLflowWrapper` | `mlflow_wrapper.py` | MLflow pyfunc wrapper for lifelines models |

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
import polars as pl
from datetime import date
from insurance_survival import (
    ExposureTransformer,
    WeibullMixtureCureFitter,
    SurvivalCLV,
    LapseTable,
)

# Step 1: transform raw policy transactions
transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
survival_df = transformer.fit_transform(transactions)

print(transformer.summary())
# {'n_policies': 50000, 'event_rate': 0.31, 'median_duration': 2.4, ...}

# Step 2: fit the cure model
fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_level", "channel_direct"],
    uncured_covariates=["ncd_level", "annual_premium_scaled"],
    penalizer=0.01,
)
fitter.fit(survival_df, duration_col="stop", event_col="event")
print(fitter.summary())

# Step 3: compute CLV for each policy
clv_model = SurvivalCLV(survival_model=fitter, horizon=5, discount_rate=0.05)
results = clv_model.predict(
    policies,
    premium_col="annual_premium",
    loss_col="expected_loss",
)
# results has: policy_id, clv, survival_integral, cure_prob, s_yr1..s_yr5

# Step 4: discount targeting
sensitivity = clv_model.discount_sensitivity(
    policies,
    discount_amounts=[25.0, 50.0, 75.0],
)
# sensitivity has: discount_amount, clv_with_discount, incremental_clv, discount_justified

# Step 5: actuarial lapse table
table = LapseTable(survival_model=fitter, radix=10_000)
df = table.generate(covariate_profile={"ncd_level": 3, "channel_direct": 1})
print(df)
```

## The cure model

The `WeibullMixtureCureFitter` estimates:

```
S(t|x) = pi(x) + (1 - pi(x)) * S_u(t|x)
```

where:

- `pi(x) = sigmoid(gamma_0 + x_cure' gamma)` is the cure fraction (never-lapse probability)
- `S_u(t|x) = exp(-(t/lambda(x))^rho)` is Weibull AFT for the uncured subgroup
- `lambda(x) = exp(beta_0 + x_uncured' beta)` is the AFT scale parameter

Parameters are estimated by EM initialisation followed by joint L-BFGS-B. The R equivalent is `flexsurvcure::flexsurvcure(mixture=TRUE, dist="weibull")`.

The key insurance insight: censored policyholders are ambiguous — they might be genuinely cured (never-lapsers) or simply not-yet-lapsed. The EM algorithm handles this correctly by treating the cure indicator for censored observations as a latent variable.

## CLV methodology

`SurvivalCLV` integrates survival probabilities with a premium/loss schedule:

```
CLV(x) = sum_{t=1}^{T} S(t|x(t)) * (P_t - C_t) / (1+r)^t
```

NCD level advances year by year via a Markov chain. `x(t)` uses the expected NCD at year `t`, computed exactly from the transition matrix (no simulation). This is the right approach for a small discrete state space.

## Consumer Duty and PS21/11

The `predict()` output is audit-friendly: it returns `S(t)` at every year, cure probability, and expected tenure alongside the headline CLV figure. The `discount_sensitivity()` output has an explicit `discount_justified` column. Together these document that discount decisions are CLV-driven, which is the evidence Consumer Duty requires.

## Use lifelines directly for standard models

```python
from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter

# These are not reimplemented here
aft = WeibullAFTFitter().fit(survival_df, duration_col="stop", event_col="event")

# But you can wrap the result in SurvivalCLV or LapseTable
from insurance_survival import SurvivalCLV
clv = SurvivalCLV(survival_model=aft, horizon=5)
```

## Development

```bash
git clone https://github.com/burning-cost/insurance-survival
cd insurance-survival
uv sync --extra dev
uv run pytest tests/ -v
```

Tests run on Databricks. See `notebooks/insurance_survival_demo.ipynb` for a full workflow on synthetic data.

## Dependencies

Required: `polars>=1.0.0`, `lifelines>=0.27.0`, `numpy>=1.24.0`, `scipy>=1.11.0`

Optional: `mlflow` (Model Registry), `openpyxl` (Excel export), `matplotlib` (plots), `catboost` (claim frequency model in SurvivalCLV)
