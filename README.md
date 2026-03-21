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
uv add insurance-survival
```

With optional extras:

```bash
uv add "insurance-survival[mlflow,plot,excel]"
```

> 💬 Questions or feedback? Start a [Discussion](https://github.com/burning-cost/insurance-survival/discussions). Found it useful? A ⭐ helps others find it.

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

ncd_years      = rng.integers(0, 9, n).astype(float)
channel_direct = rng.choice([0, 1], size=n).astype(float)
annual_premium = rng.uniform(300, 1200, n)

transactions = pl.DataFrame({
    "policy_id":        np.arange(1, n + 1),
    "transaction_date": transaction_dates,
    "transaction_type": transaction_types,
    "inception_date":   inception_dates,
    "expiry_date":      expiry_dates,
    "ncd_years":        ncd_years,
    "channel_direct":   channel_direct,
    "annual_premium":   annual_premium,
})

# Step 1: transform raw policy transactions to start/stop survival format
transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
survival_df = transformer.fit_transform(transactions)

# Step 2: fit the cure model (covariates must appear in survival_df output)
fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_years", "channel_direct"],
    uncured_covariates=["ncd_years"],
)
fitter.fit(survival_df, duration_col="stop", event_col="event")

# Step 3: CLV for each policy
# policies DataFrame needs: policy_id, annual_premium, and any CLV covariate columns
policies = pl.DataFrame({
    "policy_id":      np.arange(1, n + 1),
    "annual_premium": annual_premium,
    "expected_loss":  annual_premium * rng.uniform(0.4, 0.8, n),
    "ncd_years":      ncd_years,
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
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years + age",
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

The `SurvivalCLV.predict()` output supports CLV analysis that can form part of a fair value assessment under Consumer Duty. It returns `S(t)` at every year, cure probability, and expected tenure alongside the headline CLV figure. The `discount_sensitivity()` output has an explicit `discount_justified` column. Insurers remain responsible for the full regulatory documentation required under PRIN 12 and GIPP.

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

Run `benchmarks/benchmark.py` to reproduce these results. The benchmark uses 50,000 synthetic UK motor policies with a known 35% structural non-lapse (cure) fraction and a 5-year observation window. Three models are compared against the true data-generating process.

### Retention forecast accuracy (1–5 years)

| Method | 1-yr | 2-yr | 3-yr | 4-yr | 5-yr | MAE |
|---|---|---|---|---|---|---|
| True S_pop | 0.847 | 0.702 | 0.589 | 0.508 | 0.453 | — |
| Kaplan-Meier | 0.874 | 0.745 | 0.643 | 0.561 | 0.503 | 0.046 |
| Cox PH | 0.874 | 0.746 | 0.644 | 0.562 | 0.504 | 0.047 |
| WeibullMixtureCure | 0.878 | 0.751 | 0.645 | 0.562 | 0.499 | 0.047 |

### CLV estimation — within-window comparison (5 years) (£600/yr premium, 5% discount rate)

**Note:** This table compares models within the 5-year observation window only. The cure model's advantage is in extrapolation beyond the observation window — see Honest interpretation below.

| Method | CLV (£) | Bias vs true | Bias % |
|---|---|---|---|
| True | £1,635 | — | — |
| Kaplan-Meier | £1,752 | +£117 | +7.1% |
| Cox PH | £1,754 | +£119 | +7.3% |
| WeibullMixtureCure | £1,756 | +£121 | +7.4% |

### Cure fraction recovery

The EM algorithm recovers the cure fraction to within 0.9pp: estimated 34.1% vs true 35.0%. The EM runs for 150 iterations (max_iter default) on a 10,000-policy subsample in ~91s and exits with converged=False — the survival estimates are stable and the cure fraction accurate, but tighter convergence requires max_iter=300 or tol=1e-6. Incidence coefficients have the right signs: NCB years is negative (−0.31), meaning more NCB experience reduces susceptibility to lapse — the correct actuarial direction.

### Honest interpretation

Within the 5-year observation window, all three models produce similar MAE (~0.046). This is expected and not a failure of the cure model — it is a property of the data structure.

Within the observation window, cured policyholders look like very-long-tenure censored observations. KM correctly reads them as low-lapse-risk based on their survival history. The difference between KM and the cure model becomes decisive only when you extrapolate beyond the observation window. KM's survival function must eventually reach zero (it extrapolates toward the last observed event). The cure model's survival function correctly flattens at the cure fraction (~35%). At 10 years, the cure model would predict ~40% retention while KM would predict near-zero.

This matters for:
- **CLV projections beyond 5 years**: KM-based CLV collapses to zero; cure-model CLV converges to the annuity value of the immune subgroup.
- **Retention campaign targeting**: A Cox PH score ranks all policyholders by lapse hazard. The cure model identifies which policyholders are structurally immune — wasting retention budget on this group is the error to avoid.
- **Pricing new business**: Estimating expected lifetime value for a new customer requires extrapolating beyond the observed retention window. The cure fraction is the single most important long-run parameter.

The positive bias (+7%) in all three models reflects that the 5-year window captures most lapse events for susceptibles (Weibull scale = 36 months, shape = 1.2, so median lapse time ≈ 30 months) but not quite all. The cure model and KM are both reading the tail correctly within sample.

### When to use the cure model vs standard Cox PH

Use `WeibullMixtureCure` when:
- You need to extrapolate survival beyond the observation window
- You want to identify and score the structurally immune (non-lapse) subgroup
- Your CLV model has a horizon longer than your retention data history
- You're building retention campaign selection rules and want to exclude structural loyalists

Cox PH is adequate when:
- You only need within-sample predictions (e.g., 1-year renewal probability)
- Your dataset has no genuine cure fraction (every policyholder will eventually lapse)
- Fit time matters: Cox PH fits in 2–3 seconds on 50k policies; EM fits in 100s on 10k



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

