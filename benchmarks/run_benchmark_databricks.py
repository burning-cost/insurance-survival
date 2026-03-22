# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-survival: Cure rate model vs Cox PH for policy lapse prediction
# MAGIC
# MAGIC **The problem with standard survival models in insurance:**
# MAGIC When you fit a Cox PH or Kaplan-Meier model to lapse data, it assumes
# MAGIC everyone eventually lapses. For personal lines, that is wrong.
# MAGIC A real UK motor book contains a structural subgroup — loyal, long-tenure
# MAGIC policyholders — who will never voluntarily lapse. Standard models treat
# MAGIC these as right-censored and keep estimating downward survival curves.
# MAGIC
# MAGIC **What we test here:**
# MAGIC - Generate 15,000 synthetic motor policies with a **known 30% cure fraction**
# MAGIC - Fit three models: KM, Cox PH, and WeibullMixtureCure
# MAGIC - Measure: concordance index, cure fraction recovery, CLV estimation accuracy
# MAGIC - Be honest about when Cox PH is close and when it genuinely fails

# COMMAND ----------

import subprocess
import sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *packages])

pip_install("insurance-survival", "lifelines")

print("Dependencies installed.")

# COMMAND ----------

import warnings
import time

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from insurance_survival.cure import WeibullMixtureCure
from insurance_survival.cure.simulate import simulate_motor_panel

warnings.filterwarnings("ignore")

print("=" * 70)
print("Benchmark: insurance-survival cure model vs KM vs Cox PH")
print("Policy lapse prediction with structural non-lapsers")
print("=" * 70)

# COMMAND ----------

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------
# 15,000 motor policies: manageable on Databricks Free Edition.
# True cure fraction: 30% never lapse (structural non-lapsers).
# This is conservative — UK motor books often show 35-40% in our experience.
#
# Covariates:
#   ncd_years [0-9]: No-claims discount years. Higher NCB => more likely immune.
#   age [18-80]:     Driver age.
#   vehicle_age [0-15]: Vehicle age.
#
# The incidence model (who is immune) depends on ncd_years and age.
# The latency model (time-to-lapse for susceptibles) is Weibull(shape=1.2, scale=36mo).

TRUE_CURE_FRACTION = 0.30
N_POLICIES = 15_000
WEIBULL_SHAPE = 1.2
WEIBULL_SCALE = 36.0  # months

print(f"DGP: {N_POLICIES:,} motor policies, 5-year observation window")
print(f"     True cure fraction: {TRUE_CURE_FRACTION:.0%} structural non-lapsers")
print(f"     Latency: Weibull(shape={WEIBULL_SHAPE}, scale={WEIBULL_SCALE} months)")
print(f"     Administrative censoring: 15%/year")
print()

df = simulate_motor_panel(
    n_policies=N_POLICIES,
    n_years=5,
    cure_fraction=TRUE_CURE_FRACTION,
    weibull_shape=WEIBULL_SHAPE,
    weibull_scale=WEIBULL_SCALE,
    censoring_rate=0.15,
    seed=42,
)

# Rename to lapse terminology
df = df.rename(columns={"claimed": "lapsed", "tenure_months": "duration_months"})

n_events   = int(df["lapsed"].sum())
n_censored = len(df) - n_events
true_immune_count = int(df["is_immune"].sum())

print(f"Dataset: {len(df):,} policies")
print(f"  Observed lapses:   {n_events:,} ({n_events/len(df):.1%})")
print(f"  Censored:          {n_censored:,} ({n_censored/len(df):.1%})")
print(f"  True immune:       {true_immune_count:,} ({true_immune_count/len(df):.1%})")
print(f"  Median tenure:     {df['duration_months'].median():.0f} months")

# True population survival (analytic formula for this DGP)
eval_months = np.array([12.0, 24.0, 36.0, 48.0, 60.0])

def weibull_surv(t, shape, scale):
    return np.exp(-(t / scale) ** shape)

true_s_pop = (
    TRUE_CURE_FRACTION
    + (1 - TRUE_CURE_FRACTION) * weibull_surv(eval_months, WEIBULL_SHAPE, WEIBULL_SCALE)
)

print(f"\nTrue population S(t) — the target we're trying to recover:")
for t, s in zip(eval_months, true_s_pop):
    print(f"  {t/12:.0f}-year: {s:.4f}  ({s:.1%} retained)")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
# 80% train, 20% test.
# Test set evaluation: concordance index on held-out policies.

np.random.seed(42)
train_idx = np.random.choice(len(df), size=int(0.8 * len(df)), replace=False)
test_idx  = np.setdiff1d(np.arange(len(df)), train_idx)

df_train = df.iloc[train_idx].copy().reset_index(drop=True)
df_test  = df.iloc[test_idx].copy().reset_index(drop=True)

print(f"Train: {len(df_train):,} policies  |  Test: {len(df_test):,} policies")
print(f"Train event rate: {df_train['lapsed'].mean():.1%}")
print(f"Test event rate:  {df_test['lapsed'].mean():.1%}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Baseline 1: Kaplan-Meier
# ---------------------------------------------------------------------------
# No covariates. Non-parametric. The KM estimator is forced to zero eventually
# because it assigns all probability mass to observed event times. With a 30%
# cure fraction, this produces systematic long-term bias.

print()
print("=" * 70)
print("BASELINE 1: Kaplan-Meier")
print("=" * 70)

t0 = time.time()
kmf = KaplanMeierFitter()
kmf.fit(df_train["duration_months"], event_observed=df_train["lapsed"])
t_km = time.time() - t0

km_s = np.array([float(kmf.predict(t)) for t in eval_months])
km_mae = float(np.mean(np.abs(km_s - true_s_pop)))

# KM has no per-policy risk score, so concordance = 0.5 by construction
km_concordance = 0.5  # no covariate discrimination

print(f"  Fit time:        {t_km:.2f}s")
print(f"  Concordance:     N/A (no covariates)")
print(f"  5-year S(t):     {km_s[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  5-year bias:     {km_s[-1] - true_s_pop[-1]:+.4f}")
print(f"  MAE vs true S:   {km_mae:.4f}")
print()
print("  The KM curve never plateaus — it will hit zero with enough time.")
print("  That is structurally wrong when cure policyholders exist.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Baseline 2: Cox PH
# ---------------------------------------------------------------------------
# Semi-parametric, covariate-aware. Standard choice for lapse modelling.
# Still assumes everyone eventually lapses (S(inf) -> 0).
# Better concordance than KM because it uses NCB, age, vehicle age.
# But the survival function extrapolation is biased for the immune group.

print()
print("=" * 70)
print("BASELINE 2: Cox Proportional Hazards (lifelines)")
print("=" * 70)

cox_cols = ["duration_months", "lapsed", "ncd_years", "age", "vehicle_age"]
t0 = time.time()
cph = CoxPHFitter()
cph.fit(df_train[cox_cols], duration_col="duration_months", event_col="lapsed")
t_cox = time.time() - t0

# Concordance on test set
cox_risk = cph.predict_partial_hazard(df_test[cox_cols])
cox_concordance = concordance_index(
    df_test["duration_months"], -cox_risk, df_test["lapsed"]
)

# Population-average survival: mean over test cohort
cox_s = np.zeros(len(eval_months))
for i, t_eval in enumerate(eval_months):
    # Baseline survival at t_eval
    bl = cph.baseline_survival_
    idx = bl.index.searchsorted(t_eval, side="right") - 1
    idx = max(0, min(idx, len(bl) - 1))
    S0_t = float(bl.iloc[idx].values[0])
    # S(t|x) = S0(t)^exp(x'beta)
    lp = cph.predict_partial_hazard(df_test[cox_cols]).values
    cox_s[i] = float(np.mean(S0_t ** lp))

cox_mae = float(np.mean(np.abs(cox_s - true_s_pop)))

print(f"  Fit time:          {t_cox:.2f}s")
print(f"  Concordance (C):   {cox_concordance:.4f}")
print(f"  5-year S(t):       {cox_s[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  5-year bias:       {cox_s[-1] - true_s_pop[-1]:+.4f}")
print(f"  MAE vs true S:     {cox_mae:.4f}")
print()
print("  Cox PH can rank policyholders (good concordance) but cannot")
print("  correctly estimate long-run survival for the immune subgroup.")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Library: WeibullMixtureCure
# ---------------------------------------------------------------------------
# Mixture cure model with:
# - Incidence sub-model: logistic regression on P(immune | x)
# - Latency sub-model: Weibull AFT on time-to-lapse for susceptibles
# - EM algorithm: alternates between E-step (posterior P(immune)) and M-step
#
# The key claim: the incidence sub-model recovers the true cure fraction
# and produces P(immune) scores that correctly identify non-lapsers.
# The survival function S_pop(t) = P(immune) + P(susceptible)*S_u(t) correctly
# plateaus at the true cure fraction rather than heading to zero.

print()
print("=" * 70)
print("LIBRARY: insurance-survival WeibullMixtureCure")
print("=" * 70)

model = WeibullMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years",
    n_em_starts=3,
    max_iter=200,
    tol=1e-5,
    random_state=42,
)

t0 = time.time()
model.fit(df_train, duration_col="duration_months", event_col="lapsed")
t_cure = time.time() - t0

result = model.result_

# Population survival at eval times (average over test set)
surv_df = model.predict_population_survival(df_test, times=eval_months)
cure_s = surv_df.mean(axis=0).values

cure_mae = float(np.mean(np.abs(cure_s - true_s_pop)))

# Per-policy cure probability for concordance
# Higher P(immune) => less likely to lapse => higher expected survival
# Use (1 - P_immune) as the "risk score" for concordance
cure_prob_test = model.predict_cure_probability(df_test)
cure_concordance = concordance_index(
    df_test["duration_months"],
    -cure_prob_test,    # negate: higher cure prob = lower lapse risk = should be ranked higher survival
    df_test["lapsed"],
)

# Cure fraction recovery
est_cure = float(result.cure_fraction_mean)
cure_recovery_error = abs(est_cure - TRUE_CURE_FRACTION)

print(f"  Fit time:               {t_cure:.1f}s ({result.n_iter} EM iterations, converged={result.converged})")
print(f"  Concordance (C):        {cure_concordance:.4f}")
print(f"  5-year S(t):            {cure_s[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  5-year bias:            {cure_s[-1] - true_s_pop[-1]:+.4f}")
print(f"  MAE vs true S:          {cure_mae:.4f}")
print(f"  Est. cure fraction:     {est_cure:.4f}  (true: {TRUE_CURE_FRACTION:.4f})")
print(f"  Cure fraction error:    {cure_recovery_error:.4f}")
print()

print("  Incidence model coefficients (expected: ncd_years < 0):")
for name, coef in result.incidence_coef.items():
    direction = "correct (higher NCB = more immune)" if ("ncd" in name and coef < 0) else ""
    print(f"    {name:<15} {coef:>+8.4f}  {direction}")

# COMMAND ----------

# ---------------------------------------------------------------------------
# CLV estimation comparison
# ---------------------------------------------------------------------------
# CLV = sum_t S(t) * (premium - loss_cost) * discount_factor
# We compute a simplified version: gross premium only, ignoring loss costs.
# The question is whether Cox PH or the cure model gives closer CLV estimates.
#
# In practice, the cure model's CLV advantage concentrates in the long-tenure
# tail — exactly where Consumer Duty fair value calculations matter most.

ANNUAL_PREMIUM = 600.0   # £/year
DISCOUNT_RATE  = 0.05    # 5% annual

def compute_clv_from_survival(s_at_years, premium, discount_rate):
    """NPV of expected premium income given annual retention probability."""
    clv = 0.0
    for year in range(1, 6):
        t_idx = year - 1
        if t_idx < len(s_at_years):
            clv += premium * s_at_years[t_idx] / (1 + discount_rate) ** year
    return clv

# Eval months are annual [12, 24, 36, 48, 60] so indices map directly to years
clv_true = compute_clv_from_survival(true_s_pop, ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_km   = compute_clv_from_survival(km_s,       ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_cox  = compute_clv_from_survival(cox_s,      ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_cure = compute_clv_from_survival(cure_s,     ANNUAL_PREMIUM, DISCOUNT_RATE)

print()
print("=" * 70)
print("CLV ESTIMATION (£600/yr premium, 5% discount rate, 5yr horizon)")
print("=" * 70)
print()
print(f"  {'Method':<22} {'CLV (£)':>10} {'Bias (£)':>10} {'Bias %':>8}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*8}")
for label, clv in [
    ("True S_pop",    clv_true),
    ("Kaplan-Meier",  clv_km),
    ("Cox PH",        clv_cox),
    ("Cure model",    clv_cure),
]:
    bias     = clv - clv_true
    bias_pct = bias / clv_true * 100
    print(f"  {label:<22} {clv:>10.2f} {bias:>+10.2f} {bias_pct:>+8.1f}%")

# COMMAND ----------

# ---------------------------------------------------------------------------
# Full summary table
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)
print()
print(f"  {'Method':<22} {'C-index':>8} {'5yr S':>8} {'S bias':>8} {'S MAE':>8} {'CLV bias (£)':>13}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*13}")

rows = [
    ("True S_pop",   None,            true_s_pop[-1], 0.0,                        0.0,     0.0),
    ("Kaplan-Meier", None,            km_s[-1],       km_s[-1]-true_s_pop[-1],    km_mae,  clv_km-clv_true),
    ("Cox PH",       cox_concordance, cox_s[-1],      cox_s[-1]-true_s_pop[-1],   cox_mae, clv_cox-clv_true),
    ("Cure model",   cure_concordance,cure_s[-1],     cure_s[-1]-true_s_pop[-1],  cure_mae,clv_cure-clv_true),
]

for label, cindex, s5, s_bias, s_mae, clv_bias in rows:
    c_str = f"{cindex:.4f}" if cindex is not None else "    N/A"
    print(f"  {label:<22} {c_str:>8} {s5:>8.4f} {s_bias:>+8.4f} {s_mae:>8.4f} {clv_bias:>+13.2f}")

print()

# COMMAND ----------

# ---------------------------------------------------------------------------
# Honest interpretation
# ---------------------------------------------------------------------------

print("INTERPRETATION")
print("=" * 70)
print()
print(f"CONCORDANCE: Cox PH and the cure model rank individual lapse risk")
print(f"similarly (both use NCB and age). If you only need to rank")
print(f"policyholders for a retention campaign, Cox PH is competitive.")
print()
print(f"SURVIVAL CALIBRATION: This is where the cure model earns its place.")
print(f"Cox PH pushes all survival curves toward zero eventually. With a")
print(f"true {TRUE_CURE_FRACTION:.0%} cure fraction, the 5-year survival bias is")
print(f"{cox_s[-1]-true_s_pop[-1]:+.4f} (Cox) vs {cure_s[-1]-true_s_pop[-1]:+.4f} (cure model).")
print()
print(f"CLV ESTIMATION: Cox PH produces a CLV bias of £{clv_cox-clv_true:+.0f}/policy")
print(f"vs £{clv_cure-clv_true:+.0f}/policy for the cure model. At portfolio scale,")
print(f"a 10,000-policy book running Cox PH CLV could be misvaluing its")
print(f"renewal book by £{(clv_cox-clv_true)*10000:,.0f} — relevant for PS21/11 fair")
print(f"value assessments.")
print()
print(f"CURE FRACTION RECOVERY: The EM algorithm estimates the cure fraction")
print(f"as {est_cure:.2%} vs the true {TRUE_CURE_FRACTION:.0%} — error of")
print(f"{cure_recovery_error:.2%}. This is the parameter pricing teams care most")
print(f"about: how many of my policyholders will never leave voluntarily?")
print()
print(f"WHEN TO USE STANDARD COX PH INSTEAD:")
print(f"  - You only need risk ranking (concordance), not calibrated survival")
print(f"  - Your product has no structural non-lapsers (e.g. compulsory motor,")
print(f"    where switching is driven by price comparison, not loyalty)")
print(f"  - Your observation window is short (<2 years) — cure effects only")
print(f"    become visible when a substantial subgroup has been censored long")
print(f"    enough that you'd expect them to have lapsed if susceptible")
print()
print(f"Benchmark complete.")
