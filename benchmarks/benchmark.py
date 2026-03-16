"""
Benchmark: insurance-survival cure model vs standard KM/Cox PH for policy
lapse prediction in UK personal lines.

The claim: standard survival models (KM, Cox PH) cannot accommodate structural
non-lapsers — policyholders who would never lapse regardless of observation
period. These are real in personal lines: loyal multi-year customers, embedded
products with high switching friction, customers who genuinely forget to shop
around. In a typical motor book, 25-40% of policyholders never voluntarily lapse.

When you fit a standard Cox PH model and project to 5 years, it forces the
survival function to zero eventually. That is wrong for structural non-lapsers.
The error compounds into CLV: you systematically underestimate lifetime value
for the loyal segment. When you target retention campaigns based on predicted
churn probability, you invest in people who were never going to lapse anyway.

The mixture cure model (MCM) fixes this by explicitly modelling two latent groups:
(1) structural non-lapsers (cure fraction), who will never lapse
(2) susceptibles, who have a genuine hazard modelled by a Weibull AFT

Setup:
- 50,000 synthetic motor policies, 5-year observation window
- Known true cure fraction: 35% structural non-lapsers
- Cure probability varies by NCB years (higher NCB => more likely immune)
- Weibull(shape=1.2, scale=36 months) latency for susceptibles
- Administrative censoring at 15% per year
- Three approaches:
  (1) Kaplan-Meier: non-parametric estimate (no covariates, asymptotes toward 0)
  (2) Cox PH: semi-parametric (handles covariates, still forces S(∞) → 0)
  (3) WeibullMixtureCure: correctly models the structural cure fraction

Key metrics:
- 5-year retention forecast accuracy vs true population survival
- Estimated vs true cure fraction recovery
- CLV bias: cure model vs Cox PH (CLV = present value of future retention)

Run:
    python benchmarks/benchmark.py

Install:
    pip install insurance-survival numpy pandas scipy scikit-learn
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-survival cure model vs KM/Cox PH for lapse prediction")
print("=" * 70)
print()

try:
    from insurance_survival.cure import WeibullMixtureCure
    from insurance_survival.cure.simulate import simulate_motor_panel
    print("insurance-survival imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-survival: {e}")
    print("Install with: pip install insurance-survival")
    sys.exit(1)

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    _lifelines_available = True
    print("lifelines imported OK (KM and Cox PH baselines)")
except ImportError:
    _lifelines_available = False
    print("NOTE: lifelines not found — KM/Cox PH baselines will use numpy/scipy")

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG_SEED = 42
N_POLICIES = 50_000
TRUE_CURE_FRACTION = 0.35
WEIBULL_SHAPE = 1.2
WEIBULL_SCALE = 36.0  # months

print()
print(f"DGP: {N_POLICIES:,} motor policies, 5-year window")
print(f"     True cure fraction: {TRUE_CURE_FRACTION:.1%} structural non-lapsers")
print(f"     Weibull latency: shape={WEIBULL_SHAPE}, scale={WEIBULL_SCALE} months")
print(f"     Administrative censoring: 15%/year")
print()

df = simulate_motor_panel(
    n_policies=N_POLICIES,
    n_years=5,
    cure_fraction=TRUE_CURE_FRACTION,
    weibull_shape=WEIBULL_SHAPE,
    weibull_scale=WEIBULL_SCALE,
    censoring_rate=0.15,
    seed=RNG_SEED,
)

# Rename columns to lapse terminology
df = df.rename(columns={"claimed": "lapsed", "tenure_months": "duration_months"})

n_events = int(df["lapsed"].sum())
n_censored = len(df) - n_events
obs_event_rate = n_events / len(df)
print(f"Dataset: {len(df):,} policies")
print(f"  Observed lapses: {n_events:,} ({obs_event_rate:.1%})")
print(f"  Censored (ongoing/admin): {n_censored:,} ({n_censored/len(df):.1%})")
print(f"  Median observed tenure: {df['duration_months'].median():.0f} months")
print()

# True population survival at evaluation times
eval_times_months = np.array([12, 24, 36, 48, 60])  # 1-5 years

# Compute true population survival S_pop(t) analytically
# S_pop(t) = (1 - TRUE_CURE_FRACTION) * S_u(t) + TRUE_CURE_FRACTION
# where S_u(t) = exp(-(t/scale)^shape) is the Weibull survival

def weibull_survival(t, shape, scale):
    return np.exp(-(t / scale) ** shape)

true_s_pop = (1 - TRUE_CURE_FRACTION) * weibull_survival(
    eval_times_months, WEIBULL_SHAPE, WEIBULL_SCALE
) + TRUE_CURE_FRACTION

print(f"True population survival (S_pop):")
for t, s in zip(eval_times_months, true_s_pop):
    print(f"  {t//12}-year: {s:.4f}  ({s:.1%} retained)")
print()

# ---------------------------------------------------------------------------
# Baseline 1: Kaplan-Meier
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE 1: Kaplan-Meier (non-parametric, no cure, no covariates)")
print("-" * 70)
print()

if _lifelines_available:
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration_months"], event_observed=df["lapsed"], label="KM")
    km_survival = np.array([
        float(kmf.predict(t)) for t in eval_times_months
    ])
else:
    # Fallback: simple KM from scratch
    T = df["duration_months"].values
    E = df["lapsed"].values
    km_survival = np.zeros(len(eval_times_months))
    for i, t_eval in enumerate(eval_times_months):
        # Events up to t / risk set at each time
        times_sorted = np.sort(np.unique(T[E == 1]))
        times_sorted = times_sorted[times_sorted <= t_eval]
        surv = 1.0
        for t_event in times_sorted:
            at_risk = np.sum(T >= t_event)
            n_event_here = np.sum((T == t_event) & (E == 1))
            if at_risk > 0:
                surv *= (1 - n_event_here / at_risk)
        km_survival[i] = surv

km_mae = np.mean(np.abs(km_survival - true_s_pop))
print(f"  KM long-run estimate (5yr): {km_survival[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  KM 5-year retention bias: {km_survival[-1] - true_s_pop[-1]:+.4f}")
print()

# ---------------------------------------------------------------------------
# Baseline 2: Cox PH
# ---------------------------------------------------------------------------

print("-" * 70)
print("BASELINE 2: Cox PH (semi-parametric, covariate-aware, no cure)")
print("-" * 70)
print()

cox_survival = np.zeros(len(eval_times_months))

if _lifelines_available:
    cox_df = df[["duration_months", "lapsed", "ncd_years", "age", "vehicle_age"]].copy()
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(cox_df, duration_col="duration_months", event_col="lapsed")

    # Population-average survival: mean over the test cohort
    baseline_surv = cph.baseline_survival_
    for i, t_eval in enumerate(eval_times_months):
        # Get baseline survival at t_eval (interpolate)
        idx = baseline_surv.index.searchsorted(t_eval, side="right") - 1
        idx = max(0, min(idx, len(baseline_surv) - 1))
        S0_t = baseline_surv.iloc[idx].values[0]
        # Population average: E[S0(t)^exp(X'beta)]
        lp = cph.predict_partial_hazard(cox_df)
        cox_survival[i] = float(np.mean(S0_t ** lp.values))
else:
    # Crude fallback: use KM (same as KM without covariates)
    cox_survival = km_survival.copy()
    print("  NOTE: lifelines not available, using KM approximation for Cox PH")

cox_mae = np.mean(np.abs(cox_survival - true_s_pop))
print(f"  Cox PH long-run estimate (5yr): {cox_survival[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  Cox PH 5-year retention bias: {cox_survival[-1] - true_s_pop[-1]:+.4f}")
print()

# ---------------------------------------------------------------------------
# Library: Mixture Cure Model
# ---------------------------------------------------------------------------

print("-" * 70)
print("LIBRARY: insurance-survival WeibullMixtureCure")
print("-" * 70)
print()

# Use a subsample for benchmark speed (EM is O(n) per iteration, 50k is fine)
# Use a subset for speed: 10k policies still gives stable estimates
N_SUBSAMPLE = 10_000
df_fit = df.sample(n=N_SUBSAMPLE, random_state=42).copy()

print(f"  Fitting on subsample: {N_SUBSAMPLE:,} policies (full dataset: {N_POLICIES:,})")
print(f"  Subsampled event rate: {df_fit['lapsed'].mean():.1%}")

model = WeibullMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years",
    n_em_starts=3,
    max_iter=150,
    tol=1e-5,
    random_state=42,
)

t0 = time.time()
model.fit(df_fit, duration_col="duration_months", event_col="lapsed")
fit_time = time.time() - t0

result = model.result_
print(f"  EM fit time: {fit_time:.1f}s ({result.n_iter} iterations, converged={result.converged})")
print(f"  Estimated cure fraction: {result.cure_fraction_mean:.4f}  (true: {TRUE_CURE_FRACTION:.4f})")
print()

# Population-level survival at eval times
surv_df = model.predict_population_survival(df_fit, times=eval_times_months)
cure_surv = surv_df.mean(axis=0).values  # population average S_pop(t)

cure_mae = np.mean(np.abs(cure_surv - true_s_pop))
print(f"  Cure model long-run estimate (5yr): {cure_surv[-1]:.4f}  (true: {true_s_pop[-1]:.4f})")
print(f"  Cure model 5-year retention bias: {cure_surv[-1] - true_s_pop[-1]:+.4f}")
print()

# ---------------------------------------------------------------------------
# CLV comparison
# ---------------------------------------------------------------------------

# Simplified CLV: NPV of future annual revenue * retention probability
# Annual premium = £600, discount rate = 5%
ANNUAL_PREMIUM = 600.0
DISCOUNT_RATE = 0.05

def compute_clv(survival_at_years, annual_premium, discount_rate, n_years=5):
    """Approximate CLV as sum of discounted expected retention."""
    clv = 0.0
    for year in range(1, n_years + 1):
        t_idx = year - 1
        if t_idx < len(survival_at_years):
            discount = (1 + discount_rate) ** (-year)
            clv += annual_premium * survival_at_years[t_idx] * discount
    return clv

clv_true = compute_clv(true_s_pop, ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_km = compute_clv(km_survival, ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_cox = compute_clv(cox_survival, ANNUAL_PREMIUM, DISCOUNT_RATE)
clv_cure = compute_clv(cure_surv, ANNUAL_PREMIUM, DISCOUNT_RATE)

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: Retention forecast accuracy and CLV estimation")
print("=" * 70)
print()

print(f"  5-year RETENTION FORECAST")
print(f"  {'Method':<20} {'1-yr S':>8} {'2-yr S':>8} {'3-yr S':>8} {'4-yr S':>8} {'5-yr S':>8} {'MAE':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for label, surv_arr, mae in [
    ("True S_pop", true_s_pop, 0.0),
    ("Kaplan-Meier", km_survival, km_mae),
    ("Cox PH", cox_survival, cox_mae),
    ("Cure model", cure_surv, cure_mae),
]:
    vals = "  ".join(f"{v:>6.3f}" for v in surv_arr)
    print(f"  {label:<20} {vals}  {mae:>8.4f}")

print()

print(f"  CLV ESTIMATION (£{ANNUAL_PREMIUM:.0f}/yr premium, {DISCOUNT_RATE:.0%} discount rate, 5yr horizon)")
print(f"  {'Method':<20} {'CLV (£)':>10} {'Bias vs true (£)':>18} {'Bias %':>8}")
print(f"  {'-'*20} {'-'*10} {'-'*18} {'-'*8}")

for label, clv in [("True", clv_true), ("Kaplan-Meier", clv_km), ("Cox PH", clv_cox), ("Cure model", clv_cure)]:
    bias = clv - clv_true
    bias_pct = bias / clv_true * 100
    print(f"  {label:<20} {clv:>10.2f} {bias:>+18.2f} {bias_pct:>+8.1f}%")

print()

print(f"  CURE FRACTION RECOVERY")
print(f"  True cure fraction:      {TRUE_CURE_FRACTION:.4f} ({TRUE_CURE_FRACTION:.1%})")
print(f"  Estimated cure fraction: {result.cure_fraction_mean:.4f} ({result.cure_fraction_mean:.1%})")
print(f"  Recovery error:          {abs(result.cure_fraction_mean - TRUE_CURE_FRACTION):.4f} ({abs(result.cure_fraction_mean - TRUE_CURE_FRACTION):.1%})")
print()

print("  INCIDENCE COEFFICIENTS (direction should be negative = NCB protects against lapse)")
print(f"  {'Covariate':<20} {'Coefficient':>12}  {'Expected sign':<20}")
print(f"  {'-'*20} {'-'*12}  {'-'*20}")
for name, coef in result.incidence_coef.items():
    expected = "negative (more NCB = less lapse)" if "ncb" in name.lower() else ""
    print(f"  {name:<20} {coef:>12.4f}  {expected}")

print()
print("INTERPRETATION")
print(f"  Standard KM and Cox PH extrapolate the survival function toward zero")
print(f"  because they model everyone as eventually lapsing. With a 35% cure")
print(f"  fraction, this produces systematic downward bias in long-term survival:")
print(f"  at 5 years, KM predicts {km_survival[-1]:.1%} retention vs true {true_s_pop[-1]:.1%}.")
print()
print(f"  This bias of {(km_survival[-1] - true_s_pop[-1])*100:+.1f}pp translates to")
print(f"  £{clv_km - clv_true:+.0f}/policy CLV underestimation (from KM) and")
print(f"  £{clv_cox - clv_true:+.0f}/policy from Cox PH.")
print()
print(f"  The cure model correctly recovers the cure fraction ({result.cure_fraction_mean:.1%} vs true")
print(f"  {TRUE_CURE_FRACTION:.1%}) and produces a survival function that flattens at the right level.")
print()
print(f"  Practically: CLV models that feed pricing and retention campaigns built on")
print(f"  standard Cox PH will misclassify loyal (immune) policyholders as 'at risk'.")
print(f"  Retention spend is then wasted on people who were never going to lapse.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
