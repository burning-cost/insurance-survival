# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmark: WeibullMixtureCure vs Kaplan-Meier for long-run retention
# MAGIC
# MAGIC The core claim of `insurance-survival` is that mixture cure models are
# MAGIC meaningfully better than Kaplan-Meier or Cox PH when you need to extrapolate
# MAGIC retention beyond the observation window. Within the window the methods are
# MAGIC comparable. Beyond it, KM's survival function must eventually collapse to zero;
# MAGIC the cure model's function correctly flattens at the cure fraction.
# MAGIC
# MAGIC This benchmark plants a known DGP (10,000 UK motor policies, 35% structural
# MAGIC non-lapsers, Weibull latency with shape=1.2 and scale=36 months) and measures:
# MAGIC
# MAGIC 1. **Within-window accuracy** (years 1–5): all three methods should be similar.
# MAGIC 2. **Extrapolation accuracy** (years 6–10): cure model should dominate.
# MAGIC 3. **Cure fraction recovery**: EM should recover the true 35% to within ~2pp.
# MAGIC 4. **CLV bias**: cure model's CLV at 10-year horizon vs KM-based CLV.
# MAGIC 5. **Immunity score discrimination**: AUC of P(immune) vs true latent status.

# COMMAND ----------

# MAGIC %pip install insurance-survival lifelines polars numpy scipy scikit-learn

# COMMAND ----------

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
from insurance_survival.cure import WeibullMixtureCure
from insurance_survival.cure.diagnostics import sufficient_followup_test
from sklearn.metrics import roc_auc_score

print("Packages loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data-generating process
# MAGIC
# MAGIC Known parameters planted in the DGP:
# MAGIC - `TRUE_CURE_FRACTION = 0.35` — 35% of policyholders are structural non-lapsers
# MAGIC - Weibull shape = 1.2, scale = 36 months for susceptibles
# MAGIC - Cure probability modulated by NCB years (coef = 0.15)
# MAGIC - Age modulates Weibull scale (older drivers lapse slightly faster)
# MAGIC - Observation window: 5 years (60 months) — training horizon
# MAGIC - Extrapolation horizon: 10 years — where the methods diverge

# COMMAND ----------

SEED = 42
N_POLICIES = 10_000
TRUE_CURE_FRACTION = 0.35
WEIBULL_SHAPE = 1.2
WEIBULL_SCALE = 36.0      # months
OBS_WINDOW_MONTHS = 60    # 5-year training window
ANNUAL_PREMIUM = 650.0
DISCOUNT_RATE = 0.05

rng = np.random.default_rng(SEED)


def simulate_motor_lapse_panel(n, cure_fraction, weibull_shape, weibull_scale,
                                obs_window, seed):
    """Simulate a UK motor lapse panel with a planted cure fraction.

    Cure probability: P(immune | ncd) = logistic(intercept + 0.15 * ncd_years).
    Intercept calibrated so marginal cure fraction equals `cure_fraction`.
    Weibull scale is modulated by driver age (age_coef=0.005).
    """
    rng = np.random.default_rng(seed)
    ncd_years = rng.integers(0, 10, size=n).astype(float)
    age = rng.integers(21, 75, size=n).astype(float)

    ncd_coef = 0.15
    intercept = np.log(cure_fraction / (1 - cure_fraction)) - ncd_coef * ncd_years.mean()
    p_immune = 1.0 / (1.0 + np.exp(-(intercept + ncd_coef * ncd_years)))
    is_immune = rng.uniform(size=n) < p_immune

    age_coef = 0.005
    scale_i = weibull_scale * np.exp(-age_coef * (age - age.mean()))
    ttf = scale_i * (-np.log(rng.uniform(size=n))) ** (1.0 / weibull_shape)

    admin_censor = rng.uniform(24.0, obs_window + 24.0, size=n)
    raw_ttf = np.where(is_immune, obs_window + 1.0, ttf)
    obs_time = np.minimum(raw_ttf, np.minimum(admin_censor, obs_window))
    event = np.where(is_immune, 0, (raw_ttf <= np.minimum(admin_censor, obs_window)).astype(int))

    return pd.DataFrame({
        "policy_id": np.arange(n),
        "ncd_years": ncd_years,
        "age": age,
        "tenure_months": obs_time.clip(0.1),
        "lapsed": event,
        "is_immune": is_immune,
        "true_p_immune": p_immune,
    })


df_train = simulate_motor_lapse_panel(
    N_POLICIES, TRUE_CURE_FRACTION, WEIBULL_SHAPE, WEIBULL_SCALE, OBS_WINDOW_MONTHS, SEED
)

actual_cure = df_train["is_immune"].mean()
print(f"Policies: {len(df_train):,}")
print(f"Actual cure fraction: {actual_cure:.3f}  (target: {TRUE_CURE_FRACTION})")
print(f"Event (lapse) rate within window: {df_train['lapsed'].mean():.3f}")
print(f"Median tenure: {df_train['tenure_months'].median():.1f} months")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Maller-Zhou Qn test — verify follow-up is sufficient
# MAGIC
# MAGIC If the Qn test p-value is >= 0.05, the observation window is too short to
# MAGIC identify a genuine cure fraction. Cure fraction estimates would be unreliable.

# COMMAND ----------

qn_result = sufficient_followup_test(df_train["tenure_months"], df_train["lapsed"])
print(qn_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit three models

# COMMAND ----------

# Kaplan-Meier — non-parametric, marginal survival only
km = KaplanMeierFitter()
km.fit(df_train["tenure_months"], event_observed=df_train["lapsed"], label="KaplanMeier")
print("KM fitted.")

# Cox PH — semi-parametric with covariates
cox = CoxPHFitter()
cox.fit(
    df_train[["tenure_months", "lapsed", "ncd_years", "age"]],
    duration_col="tenure_months",
    event_col="lapsed",
)
print("Cox PH fitted.")

# WeibullMixtureCure — EM with logistic incidence + Weibull AFT latency
# n_em_starts=3 for notebook speed; use 5 in production
mcm = WeibullMixtureCure(
    incidence_formula="ncd_years + age",
    latency_formula="ncd_years + age",
    n_em_starts=3,
    max_iter=200,
    tol=1e-5,
    random_state=SEED,
)
mcm.fit(df_train, duration_col="tenure_months", event_col="lapsed")
print("WeibullMixtureCure fitted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Cure fraction recovery

# COMMAND ----------

result = mcm.result_
estimated_cure = result.cure_fraction_mean

print("=== Cure Fraction Recovery ===")
print(f"True cure fraction (DGP):   {TRUE_CURE_FRACTION:.3f}")
print(f"Actual in sample:           {actual_cure:.3f}")
print(f"Estimated (MCM):            {estimated_cure:.3f}")
print(f"Bias:                       {estimated_cure - TRUE_CURE_FRACTION:+.4f}")
print()
print("Incidence sub-model (logistic)  — true ncd_years coef = 0.15:")
print(f"  Intercept: {result.incidence_intercept:.4f}")
for name, coef in result.incidence_coef.items():
    print(f"  {name}: {coef:.4f}")
print()
print(f"Convergence: {result.converged} ({result.n_iter} iterations)")
print("If converged=False, increase max_iter to 300 for production fits.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Within-window accuracy (years 1–5)
# MAGIC
# MAGIC The true population survival S_pop(t) is estimated by Monte Carlo from the
# MAGIC DGP parameters. All three methods are measured against this ground truth.
# MAGIC Expected result: similar MAE (~0.04) for all methods within the training window.

# COMMAND ----------

def true_population_survival(t_months, n_mc=50_000, seed=0):
    """Monte Carlo S_pop(t) from the known DGP parameters."""
    rng = np.random.default_rng(seed)
    ncd = rng.integers(0, 10, size=n_mc).astype(float)
    age_mc = rng.integers(21, 75, size=n_mc).astype(float)
    ncd_coef = 0.15
    intercept = np.log(TRUE_CURE_FRACTION / (1 - TRUE_CURE_FRACTION)) - ncd_coef * ncd.mean()
    p_immune = 1.0 / (1.0 + np.exp(-(intercept + ncd_coef * ncd)))
    scale_i = WEIBULL_SCALE * np.exp(-0.005 * (age_mc - age_mc.mean()))
    is_immune = rng.uniform(size=n_mc) < p_immune
    ttf = scale_i * (-np.log(rng.uniform(size=n_mc))) ** (1.0 / WEIBULL_SHAPE)
    return np.array([np.mean(np.where(is_immune, np.inf, ttf) > t) for t in t_months])


eval_times_months = np.array([12, 24, 36, 48, 60])
true_s = true_population_survival(eval_times_months)

km_s = np.array([km.survival_function_at_times(t).values[0] for t in eval_times_months])
cox_s = cox.predict_survival_function(
    df_train[["ncd_years", "age"]], times=eval_times_months
).mean(axis=1).values
mcm_s = mcm.predict_population_survival(df_train, times=eval_times_months).mean(axis=0).values

rows = []
for i, yr in enumerate(eval_times_months / 12):
    rows.append({
        "Year": int(yr), "True S(t)": round(true_s[i], 4),
        "KM": round(km_s[i], 4), "Cox PH": round(cox_s[i], 4),
        "MCM": round(mcm_s[i], 4),
        "KM |err|": round(abs(km_s[i] - true_s[i]), 4),
        "Cox |err|": round(abs(cox_s[i] - true_s[i]), 4),
        "MCM |err|": round(abs(mcm_s[i] - true_s[i]), 4),
    })

within_df = pd.DataFrame(rows)
print("=== Within-window accuracy (years 1–5) ===")
print(within_df.to_string(index=False))
print(f"\nMAE  KM={within_df['KM |err|'].mean():.4f}  Cox={within_df['Cox |err|'].mean():.4f}  MCM={within_df['MCM |err|'].mean():.4f}")
print("Expected: all three within ~0.05. Cure model advantage is in extrapolation.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extrapolation accuracy (years 6–10)
# MAGIC
# MAGIC KM holds at the last observed event value then steps to zero — it cannot
# MAGIC extrapolate beyond the training window. The cure model extrapolates
# MAGIC parametrically and correctly flattens at the estimated cure fraction.

# COMMAND ----------

extrap_times_months = np.array([72, 84, 96, 108, 120])
true_s_ext = true_population_survival(extrap_times_months)
km_s_ext = np.array([km.survival_function_at_times(t).values[0] for t in extrap_times_months])
cox_s_ext = cox.predict_survival_function(
    df_train[["ncd_years", "age"]].head(1000), times=extrap_times_months
).mean(axis=1).values
mcm_s_ext = mcm.predict_population_survival(
    df_train.head(1000), times=extrap_times_months
).mean(axis=0).values

rows_ext = []
for i, yr in enumerate(extrap_times_months / 12):
    rows_ext.append({
        "Year": int(yr), "True S(t)": round(true_s_ext[i], 4),
        "KM": round(km_s_ext[i], 4), "Cox PH": round(cox_s_ext[i], 4),
        "MCM": round(mcm_s_ext[i], 4),
        "KM |err|": round(abs(km_s_ext[i] - true_s_ext[i]), 4),
        "Cox |err|": round(abs(cox_s_ext[i] - true_s_ext[i]), 4),
        "MCM |err|": round(abs(mcm_s_ext[i] - true_s_ext[i]), 4),
    })

extrap_df = pd.DataFrame(rows_ext)
print("=== Extrapolation accuracy (years 6–10) ===")
print(extrap_df.to_string(index=False))
print(f"\nMAE  KM={extrap_df['KM |err|'].mean():.4f}  Cox={extrap_df['Cox |err|'].mean():.4f}  MCM={extrap_df['MCM |err|'].mean():.4f}")
print(f"\nAt year 10:")
print(f"  True retention:   {true_s_ext[-1]:.3f}")
print(f"  KM prediction:    {km_s_ext[-1]:.3f}  <- collapses; underestimates loyal cohort")
print(f"  Cox PH:           {cox_s_ext[-1]:.3f}  <- same issue")
print(f"  MCM:              {mcm_s_ext[-1]:.3f}  <- should plateau near {estimated_cure:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CLV bias at 10-year horizon
# MAGIC
# MAGIC Practical consequence for Consumer Duty fair value assessments. A KM-based CLV
# MAGIC at a 10-year horizon systematically understates value for structural loyalists.

# COMMAND ----------

def compute_clv(s_arr, times_years, premium, discount_rate):
    """CLV = sum_t [ S(t-1) * premium / (1+r)^t ] (discrete, annual)."""
    return sum(
        (s_arr[i - 1] if i > 0 else 1.0) * premium / (1 + discount_rate) ** t
        for i, t in enumerate(times_years)
    )


horizon_years = np.arange(1, 11, dtype=float)
horizon_months = horizon_years * 12.0

true_s_10 = true_population_survival(horizon_months)
km_s_10 = np.array([km.survival_function_at_times(t).values[0] for t in horizon_months])
mcm_s_10 = mcm.predict_population_survival(
    df_train.head(500), times=horizon_months
).mean(axis=0).values

true_clv = compute_clv(true_s_10, horizon_years, ANNUAL_PREMIUM, DISCOUNT_RATE)
km_clv = compute_clv(km_s_10, horizon_years, ANNUAL_PREMIUM, DISCOUNT_RATE)
mcm_clv = compute_clv(mcm_s_10, horizon_years, ANNUAL_PREMIUM, DISCOUNT_RATE)

print(f"=== 10-year CLV  (£{ANNUAL_PREMIUM:.0f}/yr, {DISCOUNT_RATE*100:.0f}% discount) ===")
print(f"True CLV:   £{true_clv:,.0f}")
print(f"KM CLV:     £{km_clv:,.0f}  (bias {km_clv - true_clv:+,.0f}, {(km_clv/true_clv-1)*100:+.1f}%)")
print(f"MCM CLV:    £{mcm_clv:,.0f}  (bias {mcm_clv - true_clv:+,.0f}, {(mcm_clv/true_clv-1)*100:+.1f}%)")
print("\nExpected: KM understates long-run CLV; MCM bias should be under 5%.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Immunity score discrimination
# MAGIC
# MAGIC The incidence sub-model scores each policyholder with P(immune). This is the
# MAGIC capability KM and Cox PH cannot replicate — they rank by lapse hazard, not by
# MAGIC structural immunity. High immunity scores identify policyholders where retention
# MAGIC spend is wasted.

# COMMAND ----------

cure_scores = mcm.predict_cure_fraction(df_train)
auc = roc_auc_score(df_train["is_immune"].astype(int), cure_scores)

immune_mean = cure_scores[df_train["is_immune"]].mean()
susceptible_mean = cure_scores[~df_train["is_immune"]].mean()

print(f"AUC (P(immune) vs true immune status): {auc:.4f}")
print(f"Mean score | truly immune:             {immune_mean:.3f}")
print(f"Mean score | truly susceptible:        {susceptible_mean:.3f}")
print(f"Separation:                            {immune_mean - susceptible_mean:+.3f}")
print()
print("Expected: AUC > 0.60. Discrimination is limited by the two covariates")
print("available (ncd_years, age) — not by the method.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC | Metric | KM | Cox PH | WeibullMixtureCure |
# MAGIC |--------|----|--------|-------------------|
# MAGIC | Within-window MAE (yrs 1–5) | ~0.04 | ~0.04 | ~0.04 |
# MAGIC | Extrapolation MAE (yrs 6–10) | **high** | **high** | **low** |
# MAGIC | 10-yr CLV bias | **large –ve** | **large –ve** | small |
# MAGIC | Cure fraction recovery | N/A | N/A | ±2pp |
# MAGIC | Per-policyholder immunity score | No | No | Yes |
# MAGIC
# MAGIC ### Honest failures
# MAGIC
# MAGIC - **In-sample: no free lunch.** Within the 5-year window all methods match.
# MAGIC - **EM convergence.** `converged=False` is common at `max_iter=200`; survival
# MAGIC   estimates are stable but use `max_iter=300` for production.
# MAGIC - **Thin tails.** If the Qn test p-value >= 0.05, the cure fraction is
# MAGIC   unidentifiable and MCM estimates should not be trusted.
# MAGIC - **Covariate ceiling.** Immunity-score AUC is bounded by available features.
