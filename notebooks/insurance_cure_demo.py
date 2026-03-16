# Databricks notebook source
# insurance-cure: Mixture Cure Models for Motor Non-Claimer Scoring
# Full workflow demo on synthetic UK motor data

# COMMAND ----------
# Install the library
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "insurance-cure"], check=True)

# COMMAND ----------
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from insurance_survival.cure import WeibullMixtureCure, LogNormalMixtureCure, PromotionTimeCure
from insurance_survival.cure.diagnostics import sufficient_followup_test, CureScorecard, cure_fraction_distribution
from insurance_survival.cure.simulate import simulate_motor_panel

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Generate synthetic motor panel
# MAGIC
# MAGIC We simulate a UK motor portfolio with:
# MAGIC - 5,000 policies observed over 5 years
# MAGIC - True cure fraction: 40% (structural non-claimers)
# MAGIC - NCB years, driver age, vehicle age as covariates
# MAGIC - True incidence model: higher NCB => higher P(immune)
# MAGIC - Annual lapse rate: 15% (administrative censoring)

# COMMAND ----------
df = simulate_motor_panel(
    n_policies=5000,
    n_years=5,
    cure_fraction=0.40,
    weibull_shape=1.2,
    weibull_scale=36.0,
    censoring_rate=0.15,
    seed=42,
)
print(f"Dataset: {len(df):,} policies")
print(f"Events (claims): {df['claimed'].sum():,} ({df['claimed'].mean():.1%})")
print(f"True immune rate: {df['is_immune'].mean():.1%}")
print(f"Mean tenure: {df['tenure_months'].mean():.1f} months")
print()
print(df.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Sufficient follow-up test (Maller-Zhou Qn)
# MAGIC
# MAGIC Run this BEFORE fitting. If follow-up is insufficient, the cure fraction
# MAGIC estimate will be upwardly biased.

# COMMAND ----------
qn_result = sufficient_followup_test(df["tenure_months"], df["claimed"])
print(qn_result.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Fit Weibull mixture cure model
# MAGIC
# MAGIC The primary model. Weibull AFT latency with logistic incidence.
# MAGIC Multiple EM restarts to handle multimodality.

# COMMAND ----------
model = WeibullMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years + age",
    n_em_starts=5,
    max_iter=150,
    random_state=42,
)
model.fit(df, duration_col="tenure_months", event_col="claimed")
print(model.result_.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Cure fraction scores

# COMMAND ----------
df["cure_score"] = model.predict_cure_fraction(df)
df["susceptibility"] = model.predict_susceptibility(df)

print("Cure fraction distribution:")
dist = cure_fraction_distribution(df["cure_score"].values)
for k, v in dist.items():
    print(f"  {k:8s}: {v:.4f}")

# Correlation with true immune status
from scipy.stats import pointbiserialr
r, p = pointbiserialr(df["is_immune"].astype(float), df["cure_score"])
print(f"\nCorrelation of cure score with true immune status: r={r:.3f}, p={p:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Population survival curves

# COMMAND ----------
times = [6, 12, 18, 24, 36, 48, 60]
pop_surv = model.predict_population_survival(df, times=times)

print("Mean population survival by time:")
for t in times:
    print(f"  {t:3d} months: {pop_surv[t].mean():.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Cure scorecard — validation

# COMMAND ----------
scorecard = CureScorecard(model, bins=10)
scorecard.fit(df, duration_col="tenure_months", event_col="claimed")
print(scorecard.summary())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Compare incidence by NCB band

# COMMAND ----------
ncb_groups = df.groupby("ncd_years").agg(
    n=("policy_id", "count"),
    true_immune_rate=("is_immune", "mean"),
    predicted_cure=("cure_score", "mean"),
    event_rate=("claimed", "mean"),
).round(4)
print(ncb_groups)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Log-normal MCM for comparison

# COMMAND ----------
model_ln = LogNormalMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years + age",
    n_em_starts=3,
    max_iter=100,
    random_state=42,
)
model_ln.fit(df, duration_col="tenure_months", event_col="claimed")
print(model_ln.result_.summary())

print(f"\nWeibull log-likelihood  : {model.result_.log_likelihood:.2f}")
print(f"Log-normal log-likelihood: {model_ln.result_.log_likelihood:.2f}")
print("(Higher = better fit to this dataset)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 9: Promotion time model

# COMMAND ----------
model_pt = PromotionTimeCure(
    formula="ncd_years + age + vehicle_age",
    random_state=42,
)
model_pt.fit(df, duration_col="tenure_months", event_col="claimed")
print(f"PromotionTime cure fraction: {model_pt.result_['cure_fraction_mean']:.4f}")
print(f"PromotionTime log-likelihood: {model_pt.result_['log_likelihood']:.2f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 10: Key takeaways
# MAGIC
# MAGIC The Weibull MCM correctly identifies:
# MAGIC - Higher NCB => higher cure probability (structural non-claimer)
# MAGIC - The cure fraction estimate is close to the true 40% ground truth
# MAGIC - Cure scores discriminate between immune and susceptible policyholders
# MAGIC
# MAGIC **Production use:**
# MAGIC - Run `sufficient_followup_test` first — if p > 0.05, use with caution
# MAGIC - Use `n_em_starts >= 5` for production models
# MAGIC - Validate with `CureScorecard` — high-cure deciles must show lower observed event rates
# MAGIC - Bootstrap SEs: set `bootstrap_se=True, n_bootstrap=200` for final production model

print("Demo complete.")
