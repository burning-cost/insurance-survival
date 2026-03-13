# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-recurrent: Shared Frailty Models for Recurrent Insurance Claims
# MAGIC
# MAGIC This notebook demonstrates the full workflow for fitting frailty models to recurrent
# MAGIC insurance claims data. We simulate fleet motor data (multiple claims per policy),
# MAGIC fit Andersen-Gill with gamma frailty, and interpret the credibility scores.
# MAGIC
# MAGIC **Use cases:**
# MAGIC - Fleet motor: vehicles that claim repeatedly
# MAGIC - Pet insurance: animals with chronic conditions
# MAGIC - Home insurance: properties in flood/subsidence zones
# MAGIC
# MAGIC **What this gives you that Poisson GLM doesn't:**
# MAGIC - Unobserved heterogeneity: the latent "risk type" that covariates don't capture
# MAGIC - Credibility scores: Bühlmann-Straub credibility premiums as a model output
# MAGIC - Informative censoring handling: joint model for claims + lapse

# COMMAND ----------

# MAGIC %pip install insurance-recurrent

# COMMAND ----------

import numpy as np
import pandas as pd

from insurance_survival.recurrent import (
    AndersenGillFrailty,
    FrailtyReport,
    JointData,
    JointFrailtyModel,
    NelsonAalenFrailty,
    PWPModel,
    RecurrentEventData,
    SimulationParams,
    compare_models,
    simulate_ag_frailty,
    simulate_joint,
    simulate_pwp,
)

print(f"insurance-recurrent installed successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Format
# MAGIC
# MAGIC The library uses counting process (start, stop] format. Each row is a risk interval
# MAGIC for one policyholder. A row with event=1 means a claim occurred at time `t_stop`.

# COMMAND ----------

# Illustrative example: two fleet vehicles
example_df = pd.DataFrame({
    "vehicle_id": [101, 101, 101, 102, 102, 103],
    "t_start":    [0.0, 0.5, 1.2, 0.0, 0.8, 0.0],
    "t_stop":     [0.5, 1.2, 3.0, 0.8, 3.0, 3.0],
    "event":      [  1,   1,   0,   1,   0,   0],  # 1 = claim
    "vehicle_age": [2.0, 2.0, 2.0, 5.0, 5.0, 1.0],
    "driver_age":  [28., 28., 28., 52., 52., 35.],
})

data_example = RecurrentEventData.from_long_format(
    example_df,
    id_col="vehicle_id",
    start_col="t_start",
    stop_col="t_stop",
    event_col="event",
    covariates=["vehicle_age", "driver_age"],
)

print(data_example)
print()
print("Event count distribution:")
print(data_example.event_counts())
print()
print("Per-subject summary:")
print(data_example.per_subject_summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simulate Fleet Motor Data
# MAGIC
# MAGIC We simulate 500 policies over 3 years. True parameters:
# MAGIC - Frailty dispersion theta = 2.0 (Var[z] = 0.5, meaningful heterogeneity)
# MAGIC - Covariate effects: vehicle age (+0.3), driver age (-0.2)

# COMMAND ----------

params = SimulationParams(
    n_subjects=500,
    baseline_rate=0.35,      # 35% annual claim frequency
    beta=np.array([0.3, -0.2]),  # vehicle_age+, driver_age-
    theta=2.0,               # frailty dispersion
    follow_up=3.0,           # 3-year observation window
    random_state=42,
)

data = simulate_ag_frailty(params)
print(data)
print()
print("Event count distribution (per policy):")
print(data.event_counts())
print()
print(f"Proportion with 0 claims: {(data.per_subject_summary()['n_events'] == 0).mean():.1%}")
print(f"Proportion with 2+ claims: {(data.per_subject_summary()['n_events'] >= 2).mean():.1%}")
print(f"Maximum claims per policy: {data.per_subject_summary()['n_events'].max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit Andersen-Gill with Gamma Frailty
# MAGIC
# MAGIC The EM algorithm alternates between:
# MAGIC - **E-step**: compute E[z_i | data] for each policyholder
# MAGIC - **M-step**: update beta (weighted partial likelihood) and theta (marginal likelihood)
# MAGIC
# MAGIC For gamma frailty, the E-step is exact: conjugacy gives a closed-form posterior.

# COMMAND ----------

model = AndersenGillFrailty(frailty="gamma", max_iter=50, verbose=True).fit(data)
print()
print("Fit result:", model.result_)
print()
print("Coefficient summary:")
print(model.summary().round(4))
print()
print(f"Frailty dispersion (theta): {model.result_.theta:.4f}  (true: 2.0)")
print(f"Frailty variance (1/theta): {1/model.result_.theta:.4f}  (true: 0.5)")
print(f"Converged: {model.result_.converged}, iterations: {model.result_.n_iter}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Credibility Scores — The Key Output
# MAGIC
# MAGIC The posterior frailty E[z_i | data] is the credibility-adjusted risk multiplier.
# MAGIC This is exactly the Bühlmann-Straub credibility formula:
# MAGIC
# MAGIC     E[z_i | data] = (theta + n_i) / (theta + Lambda_i)
# MAGIC
# MAGIC A score of 2.0 means this policyholder's expected claim rate is twice the model's
# MAGIC prediction after controlling for covariates. A score of 0.5 means half.

# COMMAND ----------

scores = model.credibility_scores()
print("Credibility scores (sample):")
print(scores.head(10).round(3))
print()
print("Score distribution:")
print(scores["frailty_mean"].describe().round(3))
print()

# Verify the Bühlmann-Straub formula manually
theta = model.result_.theta
n_i = scores["n_events"].values.astype(float)
lambda_i = scores["lambda_i"].values
manual_score = (theta + n_i) / (theta + lambda_i)
print("Verification — model scores match Bühlmann-Straub formula:")
print(f"Max absolute difference: {np.abs(scores['frailty_mean'].values - manual_score).max():.2e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Diagnostics

# COMMAND ----------

report = FrailtyReport(model, data)

print("Frailty distribution summary:")
print(report.frailty_summary().round(4))
print()
print(f"AIC: {report.model_aic():.2f}")
print(f"BIC: {report.model_bic():.2f}")
print()
print("Event rate by frailty decile (should be monotone increasing):")
decile_table = report.event_rate_by_frailty_decile()
print(decile_table[["frailty_decile", "n_subjects", "mean_frailty", "mean_events_per_subject"]].round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Gamma vs Lognormal Frailty

# COMMAND ----------

model_gamma = AndersenGillFrailty(frailty="gamma", max_iter=50).fit(data)
model_lognorm = AndersenGillFrailty(frailty="lognormal", max_iter=30).fit(data)

print("Model comparison:")
comparison = compare_models(
    [model_gamma, model_lognorm],
    names=["Gamma frailty", "Lognormal frailty"],
    data=data,
)
print(comparison.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. PWP Model — Stratified by Event Number
# MAGIC
# MAGIC Use this when claim intensity depends on claim history. In motor insurance with
# MAGIC no-claims discount, having a claim changes the policyholder's future behaviour.

# COMMAND ----------

# Simulate PWP data: claim rate increases after each event
pwp_data = simulate_pwp(
    n_subjects=400,
    baseline_rates=(0.30, 0.45, 0.60),  # rates for 1st, 2nd, 3rd+ claims
    beta=np.array([0.25, -0.15]),
    random_state=0,
)

print("PWP simulation:", pwp_data)
print()

model_pwp = PWPModel(time_scale="gap", max_stratum=3).fit(pwp_data)
print("PWP model summary:")
print(model_pwp.summary().round(4))
print()
print("Events by stratum:")
print(model_pwp.result_.stratum_event_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Non-Parametric Frailty (No Covariates)
# MAGIC
# MAGIC Use NelsonAalenFrailty as a quick diagnostic: is there meaningful heterogeneity
# MAGIC in this dataset before building a covariate model?

# COMMAND ----------

na_model = NelsonAalenFrailty().fit(data)
print(f"Estimated theta (no covariates): {na_model.theta_:.4f}")
print(f"Frailty variance (1/theta): {1/na_model.theta_:.4f}")
print()
print("Score sample:")
print(na_model.credibility_scores().head(10).round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Joint Model — Handling Informative Lapse
# MAGIC
# MAGIC When policyholders who lapse are systematically riskier, the claims data from
# MAGIC survivors is a biased sample. The joint model handles this by sharing frailty
# MAGIC between the claims process and the lapse process.

# COMMAND ----------

# Simulate: risky policyholders both claim more AND lapse more
rec_data, terminal_df = simulate_joint(
    n_subjects=400,
    baseline_claim_rate=0.30,
    baseline_lapse_rate=0.25,
    theta=2.0,
    alpha=1.0,   # frailty effect on claims
    gamma=0.6,   # frailty effect on lapse (weaker)
    random_state=42,
)

print("Claims data:", rec_data)
print(f"Lapse rate: {terminal_df['lapsed'].mean():.1%}")
print()

jd = JointData(
    recurrent=rec_data,
    terminal_df=terminal_df,
    id_col="policy_id",
    terminal_time_col="lapse_time",
    terminal_event_col="lapsed",
    terminal_covariates=["x1", "x2"],
)

joint_model = JointFrailtyModel(max_iter=20, verbose=True).fit(jd)
print()
print("Joint model result:", joint_model.result_)
print()
print("Recurrent process coefficients:")
print(joint_model.result_.summary_recurrent().round(4))
print()
print("Terminal (lapse) process coefficients:")
print(joint_model.result_.summary_terminal().round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Fleet Motor Use Case: Full Pipeline
# MAGIC
# MAGIC A realistic example: you have 3 years of fleet motor data. You want to identify
# MAGIC which policies are high-frailty (latently risky) so you can apply experience rating.

# COMMAND ----------

# Simulate realistic fleet motor: 1000 vehicles, 3 years, 2 covariates
fleet_params = SimulationParams(
    n_subjects=1000,
    baseline_rate=0.40,          # 40% frequency
    beta=np.array([0.35, -0.25, 0.15]),  # vehicle_age, driver_age, annual_mileage
    theta=1.5,                   # meaningful heterogeneity
    follow_up=3.0,
    random_state=7,
)
fleet_data = simulate_ag_frailty(fleet_params)
fleet_data.covariate_cols = ["vehicle_age", "driver_age", "annual_mileage"]

# Fit model
fleet_model = AndersenGillFrailty(frailty="gamma", max_iter=50).fit(fleet_data)

print("Fleet model summary:")
print(fleet_model.summary().round(4))
print()
print(f"Estimated frailty variance: {1/fleet_model.result_.theta:.3f}")

# Experience rating bands
scores = fleet_model.credibility_scores()
scores["rating_band"] = pd.cut(
    scores["frailty_mean"],
    bins=[0, 0.7, 0.9, 1.1, 1.3, 99],
    labels=["Very Low", "Low", "Average", "High", "Very High"],
)

print("\nExperience rating band distribution:")
band_summary = (
    scores.groupby("rating_band", observed=True)
    .agg(
        n_policies=("id", "count"),
        mean_frailty=("frailty_mean", "mean"),
        mean_claims=("n_events", "mean"),
    )
    .round(3)
)
print(band_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. **The gap is real**: lifelines has no frailty, scikit-survival has no recurrent events.
# MAGIC    This library fills that gap.
# MAGIC
# MAGIC 2. **Credibility connection**: For gamma frailty, the posterior frailty IS the
# MAGIC    Bühlmann-Straub credibility premium. Frailty models and credibility theory
# MAGIC    are the same thing from different angles.
# MAGIC
# MAGIC 3. **Model choice**:
# MAGIC    - **AndersenGillFrailty (gamma)**: default, fast, interpretable
# MAGIC    - **AndersenGillFrailty (lognormal)**: more flexible, use if gamma AIC is worse
# MAGIC    - **PWPModel**: when intensity genuinely changes after each event (NCD effects)
# MAGIC    - **JointFrailtyModel**: when lapse is informative (correlated with risk)
# MAGIC    - **NelsonAalenFrailty**: diagnostic tool, no covariates needed
# MAGIC
# MAGIC 4. **UK insurance context**: Most practical use is fleet motor, pet, and home.
# MAGIC    Personal motor in the UK has 70%+ of policies with 0 claims in 3 years —
# MAGIC    the frailty signal is there but weak. Fleet data is richer.
