# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-competing-risks: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the complete workflow for competing risks analysis
# MAGIC using the `insurance-competing-risks` library. We cover:
# MAGIC
# MAGIC 1. Generating synthetic insurance retention data (lapse vs MTC vs NTU)
# MAGIC 2. Non-parametric Aalen-Johansen CIF estimation
# MAGIC 3. Gray's test for CIF comparison across groups
# MAGIC 4. Fine-Gray subdistribution hazard regression
# MAGIC 5. CIF prediction and partial effects
# MAGIC 6. Model evaluation: Brier score and C-index
# MAGIC
# MAGIC **Insurance context**: a motor insurance book with three competing exit types:
# MAGIC - Cause 1: lapse (customer did not renew)
# MAGIC - Cause 2: mid-term cancellation (MTC)
# MAGIC - Cause 3: non-taken-up (NTU)
# MAGIC
# MAGIC The event of primary interest is lapse (cause 1).

# COMMAND ----------

# MAGIC %pip install insurance-competing-risks lifelines matplotlib

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_survival.competing_risks import (
    AalenJohansenFitter,
    FineGrayFitter,
    gray_test,
    competing_risks_brier_score,
    competing_risks_c_index,
    simulate_competing_risks,
)
from insurance_survival.competing_risks.datasets import simulate_insurance_retention
from insurance_survival.competing_risks.cif import plot_stacked_cif
from insurance_survival.competing_risks.metrics import (
    calibration_curve,
    integrated_brier_score,
    plot_calibration,
)
from insurance_survival.competing_risks.plots import plot_forest, plot_cumulative_hazard

print("insurance-competing-risks loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Insurance Retention Data

# COMMAND ----------

# Generate data: 2,000 motor policies
df = simulate_insurance_retention(n=2000, seed=0)

print(f"Dataset shape: {df.shape}")
print(f"\nEvent distribution:")
print(df["E"].value_counts().sort_index().rename({
    0: "Censored",
    1: "Lapse",
    2: "MTC",
    3: "NTU",
}))
print(f"\nTime statistics:")
print(df["T"].describe())
print(f"\nFirst 5 rows:")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Non-Parametric CIF: Aalen-Johansen Estimator

# COMMAND ----------

# Fit CIF for each cause
aj_lapse = AalenJohansenFitter()
aj_lapse.fit(df["T"], df["E"], event_of_interest=1, label="Lapse")

aj_mtc = AalenJohansenFitter()
aj_mtc.fit(df["T"], df["E"], event_of_interest=2, label="MTC")

aj_ntu = AalenJohansenFitter()
aj_ntu.fit(df["T"], df["E"], event_of_interest=3, label="NTU")

# Check CIF values at key time points
for t in [0.25, 0.5, 1.0]:
    l = aj_lapse.predict(np.array([t]))[0]
    m = aj_mtc.predict(np.array([t]))[0]
    n = aj_ntu.predict(np.array([t]))[0]
    print(f"t={t:.2f}: Lapse={l:.3f}, MTC={m:.3f}, NTU={n:.3f}, Sum={l+m+n:.3f}")

# COMMAND ----------

# Plot individual CIFs with confidence bands
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

aj_lapse.plot(ax=axes[0])
axes[0].set_title("Lapse CIF (cause 1)")

aj_mtc.plot(ax=axes[1])
axes[1].set_title("MTC CIF (cause 2)")

aj_ntu.plot(ax=axes[2])
axes[2].set_title("NTU CIF (cause 3)")

plt.tight_layout()
plt.savefig("/tmp/cif_individual.png", dpi=100, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

# Stacked CIF plot: shows total exit probability by type
fig, ax = plt.subplots(figsize=(8, 5))
plot_stacked_cif(
    df["T"], df["E"],
    causes=[1, 2, 3],
    cause_labels={1: "Lapse", 2: "MTC", 3: "NTU"},
    ax=ax,
)
ax.set_title("Stacked Cumulative Incidence by Exit Type")
plt.tight_layout()
plt.savefig("/tmp/cif_stacked.png", dpi=100, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gray's Test: Does Lapse CIF Differ by Premium Uplift Band?

# COMMAND ----------

# Create premium uplift quintile groups
df["premium_band"] = pd.qcut(
    df["premium_uplift"], q=3, labels=["Low (<+5%)", "Medium", "High (>+20%)"]
)

# Gray's test for lapse CIF comparison across premium bands
result = gray_test(
    df["T"], df["E"],
    df["premium_band"],
    event_of_interest=1,
)
print(result)
print()
if result.significant:
    print("Conclusion: Lapse CIFs differ significantly across premium bands (p < 0.05)")
else:
    print("Conclusion: No significant difference in lapse CIFs across premium bands")

# COMMAND ----------

# Plot CIF by premium band
fig, ax = plt.subplots(figsize=(8, 5))

for band in sorted(df["premium_band"].unique()):
    mask = df["premium_band"] == band
    fitter = AalenJohansenFitter()
    fitter.fit(df["T"][mask], df["E"][mask], event_of_interest=1, label=str(band))
    fitter.plot(ax=ax, ci=False)

ax.set_title(f"Lapse CIF by Premium Band\n(Gray's test p={result.p_value:.4f})")
ax.legend()
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fine-Gray Regression: Modelling Lapse as Subdistribution Hazard

# COMMAND ----------

# Train/test split
train = df.iloc[:1600].copy()
test = df.iloc[1600:].copy()

covariate_cols = ["T", "E", "premium_uplift", "tenure_years", "ncd_years"]

fg = FineGrayFitter()
fg.fit(
    train[covariate_cols],
    duration_col="T",
    event_col="E",
    event_of_interest=1,
)

fg.print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpreting the subdistribution hazard ratios (SHRs)**:
# MAGIC
# MAGIC - `premium_uplift` SHR > 1: higher premium uplift increases the subdistribution hazard for lapse,
# MAGIC   meaning higher lapse probability. This is the expected direction.
# MAGIC - `tenure_years` SHR < 1: longer-tenured customers have lower subdistribution hazard — they are
# MAGIC   less likely to lapse. Classic customer loyalty effect.
# MAGIC - `ncd_years` SHR < 1: customers with more NCD years are less likely to lapse, likely because
# MAGIC   they have more to lose by switching.

# COMMAND ----------

# Forest plot of SHRs
fig, ax = plt.subplots(figsize=(8, 4))
plot_forest(fg, ax=ax)
plt.tight_layout()
plt.savefig("/tmp/forest_plot.png", dpi=100, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. CIF Prediction and Partial Effects

# COMMAND ----------

# Predict CIF for test set at 3-, 6-, 12-month marks
times = np.array([0.25, 0.5, 1.0])
cif_test = fg.predict_cumulative_incidence(test[covariate_cols], times=times)

print("Predicted lapse CIF for first 5 test subjects:")
print(cif_test.head().round(3))
print()
print("Mean predicted lapse probability by time horizon:")
for t, col in zip(times, cif_test.columns):
    print(f"  {t:.2f} years: {cif_test[col].mean():.3f}")

# COMMAND ----------

# Partial effects: how does premium uplift affect lapse probability?
fig, ax = plt.subplots(figsize=(8, 5))
fg.plot_partial_effects_on_outcome(
    "premium_uplift",
    values=[-0.05, 0.10, 0.20, 0.35],
    ax=ax,
)
ax.set_title("Partial Effect of Premium Uplift on Lapse CIF\n(all other covariates at mean)")
ax.set_xlabel("Time (policy years)")
plt.tight_layout()
plt.savefig("/tmp/partial_effects.png", dpi=100, bbox_inches="tight")
display(plt.gcf())
plt.close()

# COMMAND ----------

# What does a 10% premium uplift look like vs. 30%?
for uplift in [0.0, 0.10, 0.20, 0.30]:
    df_point = pd.DataFrame({
        "premium_uplift": [uplift],
        "tenure_years": [3.0],
        "ncd_years": [5],
    })
    cif_1yr = fg.predict_cumulative_incidence(df_point, times=[1.0]).values[0, 0]
    print(f"  Premium uplift {uplift*100:.0f}%: predicted 1-yr lapse prob = {cif_1yr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Evaluation

# COMMAND ----------

# Brier score over time
times_eval = np.linspace(0.1, 1.5, 20)
cif_eval = fg.predict_cumulative_incidence(test[covariate_cols], times=times_eval)

bs = competing_risks_brier_score(
    cif_eval,
    test["T"].values, test["E"].values,
    train["T"].values, train["E"].values,
    times_eval, event_of_interest=1
)

ibs = integrated_brier_score(
    cif_eval,
    test["T"].values, test["E"].values,
    train["T"].values, train["E"].values,
    times_eval, event_of_interest=1
)

print(f"Integrated Brier Score: {ibs:.4f}")
print(f"(Reference: 0.25 = null model predicting constant 50% probability)")

# COMMAND ----------

from insurance_survival.competing_risks.plots import plot_brier_score

fig, ax = plt.subplots(figsize=(8, 4))
plot_brier_score(times_eval, bs, ax=ax)
ax.set_title(f"IPCW Brier Score Over Time (IBS={ibs:.4f})")
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# C-index
eval_time = 0.75
c_idx = competing_risks_c_index(
    cif_eval,
    test["T"].values, test["E"].values,
    train["T"].values, train["E"].values,
    eval_time=eval_time, event_of_interest=1
)
print(f"C-index at t={eval_time}: {c_idx:.4f}")
print(f"(0.5 = random model, 1.0 = perfect discrimination)")

# COMMAND ----------

# Calibration curve
calib = calibration_curve(
    cif_eval,
    test["T"].values, test["E"].values,
    eval_time=0.5, event_of_interest=1, n_quantiles=5
)
print("\nCalibration at 6 months (5 quantile groups):")
print(calib[["mean_predicted", "observed", "n_subjects"]].round(3))

fig, ax = plt.subplots(figsize=(6, 6))
plot_calibration(calib, ax=ax, title="Lapse CIF Calibration at 6 Months")
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Competing Models: Lapse vs MTC

# COMMAND ----------

# Fit Fine-Gray for MTC (cause 2) using same covariates
fg_mtc = FineGrayFitter()
fg_mtc.fit(
    train[covariate_cols],
    duration_col="T",
    event_col="E",
    event_of_interest=2,
)
fg_mtc.print_summary()

# COMMAND ----------

# Compare SHRs across causes
print("Comparison of subdistribution hazard ratios: Lapse vs MTC")
print("=" * 60)

for cov in ["premium_uplift", "tenure_years", "ncd_years"]:
    shr_lapse = np.exp(fg.params_[cov])
    shr_mtc = np.exp(fg_mtc.params_[cov])
    print(f"{cov:20s}  Lapse SHR={shr_lapse:.3f}  MTC SHR={shr_mtc:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Classic Bone Marrow Transplant Benchmark

# COMMAND ----------

from insurance_survival.competing_risks import load_bone_marrow_transplant

bmt = load_bone_marrow_transplant()
print(f"BMT dataset: {len(bmt)} patients")
print(f"Event distribution: {bmt['E'].value_counts().to_dict()}")

# Fit CIF
aj_relapse = AalenJohansenFitter()
aj_relapse.fit(bmt["T"], bmt["E"], event_of_interest=1, label="Relapse")

aj_trd = AalenJohansenFitter()
aj_trd.fit(bmt["T"], bmt["E"], event_of_interest=2, label="TRD")

print("\nCIF at 500 days:")
print(f"  Relapse: {aj_relapse.predict(np.array([500]))[0]:.3f}")
print(f"  TRD:     {aj_trd.predict(np.array([500]))[0]:.3f}")

# Regression on group
fg_bmt = FineGrayFitter()
fg_bmt.fit(
    bmt[["T", "E", "group"]],
    duration_col="T",
    event_col="E",
    event_of_interest=1,
)
fg_bmt.print_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated the full workflow for competing risks analysis:
# MAGIC
# MAGIC | Step | Function | Output |
# MAGIC |------|----------|--------|
# MAGIC | Non-parametric CIF | `AalenJohansenFitter.fit()` | CIF with 95% confidence band |
# MAGIC | Stacked CIF | `plot_stacked_cif()` | Total exit probability by cause |
# MAGIC | CIF comparison test | `gray_test()` | Chi-squared statistic and p-value |
# MAGIC | Fine-Gray regression | `FineGrayFitter.fit()` | SHRs, SEs, p-values |
# MAGIC | Forest plot | `plot_forest()` | SHR visualisation |
# MAGIC | Prediction | `predict_cumulative_incidence()` | Per-subject CIF at given times |
# MAGIC | Partial effects | `plot_partial_effects_on_outcome()` | How one covariate shifts the CIF |
# MAGIC | Brier score | `competing_risks_brier_score()` | IPCW proper scoring rule |
# MAGIC | Integrated Brier Score | `integrated_brier_score()` | Single-number model quality |
# MAGIC | C-index | `competing_risks_c_index()` | Discrimination at a time horizon |
# MAGIC | Calibration | `calibration_curve()` | Observed vs predicted by risk decile |
# MAGIC
# MAGIC **Key finding from insurance retention example**:
# MAGIC Premium uplift has a strong positive effect on lapse (SHR > 1) but a weaker or even
# MAGIC inverse effect on MTC. This separation would be invisible with a simple logistic
# MAGIC regression that ignores competing risks — it would conflate the two exit mechanisms.
