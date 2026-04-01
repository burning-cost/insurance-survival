# Databricks notebook source
# MAGIC %md
# MAGIC # Coherent Cause-Specific Mortality Forecasting
# MAGIC
# MAGIC **Library:** insurance-survival 0.4.0 — `insurance_survival.mortality`
# MAGIC
# MAGIC **Paper:** Nigri, Shang & Ungolo (2026). *A Dirichlet-Multinomial-Poisson framework
# MAGIC for the coherent analysis and forecast of cause-specific mortality.* arXiv:2603.00973.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The problem
# MAGIC
# MAGIC Standard practice fits separate mortality improvement models to each cause of death.
# MAGIC The cause forecasts do not sum to the all-cause forecast. For UK protection/CI/annuity
# MAGIC pricing this is actuarially inconsistent — IFRS17 best estimates require internal
# MAGIC coherence between cause and aggregate.
# MAGIC
# MAGIC ## The solution
# MAGIC
# MAGIC The DMP hierarchy enforces coherence **by construction** through three levels:
# MAGIC
# MAGIC 1. **Total deaths**: Y_{a,t} ~ Poisson(E_{a,t} · m_{a,t})
# MAGIC 2. **Cause allocation**: Y_bar_{a,t} | Y_{a,t} ~ Dirichlet-Multinomial(φ · γ_{a,t}, Y_{a,t})
# MAGIC 3. **Cause probabilities**: γ_{a,t,c} = softmax(η_{a,t,c})
# MAGIC
# MAGIC This means m_{a,t,c} = m_{a,t} · γ_{a,t,c}, so sum_c m_{a,t,c} = m_{a,t} at every draw.

# COMMAND ----------

# MAGIC %pip install "insurance-survival[mortality]"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_survival.mortality import CauseSpecificMortality, MortalityForecast, HMDLoader

print("insurance-survival mortality module loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic England and Wales style data
# MAGIC
# MAGIC We use `HMDLoader.load_synthetic()` which generates plausible age-period-cause
# MAGIC mortality data based on a Lee-Carter DGP with:
# MAGIC - 18 five-year age groups (0 to 85+), matching typical HMD structure
# MAGIC - 40 calendar years (1980–2019) — pre-COVID for clean fitting
# MAGIC - 6 cause groups: neoplasms, cardiovascular, respiratory, infectious, external, other

# COMMAND ----------

deaths, exposure, age_labels, year_labels = HMDLoader.load_synthetic(
    n_ages=18,
    n_years=40,
    n_causes=6,
    seed=42,
)

print(f"Deaths array shape:    {deaths.shape}   (n_ages × n_years × n_causes)")
print(f"Exposure array shape:  {exposure.shape}  (n_ages × n_years)")
print(f"Age groups:  {age_labels}")
print(f"Year range:  {year_labels[0]} – {year_labels[-1]}")
print(f"\nTotal deaths by cause:")

cause_names = ["neoplasms", "cardiovascular", "respiratory", "infectious", "external", "other"]
for c, name in enumerate(cause_names):
    print(f"  {name:20s}: {deaths[:, :, c].sum():>10,.0f}")
print(f"  {'total':20s}: {deaths.sum():>10,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Examine cause composition by age
# MAGIC
# MAGIC Before fitting, let's look at the raw cause composition across age groups.
# MAGIC The DGP gives CVD and neoplasms dominance in older ages, external causes in younger.

# COMMAND ----------

# Compute cause fractions by age (averaged over time)
cause_fractions_by_age = deaths.sum(axis=1)  # (n_ages, n_causes)
cause_fractions_by_age = (
    cause_fractions_by_age / cause_fractions_by_age.sum(axis=1, keepdims=True)
)

fig, ax = plt.subplots(figsize=(12, 5))
bottom = np.zeros(len(age_labels))
colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]

for c, (name, colour) in enumerate(zip(cause_names, colours)):
    ax.bar(age_labels, cause_fractions_by_age[:, c], bottom=bottom,
           label=name, color=colour, alpha=0.85)
    bottom += cause_fractions_by_age[:, c]

ax.set_xlabel("Age group")
ax.set_ylabel("Cause fraction")
ax.set_title("Cause composition by age group (observed, 1980–2019)")
ax.legend(loc="upper left", fontsize=8)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/tmp/cause_composition_by_age.png", dpi=120)
plt.show()
print("Figure saved to /tmp/cause_composition_by_age.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit the LC-DM model
# MAGIC
# MAGIC The LC (Lee-Carter) variant is recommended:
# MAGIC - Shared κ_t latent trend governs both total mortality improvement and cause-mix shift
# MAGIC - More parsimonious than AP (fewer free parameters per time point)
# MAGIC - Better out-of-sample performance per the paper's rolling validation
# MAGIC
# MAGIC **Runtime note:** HMC via NumPyro/JAX. With 18 ages × 40 years × 6 causes,
# MAGIC expect ~10–20 minutes on Databricks CPU. Use n_samples=500, n_warmup=250 for
# MAGIC a quick demo; increase to n_samples=3000, n_warmup=1500 for production.

# COMMAND ----------

model = CauseSpecificMortality(
    model_type="LC",
    n_chains=2,
    n_samples=500,
    n_warmup=250,
    seed=42,
    target_accept_prob=0.8,
)

print("Fitting LC-DM model...")
print(f"  {model.n_chains} chains × {model.n_samples} samples + {model.n_warmup} warmup")
print(f"  Data: {deaths.shape[0]} age groups × {deaths.shape[1]} years × {deaths.shape[2]} causes")
print()

model.fit(
    deaths,
    exposure,
    age_labels=age_labels,
    year_labels=year_labels,
    cause_names=cause_names,
)

print("Fit complete.")
print(f"  Posterior keys: {sorted(model.posterior_.keys())}")
print(f"  Total draws available: {list(model.posterior_.values())[0].shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MCMC diagnostics
# MAGIC
# MAGIC Convergence check: R-hat < 1.05 for all parameters. If ArviZ is installed,
# MAGIC effective sample size (ESS) is also reported.

# COMMAND ----------

diag = model.diagnostics()

print(f"Diagnostic backend: {diag['backend']}")
print(f"Divergences: {diag['n_divergences']}")
print()
print("R-hat (max per parameter):")
for param, rhat_val in sorted(diag["rhat"].items()):
    flag = " <-- WARNING: R-hat > 1.05" if rhat_val > 1.05 else ""
    print(f"  {param:20s}: {rhat_val:.4f}{flag}")

if diag["ess"]:
    print()
    print("ESS (min per parameter):")
    for param, ess_val in sorted(diag["ess"].items()):
        flag = " <-- LOW ESS" if ess_val < 100 else ""
        print(f"  {param:20s}: {ess_val:.0f}{flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect the shared period trend κ_t
# MAGIC
# MAGIC In the LC-DM model, κ_t is the single latent factor driving both
# MAGIC total mortality improvement and cause-mix shift. Plotting the posterior
# MAGIC of κ_t shows the estimated mortality trend and its uncertainty.

# COMMAND ----------

raw_kappa = model.posterior_["raw_kappa"]           # (n_draws, n_years)
kappa = raw_kappa - raw_kappa.mean(axis=1, keepdims=True)  # centred

kappa_median = np.median(kappa, axis=0)
kappa_lo = np.percentile(kappa, 2.5, axis=0)
kappa_hi = np.percentile(kappa, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(year_labels, kappa_lo, kappa_hi, alpha=0.3, color="#377eb8",
                label="95% credible interval")
ax.plot(year_labels, kappa_median, color="#377eb8", linewidth=2, label="Posterior median")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Year")
ax.set_ylabel("κ_t")
ax.set_title("Shared period trend κ_t (Lee-Carter, LC-DM model)")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/kappa_trend.png", dpi=120)
plt.show()
print("Figure saved to /tmp/kappa_trend.png")
print(f"\nκ_t drift (mean annual change): {np.diff(kappa_median).mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Forecast 20 years ahead (2020–2039)
# MAGIC
# MAGIC κ_t is extended as a random walk with drift, propagating parameter
# MAGIC uncertainty through the full forecast. Coherence is guaranteed by
# MAGIC construction: cause rates = total rate × cause fractions at every draw.

# COMMAND ----------

forecast = model.forecast(horizon=20)

print(f"Forecast object:")
print(f"  total_rates shape:  {forecast.total_rates.shape}  (n_ages, horizon, n_draws)")
print(f"  cause_rates shape:  {forecast.cause_rates.shape}  (n_ages, horizon, n_causes, n_draws)")
print(f"  Future years:       {forecast.periods[0]} – {forecast.periods[-1]}")
print()

# CORE COHERENCE CHECK — this is the whole point of the DMP model
coherent = forecast.coherence_check(tol=1e-5)
print(f"Coherence check passed: {coherent}")
print("  sum_c m_{{a,t,c}} == m_{{a,t}} at every draw, by construction.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Plot forecast total mortality by age group

# COMMAND ----------

# Select a few representative age groups to plot
plot_ages = ["40-44", "55-59", "65-69", "75-79"]
plot_age_idx = [age_labels.index(a) for a in plot_ages if a in age_labels]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
axes = axes.flatten()

for i, (ai, age) in enumerate(zip(plot_age_idx[:4], [age_labels[j] for j in plot_age_idx[:4]])):
    ax = axes[i]

    # Historical: posterior mean in-sample rate
    hist_rate = deaths[ai, :, :].sum(axis=1) / exposure[ai, :]
    ax.plot(year_labels, hist_rate, color="black", linewidth=1.5, label="Observed")

    # Forecast
    fc_median = np.median(forecast.total_rates[ai, :, :], axis=1)
    fc_lo = np.percentile(forecast.total_rates[ai, :, :], 2.5, axis=1)
    fc_hi = np.percentile(forecast.total_rates[ai, :, :], 97.5, axis=1)

    ax.fill_between(forecast.periods, fc_lo, fc_hi, alpha=0.3, color="#e41a1c",
                    label="95% CI")
    ax.plot(forecast.periods, fc_median, color="#e41a1c", linewidth=2,
            label="Forecast median")
    ax.axvline(year_labels[-1], color="grey", linestyle="--", linewidth=0.8)

    ax.set_title(f"Age {age}")
    ax.set_ylabel("Central death rate")
    ax.set_xlabel("Year")
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("Total mortality forecast by age group (LC-DM model)", y=1.01)
plt.tight_layout()
plt.savefig("/tmp/mortality_forecast_by_age.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure saved to /tmp/mortality_forecast_by_age.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Plot forecast cause composition (coherent!)
# MAGIC
# MAGIC The key output: how cause-of-death mix shifts over the forecast horizon.
# MAGIC Because of the DMP structure, these cause rates **sum exactly to the total**
# MAGIC shown above — no reconciliation needed.

# COMMAND ----------

# Pick age group 65-69 as illustration
if "65-69" in age_labels:
    ai = age_labels.index("65-69")
else:
    ai = len(age_labels) // 2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: cause rate forecasts
ax = axes[0]
for c, (name, colour) in enumerate(zip(cause_names, colours)):
    rates = forecast.cause_rates[ai, :, c, :]
    median = np.median(rates, axis=1)
    lo = np.percentile(rates, 2.5, axis=1)
    hi = np.percentile(rates, 97.5, axis=1)
    ax.plot(forecast.periods, median, color=colour, linewidth=2, label=name)
    ax.fill_between(forecast.periods, lo, hi, color=colour, alpha=0.15)

ax.set_title(f"Cause-specific rates — age {age_labels[ai]}")
ax.set_xlabel("Year")
ax.set_ylabel("Cause-specific death rate")
ax.legend(fontsize=8)

# Right: cause fractions over forecast horizon
ax = axes[1]
total_fc = forecast.total_rates[ai, :, :].mean(axis=1)
bottom = np.zeros(len(forecast.periods))

for c, (name, colour) in enumerate(zip(cause_names, colours)):
    fraction = forecast.cause_rates[ai, :, c, :].mean(axis=1) / total_fc
    ax.bar(forecast.periods, fraction, bottom=bottom, color=colour,
           alpha=0.85, label=name)
    bottom += fraction

ax.set_title(f"Cause composition — age {age_labels[ai]}")
ax.set_xlabel("Year")
ax.set_ylabel("Cause fraction")
ax.legend(fontsize=8)

plt.suptitle(
    f"LC-DM model: coherent cause-specific mortality forecast (age {age_labels[ai]})",
    y=1.02
)
plt.tight_layout()
plt.savefig("/tmp/cause_specific_forecast.png", dpi=120, bbox_inches="tight")
plt.show()
print("Figure saved to /tmp/cause_specific_forecast.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary table with credible intervals

# COMMAND ----------

summary_df = forecast.summary(quantiles=(0.025, 0.5, 0.975))

# Show first two forecast years, all ages, total + causes
subset = summary_df[summary_df["period"].isin(forecast.periods[:2])]
print(f"Summary table — first 2 forecast years:")
print(f"Shape: {subset.shape}")
print()
print(subset.head(30).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. CMI-compatible improvement factors
# MAGIC
# MAGIC UK reserving actuaries work with annual improvement factors (% per year).
# MAGIC The `improvement_factors()` method post-processes the posterior into
# MAGIC CMI-compatible format with uncertainty intervals.

# COMMAND ----------

improvements = forecast.improvement_factors()

print("Annual improvement factors (posterior mean, with 95% CI):")
print("First 5 forecast years, all ages, all causes:")
print()

sample = improvements[improvements["period"] <= forecast.periods[4]]
# Show cardiovascular as example
cvd_improvements = sample[sample["cause"] == "cardiovascular"]
print(cvd_improvements.head(20).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cause fractions at a specific cell
# MAGIC
# MAGIC Quick access to posterior mean cause fractions for any age × year combination.

# COMMAND ----------

# Age group 65-69, year index 15 (mid-observed period)
year_15 = year_labels[15]
fractions_65 = model.cause_fractions(age_idx=ai, year_idx=15)

print(f"Posterior mean cause fractions — age {age_labels[ai]}, year {year_15}:")
for name, fraction in zip(cause_names, fractions_65):
    bar = "=" * int(fraction * 40)
    print(f"  {name:20s}: {fraction:.4f}  {bar}")
print(f"\n  Sum: {fractions_65.sum():.6f}  (should be ~1.0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Coherence: quantified verification
# MAGIC
# MAGIC Demonstrate the coherence property numerically — the max absolute difference
# MAGIC between sum of cause rates and total rates should be below floating-point noise.

# COMMAND ----------

cause_sum = forecast.cause_rates.sum(axis=2)  # (n_ages, horizon, n_draws)
diff = np.abs(cause_sum - forecast.total_rates)

print("Coherence verification:")
print(f"  max |sum_c m_{{a,t,c}} - m_{{a,t}}| = {diff.max():.2e}")
print(f"  mean error:                           {diff.mean():.2e}")
print(f"  99th percentile error:                {np.percentile(diff, 99):.2e}")
print()
print("These errors are floating-point noise only (~1e-7), not actuarial incoherence.")
print("Every draw satisfies coherence by the model's mathematical structure.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Comparing LC-DM and AP-DM models
# MAGIC
# MAGIC The AP (Additive Age-Period) variant is available for comparison.
# MAGIC In the paper, LC-DM achieves lower OOS RMSE for total mortality,
# MAGIC but AP-DM may fit specific cause-period patterns differently.

# COMMAND ----------

print("Fitting AP-DM model for comparison (shorter run for demo)...")

model_ap = CauseSpecificMortality(
    model_type="AP",
    n_chains=1,
    n_samples=200,
    n_warmup=100,
    seed=99,
)

model_ap.fit(
    deaths,
    exposure,
    age_labels=age_labels,
    year_labels=year_labels,
    cause_names=cause_names,
)

forecast_ap = model_ap.forecast(horizon=20, n_draws=100)

# Coherence check for AP model
assert forecast_ap.coherence_check(tol=1e-5), "AP forecast should also be coherent!"

print("AP-DM fit complete.")
print(f"AP forecast coherent: {forecast_ap.coherence_check(tol=1e-5)}")

# Compare posterior median total rates for age 65-69
lc_fc = np.median(forecast.total_rates[ai, :, :], axis=1)
ap_fc = np.median(forecast_ap.total_rates[ai, :, :], axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(forecast.periods, lc_fc, label="LC-DM", linewidth=2, color="#377eb8")
ax.plot(forecast_ap.periods, ap_fc, label="AP-DM", linewidth=2, color="#e41a1c",
        linestyle="--")
ax.set_title(f"LC-DM vs AP-DM total mortality forecast — age {age_labels[ai]}")
ax.set_xlabel("Year")
ax.set_ylabel("Central death rate (posterior median)")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/lc_vs_ap_comparison.png", dpi=120)
plt.show()
print("Figure saved to /tmp/lc_vs_ap_comparison.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Integration with FineGrayFitter (CI/LTC pipeline)
# MAGIC
# MAGIC `CauseSpecificMortality` is the population mortality layer.
# MAGIC For CI/LTC pricing, the cause-specific rates feed into a multi-state
# MAGIC model via `FineGrayFitter` as the population baseline hazard.
# MAGIC
# MAGIC The coherent mortality rates ensure that:
# MAGIC - Condition-specific improvement factors are consistent with all-cause improvement
# MAGIC - The baseline hazard passed to FineGrayFitter does not violate probability bounds
# MAGIC
# MAGIC This integration is the bridge between population mortality (DMP) and
# MAGIC insured lives (Fine-Gray, selection-adjusted).

# COMMAND ----------

print("Integration point: insurance_survival.competing_risks.FineGrayFitter")
print()
print("To use DMP rates as FineGrayFitter baseline:")
print()
print("  from insurance_survival.competing_risks import FineGrayFitter")
print("  from insurance_survival.mortality import CauseSpecificMortality, HMDLoader")
print()
print("  # 1. Get coherent cause-specific rates from DMP model")
print("  fc = dmp_model.forecast(horizon=20)")
print()
print("  # 2. Extract CVD rate for age-band of interest as baseline hazard")
print("  cvd_idx = cause_names.index('cardiovascular')")
print("  cvd_rates_65 = fc.cause_rates[ai, :, cvd_idx, :].mean(axis=1)  # posterior mean")
print()
print("  # 3. Use as external baseline in FineGrayFitter")
print("  #    (requires insured lives data with cause-of-exit indicator)")
print("  fg = FineGrayFitter()")
print("  fg.fit(insured_data, duration_col='tenure', event_col='cause', event_of_interest='cardiovascular')")
print("  # Selection-adjusted CIF on top of DMP population baseline")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The DMP framework from Nigri, Shang & Ungolo (2026) provides:
# MAGIC
# MAGIC 1. **Coherent forecasts by construction** — no post-hoc reconciliation
# MAGIC 2. **Full posterior uncertainty** — not point forecasts; credible intervals throughout
# MAGIC 3. **CMI-compatible output** — improvement factors by age and cause
# MAGIC 4. **Natural pipeline integration** — feeds directly into FineGrayFitter for CI/LTC
# MAGIC
# MAGIC For UK production use:
# MAGIC - Replace synthetic data with GBRTENW HMD download + ONS cause-of-death data
# MAGIC - Validate on 2010–2019 hold-out before applying to reserving assumptions
# MAGIC - Note COVID structural break (2020–2022): consider explicit COVID covariate
# MAGIC - Increase n_samples to 3000, n_warmup to 1500 for production-quality inference
# MAGIC
# MAGIC See `insurance_survival.mortality.HMDLoader.load()` for loading real HMD data.
