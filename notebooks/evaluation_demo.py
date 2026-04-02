# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-survival v0.6.0: CensoredForecastEvaluator Demo
# MAGIC
# MAGIC Demonstrates threshold-weighted CRPS (twCRPS) and proper scoring rules for
# MAGIC right-censored time-to-event forecasts, based on:
# MAGIC
# MAGIC **Taggart, Loveday & Louis (2026)**
# MAGIC *On the evaluation of time-to-event, survival time and first passage time forecasts*
# MAGIC arXiv:2603.14835
# MAGIC
# MAGIC ### The gap this fills
# MAGIC
# MAGIC | Library | Brier score | C-index | twCRPS | Murphy diagram | Reliability diagram |
# MAGIC |---------|-------------|---------|--------|----------------|---------------------|
# MAGIC | pycox | IPCW | Yes | No | No | No |
# MAGIC | scikit-survival | IPCW | Yes | No | No | No |
# MAGIC | lifelines | No | No | No | No | No |
# MAGIC | **insurance-survival v0.6.0** | — | — | **Yes** | **Yes** | **Yes** |
# MAGIC
# MAGIC The IPCW Brier score is NOT a proper scoring rule for the predictive
# MAGIC distribution F(t). twCRPS is.

# COMMAND ----------

# MAGIC %pip install insurance-survival[plot]

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, expon, lognorm

from insurance_survival.evaluation import CensoredForecastEvaluator, from_matrix

print("insurance-survival loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Simulate motor claims settlement data
# MAGIC
# MAGIC We simulate 1,000 claims with known settlement time distribution.
# MAGIC The insurer has a 12-month operational review window (tau = 12 months).
# MAGIC Claims still open at 12 months are administratively censored.

# COMMAND ----------

rng = np.random.default_rng(2026)
n = 1000
tau = 12.0  # 12-month evaluation horizon

# True data generating process: Weibull(shape=1.8, scale=7.5)
# In practice this would be your actual claims data
T_true = weibull_min.rvs(c=1.8, scale=7.5, size=n, random_state=rng)

# Administrative censoring at tau
T_obs = np.minimum(T_true, tau)
event = (T_true <= tau).astype(int)

print(f"n = {n}")
print(f"Event rate at tau={tau}: {event.mean():.1%}")
print(f"Censoring rate: {1 - event.mean():.1%}")
print(f"Median observed time: {np.median(T_obs):.2f} months")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define three forecasters
# MAGIC
# MAGIC - **True model**: Weibull(1.8, 7.5) — the correct DGP
# MAGIC - **Near model**: Weibull(2.0, 7.0) — slight misspecification
# MAGIC - **Exponential**: Exp(scale=7.5) — ignores the shape entirely

# COMMAND ----------

def make_weibull_surv(c, scale):
    """Survival function for Weibull(c, scale)."""
    def S(t):
        return weibull_min.sf(np.asarray(t, dtype=float), c=c, scale=scale)
    return S

def make_expo_surv(scale):
    """Survival function for Exp(scale)."""
    def S(t):
        return expon.sf(np.asarray(t, dtype=float), scale=scale)
    return S

# All observations get the same model (homogeneous predictions here;
# in practice each obs would have its own covariate-adjusted S_i)
true_fns  = [make_weibull_surv(1.8, 7.5)] * n
near_fns  = [make_weibull_surv(2.0, 7.0)] * n
expo_fns  = [make_expo_surv(7.5)] * n

print("Survival functions defined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Score with CensoredForecastEvaluator

# COMMAND ----------

ev = CensoredForecastEvaluator(tau=tau, alpha=0.5, warn=False)

comparison = ev.compare({
    "Weibull(1.8, 7.5) [true]": (true_fns, T_obs, event),
    "Weibull(2.0, 7.0) [near]": (near_fns, T_obs, event),
    "Expo(7.5) [wrong]":        (expo_fns, T_obs, event),
})

print("Model comparison (lower = better):")
print(comparison.round(4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Murphy diagram
# MAGIC
# MAGIC The Murphy diagram reveals WHERE in the time horizon each model's errors
# MAGIC concentrate. Where one model's curve lies below another, it is better
# MAGIC calibrated for events near that threshold time.

# COMMAND ----------

fig = ev.murphy_diagram(
    forecasters={
        "Weibull(1.8, 7.5) [true]": (true_fns, T_obs, event),
        "Weibull(2.0, 7.0) [near]": (near_fns, T_obs, event),
        "Expo(7.5) [wrong]":        (expo_fns, T_obs, event),
    }
)
fig.savefig("/tmp/murphy_diagram.png", dpi=120, bbox_inches="tight")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Reliability diagram
# MAGIC
# MAGIC Checks if predicted P(T > 6 months) is calibrated against the
# MAGIC Kaplan-Meier estimate within bins. Perfect calibration = 45-degree line.

# COMMAND ----------

fig2 = ev.reliability_diagram(
    surv_fns=true_fns,
    T_obs=T_obs,
    event=event,
    eval_time=6.0,
    n_bins=10,
)
fig2.savefig("/tmp/reliability_diagram.png", dpi=120, bbox_inches="tight")
display(fig2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. twCRPS profile
# MAGIC
# MAGIC Shows how the score accumulates over the time horizon. A model that
# MAGIC beats another in the early months but not the late months will show
# MAGIC crossing profiles.

# COMMAND ----------

thresholds = np.linspace(0.5, tau, 50)

profile_true = ev.twcrps_profile(true_fns, T_obs, event, thresholds=thresholds)
profile_expo = ev.twcrps_profile(expo_fns, T_obs, event, thresholds=thresholds)

fig3, ax = plt.subplots(figsize=(8, 4))
ax.plot(profile_true.index, profile_true.values, label="Weibull [true]", linewidth=2)
ax.plot(profile_expo.index, profile_expo.values, label="Expo [wrong]", linewidth=2, linestyle="--")
ax.set_xlabel("Evaluation horizon tau'")
ax.set_ylabel("Mean twCRPS(tau')")
ax.set_title("twCRPS Profile — score accumulation over the horizon")
ax.legend()
fig3.tight_layout()
fig3.savefig("/tmp/twcrps_profile.png", dpi=120, bbox_inches="tight")
display(fig3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. from_matrix helper
# MAGIC
# MAGIC If your survival model outputs a (n_samples, n_times) matrix — as pycox,
# MAGIC scikit-survival, and many neural survival models do — use from_matrix()
# MAGIC to convert it to the list-of-callables format.

# COMMAND ----------

# Simulate a pycox-style output matrix
t_grid = np.linspace(0, tau * 1.5, 100)
S_true_matrix = weibull_min.sf(t_grid[None, :], c=1.8, scale=7.5) * np.ones((n, 1))
S_expo_matrix = expon.sf(t_grid[None, :], scale=7.5) * np.ones((n, 1))

surv_fns_from_matrix = from_matrix(S_true_matrix, t_grid)
expo_fns_from_matrix = from_matrix(S_expo_matrix, t_grid)

score_true_matrix = ev.twcrps(surv_fns_from_matrix, T_obs, event)
score_expo_matrix = ev.twcrps(expo_fns_from_matrix, T_obs, event)

print(f"from_matrix — True model twCRPS: {score_true_matrix:.4f}")
print(f"from_matrix — Expo model twCRPS: {score_expo_matrix:.4f}")
print(f"True < Expo: {score_true_matrix < score_expo_matrix}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. UK insurance application note
# MAGIC
# MAGIC ### Where this matters for pricing teams
# MAGIC
# MAGIC **Motor claims**: Is the 90-day settlement probability well-calibrated
# MAGIC by claim size band? The Murphy diagram shows if the model is good at
# MAGIC short-tail claims (early theta) but poor at long-tail (late theta).
# MAGIC
# MAGIC **Income protection**: Scoring disability duration models when many claims
# MAGIC are still open at the valuation date. The IPCW Brier score systematically
# MAGIC penalises models that correctly predict long durations; twCRPS does not.
# MAGIC
# MAGIC **Lapse modelling**: Year-1 lapse model evaluation with right-censored
# MAGIC in-force policies. Standard CRPS applied to min(T, tau) gives wrong
# MAGIC rankings when censoring is heavy. twCRPS is provisionally proper.
# MAGIC
# MAGIC **IMPORTANT**: tau must be FIXED for all observations (e.g., the policy
# MAGIC anniversary, or a single valuation date). With random censoring times
# MAGIC per observation, use IPCW adjustment (not implemented here — see paper
# MAGIC Section 5 for the theoretical treatment).

print("Demo complete.")
