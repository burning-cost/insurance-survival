# Databricks notebook source
# MAGIC %pip install insurance-survival>=0.5.0

# COMMAND ----------

# MAGIC %md
# MAGIC # LifetimeBoundsCalculator — Worst/Best-Case Mortality Bounds
# MAGIC
# MAGIC **Paper**: Dupret & Motte (2026, arXiv:2603.06238)
# MAGIC
# MAGIC Life tables give us 1-year survival probabilities at integer ages. They tell us nothing
# MAGIC about *when within the year* people die. For products whose payoff depends on the
# MAGIC continuous lifetime T — annuities, death benefits, GMIB/GMDB VA riders — this matters.
# MAGIC
# MAGIC The standard solution is a *fractional age assumption* (UDD, constant force, Balducci).
# MAGIC But the choice is a model assumption, not something we can validate from the data.
# MAGIC
# MAGIC Dupret & Motte (2026) take a different approach: derive worst- and best-case contract
# MAGIC values without committing to any fractional age model. The bounds are closed-form — no
# MAGIC MCMC, no LP. They come from two extreme strategies:
# MAGIC
# MAGIC - **Upper survival bound**: mortality concentrates at year-ends. Policyholder survives
# MAGIC   as long as possible within each year. Maximises annuity value.
# MAGIC - **Lower survival bound**: mortality concentrates at year-beginnings. Minimises annuity value.
# MAGIC
# MAGIC The spread between bounds is the quantified model risk from the fractional age assumption.

# COMMAND ----------

import numpy as np
import polars as pl
from insurance_survival.mortality.bounds import LifetimeBoundsCalculator, LifetimeBoundsResult

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Basic setup — CMI S3 table, 65-year-old male

# COMMAND ----------

# Using the built-in CMI S3 table (approximate, for exploration)
# For production: use company experience or CMI S3 graduated values
calc = LifetimeBoundsCalculator.from_cmi(
    table_code="S3PML",    # pensioner male lives
    starting_age=65,
    n_years=20,
    improvement_scale="CMI_2023",
    discount_rate=0.04,    # 4% continuous rate
)

print(f"Coverage: ages {calc.starting_age} to {calc.starting_age + calc.n_years}")
print(f"Sample hat_p at age 65-70: {[round(p, 4) for p in calc.hat_p[:5]]}")
print(f"10-year survival probability: {calc.survival_upper(0.0, 10.0):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Annuity bounds — unit income while alive

# COMMAND ----------

result = calc.annuity_bounds(t=0.0, T=20.0)
print(result)
print()
print(f"Upper (worst-case annuity):  {result.upper:.4f}")
print(f"Lower (best-case annuity):   {result.lower:.4f}")
print(f"Spread:                       {result.spread:.4f}")
print(f"Spread (% of midpoint):       {result.spread_pct:.2f}%")
print(f"")
print(f"This spread quantifies the model risk from the fractional age assumption.")
print(f"A Solvency II ORSA would need to justify choosing a point within this range.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Death benefit bounds — GMDB-type rider

# COMMAND ----------

# Death benefit: GBP 100,000 on death (constant)
# Note: for constant benefit with r>0, upper < lower because
# deferring death to year-end means more discounting
r_db = calc.death_benefit_bounds(
    t=0.0,
    T=20.0,
    benefit_fn=lambda s: 100_000
)
print(r_db)
print()
print(f"If benefit increases with time (account grows), upper > lower:")
r_growing = calc.death_benefit_bounds(
    t=0.0,
    T=20.0,
    benefit_fn=lambda s: 100_000 * np.exp(0.04 * s)  # benefit grows at fund rate
)
print(r_growing)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Survival bounds schedule — visualise the step functions

# COMMAND ----------

schedule = calc.survival_bounds_schedule(0.0, 10.0, n_steps=500)
print(schedule.head(20))

# The step function structure is visible: upper is constant within each year,
# lower jumps down at the start of each year (mortality concentrated at start)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Spread as a function of contract duration

# COMMAND ----------

df_spread = calc.spread_by_duration("annuity", max_years=20)
print(df_spread)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Fractional age comparison — where do UDD/CFM/Balducci sit?

# COMMAND ----------

# This is the key actuarial question: is UDD conservative or optimistic?
df_fac = calc.fractional_age_comparison(0.0, 10.0, "annuity")
print(df_fac)

# UDD is the most common assumption in UK actuarial practice.
# This shows how much error UDD introduces relative to the theoretical bounds.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Custom income function — GMIB-type payoff

# COMMAND ----------

# Variable annuity with guaranteed minimum income benefit
# Income = max(account_value(t), guaranteed_floor)
initial_account = 100_000
guaranteed_rate = 0.03    # 3% guarantee
fund_return = 0.06        # 6% expected fund return

def gmib_income(s):
    account = initial_account * np.exp(fund_return * s)
    guarantee = initial_account * np.exp(guaranteed_rate * s)
    return max(account, guarantee)

r_gmib = calc.annuity_bounds(t=0.0, T=10.0, income_fn=gmib_income, n_steps=1000)
print(f"GMIB annuity bounds:")
print(r_gmib)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Integration with CauseSpecificMortality
# MAGIC
# MAGIC The natural workflow: use DMP-calibrated mortality to get posterior-mean hat_p,
# MAGIC then bound contract values against within-year timing uncertainty.

# COMMAND ----------

# Simulate what DMP posterior-mean rates might look like
# In production: extract from forecast.total_rates.mean(axis=2)
# Here we show the pattern
import numpy as np

# Synthetic posterior-mean mortality rates (annual force of mortality)
rng = np.random.default_rng(42)
n_years = 15
mu_age_65 = 0.008 * np.exp(0.08 * np.arange(n_years))  # Gompertz-like
mu_age_65 += rng.normal(0, 0.0003, n_years)  # posterior uncertainty captured in prior call

# Convert annual mortality rate to hat_p
hat_p_from_dmp = np.exp(-mu_age_65)
print(f"DMP posterior-mean hat_p[:5]: {hat_p_from_dmp[:5].round(4)}")

# Now bound the contract
calc_dmp = LifetimeBoundsCalculator(
    hat_p=hat_p_from_dmp,
    starting_age=65,
    discount_rate=0.04,
)
result_dmp = calc_dmp.annuity_bounds(0.0, 15.0)
print(f"\nAnnuity bounds (DMP-calibrated mortality):")
print(result_dmp)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Year-by-year breakdown via to_df()

# COMMAND ----------

r_brief = calc.annuity_bounds(0.0, 10.0)
df_detail = r_brief.to_df()
print(df_detail)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Constructing from an lx column (actuarial life table format)

# COMMAND ----------

# Standard actuarial life table format
lx_column = [100_000, 99_876, 99_734, 99_568, 99_371,
             99_135, 98_849, 98_499, 98_067, 97_530, 96_861]

calc_lx = LifetimeBoundsCalculator.from_lx(
    lx=lx_column,
    starting_age=55,
    discount_rate=0.035,
)

r_lx = calc_lx.annuity_bounds(0.0, 10.0)
print(f"10yr annuity bounds from lx column:")
print(r_lx)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The `LifetimeBoundsCalculator` gives pricing teams a defensible answer to the question:
# MAGIC "how sensitive is our annuity/DB value to the fractional age assumption?"
# MAGIC
# MAGIC For UK pensioner annuities at age 65 with a 4% discount rate:
# MAGIC - The theoretical spread is typically 1-2% of the midpoint value for 10-20yr contracts
# MAGIC - UDD sits close to the midpoint (slightly above for most ages)
# MAGIC - CFM and Balducci are very close to UDD
# MAGIC
# MAGIC For Solvency II ORSA purposes, this spread is the model risk that should be reported
# MAGIC when the firm's actuarial basis does not justify a specific fractional age assumption.
# MAGIC
# MAGIC For GMDB riders with increasing benefit functions, the bounds reverse relative to
# MAGIC annuities — higher survival concentrates death at year-end where the account is larger.

# COMMAND ----------

print("Demo complete.")
