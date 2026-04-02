# Databricks notebook source
# MAGIC %md
# MAGIC # LifetimeBoundsCalculator Demo
# MAGIC
# MAGIC ## The problem it solves
# MAGIC
# MAGIC Life tables give you 1-year survival probabilities at integer ages: l_65, l_66,
# MAGIC and so on. What they do not tell you is *when within the year* people die. For
# MAGIC a term assurance or pure savings bond, this does not matter much. For contracts
# MAGIC whose payoff depends on the continuous lifetime — annuities, GMIB riders,
# MAGIC income protection, death benefits — the within-year mortality distribution
# MAGIC affects the fair value.
# MAGIC
# MAGIC The standard actuarial solution is to pick a fractional age assumption: uniform
# MAGIC distribution of deaths (UDD), constant force of mortality (CFM), or Balducci.
# MAGIC But this choice is a model assumption, not observable from annual life tables.
# MAGIC
# MAGIC Dupret & Motte (2026, arXiv:2603.06238) derive closed-form worst- and best-case
# MAGIC contract values without committing to any fractional age assumption. The bounds
# MAGIC come from two extreme strategies:
# MAGIC
# MAGIC - **Upper survival**: mortality concentrates at year-ends. Policyholder survives
# MAGIC   as long as possible within each year. For annuities: upper > lower.
# MAGIC - **Lower survival**: mortality concentrates at year-beginnings. The policyholder
# MAGIC   dies as early as possible within each year.
# MAGIC
# MAGIC The spread between bounds is the model risk from the fractional age assumption.
# MAGIC For Solvency II ORSA and IFRS 17 sensitivity disclosures, this is quantifiable.

# COMMAND ----------
# MAGIC %pip install insurance-survival>=0.5.0 polars --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import numpy as np
import polars as pl
from insurance_survival.mortality.bounds import LifetimeBoundsCalculator, LifetimeBoundsResult

print("insurance-survival imported successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Build from a standard lx life table column
# MAGIC
# MAGIC The simplest entry point: pass a list of l_x values (survivors from a
# MAGIC radix of 100,000 at each integer age). This is the standard format from
# MAGIC published actuarial tables and CMI workbooks.

# COMMAND ----------

# Typical UK pensioner male mortality, age 65-80
# Source: approximately S3PML pattern (stylised)
lx = [
    100_000,  # age 65
     99_120,  # age 66
     98_185,  # age 67
     97_185,  # age 68
     96_105,  # age 69
     94_930,  # age 70
     93_645,  # age 71
     92_225,  # age 72
     90_650,  # age 73
     88_905,  # age 74
     86_960,  # age 75
     84_790,  # age 76
     82_365,  # age 77
     79_660,  # age 78
     76_660,  # age 79
     73_350,  # age 80
]

calc = LifetimeBoundsCalculator.from_lx(
    lx=lx,
    starting_age=65,
    discount_rate=0.04,    # 4% continuous discount rate
)

print(f"Starting age: {calc.starting_age}")
print(f"Horizon:      {calc.n_years} years (to age {calc.starting_age + calc.n_years})")
print(f"Discount rate:{calc.discount_rate:.2%}")
print(f"\nhat_p (1-year survival probs):")
for i, (p, age) in enumerate(zip(calc.hat_p, range(65, 65 + len(calc.hat_p)))):
    print(f"  age {age}: {p:.5f}  (q_x = {1-p:.5f})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Annuity bounds — unit income while alive
# MAGIC
# MAGIC An annuity-due paying £1 per year continuously while the policyholder is
# MAGIC alive. The bounds quantify how sensitive the fair value is to the within-year
# MAGIC mortality assumption.

# COMMAND ----------

result_annuity = calc.annuity_bounds(t=0.0, T=15.0)
print(result_annuity)
print()
print(f"Upper bound (mortality at year-ends):   {result_annuity.upper:.5f}")
print(f"Lower bound (mortality at year-starts): {result_annuity.lower:.5f}")
print(f"Spread:                                  {result_annuity.spread:.5f}")
print(f"Spread (% of midpoint):                  {result_annuity.spread_pct:.3f}%")
print()
print(f"Interpretation: depending on the fractional age assumption,")
print(f"the annuity fair value could be anywhere in [{result_annuity.lower:.4f}, {result_annuity.upper:.4f}].")
print(f"For a £100,000/yr annuity, this is a ±£{result_annuity.spread/2*100000:,.0f} uncertainty.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Death benefit bounds — GMDB-type rider
# MAGIC
# MAGIC A death benefit pays a lump sum on death. With a constant benefit and
# MAGIC positive discount rate, the direction of the bounds reverses: concentrating
# MAGIC death at year-ends means more discounting of the benefit, so upper survival
# MAGIC gives a *lower* death benefit value.

# COMMAND ----------

# Constant £100,000 death benefit
r_db_const = calc.death_benefit_bounds(
    t=0.0,
    T=15.0,
    benefit_fn=lambda s: 100_000,
)
print("Death benefit bounds (constant £100,000):")
print(r_db_const)
print()

# Growing benefit (linked to fund): upper > lower (earlier death => higher fund?)
# Actually: upper survival defers death => more discounting but larger fund
# Net effect depends on whether fund growth > discount rate
fund_rate = 0.06  # 6% fund growth rate (> 4% discount)
r_db_growing = calc.death_benefit_bounds(
    t=0.0,
    T=15.0,
    benefit_fn=lambda s: 100_000 * np.exp((fund_rate - calc.discount_rate) * s),
)
print("Death benefit bounds (growing benefit, r_fund=6% > r_disc=4%):")
print(r_db_growing)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Step functions — visualising the bounds
# MAGIC
# MAGIC The survival bounds are step functions: upper is constant within each year
# MAGIC (death deferred to year-end), lower jumps down at year-start.
# MAGIC The gap between them at any time s is the survival bound spread.

# COMMAND ----------

schedule = calc.survival_bounds_schedule(t=0.0, T=15.0, n_steps=500)
print(f"Bounds schedule: {schedule.shape}")
print(schedule.head(10))

# COMMAND ----------

# Visualise the step functions
s_vals = schedule["s"].to_numpy()
upper = schedule["upper"].to_numpy()
lower = schedule["lower"].to_numpy()

# For plotting only — no matplotlib in this run if not available
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s_vals, upper, color="#1f77b4", lw=2, label="Upper survival (death at year-ends)")
    ax.plot(s_vals, lower, color="#d62728", lw=2, label="Lower survival (death at year-starts)")
    ax.fill_between(s_vals, lower, upper, alpha=0.15, color="grey", label="Model risk band")
    ax.set_xlabel("Policy year s")
    ax.set_ylabel("Conditional survival probability")
    ax.set_title("Survival Bounds: Worst and Best Case Within-Year Mortality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Step function plot rendered.")
except ImportError:
    print("matplotlib not available; skipping plot.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Spread by contract duration
# MAGIC
# MAGIC How does the spread grow with contract term? Longer contracts accumulate
# MAGIC more within-year timing uncertainty.

# COMMAND ----------

df_spread = calc.spread_by_duration("annuity", max_years=15)
print("Annuity spread by contract duration:")
print(df_spread)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Fractional age comparison
# MAGIC
# MAGIC Where do UDD, CFM, and Balducci sit relative to the theoretical bounds?
# MAGIC This is the key question for an actuary choosing a method: am I above or
# MAGIC below the midpoint? By how much?

# COMMAND ----------

df_fac = calc.fractional_age_comparison(t=0.0, T=10.0, contract_type="annuity")
print("Fractional age assumption comparison (10yr annuity):")
print(df_fac)
print()
print("Key: 'in_bounds'=True means the assumption gives a value within the")
print("theoretical model-risk band. If False, the assumption is implicitly")
print("more extreme than the worst-case bound.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. GMIB-type payoff — custom income function
# MAGIC
# MAGIC A variable annuity with a guaranteed minimum income benefit (GMIB): the
# MAGIC policyholder receives max(account_value(s), guaranteed_floor(s)) while alive.
# MAGIC When the fund earns above the guarantee rate, the account dominates.

# COMMAND ----------

initial_account = 100_000.0
guaranteed_rate = 0.03   # 3% annual guarantee
fund_return = 0.06       # 6% fund return

def gmib_income(s: float) -> float:
    account = initial_account * np.exp(fund_return * s)
    guarantee = initial_account * np.exp(guaranteed_rate * s)
    return max(account, guarantee)

result_gmib = calc.annuity_bounds(t=0.0, T=10.0, income_fn=gmib_income, n_steps=1000)

print("GMIB annuity bounds (10yr, initial account £100,000):")
print(result_gmib)
print()
print(f"Model risk spread: £{result_gmib.spread * 100_000:,.0f} on a notional £100,000 account")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Integration with CMI S3 tables
# MAGIC
# MAGIC The `from_cmi` classmethod loads built-in CMI S3 survival rates directly.
# MAGIC This is the standard UK actuarial table for pensioner mortality.

# COMMAND ----------

try:
    calc_cmi = LifetimeBoundsCalculator.from_cmi(
        table_code="S3PML",       # pensioner male lives
        starting_age=65,
        n_years=20,
        discount_rate=0.04,
    )
    result_cmi = calc_cmi.annuity_bounds(0.0, 20.0)
    print("CMI S3PML table loaded successfully.")
    print(f"20yr annuity bounds (age-65 male, 4% discount):")
    print(result_cmi)
except Exception as e:
    print(f"CMI table load: {e}")
    print("(Expected if CMI data files not bundled — use from_lx instead)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Year-by-year breakdown
# MAGIC
# MAGIC `result.to_df()` gives the year-by-year contribution to the bounds:
# MAGIC which policy year drives the most spread?

# COMMAND ----------

result_detail = calc.annuity_bounds(0.0, 15.0)
df_detail = result_detail.to_df()
print("Year-by-year breakdown:")
print(df_detail)

# Early years dominate the discounted value. The spread should widen
# as survival probability declines (more mortality events means more
# uncertainty about within-year timing).

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Posterior-mean mortality from DMP model
# MAGIC
# MAGIC In production: combine with `CauseSpecificMortality` (DMP framework) to
# MAGIC obtain posterior-mean hat_p estimates, then bound contract values against
# MAGIC residual within-year uncertainty.

# COMMAND ----------

# Simulate DMP posterior-mean rates (annual force of mortality, Gompertz-like)
rng = np.random.default_rng(42)
n_years = 15
mu_annual = 0.010 * np.exp(0.09 * np.arange(n_years))   # age 65 Gompertz
mu_annual += rng.normal(0, 0.0005, n_years)              # posterior noise

# Convert force of mortality to annual survival probability
hat_p_dmp = np.exp(-mu_annual)
print(f"DMP posterior-mean hat_p (ages 65-{65+n_years-1}):")
for i, (p, age) in enumerate(zip(hat_p_dmp, range(65, 65 + n_years))):
    print(f"  age {age}: {p:.5f}")

calc_dmp = LifetimeBoundsCalculator(
    hat_p=hat_p_dmp,
    starting_age=65,
    discount_rate=0.04,
)

result_dmp = calc_dmp.annuity_bounds(0.0, 15.0)
print(f"\nAnnuity bounds using DMP-calibrated mortality:")
print(result_dmp)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Method | Use case |
# MAGIC |--------|----------|
# MAGIC | `annuity_bounds()` | GMIB, income-while-alive contracts |
# MAGIC | `death_benefit_bounds()` | GMDB, term assurance |
# MAGIC | `general_bounds()` | Arbitrary payoff callable |
# MAGIC | `spread_by_duration()` | Report materiality vs contract term |
# MAGIC | `fractional_age_comparison()` | Validate UDD/CFM/Balducci choices |
# MAGIC | `from_lx()` | Standard actuarial table input |
# MAGIC | `from_cmi()` | UK CMI S3 tables with improvement scale |
# MAGIC
# MAGIC **For Solvency II ORSA**: the spread_pct output is the model risk from the
# MAGIC fractional age assumption. For a UK pensioner annuity at age 65 with a 4%
# MAGIC discount rate, this is typically 1-2% of fair value over 10-20yr contracts.
# MAGIC This should be documented as an ORSA sensitivity alongside interest rate
# MAGIC and mortality improvement sensitivities.
# MAGIC
# MAGIC **For IFRS 17**: mortality assumption sensitivity is a required disclosure.
# MAGIC The within-year distribution is a legitimate unquantified sensitivity that
# MAGIC most firms currently do not report. LifetimeBoundsCalculator makes it easy.

# COMMAND ----------

print("Demo complete.")
dbutils.notebook.exit("LifetimeBoundsCalculator demo completed successfully.")
