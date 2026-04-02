"""
insurance_survival.mortality — Coherent cause-specific mortality forecasting
and worst/best-case lifetime bounds.

This subpackage implements two complementary mortality frameworks:

1. **Coherent cause-specific mortality forecasting** (DMP framework)
   From Nigri, Shang & Ungolo (2026, arXiv:2603.00973).
   Enforces sum_c m_{a,t,c} = m_{a,t} by construction at every posterior draw.

2. **LifetimeBoundsCalculator** (v0.5.0)
   From Dupret & Motte (2026, arXiv:2603.06238).
   Worst/best-case contract values from life table data without a fractional
   age assumption. Quantifies model risk from within-year mortality timing.

The problem solved by DMP
--------------------------
Standard actuarial practice fits separate mortality improvement models to each
cause of death (neoplasms, CVD, respiratory, etc.). The cause-specific forecasts
do not sum to the all-cause forecast. This incoherence is actuarially
problematic: IFRS17 best-estimate assumptions require internal consistency
between cause and aggregate, and CI product pricing requires cause-specific
improvement factors that are coherent with CMI all-cause improvement.

The DMP solution enforces coherence by construction via a probabilistic
hierarchy — not post-hoc adjustment:

1. Total deaths ~ Poisson(exposure * m_total)
2. Cause allocation | total ~ Dirichlet-Multinomial(phi * gamma)
3. Cause probabilities gamma = softmax(linear predictor)

This means sum_c m_{a,t,c} = m_{a,t} at every posterior draw.

Two model variants
------------------
- **LC-DM** (recommended): Lee-Carter structure, shared kappa_t latent trend
  drives both total mortality improvement and cause-mix shift.
- **AP-DM**: Additive age-period with RW2 smoothing. More flexible but can be
  unstable on shorter data runs.

Quick start — DMP
-----------------
>>> from insurance_survival.mortality import CauseSpecificMortality, HMDLoader
>>> deaths, exposure = HMDLoader.load_synthetic(n_ages=18, n_years=30, n_causes=6)
>>> model = CauseSpecificMortality(model_type="LC")
>>> model.fit(deaths, exposure)
>>> forecast = model.forecast(horizon=20)
>>> forecast.coherence_check()  # True — by construction

Quick start — LifetimeBoundsCalculator
---------------------------------------
>>> from insurance_survival.mortality import LifetimeBoundsCalculator
>>> calc = LifetimeBoundsCalculator.from_cmi("S3PML", starting_age=65, n_years=10)
>>> result = calc.annuity_bounds(t=0.0, T=10.0)
>>> print(f"Annuity spread: {result.spread_pct:.2f}%")

Integration with competing_risks
---------------------------------
The cause-specific mortality rates from CauseSpecificMortality.forecast()
provide the coherent population-level baseline hazard for calibrating
FineGrayFitter in CI/LTC pricing pipelines.

Integration between the two submodules
----------------------------------------
Use CauseSpecificMortality to produce posterior-mean mortality rates, convert
to hat_p = exp(-m_j), then pass to LifetimeBoundsCalculator to bound contract
values against within-year mortality uncertainty.

Modules
-------
mortality       CauseSpecificMortality — main estimator (requires NumPyro)
forecast        MortalityForecast — posterior forecast container
loader          HMDLoader — HMD data fetching and parsing utility
bounds          LifetimeBoundsCalculator, LifetimeBoundsResult — fractional age bounds

Dependencies
------------
NumPyro is required for CauseSpecificMortality fitting (optional install to
avoid breaking existing users). Install with:
    pip install insurance-survival[mortality]

LifetimeBoundsCalculator requires only numpy, scipy, and polars (core deps).
"""

from .mortality import CauseSpecificMortality
from .forecast import MortalityForecast
from .loader import HMDLoader
from .bounds import LifetimeBoundsCalculator, LifetimeBoundsResult

__all__ = [
    "CauseSpecificMortality",
    "MortalityForecast",
    "HMDLoader",
    "LifetimeBoundsCalculator",
    "LifetimeBoundsResult",
]
