"""
insurance_survival.mortality — Coherent cause-specific mortality forecasting.

This subpackage implements the Dirichlet-Multinomial-Poisson (DMP) framework
from Nigri, Shang & Ungolo (2026, arXiv:2603.00973) for the coherent analysis
and forecasting of cause-specific mortality rates.

The problem it solves
---------------------
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

Quick start
-----------
>>> from insurance_survival.mortality import CauseSpecificMortality, HMDLoader
>>> deaths, exposure = HMDLoader.load_synthetic(n_ages=18, n_years=30, n_causes=6)
>>> model = CauseSpecificMortality(model_type="LC")
>>> model.fit(deaths, exposure)
>>> forecast = model.forecast(horizon=20)
>>> forecast.coherence_check()  # True — by construction

Integration with competing_risks
---------------------------------
The cause-specific mortality rates from CauseSpecificMortality.forecast()
provide the coherent population-level baseline hazard for calibrating
FineGrayFitter in CI/LTC pricing pipelines.

Modules
-------
mortality       CauseSpecificMortality — main estimator
forecast        MortalityForecast — posterior forecast container
loader          HMDLoader — HMD data fetching and parsing utility

Dependencies
------------
NumPyro is required for fitting (optional install to avoid breaking existing
users). Install with: pip install insurance-survival[mortality]
"""

from .mortality import CauseSpecificMortality
from .forecast import MortalityForecast
from .loader import HMDLoader

__all__ = [
    "CauseSpecificMortality",
    "MortalityForecast",
    "HMDLoader",
]
