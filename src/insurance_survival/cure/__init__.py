"""
insurance_survival.cure — Mixture cure models for insurance non-claimer scoring.

This subpackage fills a confirmed gap: no Python package provides a
covariate-aware mixture cure model (MCM) with actuarial output.
R has smcure, flexsurvcure, cuRe. Python has nothing pip-installable
that matches this capability.

The MCM splits a population into two latent groups:

1. Structurally immune — will never experience the target event
   (e.g. a structural non-claimer who would not claim regardless of
   how long you observed them).

2. Susceptible — will eventually experience the event, modelled by
   a conditional survival distribution.

The population survival function is:

    S_pop(t | x, z) = pi(z) * S_u(t | x) + [1 - pi(z)]

where pi(z) = P(susceptible | z) is a logistic incidence sub-model
and S_u(t | x) is the latency survival for susceptibles.

Key classes
-----------
WeibullMixtureCure
    EM with Weibull AFT latency. The primary workhorse.

LogNormalMixtureCure
    EM with log-normal AFT latency. Better fit for non-monotone hazard.

CoxMixtureCure
    EM with semiparametric Cox PH latency. Most flexible baseline hazard.

PromotionTimeCure
    Non-mixture model (Tsodikov 1998). Population-level PH structure.

Quick start
-----------
>>> from insurance_survival.cure import WeibullMixtureCure
>>> from insurance_survival.cure.diagnostics import sufficient_followup_test
>>> from insurance_survival.cure.simulate import simulate_motor_panel

>>> df = simulate_motor_panel(n_policies=3000, cure_fraction=0.40, seed=42)
>>> qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
>>> model = WeibullMixtureCure(
...     incidence_formula="ncb_years + age + vehicle_age",
...     latency_formula="ncb_years + age",
...     n_em_starts=5,
... )
>>> model.fit(df, duration_col="tenure_months", event_col="claimed")
>>> cure_scores = model.predict_cure_fraction(df)

References
----------
- Farewell (1982), Biometrics 38:1041-1046
- Maller & Zhou (1996), Survival Analysis with Long-Term Survivors, Wiley
- Peng & Dear (2000), Biometrics 56:237-243
- Sy & Taylor (2000), Biometrics 56:227-236
- Tsodikov (1998), JRSS-B 60:195-207
"""

from .weibull import WeibullMixtureCure
from .lognormal import LogNormalMixtureCure
from .cox import CoxMixtureCure
from .promotion_time import PromotionTimeCure
from ._base import MCMResult
from . import diagnostics
from . import simulate

__all__ = [
    # Core model classes
    "WeibullMixtureCure",
    "LogNormalMixtureCure",
    "CoxMixtureCure",
    "PromotionTimeCure",
    # Results
    "MCMResult",
    # Submodules
    "diagnostics",
    "simulate",
]
