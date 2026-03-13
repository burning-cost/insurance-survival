"""
insurance_survival.recurrent — Shared frailty models for recurrent insurance claims.

The primary use case is fleet motor, pet, and home insurance where
policyholders make multiple claims. Standard Poisson GLMs treat each
policy-year as independent; frailty models capture unobserved heterogeneity
and produce Bühlmann-Straub credibility scores as a by-product.

Quick start
-----------
>>> from insurance_survival.recurrent import simulate_ag_frailty, AndersenGillFrailty
>>> data = simulate_ag_frailty()
>>> model = AndersenGillFrailty(frailty="gamma").fit(data)
>>> print(model.summary())
>>> scores = model.credibility_scores()

The gap
-------
Python has no shared frailty implementation for recurrent events:
- lifelines: no frailty (GitHub issue #878, closed as "maybe someday")
- scikit-survival: single-event only
- R has frailtypack/reReg but Python has nothing

Credibility connection
----------------------
For gamma frailty, the posterior mean is:
    E[z_i | data] = (theta + n_i) / (theta + Lambda_i)

This is the Bühlmann-Straub credibility formula. The frailty model and
classical credibility theory are the same thing, arrived at from different
directions.

Modules
-------
data        RecurrentEventData — counting process container
models      AndersenGillFrailty, PWPModel, NelsonAalenFrailty
frailty     GammaFrailty, LognormalFrailty, make_frailty
joint       JointFrailtyModel, JointData
simulate    simulate_ag_frailty, simulate_pwp, simulate_joint
report      FrailtyReport, compare_models
"""

from .data import RecurrentEventData
from .frailty import GammaFrailty, LognormalFrailty, make_frailty
from .joint import JointData, JointFrailtyModel, JointFrailtyResult
from .models import (
    AndersenGillFrailty,
    FrailtyFitResult,
    NelsonAalenFrailty,
    PWPFitResult,
    PWPModel,
)
from .report import FrailtyReport, compare_models
from .simulate import (
    SimulationParams,
    simulate_ag_frailty,
    simulate_joint,
    simulate_pwp,
)

__all__ = [
    # Data
    "RecurrentEventData",
    # Models
    "AndersenGillFrailty",
    "FrailtyFitResult",
    "PWPModel",
    "PWPFitResult",
    "NelsonAalenFrailty",
    # Frailty distributions
    "GammaFrailty",
    "LognormalFrailty",
    "make_frailty",
    # Joint model
    "JointFrailtyModel",
    "JointData",
    "JointFrailtyResult",
    # Report
    "FrailtyReport",
    "compare_models",
    # Simulation
    "SimulationParams",
    "simulate_ag_frailty",
    "simulate_pwp",
    "simulate_joint",
]
