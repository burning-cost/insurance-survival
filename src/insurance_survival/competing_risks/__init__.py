"""
insurance_survival.competing_risks — Fine-Gray subdistribution hazard regression
and cumulative incidence function estimation for insurance pricing.

This subpackage fills a confirmed gap in the Python ecosystem: no pip-installable
library provides Fine-Gray regression with proper IPCW weighting and a
lifelines-compatible API.

Modules
-------
cif             Aalen-Johansen non-parametric CIF estimation
fine_gray       Fine-Gray subdistribution hazard regression (core)
gray_test       Gray's K-sample test for CIF equality across groups
metrics         Competing-risks evaluation: Brier score, C-index, calibration
datasets        Example and synthetic datasets
plots           Visualisation helpers

Typical usage
-------------
>>> from insurance_survival.competing_risks import FineGrayFitter, AalenJohansenFitter
>>> fg = FineGrayFitter()
>>> fg.fit(df, duration_col="T", event_col="E", event_of_interest=1)
>>> print(fg.summary)
>>> cif = fg.predict_cumulative_incidence(df_new, times=[1, 2, 3])
"""

from .cif import AalenJohansenFitter
from .fine_gray import FineGrayFitter
from .gray_test import gray_test
from .metrics import (
    competing_risks_brier_score,
    competing_risks_c_index,
)
from .datasets import (
    load_bone_marrow_transplant,
    simulate_competing_risks,
)

__all__ = [
    "AalenJohansenFitter",
    "FineGrayFitter",
    "gray_test",
    "competing_risks_brier_score",
    "competing_risks_c_index",
    "load_bone_marrow_transplant",
    "simulate_competing_risks",
]
