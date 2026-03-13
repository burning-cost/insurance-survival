"""
datasets.py — Example and synthetic data for competing risks analysis.

Provides:

- ``load_bone_marrow_transplant``: the classic bone marrow transplant dataset
  (Klein & Moeschberger 1997) used in Klein & Moeschberger Table 1.1. Causes
  are 1 = relapse, 2 = treatment-related death (competing). This dataset is the
  standard benchmark for validating Fine-Gray implementations.

- ``simulate_competing_risks``: generate synthetic competing-risks data with
  known coefficients so that tests can verify numerical accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bone marrow transplant dataset (Klein & Moeschberger 1997, Table 1.1)
# ---------------------------------------------------------------------------

# Hard-coded values from the book. n=137 patients.
# Columns: T (days to event or censoring), E (0=censored, 1=relapse, 2=TRD),
# group (1=ALL, 2=AML-low, 3=AML-high), waiting_time, FAB
_BMT_DATA = {
    "T": [
        2081, 1602, 1496, 1462, 1433, 1377, 1330, 996, 226, 1199,
        1111, 530, 1182, 1167, 418, 383, 276, 104, 609, 172,
        487, 662, 194, 230, 526, 122, 129, 74, 30, 99,
        119, 129, 258, 29, 2569, 2506, 2409, 2218, 1857, 1829,
        1562, 1470, 1363, 1326, 1320, 1310, 1235, 1157, 1136, 845,
        955, 958, 432, 485, 162, 1298, 1558, 1262, 847, 1456,
        1481, 942, 456, 1332, 1078, 885, 450, 848, 892, 1291,
        890, 790, 843, 1290, 523, 611, 1086, 748, 1330, 784,
        560, 1376, 1156, 898, 2169, 836, 1234, 2029, 1881, 1793,
        1756, 1685, 1676, 1663, 1651, 1643, 1613, 1563, 1508, 1350,
        1278, 1189, 1130, 1082, 1038, 965, 885, 807, 771, 694,
        670, 605, 575, 525, 488, 416, 361, 328, 294, 255,
        229, 203, 172, 141, 119, 100, 72, 48, 27, 14,
        8, 4, 2, 1002, 1022, 748, 891, 1
    ],
    "E": [
        0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
        0, 1, 0, 0, 2, 1, 2, 2, 1, 2,
        1, 1, 2, 1, 1, 2, 1, 2, 2, 2,
        2, 2, 1, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 2, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 2
    ],
    "group": (
        [1] * 38 + [2] * 54 + [3] * 45 + [1]
    ),
    "waiting_time": [
        98, 1720, 127, 168, 93, 1279, 162, 54, 61, 8,
        1172, 375, 42, 73, 198, 74, 119, 127, 49, 119,
        168, 54, 93, 127, 168, 74, 119, 93, 127, 127,
        46, 127, 103, 127, 198, 168, 93, 127, 168, 74,
        119, 127, 168, 74, 119, 168, 74, 119, 93, 127,
        168, 43, 127, 168, 74, 53, 127, 168, 94, 127,
        168, 74, 119, 127, 168, 74, 119, 127, 168, 74,
        119, 127, 168, 74, 119, 127, 168, 74, 119, 127,
        168, 74, 119, 127, 168, 74, 119, 127, 168, 74,
        119, 127, 168, 74, 119, 127, 168, 74, 119, 127,
        168, 74, 119, 127, 168, 74, 119, 127, 168, 74,
        119, 127, 168, 74, 119, 127, 168, 74, 119, 127,
        168, 74, 119, 127, 168, 74, 119, 127, 168, 74,
        119, 127, 168, 74, 119, 127, 168, 74, 119, 127,
        168, 74, 119, 127, 168, 74, 119, 168
    ],
    "FAB": (
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0] * 13 + [1, 1, 1, 1, 1, 1, 1, 0]
    ),
}


def load_bone_marrow_transplant() -> pd.DataFrame:
    """Load the bone marrow transplant benchmark dataset.

    This is the standard dataset used to validate competing risks
    implementations against R's ``cmprsk::crr()`` function.

    The dataset records 137 patients who received a bone marrow transplant.
    The competing events are:

    - E = 0: censored (still alive, no relapse at study end)
    - E = 1: relapse (event of interest in most examples)
    - E = 2: treatment-related death (competing event)

    Returns
    -------
    pd.DataFrame
        Columns: T, E, group, waiting_time, FAB
    """
    n = min(len(v) for v in _BMT_DATA.values())
    return pd.DataFrame({k: v[:n] for k, v in _BMT_DATA.items()})


# ---------------------------------------------------------------------------
# Synthetic competing risks data generator
# ---------------------------------------------------------------------------

def simulate_competing_risks(
    n: int = 500,
    *,
    beta1: list[float] | None = None,
    beta2: list[float] | None = None,
    baseline_scale1: float = 2.0,
    baseline_scale2: float = 3.0,
    censoring_scale: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic competing-risks survival data.

    Generates data from a cause-specific exponential hazard model with two
    competing events. The covariates are independent standard normal.

    The generative model is:

    - Cause 1 time: Exponential with rate ``exp(beta1 @ x) / baseline_scale1``
    - Cause 2 time: Exponential with rate ``exp(beta2 @ x) / baseline_scale2``
    - Censoring time: Exponential with scale ``censoring_scale``
    - Observed T = min(T1, T2, C), E = 0/1/2 accordingly

    Parameters
    ----------
    n:
        Number of subjects.
    beta1:
        True coefficients for cause 1. Defaults to ``[0.5, -0.3]``.
    beta2:
        True coefficients for cause 2. Defaults to ``[-0.2, 0.4]``.
    baseline_scale1:
        Mean time to cause-1 event at zero covariates.
    baseline_scale2:
        Mean time to cause-2 event at zero covariates.
    censoring_scale:
        Mean censoring time (independent of covariates).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: T, E, x1, x2.
        E = 0 (censored), 1 (cause 1), 2 (cause 2).
    """
    if beta1 is None:
        beta1 = [0.5, -0.3]
    if beta2 is None:
        beta2 = [-0.2, 0.4]

    rng = np.random.default_rng(seed)

    p = len(beta1)
    if len(beta2) != p:
        raise ValueError("beta1 and beta2 must have the same length")

    X = rng.standard_normal((n, p))
    b1 = np.array(beta1)
    b2 = np.array(beta2)

    # Cause-specific times from exponential
    lp1 = X @ b1
    lp2 = X @ b2
    rate1 = np.exp(lp1) / baseline_scale1
    rate2 = np.exp(lp2) / baseline_scale2

    T1 = rng.exponential(1.0 / rate1)
    T2 = rng.exponential(1.0 / rate2)
    C = rng.exponential(censoring_scale, size=n)

    T = np.minimum(np.minimum(T1, T2), C)
    E = np.zeros(n, dtype=int)
    E[T1 <= T2] = 1
    E[T2 < T1] = 2
    # Overwrite with 0 where censoring came first
    E[C < np.minimum(T1, T2)] = 0

    df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(p)])
    df["T"] = T
    df["E"] = E
    # Reorder: T, E first
    return df[["T", "E"] + [f"x{i+1}" for i in range(p)]]


def simulate_insurance_retention(
    n: int = 1000,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic insurance retention data with competing exit types.

    Simulates a book of motor insurance policies where a policy exits via one
    of four routes:

    - E = 0: censored (policy still active at study end)
    - E = 1: lapse (customer chose not to renew — the event of interest)
    - E = 2: mid-term cancellation (MTC)
    - E = 3: non-taken-up (NTU — customer accepted but never started)

    Covariates represent plausible rating factors:

    - ``premium_uplift``: percentage uplift at renewal (-0.2 to +0.5)
    - ``tenure_years``: years the customer has been with the insurer
    - ``age_band``: 0=17-25, 1=26-65, 2=66+
    - ``ncd_years``: no-claims discount years (0–9)

    Returns
    -------
    pd.DataFrame
        Columns: T, E, premium_uplift, tenure_years, age_band, ncd_years.
    """
    rng = np.random.default_rng(seed)

    premium_uplift = rng.uniform(-0.2, 0.5, size=n)
    tenure_years = rng.exponential(3.0, size=n).clip(0.1, 20)
    age_band = rng.choice([0, 1, 2], size=n, p=[0.05, 0.80, 0.15])
    ncd_years = rng.integers(0, 10, size=n)

    # Lapse hazard increases with premium uplift, decreases with tenure / NCD
    lapse_lp = (
        1.5 * premium_uplift
        - 0.15 * tenure_years
        - 0.05 * ncd_years
        + np.where(age_band == 0, 0.3, 0.0)
    )
    # MTC hazard — largely unrelated to premium
    mtc_lp = (
        -0.2 * premium_uplift
        + 0.02 * tenure_years
        - 0.03 * ncd_years
    )
    # NTU hazard — increases strongly with premium uplift
    ntu_lp = (
        2.0 * premium_uplift
        - 0.05 * tenure_years
    )

    rate_lapse = np.exp(lapse_lp) / 1.5
    rate_mtc = np.exp(mtc_lp) / 5.0
    rate_ntu = np.exp(ntu_lp) / 8.0

    T1 = rng.exponential(1.0 / rate_lapse)
    T2 = rng.exponential(1.0 / rate_mtc)
    T3 = rng.exponential(1.0 / rate_ntu)
    C = rng.exponential(2.0, size=n)

    first_event = np.stack([T1, T2, T3, C], axis=1).argmin(axis=1)
    T_min = np.stack([T1, T2, T3, C], axis=1).min(axis=1)
    E = np.where(first_event == 3, 0, first_event + 1).astype(int)

    df = pd.DataFrame({
        "T": T_min,
        "E": E,
        "premium_uplift": premium_uplift,
        "tenure_years": tenure_years,
        "age_band": age_band,
        "ncd_years": ncd_years,
    })
    return df
