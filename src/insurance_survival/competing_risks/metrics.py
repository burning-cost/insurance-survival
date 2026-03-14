"""
metrics.py — Competing-risks model evaluation metrics.

Three core metrics:

1. **Cause-specific Brier score** (IPCW-weighted): extends the Brier score to
   the competing-risks setting. Lower is better; 0.25 is the benchmark score
   for a useless model that always predicts F_k(t) = 0.5. Integrating over
   time gives the Integrated Brier Score (IBS).

2. **Cause-specific C-index**: the probability that, for a randomly chosen pair
   where one subject experiences cause k first, the model assigns a higher
   predicted CIF to that subject. Based on Wolbers et al. (2009).

3. **CIF calibration**: compares observed empirical CIF (Aalen-Johansen) to
   mean predicted CIF across quantile groups of predicted probability.

References
----------
Brier, G.W. (1950). Verification of forecasts expressed in terms of probability.
Monthly Weather Review, 78(1), 1-3.

Graf, E., Schmoor, C., Sauerbrei, W. & Schumacher, M. (1999). Assessment and
comparison of prognostic classification schemes for survival data. Statistics
in Medicine, 18(17-18), 2529-2545.

Wolbers, M., Blanche, P., Koller, M.T., Witteman, J.C.M. & Gerds, T.A. (2009).
Concordance for prognostic models with competing risks. Biostatistics, 15(3),
526-539.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# numpy<2.0 compat: trapezoid was added in 2.0, trapz deprecated/removed in 2.0
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

from .cif import AalenJohansenFitter


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def competing_risks_brier_score(
    predicted_cif: "pd.DataFrame | np.ndarray",
    T_test: np.ndarray,
    E_test: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    times: np.ndarray,
    event_of_interest: int = 1,
) -> pd.Series:
    """IPCW Brier score for a competing-risks model at each evaluation time.

    The Brier score at time t for cause k is:

        BS(t) = (1/n) * sum_i w_i * (F_k(t|x_i) - I(T_i <= t, E_i = k))^2

    where the IPCW weight is:

        w_i = I(T_i <= t, E_i = k) / G_c(T_i)
            + I(T_i > t) / G_c(t)

    and G_c is the censoring survival function estimated on the training data.

    Parameters
    ----------
    predicted_cif:
        Predicted CIF values, shape (n_test, n_times). Columns must correspond
        to ``times``.
    T_test:
        Observed times for test subjects.
    E_test:
        Event indicators for test subjects.
    T_train:
        Observed times from the training set (used to estimate G_c).
    E_train:
        Event indicators from the training set.
    times:
        Evaluation times.
    event_of_interest:
        Cause code.

    Returns
    -------
    pd.Series
        Brier score at each time point, indexed by ``times``.
    """
    T_test = np.asarray(T_test, dtype=float)
    E_test = np.asarray(E_test, dtype=int)
    T_train = np.asarray(T_train, dtype=float)
    E_train = np.asarray(E_train, dtype=int)
    times = np.asarray(times, dtype=float)

    if isinstance(predicted_cif, pd.DataFrame):
        cif_arr = predicted_cif.values
    else:
        cif_arr = np.asarray(predicted_cif, dtype=float)

    # Estimate censoring KM on training data
    from .fine_gray import _kaplan_meier
    censoring_event_train = (E_train == 0).astype(int)
    gc_times, gc_vals = _kaplan_meier(T_train, censoring_event_train)

    def gc(t: np.ndarray) -> np.ndarray:
        return np.interp(t, gc_times, gc_vals, left=1.0, right=gc_vals[-1])

    n = len(T_test)
    brier_scores = np.zeros(len(times))

    for j, t in enumerate(times):
        gc_t = gc(np.array([t]))[0]
        gc_ti = gc(T_test)

        # Indicator variables
        ind_event_before_t = (T_test <= t) & (E_test == event_of_interest)
        ind_after_t = T_test > t

        # IPCW weights
        w = np.where(
            ind_event_before_t,
            1.0 / np.maximum(gc_ti, 1e-10),
            np.where(ind_after_t, 1.0 / max(gc_t, 1e-10), 0.0),
        )

        pred = cif_arr[:, j]
        outcome = ind_event_before_t.astype(float)
        brier_scores[j] = np.mean(w * (pred - outcome) ** 2)

    return pd.Series(brier_scores, index=times, name="brier_score")


def integrated_brier_score(
    predicted_cif: "pd.DataFrame | np.ndarray",
    T_test: np.ndarray,
    E_test: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    times: np.ndarray,
    event_of_interest: int = 1,
) -> float:
    """Integrated Brier score (IBS) — the area under the Brier score curve.

    Integrates the Brier score over the evaluation times using the trapezoid
    rule, normalised by the time range. IBS = 0 is a perfect model.

    Parameters
    ----------
    All parameters as for ``competing_risks_brier_score``.

    Returns
    -------
    float
        Integrated Brier score.
    """
    bs = competing_risks_brier_score(
        predicted_cif, T_test, E_test, T_train, E_train, times, event_of_interest
    )
    times = np.asarray(times, dtype=float)
    ibs = np.trapezoid(bs.values, times) / (times[-1] - times[0])
    return float(ibs)


# ---------------------------------------------------------------------------
# C-index
# ---------------------------------------------------------------------------

def competing_risks_c_index(
    predicted_cif: "pd.DataFrame | np.ndarray",
    T_test: np.ndarray,
    E_test: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    eval_time: Optional[float] = None,
    event_of_interest: int = 1,
) -> float:
    """Cause-specific C-index for a competing-risks model.

    For a randomly selected pair (i, j) where subject i experiences cause k
    before subject j (and before the evaluation time), the C-index is the
    probability that predicted_CIF_i(eval_time) > predicted_CIF_j(eval_time).

    Based on the time-dependent AUC framework adapted for competing risks.

    Parameters
    ----------
    predicted_cif:
        Predicted CIF values, shape (n_test, n_times). Columns correspond to
        ``eval_time`` (or a time grid if ``eval_time`` is None).
    T_test:
        Observed times for test subjects.
    E_test:
        Event indicators for test subjects.
    T_train:
        Training observed times (for censoring KM).
    E_train:
        Training event indicators.
    eval_time:
        Single time point for evaluation. If ``None``, uses the median event
        time from the training data.
    event_of_interest:
        Cause code.

    Returns
    -------
    float
        C-index in [0, 1]. 0.5 = random model, 1.0 = perfect discrimination.
    """
    T_test = np.asarray(T_test, dtype=float)
    E_test = np.asarray(E_test, dtype=int)
    T_train = np.asarray(T_train, dtype=float)
    E_train = np.asarray(E_train, dtype=int)

    if isinstance(predicted_cif, pd.DataFrame):
        cif_arr = predicted_cif.values
        time_cols = predicted_cif.columns.values.astype(float)
    else:
        cif_arr = np.asarray(predicted_cif, dtype=float)
        time_cols = None

    # Choose eval time
    if eval_time is None:
        event_times = T_train[E_train == event_of_interest]
        eval_time = float(np.median(event_times)) if len(event_times) > 0 else float(np.median(T_train))

    # Get predictions at eval_time
    if time_cols is not None:
        if eval_time in time_cols:
            idx = np.searchsorted(time_cols, eval_time)
            preds = cif_arr[:, idx]
        else:
            # Interpolate across columns
            preds = np.array([
                np.interp(eval_time, time_cols, cif_arr[i, :])
                for i in range(cif_arr.shape[0])
            ])
    else:
        # Assume cif_arr has one column
        preds = cif_arr[:, -1] if cif_arr.ndim > 1 else cif_arr

    # IPCW censoring weights
    from .fine_gray import _kaplan_meier
    censoring_event_train = (E_train == 0).astype(int)
    gc_times, gc_vals = _kaplan_meier(T_train, censoring_event_train)

    def gc(t: np.ndarray) -> np.ndarray:
        return np.interp(t, gc_times, gc_vals, left=1.0, right=gc_vals[-1])

    gc_ti = gc(T_test)

    # Subjects who experienced cause k before eval_time: "cases"
    case_mask = (T_test <= eval_time) & (E_test == event_of_interest)
    # Subjects still at risk at eval_time: "controls"
    control_mask = T_test > eval_time

    n_concordant = 0.0
    n_permissible = 0.0

    for i in np.where(case_mask)[0]:
        w_i = 1.0 / max(gc_ti[i], 1e-10)
        for j in np.where(control_mask)[0]:
            w_j = 1.0 / max(gc(np.array([eval_time]))[0], 1e-10)
            weight = w_i * w_j
            n_permissible += weight
            if preds[i] > preds[j]:
                n_concordant += weight
            elif preds[i] == preds[j]:
                n_concordant += 0.5 * weight

    if n_permissible == 0:
        return np.nan

    return float(n_concordant / n_permissible)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_curve(
    predicted_cif: "pd.DataFrame | np.ndarray",
    T_test: np.ndarray,
    E_test: np.ndarray,
    eval_time: float,
    event_of_interest: int = 1,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Compare mean predicted CIF to observed CIF by quantile of predicted risk.

    Groups subjects into ``n_quantiles`` quantile bins by their predicted
    CIF at ``eval_time``. Within each bin, computes the Aalen-Johansen
    empirical CIF at ``eval_time`` and compares it to the mean predicted CIF.

    Parameters
    ----------
    predicted_cif:
        Predicted CIF values, shape (n_test, n_times) or (n_test,) if passing
        predictions at a single time.
    T_test:
        Observed times.
    E_test:
        Event indicators.
    eval_time:
        The time at which to assess calibration.
    event_of_interest:
        Cause code.
    n_quantiles:
        Number of quantile groups.

    Returns
    -------
    pd.DataFrame
        Columns: mean_predicted, observed, n_subjects, quantile.
    """
    T_test = np.asarray(T_test, dtype=float)
    E_test = np.asarray(E_test, dtype=int)

    if isinstance(predicted_cif, pd.DataFrame):
        cif_arr = predicted_cif.values
        time_cols = predicted_cif.columns.values.astype(float)
        idx = np.argmin(np.abs(time_cols - eval_time))
        preds = cif_arr[:, idx]
    else:
        cif_arr = np.asarray(predicted_cif, dtype=float)
        preds = cif_arr[:, -1] if cif_arr.ndim > 1 else cif_arr.ravel()

    quantile_bins = pd.Series(pd.qcut(preds, q=n_quantiles, duplicates="drop"))
    rows = []

    for quantile in quantile_bins.cat.categories:
        mask = quantile_bins == quantile
        if mask.sum() < 5:
            continue

        # Empirical CIF for this subgroup at eval_time
        try:
            aj = AalenJohansenFitter()
            aj.fit(T_test[mask], E_test[mask], event_of_interest=event_of_interest)
            observed = float(aj.predict(np.array([eval_time]))[0])
        except Exception:
            observed = np.nan

        rows.append({
            "quantile": str(quantile),
            "mean_predicted": float(preds[mask].mean()),
            "observed": observed,
            "n_subjects": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def plot_calibration(
    calibration_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "CIF Calibration",
) -> plt.Axes:
    """Plot observed vs predicted CIF from ``calibration_curve()`` output.

    Parameters
    ----------
    calibration_df:
        Output of ``calibration_curve()``.
    ax:
        Matplotlib axes.
    title:
        Plot title.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        calibration_df["mean_predicted"],
        calibration_df["observed"],
        s=calibration_df["n_subjects"].clip(20, 200),
        alpha=0.7,
        label="Quantile groups",
    )

    # Perfect calibration line
    lo = min(
        calibration_df["mean_predicted"].min(),
        calibration_df["observed"].dropna().min(),
    )
    hi = max(
        calibration_df["mean_predicted"].max(),
        calibration_df["observed"].dropna().max(),
    )
    ax.plot([lo, hi], [lo, hi], "k--", label="Perfect calibration", linewidth=1)

    ax.set_xlabel("Mean predicted CIF")
    ax.set_ylabel("Observed CIF (Aalen-Johansen)")
    ax.set_title(title)
    ax.legend()
    return ax
