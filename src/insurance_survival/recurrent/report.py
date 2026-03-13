"""
Diagnostics and reporting for fitted frailty models.

FrailtyReport produces a standard set of diagnostics that actuarial teams
need when presenting a frailty model to a pricing committee:

1. Frailty distribution summary — is there meaningful heterogeneity?
2. Credibility scores by segment — do high-risk segments get high frailty?
3. Event rate by frailty decile — validates the model structure
4. Convergence diagnostics — did the EM behave sensibly?
5. Model comparison (AIC/BIC) — gamma vs lognormal, with vs without frailty

These diagnostics are intentionally simple. The goal is not academic rigour
but usefulness to a practitioner who needs to explain the model to non-statisticians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import RecurrentEventData
from .models import AndersenGillFrailty, FrailtyFitResult, NelsonAalenFrailty


@dataclass
class FrailtyReport:
    """
    Diagnostic report for a fitted AndersenGillFrailty or NelsonAalenFrailty model.

    Attributes
    ----------
    model : AndersenGillFrailty or NelsonAalenFrailty
        The fitted model.
    data : RecurrentEventData
        The data used for fitting.
    """

    model: object  # AndersenGillFrailty or NelsonAalenFrailty
    data: RecurrentEventData

    def frailty_summary(self) -> pd.DataFrame:
        """
        Summary statistics of the posterior frailty distribution.

        Returns mean, std, percentiles of E[z_i | data] across all subjects.
        The mean should be close to 1.0 (frailty is centred). The std
        measures heterogeneity — larger std means more unobserved variation.
        """
        scores = self.model.credibility_scores()
        z = scores["frailty_mean"].values
        return pd.DataFrame(
            {
                "statistic": [
                    "mean",
                    "std",
                    "p5",
                    "p25",
                    "median",
                    "p75",
                    "p95",
                    "min",
                    "max",
                ],
                "value": [
                    float(np.mean(z)),
                    float(np.std(z)),
                    float(np.percentile(z, 5)),
                    float(np.percentile(z, 25)),
                    float(np.median(z)),
                    float(np.percentile(z, 75)),
                    float(np.percentile(z, 95)),
                    float(np.min(z)),
                    float(np.max(z)),
                ],
            }
        ).set_index("statistic")

    def event_rate_by_frailty_decile(self) -> pd.DataFrame:
        """
        Observed event rate stratified by decile of posterior frailty.

        If the model is working, the event rate should increase monotonically
        with frailty decile. Flat or non-monotone patterns suggest the frailty
        is not capturing the right variation.
        """
        scores = self.model.credibility_scores()
        scores = scores.copy()
        scores["frailty_decile"] = pd.qcut(
            scores["frailty_mean"], q=10, labels=False, duplicates="drop"
        ) + 1

        result = (
            scores.groupby("frailty_decile")
            .agg(
                n_subjects=("id", "count"),
                total_events=("n_events", "sum"),
                mean_frailty=("frailty_mean", "mean"),
            )
            .reset_index()
        )
        result["mean_events_per_subject"] = (
            result["total_events"] / result["n_subjects"]
        )
        return result

    def credibility_by_group(
        self, group_col: str
    ) -> pd.DataFrame:
        """
        Credibility-weighted claim predictions by a categorical grouping variable.

        This is the actuary's table: for each segment, what does the portfolio
        prior predict, and what does the experience-adjusted estimate say?

        Parameters
        ----------
        group_col : str
            Column in data.df to group by (e.g., "region", "vehicle_age_band").

        Returns
        -------
        pd.DataFrame with columns: group, n_subjects, obs_rate, credibility_weight,
        frailty_mean.
        """
        scores = self.model.credibility_scores()
        # Join group col from data
        group_map = (
            self.data.df.groupby(self.data.id_col)[group_col]
            .first()
            .reset_index()
            .rename(columns={self.data.id_col: "id"})
        )
        merged = scores.merge(group_map, on="id")
        exposure = (
            self.data.df.groupby(self.data.id_col)
            .apply(lambda g: float((g[self.data.stop_col] - g[self.data.start_col]).sum()))
            .reset_index()
            .rename(columns={self.data.id_col: "id", 0: "exposure"})
        )
        merged = merged.merge(exposure, on="id")
        result = (
            merged.groupby(group_col)
            .agg(
                n_subjects=("id", "count"),
                total_events=("n_events", "sum"),
                total_exposure=("exposure", "sum"),
                mean_credibility=("credibility_weight", "mean"),
                mean_frailty=("frailty_mean", "mean"),
            )
            .reset_index()
        )
        result["obs_rate"] = result["total_events"] / result["total_exposure"]
        return result

    def model_aic(self) -> float:
        """Akaike Information Criterion: -2 * log_lik + 2 * n_params."""
        result = self.model.result_
        if hasattr(result, "coef"):
            n_params = len(result.coef) + 1  # +1 for theta
        else:
            n_params = 1
        ll = result.log_likelihood if hasattr(result, "log_likelihood") else result["log_likelihood"]
        return -2 * ll + 2 * n_params

    def model_bic(self) -> float:
        """Bayesian Information Criterion: -2 * log_lik + n_params * log(n_subjects)."""
        result = self.model.result_
        if hasattr(result, "coef"):
            n_params = len(result.coef) + 1
        else:
            n_params = 1
        n = self.data.n_subjects
        ll = result.log_likelihood if hasattr(result, "log_likelihood") else result["log_likelihood"]
        return -2 * ll + n_params * np.log(n)

    def convergence_summary(self) -> dict:
        """Basic EM convergence diagnostics."""
        result = self.model.result_ if hasattr(self.model, "result_") else None
        if result is None:
            return {}
        return {
            "converged": result.converged if hasattr(result, "converged") else None,
            "n_iter": result.n_iter if hasattr(result, "n_iter") else None,
            "log_likelihood": result.log_likelihood if hasattr(result, "log_likelihood") else None,
            "theta": result.theta if hasattr(result, "theta") else None,
            "frailty_variance": (
                1.0 / result.theta if hasattr(result, "theta") and result.theta > 0 else None
            ),
        }


def compare_models(
    models: list,
    names: Optional[list[str]] = None,
    data: Optional[RecurrentEventData] = None,
) -> pd.DataFrame:
    """
    Compare multiple fitted models by log-likelihood, AIC, and BIC.

    Parameters
    ----------
    models : list
        Fitted model objects.
    names : list of str, optional
        Model names for the table.
    data : RecurrentEventData, optional
        Required for AIC/BIC computation (n_subjects).

    Returns
    -------
    pd.DataFrame sorted by AIC.
    """
    if names is None:
        names = [repr(m) for m in models]

    n = data.n_subjects if data is not None else None
    rows = []
    for name, model in zip(names, models):
        result = model.result_ if hasattr(model, "result_") else None
        if result is None:
            continue
        ll = getattr(result, "log_likelihood", None)
        if ll is None:
            continue
        coef = getattr(result, "coef", np.array([]))
        n_params = len(coef) + 1

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(n) if n is not None else float("nan")

        rows.append({
            "model": name,
            "log_likelihood": ll,
            "n_params": n_params,
            "AIC": aic,
            "BIC": bic,
            "theta": getattr(result, "theta", float("nan")),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("AIC").reset_index(drop=True)
    return df
