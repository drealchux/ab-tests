"""Diagnostics the original analysis skipped: does the pooled result hold
up inside every segment, and is the effect stable over the life of the
experiment?

Both checks matter because a pooled A/B result can be misleading in two
specific, well-documented ways:

1. Simpson's paradox: the pooled effect can differ in sign or magnitude
   from the effect in every individual subgroup if segment sizes are
   unbalanced across arms.
2. Novelty/primacy effects: a treatment effect that decays (or grows) over
   the course of the experiment means the pooled average is not a stable
   estimate of the steady-state effect a permanent launch would produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


@dataclass
class SimpsonsCheck:
    pooled_direction: str  # "treatment_higher" | "control_higher" | "tie"
    segment_directions: dict[str, str]
    reversal_detected: bool


def conversion_by_segment(
    df: pd.DataFrame, segment_col: str, group_col: str = "group", outcome_col: str = "converted"
) -> pd.DataFrame:
    table = (
        df.groupby([segment_col, group_col])[outcome_col]
        .agg(["mean", "count", "sum"])
        .rename(columns={"mean": "conversion_rate", "count": "n", "sum": "conversions"})
        .reset_index()
    )
    return table


def chi_square_independence(
    df: pd.DataFrame, segment_col: str, group_col: str = "group"
) -> tuple[float, float, int]:
    """Tests whether experiment group assignment is independent of the
    segment (e.g. country) — i.e. whether randomization was balanced
    across segments. A significant result here means the randomization
    itself may be confounded with the segment, not that the segment
    affects conversion.
    """
    contingency = pd.crosstab(df[segment_col], df[group_col])
    chi2, p, dof, _ = chi2_contingency(contingency)
    return float(chi2), float(p), int(dof)


def check_simpsons_paradox(
    df: pd.DataFrame,
    segment_col: str,
    group_col: str = "group",
    outcome_col: str = "converted",
    control_label: str = "control",
    treatment_label: str = "treatment",
) -> SimpsonsCheck:
    pooled = df.groupby(group_col)[outcome_col].mean()
    pooled_diff = pooled.get(treatment_label, np.nan) - pooled.get(control_label, np.nan)
    pooled_direction = _direction(pooled_diff)

    segment_directions: dict[str, str] = {}
    for segment_value, sub in df.groupby(segment_col):
        rates = sub.groupby(group_col)[outcome_col].mean()
        if control_label not in rates or treatment_label not in rates:
            continue
        diff = rates[treatment_label] - rates[control_label]
        segment_directions[str(segment_value)] = _direction(diff)

    reversal = any(
        d != pooled_direction and d != "tie" and pooled_direction != "tie"
        for d in segment_directions.values()
    )
    return SimpsonsCheck(pooled_direction, segment_directions, reversal)


def _direction(diff: float) -> str:
    if abs(diff) < 1e-9:
        return "tie"
    return "treatment_higher" if diff > 0 else "control_higher"


def novelty_effect_check(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    group_col: str = "group",
    outcome_col: str = "converted",
    n_bins: int = 7,
) -> pd.DataFrame:
    """Bins the experiment window into equal-width time periods and reports
    the treatment-vs-control conversion gap in each. A gap that trends
    toward (or away from) zero over time is evidence of a novelty/primacy
    effect that the overall pooled average would mask.
    """
    working = df.copy()
    working["_time_bin"] = pd.cut(working[timestamp_col], bins=n_bins, labels=False)
    pivot = (
        working.groupby(["_time_bin", group_col])[outcome_col]
        .mean()
        .unstack(group_col)
    )
    pivot["gap"] = pivot.get("treatment") - pivot.get("control")
    bin_edges = pd.cut(working[timestamp_col], bins=n_bins).cat.categories
    pivot["period_start"] = [bin_edges[i].left for i in pivot.index]
    pivot["period_end"] = [bin_edges[i].right for i in pivot.index]
    return pivot.reset_index()
