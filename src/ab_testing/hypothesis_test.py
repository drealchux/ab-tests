"""Two-proportion hypothesis testing for the conversion experiment.

Two independent tests are provided on purpose:

- ``two_proportion_ztest``: the standard parametric test (normal
  approximation to the binomial), fast and exact enough at this sample size.
- ``permutation_test``: a nonparametric test that shuffles the observed
  treatment labels directly, rather than drawing from a parametric normal
  distribution. This is the statistically correct version of the resampling
  approach: it reuses the real data instead of approximating it, so it
  makes no distributional assumption at all.

Agreement between the two is the robustness check: if the parametric and
nonparametric tests disagree, that is itself a finding worth reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


@dataclass
class TestResult:
    method: str
    control_rate: float
    treatment_rate: float
    absolute_difference: float  # treatment - control
    relative_lift: float
    z_score: float | None
    p_value: float
    alternative: str
    ci_low: float
    ci_high: float
    alpha: float

    def reject_null(self) -> bool:
        return self.p_value < self.alpha


def conversion_rates(df: pd.DataFrame, group_col: str = "group", outcome_col: str = "converted"):
    rates = df.groupby(group_col)[outcome_col].mean()
    counts = df.groupby(group_col)[outcome_col].count()
    return rates, counts


def two_proportion_ztest(
    df: pd.DataFrame,
    group_col: str = "group",
    outcome_col: str = "converted",
    control_label: str = "control",
    treatment_label: str = "treatment",
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    rates, counts = conversion_rates(df, group_col, outcome_col)
    control_conv = int(df.loc[df[group_col] == control_label, outcome_col].sum())
    treatment_conv = int(df.loc[df[group_col] == treatment_label, outcome_col].sum())
    n_control = int(counts[control_label])
    n_treatment = int(counts[treatment_label])

    z, p = proportions_ztest(
        count=[treatment_conv, control_conv],
        nobs=[n_treatment, n_control],
        alternative=("two-sided" if alternative == "two-sided" else "larger"),
    )

    diff = rates[treatment_label] - rates[control_label]
    se_diff = np.sqrt(
        rates[control_label] * (1 - rates[control_label]) / n_control
        + rates[treatment_label] * (1 - rates[treatment_label]) / n_treatment
    )
    crit = norm.ppf(1 - alpha / 2)
    ci_low, ci_high = diff - crit * se_diff, diff + crit * se_diff

    return TestResult(
        method="two_proportion_ztest",
        control_rate=float(rates[control_label]),
        treatment_rate=float(rates[treatment_label]),
        absolute_difference=float(diff),
        relative_lift=float(diff / rates[control_label]),
        z_score=float(z),
        p_value=float(p),
        alternative=alternative,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
    )


def permutation_test(
    df: pd.DataFrame,
    group_col: str = "group",
    outcome_col: str = "converted",
    control_label: str = "control",
    treatment_label: str = "treatment",
    n_permutations: int = 10_000,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[TestResult, np.ndarray]:
    """Simulate the label-shuffling null distribution of the conversion gap.

    Physically shuffling ~290k rows ``n_permutations`` times is wasteful:
    since the outcome is binary, the only thing a shuffle can change is
    *how many* of the ``K`` total conversions land in the treatment group,
    and that count's distribution under random shuffling is exactly
    Hypergeometric(pool_size, total_conversions, treatment_size) — the
    standard combinatorial result for sampling without replacement. Drawing
    from that hypergeometric distribution directly is mathematically
    identical to shuffling labels and recomputing group means, just
    vectorized instead of an O(n_permutations * n) Python loop.
    """
    rng = np.random.default_rng(random_state)
    mask = df[group_col].isin([control_label, treatment_label])
    sub = df.loc[mask, [group_col, outcome_col]]

    labels = sub[group_col].to_numpy()
    outcomes = sub[outcome_col].to_numpy()

    n_treatment = int((labels == treatment_label).sum())
    n_control = int((labels == control_label).sum())
    n_total = n_treatment + n_control
    total_conversions = int(outcomes.sum())

    observed_diff = outcomes[labels == treatment_label].mean() - outcomes[labels == control_label].mean()

    treatment_conversions_sim = rng.hypergeometric(
        ngood=total_conversions,
        nbad=n_total - total_conversions,
        nsample=n_treatment,
        size=n_permutations,
    )
    control_conversions_sim = total_conversions - treatment_conversions_sim
    treatment_rate_sim = treatment_conversions_sim / n_treatment
    control_rate_sim = control_conversions_sim / n_control
    perm_diffs = treatment_rate_sim - control_rate_sim

    if alternative == "two-sided":
        p_value = float((np.abs(perm_diffs) >= abs(observed_diff)).mean())
    elif alternative == "larger":
        p_value = float((perm_diffs >= observed_diff).mean())
    else:
        p_value = float((perm_diffs <= observed_diff).mean())
    p_value = max(p_value, 1 / n_permutations)  # cannot claim p < 1/n_permutations resolution

    control_rate = float(outcomes[labels == control_label].mean())
    treatment_rate = float(outcomes[labels == treatment_label].mean())

    # The permutation null is centered on 0 by construction, so its
    # percentiles are not a confidence interval for the observed
    # difference — that requires a *separate* bootstrap that resamples
    # each group under the actually-observed rates, not the shared null.
    ci_low, ci_high = bootstrap_ci(
        control_rate, n_control, treatment_rate, n_treatment, random_state=random_state
    )

    return TestResult(
        method="permutation_test",
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_difference=float(observed_diff),
        relative_lift=float(observed_diff / control_rate),
        z_score=None,
        p_value=p_value,
        alternative=alternative,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
    ), perm_diffs


def bootstrap_ci(
    control_rate: float,
    n_control: int,
    treatment_rate: float,
    n_treatment: int,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the treatment-minus-control difference.

    Resampling a binary column with replacement and taking the mean is
    equivalent, in distribution, to drawing ``Binomial(n, rate) / n`` —
    so the whole bootstrap is generated in one vectorized call instead of
    resampling raw rows ``n_bootstrap`` times.
    """
    rng = np.random.default_rng(random_state)
    control_boot = rng.binomial(n_control, control_rate, size=n_bootstrap) / n_control
    treatment_boot = rng.binomial(n_treatment, treatment_rate, size=n_bootstrap) / n_treatment
    diffs = treatment_boot - control_boot
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilson_confidence_interval(successes: int, n: int, alpha: float = 0.05):
    return proportion_confint(successes, n, alpha=alpha, method="wilson")


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference of two proportions, comparable across
    experiments with different baseline rates (unlike the raw percentage-point
    difference, which is scale-dependent).
    """
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
