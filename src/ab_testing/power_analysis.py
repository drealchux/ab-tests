"""Power analysis for two-proportion experiments.

Power should be established *before* interpreting a result, not after.
An underpowered "fail to reject" is not evidence of no effect — it is
evidence of nothing. This module lets the notebook check, given the
sample size the experiment actually collected, what minimum effect it
was even capable of detecting, and to size a rerun if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


@dataclass
class PowerResult:
    baseline_rate: float
    alpha: float
    power_target: float
    n_per_group: int
    minimum_detectable_effect: float  # absolute, in proportion points
    achieved_power_for_observed_effect: float | None = None


def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Sample size needed *per group* to detect an absolute lift of
    ``minimum_detectable_effect`` on top of ``baseline_rate`` with the
    given significance level and power, using a two-proportion z-test.
    """
    effect_size = proportion_effectsize(
        baseline_rate + minimum_detectable_effect, baseline_rate
    )
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=abs(effect_size), alpha=alpha, power=power, ratio=1.0
    )
    return int(np.ceil(n))


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Inverse of ``required_sample_size``: given the sample size actually
    collected, what absolute lift could the test detect at the target power?
    Search over a fine grid since statsmodels has no closed-form inverse
    for effect size given n.
    """
    analysis = NormalIndPower()
    grid = np.linspace(0.0005, 0.5, 2000)
    for mde in grid:
        effect_size = abs(proportion_effectsize(baseline_rate + mde, baseline_rate))
        achieved = analysis.solve_power(
            effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0
        )
        if achieved >= power:
            return float(mde)
    return float(grid[-1])


def achieved_power(
    baseline_rate: float,
    observed_effect: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Power the test actually had, post hoc, to detect the effect size
    that was observed. Reported for transparency, not used to justify
    the conclusion (post-hoc "observed power" is not a substitute for
    a pre-registered MDE, and is reported here only as context).
    """
    effect_size = abs(proportion_effectsize(baseline_rate + observed_effect, baseline_rate))
    analysis = NormalIndPower()
    return float(
        analysis.solve_power(effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=1.0)
    )


def summarize_power(
    baseline_rate: float,
    n_per_group: int,
    observed_effect: float | None = None,
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> PowerResult:
    mde = minimum_detectable_effect(baseline_rate, n_per_group, alpha, power_target)
    result = PowerResult(
        baseline_rate=baseline_rate,
        alpha=alpha,
        power_target=power_target,
        n_per_group=n_per_group,
        minimum_detectable_effect=mde,
    )
    if observed_effect is not None:
        result.achieved_power_for_observed_effect = achieved_power(
            baseline_rate, observed_effect, n_per_group, alpha
        )
    return result
