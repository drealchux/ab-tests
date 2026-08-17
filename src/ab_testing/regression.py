"""Logistic regression modeling of conversion, with the diagnostics the
original notebook skipped: odds ratios reported with confidence intervals
(not just a point estimate, and without sign errors), a variance inflation
factor table before trusting any interaction term, and a likelihood-ratio
test to check whether adding predictors actually improves the model rather
than just adding noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass
class FittedModel:
    name: str
    result: sm.discrete.discrete_model.BinaryResultsWrapper
    predictors: list[str]

    @property
    def llf(self) -> float:
        return self.result.llf

    @property
    def aic(self) -> float:
        return self.result.aic

    @property
    def bic(self) -> float:
        return self.result.bic

    @property
    def pseudo_r2(self) -> float:
        """McFadden's pseudo R^2: 1 - (log-likelihood of fitted model /
        log-likelihood of intercept-only model). Unlike OLS R^2, values of
        0.2-0.4 are already considered a strong fit for this statistic.
        """
        return float(self.result.prsquared)


def fit_logistic(
    df: pd.DataFrame, outcome_col: str, predictor_cols: list[str], add_intercept: bool = True
) -> FittedModel:
    X = df[predictor_cols].astype(float).copy()
    if add_intercept:
        X = sm.add_constant(X, has_constant="add")
    y = df[outcome_col].astype(float)
    result = sm.Logit(y, X).fit(disp=0)
    name = " + ".join(predictor_cols) if predictor_cols else "intercept_only"
    return FittedModel(name=name, result=result, predictors=list(X.columns))


def odds_ratio_table(model: FittedModel, alpha: float = 0.05) -> pd.DataFrame:
    """Odds ratios with confidence intervals, computed the correct way:
    exponentiate the coefficient and its CI endpoints directly. (The
    original notebook's `1/np.exp(-coef)` trick for negative coefficients
    is mathematically equivalent to `np.exp(coef)` and was applied
    inconsistently across sections, producing contradictory-looking
    interpretations for coefficients of the same sign.)
    """
    params = model.result.params
    conf = model.result.conf_int(alpha=alpha)
    conf.columns = ["ci_low", "ci_high"]
    table = pd.DataFrame(
        {
            "coefficient": params,
            "std_err": model.result.bse,
            "p_value": model.result.pvalues,
            "odds_ratio": np.exp(params),
            "odds_ratio_ci_low": np.exp(conf["ci_low"]),
            "odds_ratio_ci_high": np.exp(conf["ci_high"]),
        }
    )
    return table.reset_index().rename(columns={"index": "predictor"})


def vif_table(df: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    """Variance inflation factor per predictor. VIF > 5 (some practitioners
    use 10) flags a predictor whose standard error is inflated by
    collinearity with the others — relevant here because interaction terms
    like ``UK_new_page`` are, by construction, correlated with their main
    effects (``new_page``, ``UK``).
    """
    X = sm.add_constant(df[predictor_cols].astype(float), has_constant="add")
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.to_numpy(), i)
        rows.append({"predictor": col, "vif": float(vif)})
    return pd.DataFrame(rows)


def likelihood_ratio_test(restricted: FittedModel, full: FittedModel) -> dict:
    """Tests whether the extra predictors in ``full`` (nested inside
    ``restricted``) significantly improve fit, via -2*(LL_restricted -
    LL_full) ~ chi-square(df = extra params). More rigorous than comparing
    AIC/BIC alone when the goal is a formal significance statement about
    whether a group of added variables belongs in the model.
    """
    lr_stat = 2 * (full.llf - restricted.llf)
    df_diff = len(full.predictors) - len(restricted.predictors)
    p_value = float(stats.chi2.sf(lr_stat, df_diff))
    return {
        "lr_statistic": float(lr_stat),
        "df_difference": df_diff,
        "p_value": p_value,
        "significant_improvement": p_value < 0.05,
    }
