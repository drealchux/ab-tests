import numpy as np
import pandas as pd
import pytest

from ab_testing.regression import fit_logistic, likelihood_ratio_test, odds_ratio_table, vif_table


@pytest.fixture
def logistic_df():
    rng = np.random.default_rng(11)
    n = 4000
    x1 = rng.integers(0, 2, n).astype(float)
    logit_p = -2.2 + 0.8 * x1
    p = 1 / (1 + np.exp(-logit_p))
    y = rng.binomial(1, p)
    return pd.DataFrame({"treatment": x1, "converted": y})


def test_fit_logistic_recovers_known_coefficient_sign(logistic_df):
    model = fit_logistic(logistic_df, outcome_col="converted", predictor_cols=["treatment"])
    coef = model.result.params["treatment"]
    assert coef > 0  # true effect was +0.8


def test_odds_ratio_table_matches_manual_exponentiation(logistic_df):
    model = fit_logistic(logistic_df, outcome_col="converted", predictor_cols=["treatment"])
    table = odds_ratio_table(model)
    row = table.set_index("predictor").loc["treatment"]
    assert row["odds_ratio"] == pytest.approx(np.exp(row["coefficient"]), rel=1e-6)
    assert row["odds_ratio_ci_low"] < row["odds_ratio"] < row["odds_ratio_ci_high"]


def test_odds_ratio_sign_consistent_for_negative_coefficient(logistic_df):
    # Flip the sign of the predictor; the odds ratio for the flipped
    # predictor should be the reciprocal of the original, not something
    # inconsistent (this is the exact bug the original notebook had).
    flipped = logistic_df.copy()
    flipped["treatment"] = 1 - flipped["treatment"]
    model_orig = fit_logistic(logistic_df, outcome_col="converted", predictor_cols=["treatment"])
    model_flip = fit_logistic(flipped, outcome_col="converted", predictor_cols=["treatment"])
    or_orig = odds_ratio_table(model_orig).set_index("predictor").loc["treatment", "odds_ratio"]
    or_flip = odds_ratio_table(model_flip).set_index("predictor").loc["treatment", "odds_ratio"]
    assert or_orig == pytest.approx(1 / or_flip, rel=1e-3)


def test_vif_flags_collinear_predictors():
    rng = np.random.default_rng(2)
    n = 2000
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.01, size=n)  # near-perfect collinearity with a
    c = rng.normal(size=n)  # independent
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    table = vif_table(df, ["a", "b", "c"]).set_index("predictor")
    assert table.loc["a", "vif"] > 10
    assert table.loc["b", "vif"] > 10
    assert table.loc["c", "vif"] < 5


def test_likelihood_ratio_test_detects_useful_predictor(logistic_df):
    restricted = fit_logistic(logistic_df, outcome_col="converted", predictor_cols=[])
    full = fit_logistic(logistic_df, outcome_col="converted", predictor_cols=["treatment"])
    lr = likelihood_ratio_test(restricted, full)
    assert lr["significant_improvement"] is True
    assert lr["p_value"] < 0.05


def test_likelihood_ratio_test_no_improvement_for_noise_predictor(logistic_df):
    rng = np.random.default_rng(99)
    df = logistic_df.copy()
    df["noise"] = rng.normal(size=len(df))
    restricted = fit_logistic(df, outcome_col="converted", predictor_cols=["treatment"])
    full = fit_logistic(df, outcome_col="converted", predictor_cols=["treatment", "noise"])
    lr = likelihood_ratio_test(restricted, full)
    assert lr["p_value"] > 0.05
