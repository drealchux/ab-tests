import numpy as np
import pandas as pd

from ab_testing.segmentation import check_simpsons_paradox, chi_square_independence, novelty_effect_check


def test_no_reversal_when_segments_agree_with_pooled():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame(
        {
            "group": ["control"] * n + ["treatment"] * n,
            "segment": (["A"] * (n // 2) + ["B"] * (n // 2)) * 2,
            "converted": np.concatenate(
                [rng.binomial(1, 0.10, n), rng.binomial(1, 0.14, n)]
            ),
        }
    )
    check = check_simpsons_paradox(df, segment_col="segment")
    assert check.reversal_detected is False
    assert check.pooled_direction == "treatment_higher"


def test_reversal_detected_in_constructed_simpsons_paradox():
    # Classic constructed Simpson's paradox: treatment wins in BOTH segments
    # individually (A: 90% vs 80%, B: 30% vs 20%), but control wins pooled
    # because treatment's sample is concentrated in the low-converting
    # segment B (pooled control 82/110=74.5%, pooled treatment 39/110=35.5%).
    rows = []
    rows += [{"group": "control", "segment": "A", "converted": 1}] * 80
    rows += [{"group": "control", "segment": "A", "converted": 0}] * 20
    rows += [{"group": "treatment", "segment": "A", "converted": 1}] * 9
    rows += [{"group": "treatment", "segment": "A", "converted": 0}] * 1
    rows += [{"group": "control", "segment": "B", "converted": 1}] * 2
    rows += [{"group": "control", "segment": "B", "converted": 0}] * 8
    rows += [{"group": "treatment", "segment": "B", "converted": 1}] * 30
    rows += [{"group": "treatment", "segment": "B", "converted": 0}] * 70

    df = pd.DataFrame(rows)
    check = check_simpsons_paradox(df, segment_col="segment")
    assert check.segment_directions["A"] == "treatment_higher"
    assert check.segment_directions["B"] == "treatment_higher"
    assert check.pooled_direction == "control_higher"
    assert check.reversal_detected is True


def test_chi_square_independence_flags_imbalanced_randomization():
    n = 1000
    # Deliberately imbalanced: segment "A" is almost all control.
    df = pd.DataFrame(
        {
            "group": ["control"] * 900 + ["treatment"] * 100 + ["control"] * 100 + ["treatment"] * 900,
            "segment": ["A"] * 1000 + ["B"] * 1000,
        }
    )
    chi2, p, dof = chi_square_independence(df, segment_col="segment")
    assert p < 0.001


def test_chi_square_independence_high_pvalue_when_balanced():
    # Deterministic, exactly balanced 2x3 table (not randomly sampled --
    # a random draw is legitimately expected to trip p<0.05 about 5% of
    # the time under a true null, which would make this test flaky).
    df = pd.DataFrame(
        {
            "group": ["control"] * 300 + ["treatment"] * 300,
            "segment": (["A"] * 100 + ["B"] * 100 + ["C"] * 100) * 2,
        }
    )
    chi2, p, dof = chi_square_independence(df, segment_col="segment")
    assert p > 0.05


def test_novelty_effect_check_returns_expected_columns():
    rng = np.random.default_rng(5)
    n = 1400
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2017-01-01", periods=n, freq="h"),
            "group": (["control", "treatment"] * (n // 2)),
            "converted": rng.integers(0, 2, n),
        }
    )
    result = novelty_effect_check(df, n_bins=4)
    assert "gap" in result.columns
    assert len(result) == 4
