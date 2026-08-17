import numpy as np

from ab_testing.hypothesis_test import (
    bootstrap_ci,
    cohens_h,
    permutation_test,
    two_proportion_ztest,
    wilson_confidence_interval,
)


def test_ztest_detects_known_effect(clean_experiment_df):
    result = two_proportion_ztest(clean_experiment_df, alternative="two-sided")
    assert result.p_value < 0.05
    assert result.reject_null()
    assert result.absolute_difference > 0  # treatment (12%) > control (10%)


def test_ztest_fails_to_reject_when_no_effect(null_experiment_df):
    result = two_proportion_ztest(null_experiment_df, alternative="two-sided")
    assert result.p_value > 0.05
    assert not result.reject_null()


def test_permutation_test_agrees_with_ztest_direction(clean_experiment_df):
    z_result = two_proportion_ztest(clean_experiment_df)
    perm_result, perm_diffs = permutation_test(clean_experiment_df, n_permutations=5000)
    assert perm_result.reject_null() == z_result.reject_null()
    assert np.sign(perm_result.absolute_difference) == np.sign(z_result.absolute_difference)


def test_permutation_null_distribution_centered_near_zero(clean_experiment_df):
    _, perm_diffs = permutation_test(clean_experiment_df, n_permutations=5000)
    assert abs(perm_diffs.mean()) < 0.01


def test_permutation_pvalue_uniform_under_null(null_experiment_df):
    result, _ = permutation_test(null_experiment_df, n_permutations=5000)
    # Under a true null, p-value should not be extreme/tiny.
    assert result.p_value > 0.05


def test_bootstrap_ci_contains_true_difference_direction():
    lo, hi = bootstrap_ci(control_rate=0.10, n_control=5000, treatment_rate=0.12, n_treatment=5000)
    assert lo < 0.02 < hi  # true diff (0.02) should fall inside a 95% CI most of the time
    assert lo < hi


def test_wilson_ci_narrower_with_more_data():
    lo_small, hi_small = wilson_confidence_interval(successes=50, n=500)
    lo_large, hi_large = wilson_confidence_interval(successes=5000, n=50000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_cohens_h_zero_when_equal():
    assert abs(cohens_h(0.1, 0.1)) < 1e-9


def test_cohens_h_positive_when_first_larger():
    assert cohens_h(0.15, 0.10) > 0
    assert cohens_h(0.10, 0.15) < 0
