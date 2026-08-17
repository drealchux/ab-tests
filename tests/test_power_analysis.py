from ab_testing.power_analysis import (
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
    summarize_power,
)


def test_required_sample_size_matches_known_textbook_case():
    # Baseline 10%, detect +2pp lift, alpha=0.05, power=0.80: standard
    # references (e.g. Evan Miller's sample size calculator) give ~3,840
    # per group for this exact configuration.
    n = required_sample_size(baseline_rate=0.10, minimum_detectable_effect=0.02)
    assert 3600 <= n <= 4100


def test_larger_effect_requires_smaller_sample():
    n_small_effect = required_sample_size(0.10, 0.01)
    n_large_effect = required_sample_size(0.10, 0.05)
    assert n_large_effect < n_small_effect


def test_mde_and_required_sample_size_are_inverses():
    n = required_sample_size(baseline_rate=0.10, minimum_detectable_effect=0.02)
    mde = minimum_detectable_effect(baseline_rate=0.10, n_per_group=n)
    assert abs(mde - 0.02) < 0.003


def test_achieved_power_high_when_n_far_exceeds_requirement():
    power = achieved_power(baseline_rate=0.10, observed_effect=0.05, n_per_group=50_000)
    assert power > 0.99


def test_achieved_power_low_when_underpowered():
    power = achieved_power(baseline_rate=0.10, observed_effect=0.001, n_per_group=200)
    assert power < 0.2


def test_summarize_power_returns_consistent_result():
    result = summarize_power(baseline_rate=0.1196, n_per_group=145000, observed_effect=-0.0016)
    assert result.n_per_group == 145000
    assert result.minimum_detectable_effect > 0
    assert result.achieved_power_for_observed_effect is not None
