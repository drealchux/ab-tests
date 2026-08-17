import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_ab_df():
    """A small synthetic dataset with known, injectable data-quality issues:
    2 mismatched assignment rows and 2 duplicate user_id rows, on top of a
    clean core of 20 users so cleaning logic has a known ground truth to
    check against.
    """
    rng = np.random.default_rng(0)
    n = 20
    user_ids = np.arange(1000, 1000 + n)
    groups = ["control"] * (n // 2) + ["treatment"] * (n // 2)
    landing_pages = ["old_page"] * (n // 2) + ["new_page"] * (n // 2)
    converted = rng.integers(0, 2, size=n)
    timestamps = pd.date_range("2017-01-01", periods=n, freq="h")

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "timestamp": timestamps,
            "group": groups,
            "landing_page": landing_pages,
            "converted": converted,
        }
    )

    mismatch_rows = pd.DataFrame(
        {
            "user_id": [2000, 2001],
            "timestamp": [pd.Timestamp("2017-01-05"), pd.Timestamp("2017-01-06")],
            "group": ["treatment", "control"],
            "landing_page": ["old_page", "new_page"],
            "converted": [0, 1],
        }
    )
    duplicate_rows = df.iloc[[0, 1]].copy()

    return pd.concat([df, mismatch_rows, duplicate_rows], ignore_index=True)


@pytest.fixture
def countries_df():
    return pd.DataFrame(
        {
            "user_id": list(range(1000, 1020)),
            "country": (["US"] * 10 + ["UK"] * 7 + ["CA"] * 3),
        }
    )


@pytest.fixture
def clean_experiment_df():
    """Larger synthetic experiment with a real, known effect: treatment
    converts at 12%, control at 10%, n=5000 per group. Used to check that
    hypothesis tests correctly detect a real, specified effect.
    """
    rng = np.random.default_rng(42)
    n_per_group = 5000
    control_converted = rng.binomial(1, 0.10, n_per_group)
    treatment_converted = rng.binomial(1, 0.12, n_per_group)

    df = pd.DataFrame(
        {
            "user_id": range(n_per_group * 2),
            "group": ["control"] * n_per_group + ["treatment"] * n_per_group,
            "converted": np.concatenate([control_converted, treatment_converted]),
            "timestamp": pd.date_range("2017-01-01", periods=n_per_group * 2, freq="min"),
        }
    )
    return df


@pytest.fixture
def null_experiment_df():
    """Synthetic experiment with NO true effect: both arms convert at 10%,
    n=3000 per group. Used to check that tests correctly fail to reject.
    """
    rng = np.random.default_rng(7)
    n_per_group = 3000
    control_converted = rng.binomial(1, 0.10, n_per_group)
    treatment_converted = rng.binomial(1, 0.10, n_per_group)

    df = pd.DataFrame(
        {
            "user_id": range(n_per_group * 2),
            "group": ["control"] * n_per_group + ["treatment"] * n_per_group,
            "converted": np.concatenate([control_converted, treatment_converted]),
            "timestamp": pd.date_range("2017-01-01", periods=n_per_group * 2, freq="min"),
        }
    )
    return df
