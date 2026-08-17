"""Load, validate, and clean the raw experiment logs.

Every cleaning step is counted and returned in a report dict rather than
silently applied, so the notebook/report can disclose exactly how much
data was dropped and why (a common source of unreported bias in A/B
test write-ups).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleaningReport:
    rows_loaded: int
    mismatched_assignment_dropped: int
    duplicate_user_id_dropped: int
    rows_after_cleaning: int
    countries_matched: int = 0
    countries_unmatched: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_loaded": self.rows_loaded,
            "mismatched_assignment_dropped": self.mismatched_assignment_dropped,
            "duplicate_user_id_dropped": self.duplicate_user_id_dropped,
            "rows_after_cleaning": self.rows_after_cleaning,
            "countries_matched": self.countries_matched,
            "countries_unmatched": self.countries_unmatched,
            "notes": self.notes,
        }


def load_ab_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"user_id", "timestamp", "group", "landing_page", "converted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ab_data is missing required columns: {missing}")
    return df


def load_countries(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_ab_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove assignment/exposure mismatches and duplicate user records.

    Two integrity issues are checked, both of which threaten the
    randomization guarantee the whole test depends on if left in:

    1. Assignment/exposure mismatch: a user marked ``treatment`` who was
       actually served ``old_page`` (or vice versa). This is a tracking
       bug, not a real experiment outcome, and mixing it in dilutes the
       measured effect toward zero.
    2. Duplicate ``user_id`` rows: the same user logged more than once.
       Since the unit of randomization is the user, duplicates violate
       the independence assumption behind every test below.
    """
    rows_loaded = len(df)

    mismatch_mask = (df["group"] == "treatment") != (df["landing_page"] == "new_page")
    mismatched = int(mismatch_mask.sum())
    df = df.loc[~mismatch_mask].copy()

    dup_mask = df.duplicated(subset="user_id", keep="first")
    duplicates = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    report = CleaningReport(
        rows_loaded=rows_loaded,
        mismatched_assignment_dropped=mismatched,
        duplicate_user_id_dropped=duplicates,
        rows_after_cleaning=len(df),
    )
    if mismatched / rows_loaded > 0.05:
        report.notes.append(
            "Mismatch rate exceeds 5% of raw rows — investigate the "
            "assignment/exposure logging pipeline before trusting results."
        )
    return df.reset_index(drop=True), report


def merge_countries(
    df: pd.DataFrame, countries_df: pd.DataFrame, report: CleaningReport
) -> tuple[pd.DataFrame, CleaningReport]:
    merged = df.merge(countries_df, on="user_id", how="left")
    report.countries_matched = int(merged["country"].notna().sum())
    report.countries_unmatched = int(merged["country"].isna().sum())
    if report.countries_unmatched:
        report.notes.append(
            f"{report.countries_unmatched} users had no country match and "
            "will be excluded from country-segmented analysis only."
        )
    return merged, report


def prepare_dataset(ab_path: str, countries_path: str) -> tuple[pd.DataFrame, CleaningReport]:
    """End-to-end load -> clean -> merge, used by both the notebook and the CLI skill."""
    raw = load_ab_data(ab_path)
    clean, report = clean_ab_data(raw)
    countries = load_countries(countries_path)
    merged, report = merge_countries(clean, countries, report)
    return merged, report
