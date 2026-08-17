#!/usr/bin/env python
"""CLI: run the full statistical pipeline (cleaning -> power -> hypothesis
test -> segmentation -> regression) on a two-arm conversion experiment and
write the results to a JSON file.

This is the computation half of the pipeline; `generate_report.py` turns
its JSON output into a business-readable markdown report. Splitting them
lets the same statistical run feed multiple report formats without
recomputing anything.

Usage:
    python scripts/run_experiment.py \
        --ab-data data/raw/ab_data.csv \
        --countries data/raw/countries.csv \
        --mde 0.01 \
        --out reports/experiment_results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from ab_testing import data_prep, hypothesis_test, power_analysis, regression, segmentation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a rigorous A/B test analysis pipeline.")
    p.add_argument("--ab-data", required=True, help="Path to the experiment CSV (user_id, timestamp, group, landing_page, converted).")
    p.add_argument("--countries", default=None, help="Optional path to a user_id -> country/segment CSV.")
    p.add_argument("--segment-col", default="country", help="Column name to segment on, if --countries is given.")
    p.add_argument("--group-col", default="group")
    p.add_argument("--outcome-col", default="converted")
    p.add_argument("--control-label", default="control")
    p.add_argument("--treatment-label", default="treatment")
    p.add_argument("--mde", type=float, default=0.01, help="Business-meaningful minimum detectable effect (absolute), used to size the pre-registered power check.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--n-permutations", type=int, default=10_000)
    p.add_argument("--out", required=True, help="Path to write the JSON results file.")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.countries:
        df, cleaning_report = data_prep.prepare_dataset(args.ab_data, args.countries)
    else:
        df = data_prep.load_ab_data(args.ab_data)
        df, cleaning_report = data_prep.clean_ab_data(df)

    n_control = int((df[args.group_col] == args.control_label).sum())
    n_treatment = int((df[args.group_col] == args.treatment_label).sum())
    n_per_group = min(n_control, n_treatment)
    baseline_rate = float(df.loc[df[args.group_col] == args.control_label, args.outcome_col].mean())

    power_result = power_analysis.summarize_power(
        baseline_rate, n_per_group=n_per_group, alpha=args.alpha, power_target=args.power
    )
    required_n = power_analysis.required_sample_size(baseline_rate, args.mde, args.alpha, args.power)

    z_result = hypothesis_test.two_proportion_ztest(
        df, args.group_col, args.outcome_col, args.control_label, args.treatment_label,
        alternative="two-sided", alpha=args.alpha,
    )
    perm_result, _ = hypothesis_test.permutation_test(
        df, args.group_col, args.outcome_col, args.control_label, args.treatment_label,
        n_permutations=args.n_permutations, alternative="two-sided", alpha=args.alpha,
    )
    effect_size_h = hypothesis_test.cohens_h(z_result.treatment_rate, z_result.control_rate)
    post_hoc_power = power_analysis.achieved_power(
        baseline_rate, z_result.absolute_difference, n_per_group, args.alpha
    )

    results: dict = {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "baseline_rate": baseline_rate,
        "cleaning_report": cleaning_report.as_dict(),
        "power_analysis": {
            **dataclasses.asdict(power_result),
            "required_n_per_group_for_mde": required_n,
            "mde_input": args.mde,
            "sample_size_ratio_vs_required": n_per_group / required_n,
        },
        "hypothesis_test": {
            "z_test": dataclasses.asdict(z_result),
            "permutation_test": dataclasses.asdict(perm_result),
            "cohens_h": effect_size_h,
            "post_hoc_power_for_observed_effect": post_hoc_power,
            "agree_on_reject_null": z_result.reject_null() == perm_result.reject_null(),
        },
    }

    if args.countries and args.segment_col in df.columns:
        chi2, p_random, dof = segmentation.chi_square_independence(df, args.segment_col, args.group_col)
        simpsons = segmentation.check_simpsons_paradox(
            df, args.segment_col, args.group_col, args.outcome_col, args.control_label, args.treatment_label
        )
        seg_table = segmentation.conversion_by_segment(df, args.segment_col, args.group_col, args.outcome_col)

        per_segment_tests = []
        for segment_value, sub in df.groupby(args.segment_col):
            r = hypothesis_test.two_proportion_ztest(
                sub, args.group_col, args.outcome_col, args.control_label, args.treatment_label, alternative="two-sided", alpha=args.alpha
            )
            per_segment_tests.append({"segment": str(segment_value), **dataclasses.asdict(r)})

        results["segmentation"] = {
            "randomization_balance_chi2": chi2,
            "randomization_balance_p_value": p_random,
            "randomization_balanced": p_random > args.alpha,
            "simpsons_check": dataclasses.asdict(simpsons),
            "conversion_by_segment": seg_table.to_dict(orient="records"),
            "per_segment_significance_tests": per_segment_tests,
        }

        if "timestamp" in df.columns:
            novelty = segmentation.novelty_effect_check(df, "timestamp", args.group_col, args.outcome_col)
            results["novelty_effect"] = novelty[["gap"]].to_dict(orient="records")

        df = df.copy()
        df["ab_page"] = (df[args.group_col] == args.treatment_label).astype(int)
        dummies = pd.get_dummies(df[args.segment_col]).astype(int)
        segment_dummy_cols = [c for c in dummies.columns][1:]  # drop first as baseline
        for c in segment_dummy_cols:
            df[c] = dummies[c]

        m1 = regression.fit_logistic(df, args.outcome_col, ["ab_page"])
        m2 = regression.fit_logistic(df, args.outcome_col, ["ab_page", *segment_dummy_cols])

        results["regression"] = {
            "m1_treatment_only": {
                "odds_ratios": regression.odds_ratio_table(m1).to_dict(orient="records"),
                "aic": m1.aic, "bic": m1.bic, "pseudo_r2": m1.pseudo_r2,
            },
            "m2_with_segment": {
                "odds_ratios": regression.odds_ratio_table(m2).to_dict(orient="records"),
                "aic": m2.aic, "bic": m2.bic, "pseudo_r2": m2.pseudo_r2,
            },
            "likelihood_ratio_test_m1_vs_m2": regression.likelihood_ratio_test(m1, m2),
        }
    else:
        df = df.copy()
        df["ab_page"] = (df[args.group_col] == args.treatment_label).astype(int)
        m1 = regression.fit_logistic(df, args.outcome_col, ["ab_page"])
        results["regression"] = {
            "m1_treatment_only": {
                "odds_ratios": regression.odds_ratio_table(m1).to_dict(orient="records"),
                "aic": m1.aic, "bic": m1.bic, "pseudo_r2": m1.pseudo_r2,
            }
        }

    return results


def main() -> None:
    args = parse_args()
    results = run(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote results to {out_path}")
    print(f"  n_control={results['n_control']:,}  n_treatment={results['n_treatment']:,}")
    print(f"  z-test p={results['hypothesis_test']['z_test']['p_value']:.4f}  "
          f"reject_null={results['hypothesis_test']['z_test']['p_value'] < args.alpha}")


if __name__ == "__main__":
    main()
