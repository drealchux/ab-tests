#!/usr/bin/env python
"""CLI: turn a JSON results file produced by run_experiment.py into a
business-readable markdown report with an explicit, rule-based
recommendation (ship / don't ship / run longer).

Usage:
    python scripts/generate_report.py --results reports/experiment_results.json --out reports/experiment_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a markdown report from run_experiment.py output.")
    p.add_argument("--results", required=True, help="Path to the JSON file produced by run_experiment.py")
    p.add_argument("--out", required=True, help="Path to write the markdown report")
    return p.parse_args()


def recommend(results: dict) -> tuple[str, str]:
    """Rule-based recommendation. Deliberately conservative: it will not
    recommend shipping on a significant-but-underpowered result, and will
    not claim a confident "no effect" on an underpowered null result.
    """
    ht = results["hypothesis_test"]
    pa = results["power_analysis"]
    reject = ht["z_test"]["p_value"] < 0.05
    agree = ht["agree_on_reject_null"]
    well_powered = pa["sample_size_ratio_vs_required"] >= 1.0

    if not agree:
        return (
            "INCONCLUSIVE",
            "The parametric (z-test) and nonparametric (permutation) tests disagree on "
            "statistical significance. Do not act on this result — investigate why the two "
            "methods diverge (likely a distributional assumption violation) before deciding.",
        )
    if reject and well_powered:
        direction = "higher" if ht["z_test"]["absolute_difference"] > 0 else "lower"
        return (
            "SHIP" if direction == "higher" else "DO NOT SHIP",
            f"Treatment conversion is statistically significantly {direction} than control "
            f"(p={ht['z_test']['p_value']:.4f}), and the experiment was adequately powered.",
        )
    if not reject and well_powered:
        return (
            "DO NOT SHIP",
            f"No statistically significant difference was found (p={ht['z_test']['p_value']:.4f}), "
            f"and the experiment collected {pa['sample_size_ratio_vs_required']:.1f}x the sample size "
            "needed to detect the pre-specified minimum meaningful effect — this is a well-powered "
            "null result, not an inconclusive one.",
        )
    return (
        "RUN LONGER",
        f"The experiment collected only {pa['sample_size_ratio_vs_required']:.1f}x the sample size "
        "needed to detect the pre-specified minimum meaningful effect. A 'fail to reject' result here "
        "cannot be distinguished from 'not enough data' — extend the experiment before deciding.",
    )


def render(results: dict) -> str:
    ht = results["hypothesis_test"]
    pa = results["power_analysis"]
    z = ht["z_test"]
    verdict, rationale = recommend(results)

    lines = []
    lines.append("# A/B Test Experiment Report")
    lines.append("")
    lines.append(f"**Recommendation: {verdict}**")
    lines.append("")
    lines.append(rationale)
    lines.append("")
    lines.append("## Sample")
    lines.append(f"- Control: {results['n_control']:,} users, baseline conversion {results['baseline_rate']:.4%}")
    lines.append(f"- Treatment: {results['n_treatment']:,} users, conversion {z['treatment_rate']:.4%}")
    lines.append("")
    lines.append("## Data Cleaning")
    cr = results["cleaning_report"]
    lines.append(f"- Rows loaded: {cr['rows_loaded']:,}")
    lines.append(f"- Dropped (assignment/exposure mismatch): {cr['mismatched_assignment_dropped']:,}")
    lines.append(f"- Dropped (duplicate user_id): {cr['duplicate_user_id_dropped']:,}")
    lines.append(f"- Rows analyzed: {cr['rows_after_cleaning']:,}")
    for note in cr.get("notes", []):
        lines.append(f"- ⚠️ {note}")
    lines.append("")
    lines.append("## Power Analysis")
    lines.append(f"- Minimum detectable effect at {pa['power_target']:.0%} power, given actual sample size: **{pa['minimum_detectable_effect']:.4%}**")
    lines.append(f"- Required sample size per arm for the specified MDE ({pa['mde_input']:.2%}): {pa['required_n_per_group_for_mde']:,}")
    lines.append(f"- Actual sample vs. required: **{pa['sample_size_ratio_vs_required']:.1f}x**")
    lines.append("")
    lines.append("## Hypothesis Test")
    lines.append("| Method | Difference (treat-control) | p-value | 95% CI | Reject H0 |")
    lines.append("|---|---|---|---|---|")
    for method_key, label in [("z_test", "Two-proportion z-test"), ("permutation_test", "Permutation test")]:
        r = ht[method_key]
        lines.append(
            f"| {label} | {r['absolute_difference']:+.4%} | {r['p_value']:.4f} "
            f"| [{r['ci_low']:+.4%}, {r['ci_high']:+.4%}] | {r['p_value'] < 0.05} |"
        )
    lines.append("")
    lines.append(f"- Effect size (Cohen's h): {ht['cohens_h']:.4f}")
    lines.append(f"- Both tests agree: **{ht['agree_on_reject_null']}**")
    lines.append("")

    if "segmentation" in results:
        seg = results["segmentation"]
        lines.append("## Segmentation")
        lines.append(
            f"- Randomization balanced across segments: **{seg['randomization_balanced']}** "
            f"(χ²={seg['randomization_balance_chi2']:.3f}, p={seg['randomization_balance_p_value']:.4f})"
        )
        sc = seg["simpsons_check"]
        lines.append(f"- Pooled direction: {sc['pooled_direction']}")
        lines.append(f"- Directional reversal flagged: {sc['reversal_detected']}")
        lines.append("- Per-segment significance (directional flags are NOT the same as significant differences):")
        lines.append("")
        segment_n = {}
        for row in seg["conversion_by_segment"]:
            key = str(row[next(k for k in row if k not in ("group", "conversion_rate", "n", "conversions"))])
            segment_n[key] = segment_n.get(key, 0) + row["n"]

        lines.append("| Segment | n | Difference | p-value | Significant |")
        lines.append("|---|---|---|---|---|")
        for t in seg["per_segment_significance_tests"]:
            n = segment_n.get(t["segment"])
            n_str = f"{n:,}" if n is not None else "-"
            lines.append(f"| {t['segment']} | {n_str} | {t['absolute_difference']:+.4%} | {t['p_value']:.4f} | {t['p_value'] < 0.05} |")
        lines.append("")

    if "regression" in results:
        reg = results["regression"]
        lines.append("## Logistic Regression")
        lines.append("**Preferred model (M1: treatment only)**")
        lines.append("")
        lines.append("| Predictor | Odds Ratio | 95% CI | p-value |")
        lines.append("|---|---|---|---|")
        for row in reg["m1_treatment_only"]["odds_ratios"]:
            lines.append(
                f"| {row['predictor']} | {row['odds_ratio']:.4f} "
                f"| [{row['odds_ratio_ci_low']:.4f}, {row['odds_ratio_ci_high']:.4f}] | {row['p_value']:.4f} |"
            )
        lines.append("")
        if "likelihood_ratio_test_m1_vs_m2" in reg:
            lr = reg["likelihood_ratio_test_m1_vs_m2"]
            lines.append(
                f"- Adding segment predictors: LR p={lr['p_value']:.4f} "
                f"({'significant improvement' if lr['significant_improvement'] else 'no significant improvement, prefer simpler model'})"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/generate_report.py` from `scripts/run_experiment.py` output. "
                  "See `notebooks/experiment_analysis.ipynb` for the full narrative analysis and diagnostics.*")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = render(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
