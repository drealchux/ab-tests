---
name: run-ab-experiment
description: Run the statistically rigorous A/B test pipeline (data cleaning, power analysis, two-proportion z-test + permutation test, segmentation diagnostics, logistic regression) on any two-arm conversion dataset and produce a structured JSON results file. Use when the user has experiment/conversion data (a CSV with a group column and a binary outcome column) and wants it analyzed correctly, or wants to re-run this project's pipeline on new data.
---

# Run A/B Experiment

Runs the full analysis pipeline implemented in `src/ab_testing/` against a
two-arm conversion dataset, via `scripts/run_experiment.py`. This is the
same pipeline documented and validated in `notebooks/experiment_analysis.ipynb`,
exposed as a reusable, scriptable step so it can run against new data without
copy-pasting notebook cells.

## When to use this

- The user provides (or points to) a CSV with a treatment/control column and
  a binary outcome column and asks for an A/B test / experiment analysis.
- The user wants to re-run this project's methodology on a *different*
  dataset (new experiment, new time window, a colleague's export).
- The user asks specifically to "run the pipeline," "regenerate results,"
  or "check if this experiment is well-powered."

Do **not** use this to answer general statistics questions with no data
attached — that doesn't need the pipeline.

## Required input shape

A CSV with at minimum:
- a group/arm column (default name `group`, values like `control`/`treatment`)
- a binary outcome column (default name `converted`, values 0/1)

Optionally, a second CSV mapping an id column to a segment (default: `country`)
for the segmentation diagnostics and the country-augmented regression model.

If the user's column names differ from the defaults, pass the matching flags
(see below) rather than renaming their data.

## How to run it

```bash
.venv/Scripts/python.exe scripts/run_experiment.py \
    --ab-data <path-to-experiment-csv> \
    --countries <path-to-segment-csv>   \  # optional
    --segment-col country               \  # optional, default "country"
    --group-col group                   \  # optional
    --outcome-col converted             \  # optional
    --control-label control             \  # optional
    --treatment-label treatment         \  # optional
    --mde 0.01                          \  # business-meaningful absolute effect size for the power check
    --alpha 0.05 --power 0.80           \  # optional, these are the defaults
    --out reports/experiment_results.json
```

On the project's own bundled data (`data/raw/ab_data.csv` + `data/raw/countries.csv`),
just run it with no optional flags beyond `--out`.

## What it does, step by step

1. Loads and cleans the data (`ab_testing.data_prep`): drops assignment/exposure
   mismatches and duplicate ids, reporting exact counts — never silently.
2. Runs a pre-registered power analysis (`ab_testing.power_analysis`): computes
   the minimum detectable effect the actual sample size supports at 80% power,
   and compares it against the sample size required for the `--mde` the user
   specified. **This determines whether a null result later means anything.**
3. Runs both a two-proportion z-test and a permutation test
   (`ab_testing.hypothesis_test`) and checks that they agree — disagreement
   is itself flagged as a result worth investigating, not silently resolved.
4. If a segment file was given, runs segmentation diagnostics
   (`ab_testing.segmentation`): checks randomization balance across segments
   (chi-square independence), flags directional Simpson's-paradox-style
   reversals, and runs a per-segment significance test so a directional flag
   is never confused with an actual significant reversal.
5. Fits logistic regression models (`ab_testing.regression`) with correctly
   signed odds ratios and confidence intervals, and a likelihood-ratio test
   for whether adding segment predictors actually improves the model.

## After running

The output JSON is meant to be read by a human or piped into
`generate-experiment-report` (see that skill) for a business-readable
markdown write-up — don't hand-summarize the raw JSON when that skill exists
to do it consistently.

If `--mde` wasn't specified by the user, ask what a business-meaningful lift
would be before running, rather than silently defaulting — the power
analysis is only useful if the MDE reflects something the business actually
cares about. If they don't know, state the default (1 percentage point) and
proceed rather than blocking.
