---
name: generate-experiment-report
description: Turn a JSON results file from run-ab-experiment into a business-readable markdown report with an explicit, rule-based ship/don't-ship/run-longer recommendation. Use after running the experiment pipeline, or when the user has an existing results JSON and wants a written summary/recommendation instead of raw numbers.
---

# Generate Experiment Report

Renders `scripts/run_experiment.py`'s JSON output into a markdown report via
`scripts/generate_report.py`. Keeps computation and presentation separate: the
same results JSON can be re-rendered without recomputing any statistics, and
the recommendation logic lives in one auditable place instead of being
reasoned about freshly (and potentially inconsistently) each time.

## When to use this

- Right after `run-ab-experiment` has produced a results JSON.
- The user hands you an existing `experiment_results.json` (from this
  project or a prior run) and asks "so what should we do?" or wants a
  written summary rather than raw numbers.

## How to run it

```bash
.venv/Scripts/python.exe scripts/generate_report.py \
    --results reports/experiment_results.json \
    --out reports/experiment_report.md
```

## Recommendation logic (what `recommend()` in the script actually checks)

The verdict is rule-based, not vibes-based — read `scripts/generate_report.py::recommend()`
before describing *why* a verdict was reached, rather than guessing:

- **INCONCLUSIVE** — the z-test and permutation test disagree on significance.
  This should never be papered over; it means a distributional assumption is
  likely violated and needs investigating before anyone acts on the result.
- **SHIP / DO NOT SHIP** — a statistically significant difference was found
  *and* the experiment was adequately powered for the specified MDE. Direction
  of the significant effect determines which of the two.
- **DO NOT SHIP** (null-result case) — no significant difference, but the
  sample size met or exceeded what the power analysis required. This is a
  confident "no meaningful effect," not a shrug.
- **RUN LONGER** — no significant difference, and the sample size fell short
  of what the power analysis required. The experiment cannot yet distinguish
  "no effect" from "not enough data."

## After running

Report the verdict and the one-line rationale from the top of the generated
markdown directly to the user — don't re-derive a different-sounding
recommendation from the same numbers. If the user pushes back on the
verdict, that's a signal to open `scripts/generate_report.py` and check
whether their disagreement points to a real edge case in `recommend()`
worth fixing, rather than just restating the number.
