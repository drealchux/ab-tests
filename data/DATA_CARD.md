---
name: data-card
description: Provenance, structure, and known limitations of the raw experiment data used in this project.
---

# Data Card: `ab_data.csv` + `countries.csv`

## What this is

Two CSV files simulating a landing-page conversion experiment for an unnamed e-commerce site:

- **`ab_data.csv`** (294,478 rows): one row per user session — `user_id`, `timestamp`, `group` (`control`/`treatment`), `landing_page` (`old_page`/`new_page`), `converted` (0/1).
- **`countries.csv`** (290,584 rows): `user_id` → `country` (`US`, `UK`, `CA`), joined in for the segmentation analysis.

## Provenance — what is and isn't verified

This is a well-known, widely-mirrored pedagogical dataset originating from **Udacity's "Analyze A/B Test Results" project** (Data Analyst Nanodegree, A/B Testing course, circa 2016–2017). It appears identically across dozens of public GitHub repositories and Kaggle uploads, all tracing back to the same course material.

**No real company or production system is identified as the source**, and no primary disclosure (paper, engineering blog post, company statement) establishing this as real production telemetry could be located. Treat any "this proves a real product decision" framing of this dataset with skepticism — including in this repo's own earlier version, which did not flag this.

## Evidence this is synthetic, not organic production data

| Observation | Why it's a synthetic-data signature |
|---|---|
| `landing_page` splits 147,239 / 147,239 between `old_page` and `new_page` | A dead-even split like this essentially never occurs in real production traffic; real randomizers produce ratios like 50.1/49.9 with sampling noise, not exact ties |
| `group` splits 147,202 / 147,276 — near-identical pattern | Same signature as above |
| Timestamps span exactly Jan 2–24, 2017 (23 days), no gaps, no anomalies | Consistent with a generated date range rather than logged organic traffic (no holidays, outages, weekday/weekend traffic swings visible) |
| 3,894 duplicate `user_id` rows and a `group`/`landing_page` mismatch subset, cleanly separable | Reads as a deliberately injected data-cleaning exercise for the course, not organic pipeline noise |

## How this project treats it

Given the above, this project:

1. **Does not claim the numeric results generalize to any real business decision.** The value of this repo is the *methodology* — correct power analysis, correct hypothesis testing, correct regression diagnostics — demonstrated end-to-end on a realistic-shaped dataset, not a claim about a real product's conversion rate.
2. **Still analyzes it with full statistical rigor**, because the data-quality issues (duplicates, mismatches) and the analytical traps (underpowering, Simpson's paradox, novelty effects, multicollinearity) are real methodological hazards regardless of whether the rows are synthetic — and demonstrating how to catch them is the point.
3. **Documents every cleaning step with counts** (see `src/ab_testing/data_prep.py::CleaningReport`) so nothing is silently dropped.

## Schema

**`ab_data.csv`**

| column | type | notes |
|---|---|---|
| `user_id` | int | not unique in raw file — 3,894 duplicates |
| `timestamp` | datetime | Jan 2 – Jan 24, 2017 |
| `group` | categorical | `control`, `treatment` |
| `landing_page` | categorical | `old_page`, `new_page` |
| `converted` | binary | outcome variable |

**`countries.csv`**

| column | type | notes |
|---|---|---|
| `user_id` | int | join key to `ab_data.csv` |
| `country` | categorical | `US` (70%), `UK` (25%), `CA` (5%) |
