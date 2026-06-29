# Reviewer Positioning Notes

This file records how to frame the conference paper without weakening the future journal dataset paper.

## What The Conference Paper Is

The conference paper is:

```text
An empirical study of cross-view vehicle Re-ID under weather and lighting variations.
```

It contributes:

- problem formulation
- leakage-audited evaluation protocol
- baseline comparison
- weather/time analysis

## What The Conference Paper Is Not

The conference paper is not:

- a public dataset release paper
- a new architecture paper
- a comprehensive benchmark paper
- the final full dataset paper

## How To Mention The Data

Use:

```text
Our experiments use a two-view multi-weather traffic dataset collected and annotated in a real traffic monitoring scenario.
```

Do not use:

```text
private dataset
```

```text
reserved for journal
```

```text
full public dataset release
```

## If Reviewer Asks About Reproducibility

Safe response:

```text
The paper provides the evaluation protocol, split statistics, leakage audit criteria, implementation details, and representative qualitative examples. The current conference version focuses on the empirical evaluation protocol and baseline analysis.
```

If sample data is provided:

```text
A small sample subset is provided to illustrate the data format and annotation protocol.
```

## How To Preserve Journal Novelty

Conference:

- 1,200-ID condition-balanced subset
- Re-ID baselines
- protocol/audit
- limited qualitative analysis

Journal:

- full dataset release
- full 2,307 shared identities / 100,952 boxes
- detection benchmark
- classification benchmark if useful
- full condition-wise/domain-shift experiments
- dataset comparison table
- more qualitative examples
- formal dataset card and citation

## Claims To Avoid

Avoid:

- "large-scale public dataset"
- "we release a dataset"
- "comprehensive dataset benchmark"
- "first dataset"
- "state-of-the-art"

Use:

- "practical cross-view setting"
- "condition-balanced subset"
- "identity-disjoint protocol"
- "leakage-audited evaluation"
- "representative baseline comparison"
- "weather/time analysis"
