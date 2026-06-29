# Paper Documentation Index

This folder contains working notes for preparing a dataset paper.

| File | Purpose |
| --- | --- |
| `data.md` | Current dataset notes and validation statistics |
| `dataset_statistics.md` | Paper-ready statistics tables |
| `paper_brief.md` | Dataset story, contributions, and positioning |
| `experiment_plan.md` | Recommended benchmarks and result tables |
| `paper_template.md` | Scopus-style paper outline and writing template |
| `q2_checklist.md` | Submission-readiness checklist |
| `server_commands.md` | Commands to collect final results from the server |
| `figures/` | Generated paper figures after running `scripts/make_paper_figures.py` |

## Current Priority

1. Export full crops for the latest annotation version.
2. Rebuild identity-disjoint splits.
3. Audit split leakage.
4. Train/evaluate Re-ID baselines.
5. Add detection benchmark.
6. Fill missing result tables in the paper.

## Notes

The exact format depends on the target journal. After choosing a journal, adapt the manuscript to that journal's author guidelines and template.
