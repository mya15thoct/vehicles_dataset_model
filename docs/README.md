# Paper Documentation Index

This folder contains working notes for preparing the journal paper (dataset +
WICV-Net method, target: IEEE Access).

| File | Purpose |
| --- | --- |
| `data.md` | Current dataset notes and validation statistics |
| `dataset_statistics.md` | Paper-ready statistics tables |
| `reid_split_statistics.md` | Final Re-ID train/val/test split and leakage audit statistics |
| `paper_brief.md` | Dataset + method story, contributions, and positioning |
| `paper_template.md` | IEEE Access-style paper outline and writing template (includes the WICV-Net method section) |
| `q2_checklist.md` | Submission-readiness checklist |
| `journal_experiment_plan.md` | Prioritized experiment playbook: every paper table mapped to its command, live status |
| `server_commands.md` | Commands to collect dataset/annotation-side results from the server |
| `writer_prompt.md` | Short handoff prompt for drafting the manuscript once results are in |
| `figures/` | Generated dataset figures after running `scripts/make_paper_figures.py` |
| `../methods/wicv/README.md` | WICV-Net method description and novelty positioning (source of truth for the method section) |
| `../conference/` | Separate conference submission — not part of this journal paper |

## Current Priority

Experiments are running on the server; see `journal_experiment_plan.md` for
live status and the exact commands per remaining table. Do not start writing
result-dependent sections (6.2 onward in `paper_template.md`) until the
corresponding `results/*/summary.csv` exists.

1. Wait for the running experiment suite (sensitivity, baselines, seeds,
   cross-condition, transformer backbone, re-ranking, figures).
2. As each table's source file lands, fill it into `paper_template.md`.
3. Sections 1-5 (Introduction, Related Work, Dataset, Benchmark Protocol,
   Method) do not depend on results and can be drafted now.
4. Once all tables are in, write Discussion and Conclusion last.
