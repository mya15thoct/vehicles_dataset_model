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
| `journal_experiment_plan.md` | Prioritized experiment playbook: every paper table mapped to its command, live status (this is the one source of truth for what to run) |
| `final_results.md` | **All experiments done (2026-07-22).** Every final number mapped directly to paper Table 7-11, ready to copy into the manuscript. |
| `server_commands.md` | Data-pipeline setup only: env, annotation validation, crop export, split build/audit. Training commands live in `journal_experiment_plan.md` instead. |
| `writer_prompt.md` | Short handoff prompt for drafting the manuscript once results are in |
| `figures/` | Generated dataset figures after running `scripts/make_paper_figures.py` |
| `../methods/wicv/README.md` | WICV-Net method description and novelty positioning (source of truth for the method section) |
| `../conference/` | Separate conference submission — not part of this journal paper |

## Current Priority

All experiments finished 2026-07-22 -- see `final_results.md` for every
number, already mapped to the paper's Table 7-11.

1. Fill Tables 7-11 and the abstract using `final_results.md`.
2. Resolve the 2 pending citations (DW-ReID, DualDis) -- info already found,
   see `methods/wicv/README.md`.
3. Verify the camera-mounting description in Section III-A before writing
   it as fact (flagged in `journal_experiment_plan.md`).
4. Run `scripts/make_paper_figures.py` on the server to generate the 2 new
   statistics figures (view asymmetry, cross-view ID coverage).
5. Write Discussion using the 5 points listed at the end of
   `final_results.md`, then Conclusion last.
