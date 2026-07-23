# Q2 / IEEE Access Submission Checklist

This checklist targets IEEE Access (Scopus Q2) for a **dataset + method**
paper (dataset + WICV-Net). Live experiment status: `journal_experiment_plan.md`.

## Dataset Completeness

- [x] Four weather/time conditions are annotated.
- [x] Two views are available for every condition.
- [x] Bounding boxes are annotated.
- [x] Vehicle class labels are annotated.
- [x] Cross-view identity IDs are annotated.
- [x] Annotation validation reports no missing IDs.
- [x] Annotation validation reports no invalid boxes.
- [x] Annotation validation reports no cross-view label mismatch.
- [x] Dataset card exists on Hugging Face.
- [x] Code repository exists on GitHub.

## Reproducibility

- [x] Annotation validation script.
- [x] Crop export script.
- [x] Identity-disjoint split script.
- [x] Leakage audit script.
- [x] Baseline training/evaluation scripts.
- [x] Final split CSV files archived or reproducible by seed.
- [x] Final baseline results archived (`results/baselines_full_e100/summary.csv`, all 6 models).
- [ ] Environment/package versions documented.

## Required Paper Tables

All numbers ready in `docs/final_results.md`, mapped to Table 7-11.

- [x] Dataset statistics by condition/view.
- [x] Class distribution.
- [x] Cross-view identity coverage.
- [ ] Comparison with existing datasets. (Table 1 in draft has the structure; verify facts before submission)
- [ ] Comparison with recent invariance/cross-view Re-ID methods (Table 2; DW-ReID/DualDis citations still TODO, see `methods/wicv/README.md`)
- [x] Re-ID baseline table (CE/triplet, multiple backbones) -- done, 6 models.
- [x] Main table: baselines vs. WICV-Net, multi-seed mean +/- std -- done (osnet_x1_0 3-seed, resnet50 + tv_swin_t single-run).
- [x] Component ablation table (ce_only / plain_triplet / no_adv / no_cvpa / no_cvtri / full) -- done, caveat on untuned w_adv documented.
- [x] Loss-weight sensitivity table (w_adv, w_cvpa) -- done.
- [x] Per-condition Re-ID table -- done.
- [x] Cross-condition generalization table (4 protocols) -- done, WICV-Net wins 4/4.
- [x] Re-ranking add-on row -- done.
- [ ] Detection baseline table (optional, not run -- omit section rather than leave TODO).

## Required Figures

- [ ] Dataset sample grid: all four conditions and both views. (`scripts/make_paper_figures.py` -> `figure_01_dataset_overview.jpg`)
- [ ] Camera/view diagram (not auto-generated; draw manually).
- [ ] Annotation examples with bounding boxes and IDs. (`figure_02_annotation_examples.jpg`)
- [ ] Class distribution chart. (`figure_03a_class_distribution.png`)
- [ ] Per-condition/view box counts + after/before ratio. (`figure_05_view_asymmetry.png`, code ready, run on server)
- [ ] Cross-view identity coverage (shared vs. before-only). (`figure_06_crossview_id_coverage.png`, code ready, run on server)
- [ ] WICV-Net architecture/method diagram (not auto-generated; draw manually).
- [ ] Loss-weight sensitivity plot (data in `results/wicv_sensitivity/summary.csv`; not yet plotted).
- [x] Re-ID success/failure examples (per condition) -- done, 8 files in `docs/figures/retrieval/`.
- [ ] Detection examples (optional, not run).

## Benchmark Quality

- [x] Identity-disjoint split design.
- [x] Query/gallery direction defined.
- [x] Leakage audit script available.
- [x] Final audit output included in supplementary or repository.
- [x] Baselines trained with validation-based model selection.
- [x] Same split used for all models (baseline and WICV-Net).
- [x] Per-condition evaluation performed.
- [x] Ablation result is internally consistent (resolved: tuned full beats no_adv; untuned full does not -- both reported honestly).
- [x] Result validated on >= 2 backbone families (CNN + transformer: osnet, resnet50, tv_swin_t).
- [x] Result validated across >= 3 random seeds (osnet_x1_0 only; resnet50/tv_swin_t are single-run -- state this scope limit).

## Writing Quality

- [ ] Clear problem statement.
- [ ] Clear dataset novelty.
- [ ] Dataset collection details complete.
- [ ] Annotation protocol complete.
- [ ] Limitations discussed honestly.
- [ ] License/access terms stated.
- [ ] Citation format added.
- [ ] English grammar checked.
- [ ] Target journal template applied.

## Must-Fill Missing Details

These details should be filled before submission:

1. Dataset name.
2. Exact collection location description, without exposing sensitive information.
3. Camera resolution.
4. Frame rate or frame sampling rate.
5. Duration of each condition.
6. Hardware used for training (GPU model -- record from the server before submission).
7. ~~Final Re-ID results.~~ Done, see `docs/final_results.md`.
8. Final detection results (not run; omit detection section from the paper).
9. License.
10. Citation/BibTeX (2 pending: DW-ReID, DualDis -- info in `methods/wicv/README.md`).

## Recommended Final Claim

Use careful wording:

```text
The dataset supports multiple vehicle perception tasks, including detection, classification, and cross-view re-identification. In this work, we provide benchmark experiments for vehicle re-identification and vehicle detection, together with weather/time-based robustness analysis.
```

Avoid claiming that the dataset solves every task unless benchmark results are provided.
