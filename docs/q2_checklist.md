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
- [ ] Final baseline results archived.
- [ ] Environment/package versions documented.

## Required Paper Tables

- [x] Dataset statistics by condition/view.
- [x] Class distribution.
- [x] Cross-view identity coverage.
- [ ] Comparison with existing datasets.
- [ ] Comparison with recent invariance/cross-view Re-ID methods (positioning table).
- [ ] Re-ID baseline table (CE/triplet, multiple backbones).
- [ ] Main table: baselines vs. WICV-Net, multi-seed mean +/- std.
- [ ] Component ablation table (ce_only / plain_triplet / no_adv / no_cvpa / no_cvtri / full).
- [ ] Loss-weight sensitivity table (w_adv, w_cvpa).
- [ ] Per-condition Re-ID table.
- [ ] Cross-condition generalization table (4 protocols).
- [ ] Re-ranking add-on row.
- [ ] Detection baseline table (optional, only if run).

## Required Figures

- [ ] Dataset sample grid: all four conditions and both views.
- [ ] Camera/view diagram.
- [ ] Annotation examples with bounding boxes and IDs.
- [ ] Class distribution chart.
- [ ] WICV-Net architecture/method diagram.
- [ ] Loss-weight sensitivity plot.
- [ ] Re-ID success/failure examples (per condition).
- [ ] Detection examples (optional).

## Benchmark Quality

- [x] Identity-disjoint split design.
- [x] Query/gallery direction defined.
- [x] Leakage audit script available.
- [x] Final audit output included in supplementary or repository.
- [ ] Baselines trained with validation-based model selection.
- [ ] Same split used for all models (baseline and WICV-Net).
- [ ] Per-condition evaluation performed.
- [ ] Ablation result is internally consistent (full model matches or beats every single-component-removed variant; if not, the sensitivity study explains why and the paper reports it honestly).
- [ ] Result validated on >= 2 backbone families (CNN + transformer).
- [ ] Result validated across >= 3 random seeds.

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
6. Hardware used for training.
7. Final Re-ID results.
8. Final detection results.
9. License.
10. Citation/BibTeX.

## Recommended Final Claim

Use careful wording:

```text
The dataset supports multiple vehicle perception tasks, including detection, classification, and cross-view re-identification. In this work, we provide benchmark experiments for vehicle re-identification and vehicle detection, together with weather/time-based robustness analysis.
```

Avoid claiming that the dataset solves every task unless benchmark results are provided.
