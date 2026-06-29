# Q2 Submission Checklist

This checklist is designed for a dataset paper targeting a Scopus Q2-level venue.

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
- [ ] Final split CSV files archived or reproducible by seed.
- [ ] Final baseline results archived.
- [ ] Environment/package versions documented.

## Required Paper Tables

- [x] Dataset statistics by condition/view.
- [x] Class distribution.
- [x] Cross-view identity coverage.
- [ ] Comparison with existing datasets.
- [ ] Re-ID baseline table.
- [ ] Per-condition Re-ID table.
- [ ] Detection baseline table.
- [ ] Per-class detection table.
- [ ] Ablation/domain-shift table, if time permits.

## Required Figures

- [ ] Dataset sample grid: all four conditions and both views.
- [ ] Camera/view diagram.
- [ ] Annotation examples with bounding boxes and IDs.
- [ ] Class distribution chart.
- [ ] Re-ID success/failure examples.
- [ ] Detection examples.

## Benchmark Quality

- [x] Identity-disjoint split design.
- [x] Query/gallery direction defined.
- [x] Leakage audit script available.
- [ ] Final audit output included in supplementary or repository.
- [ ] Baselines trained with validation-based model selection.
- [ ] Same split used for all models.
- [ ] Per-condition evaluation performed.

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
