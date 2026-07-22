# Paper Writing Template

IEEE Access follows an IMRaD-style structure with a strong emphasis on
reproducibility and thorough experimental validation. The content below is
the outline for a **dataset + method** paper: the dataset (multi-weather
two-view traffic Re-ID) and WICV-Net (the proposed training framework).

## Suggested Title

WICV-Net: A Weather-Invariant Cross-View Framework for Multi-Weather Traffic Vehicle Re-Identification

## Abstract Template

```text
Vehicle re-identification in real-world traffic scenes remains challenging due to viewpoint changes, lighting variation, adverse weather, and visually similar vehicles. This paper makes two contributions. First, we introduce [DATASET NAME], a multi-weather traffic vehicle dataset collected from two synchronized camera views, containing 42,254 frames and 100,952 annotated vehicle bounding boxes across four weather/time conditions: morning no-rain, evening no-rain, morning rain, and evening rain, with cross-view identity annotations. Second, we propose WICV-Net, a backbone-agnostic training framework that addresses two properties standard Re-ID training ignores on this benchmark: the extreme front/rear camera gap and the unused weather/time condition labels. WICV-Net combines cross-view batch-hard triplet mining, cross-view prototype alignment, and factorized time/weather condition-adversarial learning. Across [N] backbones, WICV-Net improves Rank-1/mAP by [MAIN RESULT] over CE/triplet baselines, with the largest gains under the most challenging conditions (evening/rain). We validate the contribution of each component with a full ablation, a loss-weight sensitivity analysis, a cross-condition generalization protocol, and multi-seed statistical testing. The dataset and code are publicly available at [LINKS].
```

## 1. Introduction

Include:

- Importance of vehicle Re-ID in intelligent transportation systems.
- Difficulty of matching vehicles across different views, compounded by weather/time variation.
- Gap 1: limited public datasets with synchronized two-view traffic data and weather/time conditions.
- Gap 2: standard Re-ID training does not exploit the two-camera structure or free condition labels available in such data (cite the novelty positioning in `../methods/wicv/README.md`: CLIP-based/prompt methods need VLMs; disentanglement methods don't target this two-view setting; adversarial invariance exists for person Re-ID modality/illumination but not combined with cross-view alignment for vehicles).
- Your two contributions: the dataset, and WICV-Net.

Suggested contribution paragraph:

```text
The main contributions of this work are as follows:
1. We introduce a multi-weather two-view traffic vehicle dataset with bounding boxes, vehicle classes, and cross-view identity annotations.
2. We identify that standard Re-ID training ignores the extreme cross-view gap and the free weather/time labels available in this setting.
3. We propose WICV-Net: cross-view batch-hard triplet mining, cross-view prototype alignment (EMA memory), and factorized condition-adversarial learning, applicable to any existing Re-ID backbone.
4. We validate WICV-Net with a full component ablation, loss-weight sensitivity analysis, cross-condition generalization protocol, and multi-seed statistical testing across multiple backbones (CNN and transformer).
```

## 2. Related Work

Recommended subsections:

- Vehicle re-identification datasets
- Vehicle Re-ID methods (CNN and transformer backbones)
- Weather/illumination-invariant and cross-view/viewpoint-aware Re-ID (cite CLIP-ReID, CLIP-driven view-aware prompt learning, DW-ReID, DualDis, IBNT-Net -- see `../methods/wicv/README.md` novelty section for the full list and how each differs from WICV-Net)
- Adversarial/disentangled representation learning for invariance

Dataset comparison table to prepare:

| Dataset | Year | Vehicle Re-ID | Detection boxes | Weather variation | Night/evening | Cross-view IDs | Public |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| VeRi-776 | TODO | Yes | TODO | TODO | TODO | Yes | Yes |
| VehicleID | TODO | Yes | TODO | TODO | TODO | TODO | Yes |
| CityFlow-ReID | TODO | Yes | TODO | TODO | TODO | Yes | Yes |
| UA-DETRAC | TODO | No/Reformulated | Yes | TODO | TODO | No | Yes |
| Ours | 2026 | Yes | Yes | Yes | Yes | Yes | Yes |

Verify each external dataset fact before final submission.

Method comparison table to prepare (positions WICV-Net against recent invariance/cross-view approaches):

| Method | Venue/Year | Cross-view aware | Weather/condition invariant | Needs VLM/prompts | Backbone-agnostic |
| --- | --- | --- | --- | --- | --- |
| CLIP-ReID | TODO | No | No | Yes | No (CLIP-tied) |
| DW-ReID | 2026 | No | Yes (person) | Yes | No |
| DualDis | 2026 | Partial (component-level) | No | No | TODO |
| WICV-Net (ours) | 2026 | Yes | Yes | No | Yes |

## 3. Dataset

Recommended subsections:

### 3.1 Data Collection

Describe:

- Camera setup: two synchronized views, `before` and `after`.
- Location type: traffic scene.
- Weather/time conditions: morning/evening and rain/no-rain.
- Frame extraction.
- Dataset storage and release.

Missing details to fill:

- FPS or frame sampling interval.
- Approximate duration per condition.
- Camera resolution.
- Whether cameras are fixed.
- Whether timestamps are synchronized manually or by recording setup.

### 3.2 Annotation Protocol

Describe:

- CVAT annotation.
- Bounding boxes around visible vehicles.
- Vehicle classes: bus, car, motorbike, truck.
- Identity assignment across `before` and `after`.
- Quality control: removed inconsistent IDs, validated no missing IDs, no invalid boxes, no label mismatches.

### 3.3 Dataset Statistics

Use tables from `docs/dataset_statistics.md`.

### 3.4 Supported Tasks

Describe:

- Re-ID / cross-view matching.
- Detection.
- Classification.
- Robustness analysis.

## 4. Benchmark Protocol

### 4.1 Re-ID Protocol

Describe:

- Identity-disjoint train/validation/test split.
- Query = `after`.
- Gallery = `before`.
- Metrics: Rank-1, Rank-5, mAP.
- Split leakage audit.

### 4.2 Detection Protocol (optional section)

Describe:

- Convert CVAT XML to YOLO or COCO.
- Train/val/test split.
- Metrics: mAP@50, mAP@50:95, precision, recall.
- Per-class AP.

## 5. Proposed Method: WICV-Net

Source: `../methods/wicv/README.md` has the full technical description; keep
this section aligned with it as the implementation evolves.

### 5.1 Motivation

State the two properties standard training ignores: the extreme cross-view
gap (query=after, gallery=before) and the unused free time/weather labels.

### 5.2 Cross-View Batch-Hard Triplet (CV-Tri)

Describe the cross-view positive mining rule and the single-view fallback.

### 5.3 Cross-View Prototype Alignment (CVPA)

Describe the EMA per-(identity, view) prototype memory and the InfoNCE
opposite-view alignment loss.

### 5.4 Factorized Condition-Adversarial Learning (FCA)

Describe the time/weather factorization, the gradient-reversal heads, and the
GRL warmup schedule. State the tuned loss weight found in the sensitivity
study and be upfront that naive weighting degraded performance (this is a
legitimate finding to report, not something to hide).

### 5.5 Overall Objective And Training Details

Give the total loss formula and the cross-view balanced PK sampler.

## 6. Experiments

### 6.1 Implementation Details

Fill after final runs:

- GPU model.
- Batch size (PK sampling: P identities x K instances).
- Epochs, early stopping patience.
- Optimizer, learning rate, weight decay, schedule.
- Image size, augmentation.
- Backbones evaluated (CNN + transformer).

### 6.2 Main Results: Baselines vs. WICV-Net

Insert the main comparison table: CE-only / plain-triplet baseline vs.
WICV-Net (full), per backbone, with multi-seed mean +/- std. Source:
`results/baselines_full_e100/summary.csv` + `results/wicv/*/eval.json` +
`results/wicv_seeds/*_seeds_summary.json`.

### 6.3 Component Ablation

Insert the six-variant ablation table (ce_only / plain_triplet / no_adv /
no_cvpa / no_cvtri / full). Source: `results/wicv_ablation/summary.csv`.
Discuss which component contributes most (report honestly, including any
component that requires careful weighting).

### 6.4 Loss-Weight Sensitivity

Insert the sweep table/figure for w_adv, w_cvpa (and w_tri/temperature if
run). Source: `results/wicv_sensitivity/summary.csv`. This section directly
supports the claim in 5.4 about tuned vs. naive adversarial weighting.

### 6.5 Per-Condition (Weather/Time) Analysis

Insert per-condition results for the main model. Source: `eval.json`
`per_condition` field. Highlight where WICV-Net's gain over baseline is
largest (expected: evening/rain, the hardest domain gap).

### 6.6 Cross-Condition Generalization

Insert the four-protocol table (train no-rain/test rain, and its reverse;
train morning/test evening, and its reverse). Source:
`results/wicv_cross_condition/summary.csv`. This is the strongest evidence
for the "weather-invariant" claim in the title.

### 6.7 Re-Ranking (optional add-on row)

Report the +k-reciprocal re-ranking numbers as an additional row, not the
headline result. Source: `eval_rerank.json`.

### 6.8 Detection Results (optional section)

Insert detection benchmark only if run; otherwise omit this subsection
entirely rather than leaving a TODO in a submitted manuscript.

### 6.9 Qualitative Analysis

Show success and failure retrieval cases per condition. Source:
`docs/figures/retrieval/*.jpg` (generated by
`methods/wicv/make_retrieval_figures.py`).

## 7. Discussion

Discuss:

- Why CVPA (opposite-view prototypes) matters most for this cross-view setup.
- Why naive adversarial weighting hurt, and what that implies for applying condition-adversarial learning elsewhere.
- Rain and evening difficulty; where WICV-Net closes the gap and where it doesn't.
- Class imbalance, visual similarity among vehicles, occlusion and partial views.
- Dataset and method limitations (e.g., two fixed camera views only; factorization assumes binary time/weather).

## 8. Conclusion

Summarize:

- Dataset release and annotation scale.
- WICV-Net's contribution and validated gains.
- Supported tasks.
- Future work (e.g., more camera views, continuous weather severity instead of binary factors).

## Ethical And Privacy Notes

Include a short note if needed:

- The dataset focuses on vehicles in public traffic scenes.
- No person identity annotation is provided.
- License and access conditions are specified on the dataset page.

## Citation

Placeholder until the paper is accepted:

```bibtex
@article{yourkey2026multiweather,
  title={Multi-Weather Traffic Vehicle Re-Identification Dataset for Cross-View Matching},
  author={TODO},
  journal={TODO},
  year={2026}
}
```
