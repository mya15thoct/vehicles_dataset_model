# Experiment Plan

This file lists the experiments recommended for a dataset paper targeting a Scopus Q2-level submission.

## Main Benchmark: Vehicle Re-Identification

### Task Definition

Given a vehicle crop from the `after` view, retrieve the same vehicle identity from the `before` view.

```text
query   = after-view crops
gallery = before-view crops
```

### Split Protocol

Use identity-disjoint train/validation/test splits.

```text
train.csv       -> before + after crops from train identities
val_query.csv   -> after crops from validation identities
val_gallery.csv -> before crops from validation identities
query.csv       -> after crops from test identities
gallery.csv     -> before crops from test identities
```

No identity should appear in more than one split.

### Metrics

Report:

- Rank-1
- Rank-5
- mAP

### Recommended Baselines

Use existing Re-ID models instead of proposing a new model:

| Model | Role |
| --- | --- |
| `osnet_x1_0` | Strong lightweight Re-ID baseline |
| `osnet_ain_x1_0` | Re-ID baseline with instance normalization variants |
| `osnet_ibn_x1_0` | Re-ID baseline with IBN-style robustness |
| `resnet50` | Standard CNN baseline |
| `resnet101` | Deeper CNN baseline |
| `mobilenetv2_x1_0` | Lightweight mobile baseline |

### Main Tables

Table 1: Overall Re-ID benchmark.

| Model | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| OSNet | TODO | TODO | TODO |
| OSNet-AIN | TODO | TODO | TODO |
| OSNet-IBN | TODO | TODO | TODO |
| ResNet-50 | TODO | TODO | TODO |
| ResNet-101 | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO |

Table 2: Per-condition Re-ID analysis using the best model or all models.

| Test condition | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| `morning_norain` | TODO | TODO | TODO |
| `evening_norain` | TODO | TODO | TODO |
| `morning_rain` | TODO | TODO | TODO |
| `evening_rain` | TODO | TODO | TODO |

## Secondary Benchmark: Vehicle Detection

### Task Definition

Detect and classify vehicles in each frame.

Classes:

```text
bus, car, motorbike, truck
```

### Recommended Baselines

Pick one or two detector baselines to keep the paper focused:

- YOLOv8 or YOLOv11
- Faster R-CNN with ResNet-50-FPN

### Metrics

Report:

- mAP@50
- mAP@50:95
- precision
- recall
- per-class AP

### Main Tables

Detection overall result:

| Model | mAP@50 | mAP@50:95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| YOLO baseline | TODO | TODO | TODO | TODO |
| Faster R-CNN baseline | TODO | TODO | TODO | TODO |

Per-class AP:

| Class | AP@50 | AP@50:95 |
| --- | ---: | ---: |
| bus | TODO | TODO |
| car | TODO | TODO |
| motorbike | TODO | TODO |
| truck | TODO | TODO |

## Robustness Analysis

This is important because the dataset is multi-weather and multi-time.

Recommended analyses:

1. Train on all conditions, test each condition separately.
2. Train no-rain, test rain.
3. Train rain, test no-rain.
4. Train morning, test evening.
5. Train evening, test morning.

Minimum acceptable analysis:

- Overall Re-ID result
- Per-condition Re-ID result
- Overall detection result
- Per-condition detection result if time permits

## Qualitative Figures

Recommended figures:

1. Dataset examples: before/after frames for all four conditions.
2. Annotation examples: bounding boxes and ID labels.
3. Re-ID success cases: query and top gallery matches.
4. Re-ID failure cases: visually similar vehicles, occlusion, rain/night artifacts.
5. Detection examples: per-condition predictions.

## Leakage And Validity Checks

Must report:

- No train/val/test identity overlap.
- No crop overlap between splits.
- No query/gallery crop duplication.
- No label mismatch for shared IDs.
- Every query ID has at least one matching gallery ID.

Use `scripts/audit_reid_splits.py` after building splits.
