# Conference Experiment Results

This file records the current conference-subset results for the short paper.

## Evaluation Setting

- Subset: condition-balanced conference subset with 1,200 cross-view identities.
- Split: 70/10/20 by identity.
- Test protocol: after-view crops are used as queries and before-view crops are used as galleries.
- Test size: 5,301 query crops and 5,018 gallery crops.
- Metrics: Rank-1, Rank-5, and mAP.

## Overall Baseline Results

Completed models:

| Model | Query | Gallery | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: |
| OSNet-AIN | 5,301 | 5,018 | 85.80 | 89.85 | 81.66 |
| OSNet | 5,301 | 5,018 | 86.36 | 90.85 | 81.46 |
| OSNet-IBN | 5,301 | 5,018 | 84.14 | 88.98 | 78.68 |
| ResNet-101 | 5,301 | 5,018 | 75.38 | 85.89 | 66.12 |
| ResNet-50 | 5,301 | 5,018 | 70.36 | 82.49 | 62.26 |
| MobileNetV2 | 5,301 | 5,018 | 66.38 | 76.34 | 53.01 |

Current best model by mAP:

```text
OSNet-AIN: mAP = 81.66%
```

Current best model by Rank-1:

```text
OSNet: Rank-1 = 86.36%
```

For the condition-wise and class-wise analysis, use OSNet-AIN because it has the highest current mAP.

## Condition-Wise Results

Model: OSNet-AIN.

| Test condition | Query | Gallery | IDs | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Morning no-rain | 1,145 | 1,122 | 60 | 91.53 | 94.50 | 90.54 |
| Evening no-rain | 1,329 | 1,034 | 60 | 71.71 | 79.01 | 70.51 |
| Morning rain | 2,025 | 1,577 | 60 | 95.41 | 97.14 | 91.08 |
| Evening rain | 802 | 1,285 | 60 | 80.92 | 86.91 | 74.55 |

Paper-ready observations:

- Morning rain gives the strongest result among the four conditions.
- Evening no-rain is the hardest condition, suggesting that illumination change is a major challenge.
- Evening rain remains lower than morning conditions, showing that nighttime appearance change and wet-road reflections affect ranking quality.

## Class-Wise Results

Model: OSNet-AIN.

| Class | Query | Gallery | IDs | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bus | 53 | 49 | 2 | 100.00 | 100.00 | 100.00 |
| Car | 1,138 | 1,070 | 39 | 95.96 | 97.01 | 91.57 |
| Motorbike | 1,528 | 1,796 | 140 | 74.48 | 81.87 | 71.54 |
| Truck | 2,582 | 2,103 | 59 | 88.46 | 91.87 | 84.45 |

Paper-ready observations:

- Bus performance should not be over-interpreted because only 2 identities appear in the test split.
- Motorbike is the most difficult major class, despite having many samples. This supports the discussion that mixed traffic with many visually similar motorcycles is challenging for cross-view Re-ID.
- Car and truck are easier than motorbike under this protocol.

## Raw Commands To Recheck

Overall results:

```bash
for m in osnet_x1_0 osnet_ain_x1_0 osnet_ibn_x1_0 resnet50 resnet101 mobilenetv2_x1_0; do
  echo "== $m =="
  if [ -f results/conference_50_e100/$m/eval.json ]; then
    cat results/conference_50_e100/$m/eval.json | grep -E '"num_query_images"|"num_gallery_images"|"rank1"|"rank5"|"mAP"'
  else
    echo "missing eval.json"
  fi
done
```

Breakdown results:

```bash
cat results/conference_50_breakdowns/breakdown_summary.csv
```

## Status

- Overall model comparison: 6/6 models completed.
- Condition-wise analysis: completed for OSNet-AIN.
- Class-wise analysis: completed for OSNet-AIN.
- Pending models: none.
