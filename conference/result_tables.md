# Conference Result Tables

Fill this file after `results/conference_50_e100/summary.csv` is available.

## Overall Re-ID Baseline Results

Metrics:

- Rank-1
- Rank-5
- mAP

| Model | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| OSNet | TODO | TODO | TODO |
| OSNet-AIN | TODO | TODO | TODO |
| OSNet-IBN | TODO | TODO | TODO |
| ResNet-50 | TODO | TODO | TODO |
| ResNet-101 | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO |

Suggested caption:

```text
Overall Re-ID performance on the condition-balanced conference subset. The best result in each metric is highlighted in bold.
```

## Validation Results

Use `best_val.json` from each model folder.

| Model | Best Epoch | Val Rank-1 | Val Rank-5 | Val mAP |
| --- | ---: | ---: | ---: | ---: |
| OSNet | TODO | TODO | TODO | TODO |
| OSNet-AIN | TODO | TODO | TODO | TODO |
| OSNet-IBN | TODO | TODO | TODO | TODO |
| ResNet-50 | TODO | TODO | TODO | TODO |
| ResNet-101 | TODO | TODO | TODO | TODO |
| MobileNetV2 | TODO | TODO | TODO | TODO |

## Condition-Wise Results

If time permits, evaluate the best checkpoint on condition-specific query/gallery subsets.

| Test Condition | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| `morning_norain` | TODO | TODO | TODO |
| `evening_norain` | TODO | TODO | TODO |
| `morning_rain` | TODO | TODO | TODO |
| `evening_rain` | TODO | TODO | TODO |

Suggested caption:

```text
Condition-wise retrieval performance of the best-performing model. Results show the effect of weather and time-of-day variation on cross-view matching.
```

## Paper Wording Templates

Overall result:

```text
Table X compares the evaluated Re-ID baselines on the condition-balanced conference subset. Among the tested models, [MODEL] achieves the best overall performance with [RANK-1]% Rank-1 and [MAP]% mAP.
```

Model comparison:

```text
The OSNet-based models generally provide strong retrieval performance, while lightweight architectures such as MobileNetV2 offer a lower-complexity alternative with reduced accuracy.
```

Condition-wise analysis:

```text
The condition-wise results indicate that rain and evening illumination affect cross-view matching performance. These conditions introduce reflections, lower contrast, and appearance degradation, which make vehicle retrieval more challenging.
```
