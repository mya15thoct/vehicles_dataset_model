# Conference Subset Statistics

This file stores the current conference subset statistics for paper writing.

Source:

```text
/mnt/ngan/vehicles/reid_benchmark_conference_50
```

Subset construction:

```text
300 shared cross-view identities per condition
1,200 shared identities in total
70/10/20 train/validation/test split by identity
```

## Identity Split

| Split | Identities |
| --- | ---: |
| Train | 840 |
| Validation | 120 |
| Test | 240 |
| **Total** | **1,200** |

Per condition:

| Condition | Train IDs | Val IDs | Test IDs | Total IDs |
| --- | ---: | ---: | ---: | ---: |
| `morning_norain` | 210 | 30 | 60 | 300 |
| `evening_norain` | 210 | 30 | 60 | 300 |
| `morning_rain` | 210 | 30 | 60 | 300 |
| `evening_rain` | 210 | 30 | 60 | 300 |
| **Total** | **840** | **120** | **240** | **1,200** |

## Image/Crop Counts

| Split | Images/Crops | Identities |
| --- | ---: | ---: |
| Train | 33,791 | 840 |
| `val_query` | 2,644 | 120 |
| `val_gallery` | 2,721 | 120 |
| Validation total | 5,365 | 120 |
| Query | 5,301 | 240 |
| Gallery | 5,018 | 240 |
| Test total | 10,319 | 240 |

## Split Counts By Condition

| Split | `evening_norain` | `evening_rain` | `morning_norain` | `morning_rain` |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,277 | 6,937 | 7,696 | 10,881 |
| `val_query` | 457 | 558 | 544 | 1,085 |
| `val_gallery` | 453 | 732 | 690 | 846 |
| Query | 1,329 | 802 | 1,145 | 2,025 |
| Gallery | 1,034 | 1,285 | 1,122 | 1,577 |
| Validation total | 910 | 1,290 | 1,234 | 1,931 |
| Test total | 2,363 | 2,087 | 2,267 | 3,602 |

## Split Counts By Class

| Split | Bus | Car | Motorbike | Truck |
| --- | ---: | ---: | ---: | ---: |
| Train | 456 | 6,208 | 10,988 | 16,139 |
| `val_query` | 7 | 346 | 709 | 1,582 |
| `val_gallery` | 11 | 329 | 808 | 1,573 |
| Query | 53 | 1,138 | 1,528 | 2,582 |
| Gallery | 49 | 1,070 | 1,796 | 2,103 |
| Validation total | 18 | 675 | 1,517 | 3,155 |
| Test total | 102 | 2,208 | 3,324 | 4,685 |

## Leakage Audit

The audit result is:

```json
"passed": true
```

All checked leakage values are zero:

| Check | Value |
| --- | ---: |
| `identity_overlap_train_val` | 0 |
| `identity_overlap_train_test` | 0 |
| `identity_overlap_val_test` | 0 |
| `crop_overlap_train_val` | 0 |
| `crop_overlap_train_test` | 0 |
| `crop_overlap_val_test` | 0 |
| `crop_overlap_query_gallery` | 0 |
| `crop_overlap_val_query_val_gallery` | 0 |
| `val_identity_missing_gallery` | 0 |
| `val_identity_missing_query` | 0 |
| `test_identity_missing_gallery` | 0 |
| `test_identity_missing_query` | 0 |
| `query_non_after_rows` | 0 |
| `gallery_non_before_rows` | 0 |
| `val_query_non_after_rows` | 0 |
| `val_gallery_non_before_rows` | 0 |

## Paper-Ready Text

```text
For the conference experiments, we construct a condition-balanced subset containing 1,200 shared cross-view vehicle identities, with 300 identities sampled from each weather/time condition. The identities are split into 840 training identities, 120 validation identities, and 240 test identities. This results in 33,791 training crops, 5,365 validation crops, and 10,319 test crops. Query crops are taken from the after view and gallery crops from the before view. An automated audit confirms that there is no identity overlap or crop overlap between training, validation, and test splits.
```
