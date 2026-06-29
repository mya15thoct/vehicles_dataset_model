# Re-ID Split Statistics

This file records the final identity-disjoint Re-ID split built from the full crop manifest.

Source files:

```text
Manifest: /mnt/ngan/vehicles/reid_crops_full/manifest.csv
Split root: /mnt/ngan/vehicles/reid_benchmark_identity_full
Audit: /mnt/ngan/vehicles/reid_benchmark_identity_full/audit.json
```

## Identity Split

The split uses only identities shared between `before` and `after` views.

| Condition | Shared IDs | Train IDs | Val IDs | Test IDs |
| --- | ---: | ---: | ---: | ---: |
| `evening_norain` | 537 | 376 | 54 | 107 |
| `evening_rain` | 581 | 407 | 58 | 116 |
| `morning_norain` | 571 | 400 | 57 | 114 |
| `morning_rain` | 618 | 433 | 62 | 123 |
| **Total** | **2,307** | **1,616** | **231** | **460** |

## Split Image Counts

| Split | Images | Identities |
| --- | ---: | ---: |
| `train` | 66,905 | 1,616 |
| `val_query` | 4,888 | 231 |
| `val_gallery` | 5,446 | 231 |
| `query` | 9,813 | 460 |
| `gallery` | 9,477 | 460 |
| **Validation total** | **10,334** | **231** |
| **Test total** | **19,290** | **460** |

## Split Counts By Condition

| Split | `evening_norain` | `evening_rain` | `morning_norain` | `morning_rain` |
| --- | ---: | ---: | ---: | ---: |
| `train` | 14,484 | 13,869 | 15,068 | 23,484 |
| `val_query` | 1,156 | 1,225 | 970 | 1,537 |
| `val_gallery` | 924 | 1,234 | 1,396 | 1,892 |
| `query` | 2,370 | 1,786 | 2,588 | 3,069 |
| `gallery` | 2,030 | 2,022 | 2,370 | 3,055 |

## Split Counts By Class

| Split | Bus | Car | Motorbike | Truck |
| --- | ---: | ---: | ---: | ---: |
| `train` | 1,005 | 12,527 | 21,043 | 32,330 |
| `val_query` | 105 | 1,052 | 1,377 | 2,354 |
| `val_gallery` | 113 | 1,000 | 1,551 | 2,782 |
| `query` | 270 | 2,142 | 2,684 | 4,717 |
| `gallery` | 427 | 1,773 | 3,173 | 4,104 |

## Leakage Audit

The final audit passed:

```json
"passed": true
```

All checked leakage indicators are zero:

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

## Paper Text

The Re-ID benchmark uses identity-disjoint training, validation, and test splits. We first retain identities that appear in both camera views, resulting in 2,307 shared identities. These identities are split into 1,616 training identities, 231 validation identities, and 460 test identities. Query images are sampled from the `after` view, while gallery images are sampled from the `before` view. The final audit confirms no identity overlap or crop overlap between splits, and every query identity has at least one matching gallery identity.
