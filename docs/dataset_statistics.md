# Dataset Statistics For Paper

Use this file to copy tables into the paper. Numbers are based on the current validated annotations in `annotation/`.

## Overall Size

| Item | Count |
| --- | ---: |
| Conditions | 4 |
| Views per condition | 2 |
| Annotated frame streams | 8 |
| Frames | 42,254 |
| Vehicle boxes / crops | 100,952 |
| Condition-level identity pairs | 2,499 |
| Vehicle classes | 4 |

## Per-Condition And Per-View Statistics

| Condition | View | Frames | Boxes | IDs |
| --- | --- | ---: | ---: | ---: |
| `morning_norain` | `before` | 4,923 | 12,784 | 628 |
| `morning_norain` | `after` | 5,207 | 10,785 | 571 |
| `evening_norain` | `before` | 4,597 | 10,798 | 567 |
| `evening_norain` | `after` | 5,037 | 10,631 | 537 |
| `morning_rain` | `before` | 6,671 | 18,445 | 669 |
| `morning_rain` | `after` | 6,700 | 16,074 | 618 |
| `evening_rain` | `before` | 4,561 | 12,807 | 635 |
| `evening_rain` | `after` | 4,558 | 8,628 | 581 |
| **Total** |  | **42,254** | **100,952** |  |

## Class Distribution

| Class | Boxes | Percentage |
| --- | ---: | ---: |
| `truck` | 48,801 | 48.34% |
| `motorbike` | 30,360 | 30.07% |
| `car` | 19,439 | 19.26% |
| `bus` | 2,352 | 2.33% |
| **Total** | **100,952** | **100.00%** |

## Condition Distribution

| Condition | Boxes | Percentage |
| --- | ---: | ---: |
| `morning_norain` | 23,569 | 23.35% |
| `evening_norain` | 21,429 | 21.23% |
| `morning_rain` | 34,519 | 34.19% |
| `evening_rain` | 21,435 | 21.23% |
| **Total** | **100,952** | **100.00%** |

## View Distribution

| View | Boxes | Percentage |
| --- | ---: | ---: |
| `before` | 54,834 | 54.32% |
| `after` | 46,118 | 45.68% |
| **Total** | **100,952** | **100.00%** |

## Cross-View Identity Coverage

| Condition | Before IDs | After IDs | Shared IDs | Before-only IDs | After-only IDs | Label mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `morning_norain` | 628 | 571 | 571 | 57 | 0 | 0 |
| `evening_norain` | 567 | 537 | 537 | 30 | 0 | 0 |
| `morning_rain` | 669 | 618 | 618 | 51 | 0 | 0 |
| `evening_rain` | 635 | 581 | 581 | 54 | 0 | 0 |

## Suggested Text For Paper

The dataset contains 42,254 frames and 100,952 annotated vehicle bounding boxes across four weather/time conditions and two synchronized camera views. Each vehicle annotation includes a bounding box, a vehicle class, and an identity ID. The identity IDs are consistent across the two views, enabling cross-view vehicle re-identification. The dataset contains four vehicle categories: bus, car, motorbike, and truck. The class distribution is naturally imbalanced, with trucks and motorbikes appearing more frequently than buses, reflecting real-world traffic composition at the collection site.

## Notes

- `after_only_ids` is zero for every condition, so every query identity can be matched in the `before` gallery.
- `before_only_ids` are kept in the dataset but should not be used as valid Re-ID query identities.
- The class imbalance should be explicitly reported for detection/classification experiments.
