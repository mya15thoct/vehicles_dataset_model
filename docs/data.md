# Dataset Notes

This file summarizes the current validated state of the dataset used by this repository.

## Dataset Location

Public dataset:

```text
https://huggingface.co/datasets/mya15thoct/multi-weather_traffic_data
```

Local server example:

```text
/mnt/ngan/vehicles/multi-weather_traffic_data
```

Annotation copies used by the code repository:

```text
annotation/
```

## Conditions

| Condition | Views | Status |
| --- | --- | --- |
| `morning_norain` | `before`, `after` | Completed |
| `evening_norain` | `before`, `after` | Completed |
| `morning_rain` | `before`, `after` | Completed |
| `evening_rain` | `before`, `after` | Completed |

## Annotation Format

Annotations are stored in CVAT XML format. Each vehicle box contains:

- `box.label`: vehicle class
- `xtl`, `ytl`, `xbr`, `ybr`: bounding box coordinates
- `attribute name="id"`: vehicle identity ID

Example:

```xml
<image id="1" name="frame_000001.jpg" width="1080" height="1920">
  <box label="motorbike" xtl="70.08" ytl="247.49" xbr="127.92" ybr="355.39">
    <attribute name="id">2</attribute>
  </box>
</image>
```

Identity rule:

```text
same physical vehicle  -> same id
different vehicle      -> different id
```

Vehicle classes:

```text
bus, car, motorbike, truck
```

## Current Validated Statistics

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

The dataset is class-imbalanced, especially for `bus`. This should be reported transparently in the paper and considered when interpreting detection/classification results.

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

Every identity in the `after` view has a corresponding identity in the `before` view. Some vehicles appear only in the `before` view; these are kept for detection/classification and excluded from query-gallery matching.

## Validation Status

Current validation checks:

```text
missing_id = 0
invalid_boxes = 0
label_mismatch = 0
after_only_ids = 0
```

Known inconsistent IDs found during annotation cleaning were removed before final validation.
