# Dataset Description and Statistics

## Overview

This dataset is designed for vehicle re-identification across two synchronized camera views:

- `before`: front-side/front-view camera
- `after`: rear-side/back-view camera

The same physical vehicle appearing in both views is assigned the same identity using the annotation attribute `id`.

## Conditions

The full dataset is planned to contain four conditions:

| Condition | Annotation Status | Views |
| --- | --- | --- |
| `morning_norain` | Completed | `before`, `after` |
| `morning_rain` | In progress | `before`, `after` |
| `evening_norain` | Completed | `before`, `after` |
| `evening_rain` | In progress | `before`, `after` |

Frame images are stored on the server at:

```text
/mnt/ngan/vehicles
```

Annotation XML files are stored on the server at:

```text
/mnt/ngan/vehicles/annotation
```

The local XML files in this workspace are stored in `annotation/` as working copies for inspection and editing.
Each XML file should be paired with the image folder that has the same base name.

| Annotation file | Server image folder |
| --- | --- |
| `/mnt/ngan/vehicles/annotation/morning_norain_before.xml` | `/mnt/ngan/vehicles/morning_norain_before/` |
| `/mnt/ngan/vehicles/annotation/morning_norain_after.xml` | `/mnt/ngan/vehicles/morning_norain_after/` |
| `/mnt/ngan/vehicles/annotation/evening_norain_before.xml` | `/mnt/ngan/vehicles/evening_norain_before/` |
| `/mnt/ngan/vehicles/annotation/evening_norain_after.xml` | `/mnt/ngan/vehicles/evening_norain_after/` |
| `/mnt/ngan/vehicles/annotation/morning_rain_before.xml` | `/mnt/ngan/vehicles/morning_rain_before/` |
| `/mnt/ngan/vehicles/annotation/morning_rain_after.xml` | `/mnt/ngan/vehicles/morning_rain_after/` |
| `/mnt/ngan/vehicles/annotation/evening_rain_before1.xml` | `/mnt/ngan/vehicles/evening_rain_before1/` |
| `/mnt/ngan/vehicles/annotation/evening_rain_before2.xml` | `/mnt/ngan/vehicles/evening_rain_before2/` |
| `/mnt/ngan/vehicles/annotation/evening_rain_after1.xml` | `/mnt/ngan/vehicles/evening_rain_after1/` |
| `/mnt/ngan/vehicles/annotation/evening_rain_after2.xml` | `/mnt/ngan/vehicles/evening_rain_after2/` |

## Annotation Format

Annotations are stored in CVAT XML format. Each frame contains vehicle bounding boxes:

```xml
<image id="1" name="frame_000001.jpg" width="1080" height="1920">
  <box label="motorbike" xtl="70.08" ytl="247.49" xbr="127.92" ybr="355.39">
    <attribute name="id">2</attribute>
  </box>
</image>
```

Important fields:

| Field | Meaning |
| --- | --- |
| `image.name` | Frame filename |
| `box.label` | Vehicle class |
| `xtl`, `ytl`, `xbr`, `ybr` | Bounding box coordinates |
| `attribute id` | Vehicle identity |

Current vehicle classes:

```text
motorbike, car, truck, Bus
```

For Re-ID, the key identity rule is:

```text
same physical vehicle  -> same id
different vehicle      -> different id
```

## Current Annotation Files

Currently completed local XML files:

```text
morning_norain_before.xml
morning_norain_after.xml
evening_norain_before.xml
evening_norain_after.xml
```

## Current Statistics

| XML file | Frames | Boxes | Unique IDs |
| --- | ---: | ---: | ---: |
| `morning_norain_before.xml` | 3019 | 9448 | 371 |
| `morning_norain_after.xml` | 3305 | 7640 | 343 |
| `evening_norain_before.xml` | 4597 | 10798 | 567 |
| `evening_norain_after.xml` | 5037 | 10631 | 537 |

## Cross-View Identity Coverage

| Condition | Before IDs | After IDs | Shared IDs | Before-only IDs | After-only IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `morning_norain` | 371 | 343 | 343 | 28 | 0 |
| `evening_norain` | 567 | 537 | 537 | 30 | 0 |

This means every vehicle ID in the `after` view currently has a matching ID in the `before` view.

## Removed Label Issues

The following inconsistent IDs were removed from the XML annotation files.
After removal, no shared ID has a different class label between `before` and `after` in the currently completed conditions.

### `morning_norain`

| Removed ID | Previous `before` label/count/frames | Previous `after` label/count/frames |
| ---: | --- | --- |
| 314 | car, 15 boxes, frames 2590-2604 | motorbike, 27 boxes, frames 2843-2869 |
| 256 | car, 42 boxes, frames 2164-2205 | motorbike, 28 boxes, frames 2414-2441 |
| 243 | motorbike, 14 boxes, frames 1999-2012 | car, 18 boxes, frames 2196-2213 |
| 138 | car, 16 boxes, frames 986-1001 | motorbike, 17 boxes, frames 1095-1111 |

### `evening_norain`

| Removed ID | Previous `before` label/count/frames | Previous `after` label/count/frames |
| ---: | --- | --- |
| 320 | car, 18 boxes, frames 2320-2337 | motorbike, 17 boxes, frames 2587-2603 |
| 32 | car, 16 boxes, frames 351-366 | truck, 11 boxes, frames 417-427 |
| 1 | truck, 30 boxes + car, 20 boxes, frames 0-2410 | truck, 47 boxes + motorbike, 12 boxes, frames 171-1579 |

For ID 1 in `evening_norain`, the same ID appeared with multiple labels inside the same view, so it was probably an ID reuse or merge error.

## Planned Benchmark Setup

Recommended Re-ID setup:

```text
query   = after view crops
gallery = before view crops
```

Suggested metrics:

```text
Rank-1, Rank-5, mAP
```

Since the main contribution is the dataset, the paper can evaluate existing Re-ID baselines instead of proposing a new model.

## Current Pipeline Files

The initial data pipeline files are:

```text
configs/dataset.json
scripts/validate_annotations.py
scripts/export_reid_crops.py
```

Validate the completed local XML working copies:

```bash
python scripts/validate_annotations.py --annotation-root annotation
```

Export Re-ID crops on the server:

```bash
python scripts/export_reid_crops.py \
  --config configs/dataset.json \
  --completed-only \
  --output-root /mnt/ngan/vehicles/reid_crops
```
