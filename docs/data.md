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
/mnt/ngan/vehicles/multi-weather_traffic_data
```

Annotation XML files are stored in this code repository at:

```text
annotation
```

Each XML file should be paired with the image folder that has the same base name.

| Annotation file | Server image folder |
| --- | --- |
| `annotation/morning_norain_before.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/morning_norain_before/` |
| `annotation/morning_norain_after.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/morning_norain_after/` |
| `annotation/evening_norain_before.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_norain_before/` |
| `annotation/evening_norain_after.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_norain_after/` |
| `annotation/morning_rain_before.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/morning_rain_before/` |
| `annotation/morning_rain_after.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/morning_rain_after/` |
| `annotation/evening_rain_before1.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_rain_before1/` |
| `annotation/evening_rain_before2.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_rain_before2/` |
| `annotation/evening_rain_after1.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_rain_after1/` |
| `annotation/evening_rain_after2.xml` | `/mnt/ngan/vehicles/multi-weather_traffic_data/evening_rain_after2/` |

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
scripts/build_reid_split.py
scripts/build_train_test_split.py
```

Validate the completed local XML working copies:

```bash
python scripts/validate_annotations.py
```

Export Re-ID crops on the server:

```bash
python scripts/export_reid_crops.py \
  --config configs/dataset.json \
  --completed-only \
  --output-root /mnt/ngan/vehicles/reid_crops
```

Build query/gallery CSV files:

```bash
python scripts/build_reid_split.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark
```

Build identity-disjoint train/test split for the main paper benchmark:

```bash
nohup python -u scripts/build_train_test_split.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark_identity \
  --train-ratio 0.7 \
  --seed 42 \
  > build_train_test_split.log 2>&1 &
```

Run the first pretrained OSNet sanity-check baseline:

```bash
nohup python -u baselines/osnet/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark/gallery.csv \
  --output results/osnet_pretrained.json \
  > osnet_eval.log 2>&1 &
```

Train/fine-tune OSNet for the main benchmark:

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --model-name osnet_x1_0 \
  --output-dir results/osnet_finetuned \
  --epochs 20 \
  --batch-size 64 \
  > osnet_train.log 2>&1 &
```

Evaluate fine-tuned OSNet:

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --model-name osnet_x1_0 \
  --weights results/osnet_finetuned/model_last.pth \
  --output results/osnet_finetuned/eval.json \
  > osnet_eval_finetuned.log 2>&1 &
```

Train/fine-tune ResNet50:

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --model-name resnet50 \
  --output-dir results/resnet50_finetuned \
  --epochs 20 \
  --batch-size 64 \
  > resnet50_train.log 2>&1 &
```

Evaluate fine-tuned ResNet50:

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --model-name resnet50 \
  --weights results/resnet50_finetuned/model_last.pth \
  --output results/resnet50_finetuned/eval.json \
  > resnet50_eval_finetuned.log 2>&1 &
```

Run three Torchreid baselines and aggregate results:

```bash
nohup python -u baselines/torchreid/run_all.py \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --results-root results/baselines \
  --epochs 20 \
  --batch-size 64 \
  > run_all_baselines.log 2>&1 &
```
