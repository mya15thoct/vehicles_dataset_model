# Multi-Weather Traffic Vehicle Re-Identification

This repository contains data-processing and baseline evaluation code for a multi-weather traffic vehicle re-identification dataset. The dataset is collected from two synchronized camera views of the same traffic scene:

- `before`: front-side/front-view stream
- `after`: rear-side/back-view stream

Each vehicle is annotated with a bounding box, vehicle class, and identity ID. The same physical vehicle appearing in both views is assigned the same identity ID, enabling cross-view vehicle re-identification.

## Dataset

The full image dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/mya15thoct/multi-weather_traffic_data
```

The dataset contains four traffic conditions:

| Condition | Views | Description |
| --- | --- | --- |
| `morning_norain` | `before`, `after` | Morning traffic without rain |
| `morning_rain` | `before`, `after` | Morning traffic with rain |
| `evening_norain` | `before`, `after` | Evening/night traffic without rain |
| `evening_rain` | `before`, `after` | Evening/night traffic with rain |

Expected dataset layout after downloading from Hugging Face:

```text
multi-weather_traffic_data/
  annotation/
    morning_norain_before.xml
    morning_norain_after.xml
    morning_rain_before.xml
    morning_rain_after.xml
    evening_norain_before.xml
    evening_norain_after.xml
    evening_rain_before.xml
    evening_rain_after.xml
  morning_norain_before/
  morning_norain_after/
  morning_rain_before/
  morning_rain_after/
  evening_norain_before/
  evening_norain_after/
  evening_rain_before/
  evening_rain_after/
```

Annotations are provided in CVAT XML format. Each annotated vehicle box contains a class label and an identity ID:

```xml
<image id="1" name="frame_000001.jpg" width="1080" height="1920">
  <box label="motorbike" xtl="70.08" ytl="247.49" xbr="127.92" ybr="355.39">
    <attribute name="id">2</attribute>
  </box>
</image>
```

The identity rule is:

```text
same physical vehicle  -> same id
different vehicle      -> different id
```

## Annotation Statistics

Current validated annotation statistics:

| Condition | View | Boxes | IDs |
| --- | --- | ---: | ---: |
| `morning_norain` | `before` | 12,784 | 628 |
| `morning_norain` | `after` | 10,785 | 571 |
| `evening_norain` | `before` | 10,798 | 567 |
| `evening_norain` | `after` | 10,631 | 537 |
| `morning_rain` | `before` | 18,445 | 669 |
| `morning_rain` | `after` | 16,074 | 618 |
| `evening_rain` | `before` | 12,807 | 635 |
| `evening_rain` | `after` | 8,628 | 581 |
| **Total** |  | **100,952** |  |

Cross-view identity consistency:

| Condition | Shared IDs | Before-only IDs | After-only IDs | Label mismatches |
| --- | ---: | ---: | ---: | ---: |
| `morning_norain` | 571 | 57 | 0 | 0 |
| `evening_norain` | 537 | 30 | 0 | 0 |
| `morning_rain` | 618 | 51 | 0 | 0 |
| `evening_rain` | 581 | 54 | 0 | 0 |

Vehicle classes:

```text
bus, car, motorbike, truck
```

## Repository Structure

```text
annotation/                 # Annotation copies used by this repository
configs/dataset.json        # Dataset configuration
docs/data.md                # Dataset notes and statistics
scripts/
  validate_annotations.py   # Validate XML labels and cross-view identity consistency
  export_reid_crops.py      # Export vehicle crops from frame images and XML
  build_reid_split.py       # Build query/gallery split for zero-shot checks
  build_train_test_split.py # Build identity-disjoint train/val/test split
  audit_reid_splits.py      # Check split CSV files for data leakage
baselines/
  osnet/                    # Pretrained OSNet sanity-check evaluator
  torchreid/                # Fine-tuning/evaluation baselines
methods/
  wicv/                     # Proposed WICV-Net training framework (see methods/wicv/README.md)
```

## Setup

Create a Python environment with Python 3.10+.

Install PyTorch and TorchVision for your CUDA or CPU setup by following the official PyTorch instructions:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies:

```bash
pip install pillow gdown torchreid huggingface_hub
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torchreid; print('torchreid ok')"
```

## Download Data

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Download the dataset:

```bash
hf download mya15thoct/multi-weather_traffic_data \
  --repo-type dataset \
  --local-dir /path/to/multi-weather_traffic_data
```

If the dataset is gated, request access on the Hugging Face dataset page and log in before downloading:

```bash
hf auth login
```

In the examples below, set these paths for your machine:

```bash
DATA_ROOT=/path/to/multi-weather_traffic_data
CROP_ROOT=/path/to/reid_crops
SPLIT_ROOT=/path/to/reid_benchmark_identity
RESULT_ROOT=results/baselines_final
```

## Data Pipeline

Validate annotations:

```bash
python scripts/validate_annotations.py \
  --config configs/dataset.json \
  --annotation-root "$DATA_ROOT/annotation"
```

Export vehicle crops:

```bash
python -u scripts/export_reid_crops.py \
  --config configs/dataset.json \
  --image-root "$DATA_ROOT" \
  --annotation-root "$DATA_ROOT/annotation" \
  --completed-only \
  --output-root "$CROP_ROOT"
```

Build identity-disjoint train/validation/test splits:

```bash
python -u scripts/build_train_test_split.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --output-root "$SPLIT_ROOT" \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
```

Audit the split for leakage:

```bash
python scripts/audit_reid_splits.py \
  --train "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --output "$SPLIT_ROOT/audit.json"
```

A clean split should report:

```json
"passed": true
```

## Evaluation Protocol

The main benchmark uses identity-disjoint splits:

```text
train.csv       -> before + after crops from train identities
val_query.csv   -> after crops from validation identities
val_gallery.csv -> before crops from validation identities
query.csv       -> after crops from test identities
gallery.csv     -> before crops from test identities
```

The retrieval task is:

```text
Given a vehicle crop from the after view, retrieve the same vehicle from the before-view gallery.
```

Recommended metrics:

```text
Rank-1, Rank-5, mAP
```

## Baselines

The Torchreid baseline runner trains multiple models, selects the best checkpoint using validation mAP, and evaluates the selected checkpoint on the test split.

Default baseline models:

```text
osnet_x1_0
osnet_ain_x1_0
osnet_ibn_x1_0
resnet50
resnet101
mobilenetv2_x1_0
```

Run all default baselines:

```bash
python -u baselines/torchreid/run_all.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --results-root "$RESULT_ROOT" \
  --epochs 100 \
  --eval-every 5 \
  --patience 4 \
  --batch-size 64 \
  --num-workers 4 \
  --no-auto-split
```

Outputs:

```text
results/baselines_final/summary.csv
results/baselines_final/summary.json
results/baselines_final/<model_name>/model_best.pth
results/baselines_final/<model_name>/best_val.json
results/baselines_final/<model_name>/eval.json
```

## Notes for Paper Experiments

Recommended result tables:

1. Overall benchmark across all conditions.
2. Per-condition benchmark using the same trained model.
3. Dataset statistics by condition and class.
4. Split statistics for train/validation/test.
5. Qualitative success and failure examples.

For per-condition results, train the model once on the full training split, then evaluate the selected checkpoint on condition-specific query/gallery subsets.

## License

Please check the Hugging Face dataset page for the dataset license and usage terms.
