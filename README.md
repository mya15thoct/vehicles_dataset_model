# Multi-Weather Traffic Vehicle Re-Identification

This repository contains the data-processing and baseline evaluation code for a multi-weather vehicle re-identification dataset. The dataset is collected from two synchronized camera views of the same traffic scene:

- `before`: front-side/front-view stream
- `after`: rear-side/back-view stream

Each vehicle is annotated with a bounding box, vehicle class, and identity ID. The same physical vehicle appearing in both views is assigned the same identity ID, enabling cross-view vehicle re-identification.

## Dataset

The full image dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/mya15thoct/multi-weather_traffic_data
```

The dataset contains four conditions:

| Condition | Views | Description |
| --- | --- | --- |
| `morning_norain` | `before`, `after` | Morning, no rain |
| `morning_rain` | `before`, `after` | Morning, rain |
| `evening_norain` | `before`, `after` | Evening/night, no rain |
| `evening_rain` | `before`, `after` | Evening/night, rain |

Expected Hugging Face dataset layout:

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

The annotation format is CVAT XML. Each frame contains vehicle boxes:

```xml
<image id="1" name="frame_000001.jpg" width="1080" height="1920">
  <box label="motorbike" xtl="70.08" ytl="247.49" xbr="127.92" ybr="355.39">
    <attribute name="id">2</attribute>
  </box>
</image>
```

The Re-ID identity rule is:

```text
same physical vehicle  -> same id
different vehicle      -> different id
```

## Repository Structure

```text
annotation/                 # Working annotation copies used by this repo
configs/dataset.json        # Dataset path/configuration
docs/data.md                # Dataset notes and current statistics
scripts/
  validate_annotations.py   # Validate XML labels and cross-view identity consistency
  export_reid_crops.py      # Export vehicle crops from frame images and XML
  build_reid_split.py       # Build query/gallery split for zero-shot checks
  build_train_test_split.py # Build identity-disjoint train/val/test split
  audit_reid_splits.py      # Check for data leakage in split CSV files
baselines/
  osnet/                    # Pretrained OSNet sanity-check evaluator
  torchreid/                # Fine-tuning/evaluation baselines
```

## Setup

Create or activate a Python environment with PyTorch, TorchVision, Pillow, and Torchreid:

```bash
conda activate recognition
pip install pillow gdown
pip install torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
pip install torchreid
```

Adjust the PyTorch/TorchVision install command to match your CUDA and PyTorch version if needed.

## Download Data

Example using Hugging Face CLI:

```bash
huggingface-cli download mya15thoct/multi-weather_traffic_data \
  --repo-type dataset \
  --local-dir /mnt/ngan/vehicles/multi-weather_traffic_data
```

Or clone with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/mya15thoct/multi-weather_traffic_data \
  /mnt/ngan/vehicles/multi-weather_traffic_data
```

Set paths consistently:

```text
Image root      : /mnt/ngan/vehicles/multi-weather_traffic_data
Annotation root : /mnt/ngan/vehicles/multi-weather_traffic_data/annotation
```

## Data Pipeline

Validate annotations:

```bash
python scripts/validate_annotations.py \
  --annotation-root /mnt/ngan/vehicles/multi-weather_traffic_data/annotation \
  --include-in-progress
```

Export vehicle crops:

```bash
nohup python -u scripts/export_reid_crops.py \
  --config configs/dataset.json \
  --image-root /mnt/ngan/vehicles/multi-weather_traffic_data \
  --annotation-root /mnt/ngan/vehicles/multi-weather_traffic_data/annotation \
  --completed-only \
  --output-root /mnt/ngan/vehicles/reid_crops \
  > export_reid_crops.log 2>&1 &
```

Build identity-disjoint train/validation/test splits:

```bash
nohup python -u scripts/build_train_test_split.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark_identity \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42 \
  > build_train_test_split.log 2>&1 &
```

Audit the split for leakage:

```bash
python scripts/audit_reid_splits.py \
  --train /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_identity/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_identity/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --output results/split_audit.json
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

Metrics:

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
nohup python -u baselines/torchreid/run_all.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_identity/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_identity/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --results-root results/baselines_final \
  --epochs 100 \
  --eval-every 5 \
  --patience 3 \
  --min-delta 0.001 \
  --batch-size 64 \
  > run_all_baselines.log 2>&1 &
```

Outputs:

```text
results/baselines_final/summary.csv
results/baselines_final/summary.json
results/baselines_final/<model_name>/model_best.pth
results/baselines_final/<model_name>/best_val.json
results/baselines_final/<model_name>/eval.json
```

## Current Validated Subset

The initial validated subset contains the two no-rain conditions:

| Split | Images | Identities |
| --- | ---: | ---: |
| Train | 25,754 | 616 |
| Validation | 3,785 | 88 |
| Test | 7,837 | 176 |

No identity or crop overlap was found between train, validation, and test in the audited split.

## Notes for Paper Experiments

Recommended result tables:

1. Overall benchmark across all conditions.
2. Per-condition benchmark using the same trained model.
3. Dataset statistics by condition and class.
4. Split statistics for train/validation/test.
5. Qualitative success and failure examples.

For per-condition results, train the model once on the full training split, then evaluate the selected checkpoint on condition-specific query/gallery subsets.

## Citation

If you use this dataset or code, please cite the associated paper when available.

```bibtex
@misc{multi_weather_traffic_reid,
  title        = {Multi-Weather Traffic Vehicle Re-Identification Dataset},
  author       = {Tran, Ngan and Contributors},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/mya15thoct/multi-weather_traffic_data}}
}
```

## License

Please check the Hugging Face dataset page for the dataset license and usage terms.
