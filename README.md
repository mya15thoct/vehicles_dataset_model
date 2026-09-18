# Multi-Weather Traffic Vehicle Re-Identification

## Overview

This repository is the official code release for:

> **WICV-Net: A Cross-View Alignment Framework for Multi-Weather Traffic Vehicle Re-Identification**
> Thi Kim Ngan Tran, Trong Hoang Dung Le, Huy Kien Phan, Trong-Hop Do
> University of Information Technology, Ho Chi Minh City, Vietnam; Vietnam National University, Ho Chi Minh City, Vietnam — submitted to *IEEE Access*.

It makes two contributions:

1. **VN2V-Weather**, to our knowledge the first public cross-view vehicle Re-ID
   benchmark to combine paired elevated front/rear identity supervision,
   explicit weather and time-of-day annotations, and heterogeneous mixed
   traffic (cars, motorbikes, trucks, buses). It contains 42,254 frames and
   100,952 annotated vehicle boxes across four weather/time conditions.
2. **WICV-Net**, a backbone-agnostic training framework for the resulting
   extreme opposite-view retrieval protocol (rear-view query → front-view
   gallery). On an identity-disjoint, leakage-audited split, WICV-Net
   improves mAP from 71.18% to 83.34% (± 0.21 over three seeds) over a
   seed-matched control, gains up to +17.7 mAP over same-backbone CE/triplet
   baselines, reaches 84.87% mAP with a Swin-T backbone, and improves all
   four unseen cross-condition transfer protocols by 3.6–9.1 mAP without
   using weather/time labels during training.

The dataset is collected from two paired camera views of the same traffic
scene, both mounted on a pedestrian footbridge over a national highway so the
two views share an elevated, roughly top-down geometry. The two streams cover
the same traffic window but are **not hardware-synchronized**; cross-view
correspondence between the two streams is established through manual
identity annotation rather than timestamp alignment:

- `before`: front-view stream
- `after`: rear-view stream

Each vehicle is annotated with a bounding box, vehicle class, and identity
ID. The same physical vehicle appearing in both views is assigned the same
identity ID, enabling cross-view vehicle re-identification.

## Method

WICV-Net keeps the same backbones as the baselines (OSNet, ResNet, Swin-T,
... from Torchreid/TorchVision) and changes only the training framework, on
top of a standard BNNeck identity head:

```text
                          +--> BNNeck --> ID classifier ---> L_id (CE + label smoothing)
crop --> backbone --> f --+--> cross-view prototype memory ---> L_cvpa
                          +--> cross-view batch-hard triplet -> L_cv-tri
```

- **Cross-View Prototype Alignment (CVPA)** maintains an EMA memory of one
  L2-normalized prototype per `(identity, view)` pair and pulls each
  embedding toward its identity's opposite-view prototype via an InfoNCE
  loss, giving a stable, dataset-wide cross-view anchor. The three-seed
  component ablation attributes most of WICV-Net's gain to CVPA.
- **Cross-View Batch-Hard Triplet (CV-Tri)** restricts the hardest positive
  for each anchor to the *opposite* camera view, aligning training with the
  actual retrieval protocol instead of letting the loss be satisfied by easy
  same-view positives. On top of CVPA its marginal contribution is small
  (within run-to-run noise in the three-seed ablation); it is retained
  because it does not hurt and appears to reduce variance.
- A view-balanced PK sampler splits each identity's instances evenly between
  the two views whenever both are present in the batch, so CV-Tri and CVPA
  receive cross-view positives for every identity that has them. When only
  one view of an identity is present, that identity's loss term falls back
  to standard same-view mining instead.

A factorized condition-adversarial term (FCA) was also investigated (a
gradient-reversed classifier discouraging the identity feature from encoding
weather/time information); it gave no additional gain over cross-view
alignment alone and is **not** part of the final objective. See
`methods/wicv/README.md` for the full design discussion and ablations. The
final reported objective is adversarial-free (identity CE + CVPA + CV-Tri) —
see [Main Results](#main-results).

## Repository Structure

```text
annotation/                 # Annotation copies used by this repository
configs/dataset.json        # Dataset configuration
reid_common/                # Shared CSV/path/eval/plotting primitives (pip install -e .)
docs/                       # Dataset notes and statistics (see docs/README.md)
scripts/
  validate_annotations.py   # Validate XML labels and cross-view identity consistency
  export_reid_crops.py      # Export vehicle crops from frame images and XML
  build_reid_split.py       # Simple query/gallery split, no train/val — not the benchmark split
  build_train_test_split.py # Build the identity-disjoint train/val/test split used for all reported results
  build_size_matched_split.py # Filter an existing split to one common min-box-size threshold (see Reproducing Results)
  audit_reid_splits.py      # Check split CSV files for data leakage
baselines/
  osnet/                    # Pretrained OSNet sanity-check evaluator
  torchreid/                # Fine-tuning/evaluation baselines
methods/
  wicv/                     # Proposed WICV-Net training framework (see methods/wicv/README.md)
```

## Dataset

The full image dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/vehicle-research/multi-weather_traffic_data
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

Annotations are provided in CVAT XML format. Each annotated vehicle box
contains a class label and an identity ID:

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

### Annotation Statistics

Current validated annotation statistics:

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

Cross-view identity consistency (identities are scoped to a condition — a
vehicle observed in `morning_rain` and one observed in `evening_rain` never
share an identity — so matching is defined within a condition, not across
conditions; the 2,307 cross-view-matchable identities used for the benchmark
split are the sum of the four "Shared IDs" values below):

| Condition | Shared IDs | Before-only IDs | After-only IDs | Label mismatches |
| --- | ---: | ---: | ---: | ---: |
| `morning_norain` | 571 | 57 | 0 | 0 |
| `evening_norain` | 537 | 30 | 0 | 0 |
| `morning_rain` | 618 | 51 | 0 | 0 |
| `evening_rain` | 581 | 54 | 0 | 0 |

Vehicle class distribution (naturally imbalanced — trucks and motorbikes
dominate, unlike car-centric benchmarks such as VeRi-776 and VehicleID):

| Class | Boxes | Share |
| --- | ---: | ---: |
| `truck` | 48,801 | 48.34% |
| `motorbike` | 30,360 | 30.07% |
| `car` | 19,439 | 19.26% |
| `bus` | 2,352 | 2.33% |

## Environment Setup

Create a Python environment with Python 3.10+.

Install PyTorch and TorchVision for your CUDA or CPU setup by following the
official PyTorch instructions:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependencies and the shared `reid_common` package:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torchreid; print('torchreid ok')"
python -c "from reid_common.reid_eval import CropDataset; print('reid_common ok')"
```

## Dataset Setup (Hugging Face)

Install the Hugging Face CLI if needed:

```bash
pip install -U huggingface_hub
```

Download the dataset:

```bash
hf download vehicle-research/multi-weather_traffic_data \
  --repo-type dataset \
  --local-dir /path/to/multi-weather_traffic_data
```

The dataset is public — no access request is required.

In the examples below, set these paths for your machine:

```bash
DATA_ROOT=/path/to/multi-weather_traffic_data
CROP_ROOT=/path/to/reid_crops
SPLIT_ROOT=/path/to/reid_benchmark_identity
RESULT_ROOT=results/baselines_final
```

## Reproducing Results

All paper results were produced on a single NVIDIA RTX A6000 GPU (48 GB)
with CUDA 12.6.

### 1. Data pipeline

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

A clean split should report `"passed": true`.

### 2. Train / evaluate the CE-triplet baselines

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

Default baseline models: `osnet_x1_0`, `osnet_ain_x1_0`, `osnet_ibn_x1_0`,
`resnet50`, `resnet101`, `mobilenetv2_x1_0`. See `baselines/README.md` and
`baselines/torchreid/README.md` for single-model commands.

### 3. Train / evaluate WICV-Net (fixed 60-epoch schedule)

The paper uses two distinct training schedules: an **early-stopping**
schedule (validate every 5 epochs, stop after 4 non-improving validations —
used only for the loss-weight sweep and is what `--patience 4` below would
give you) and the **fixed** schedule (run the full 60 epochs, no early
stopping, then evaluate the best-validation-mAP checkpoint). Every reported
WICV-Net result uses the fixed schedule, i.e. `--patience 0`. Both schedules
select the checkpoint the same way (best validation mAP); they differ only in
whether training can stop early. `--no-adv` is required to get the
adversarial-free final objective — omitting it trains with the
condition-adversarial term (FCA) at its default weight, which is the
configuration the paper's ablation rejected, not the one reported in
[Main Results](#main-results):

```bash
python -u methods/wicv/train.py \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --model-name osnet_x1_0 \
  --output-root results/wicv/osnet_x1_0_final \
  --epochs 60 --eval-every 5 --patience 0 \
  --no-adv --w-tri 1.0 --w-cvpa 0.5 \
  --seed 42

python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_final/model_best.pth \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv"
```

`evaluate.py` reports overall mAP/Rank-1/Rank-5 **and** a per-condition
breakdown for every condition present in the query CSV automatically (no
separate script needed) — see the `per_condition` section of the written
`eval.json`.

For the three-seed OSNet result in [Main Results](#main-results)
(83.34 ± 0.21 mAP), repeat with `--seed 43` and `--seed 44` into separate
`--output-root` directories, or use `methods/wicv/run_seeds.py` (see
`methods/wicv/README.md`) to run and aggregate all three seeds in one
command. See `methods/wicv/README.md` also for the ablation/sensitivity
runners (`run_ablation.py`, `run_sensitivity.py`) and the investigated-but-
unpublished v2 structural-module variants (`--use-cvt`, `--use-can`).

#### Size-matched evaluation

Native per-condition results are confounded by condition-dependent minimum
annotated vehicle size (small, dark vehicles are harder to annotate, so the
effective size cutoff differs by condition). To reproduce the paper's
size-matched protocol, which applies one common threshold across conditions:

```bash
python -u scripts/build_size_matched_split.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --split-root "$SPLIT_ROOT" \
  --output-root "$SPLIT_ROOT/size_matched" \
  --min-size auto

python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_final/model_best.pth \
  --query "$SPLIT_ROOT/size_matched/query.csv" \
  --gallery "$SPLIT_ROOT/size_matched/gallery.csv"
```

`--min-size auto` uses the strictest per-stream minimum observed in the
split as the common threshold; the paper reports this as ≈143.6 px
equal-area edge length on its data, covering 88.8% of queries.

### 4. Cross-condition generalization

```bash
CROSS_CONDITION_ROOT=/path/to/reid_cross_condition

python -u scripts/build_cross_condition_splits.py \
  --split-root "$SPLIT_ROOT" \
  --output-root "$CROSS_CONDITION_ROOT"

python -u methods/wicv/run_cross_condition.py \
  --protocol-root "$CROSS_CONDITION_ROOT" \
  --model-name osnet_x1_0
```

### 5. k-Reciprocal re-ranking (optional add-on)

```bash
python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_final/model_best.pth \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --rerank
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

Recommended metrics: Rank-1, Rank-5, mAP.

## Main Results

**Table 7 — WICV-Net vs. CE/triplet baseline, per backbone (%).** WICV-Net
rows use the final adversarial-free objective under a fixed 60-epoch
schedule; the OSNet row is mean ± std over 3 seeds. CE/triplet baselines use
the strong Re-ID baseline recipe of Luo et al., "Bag of Tricks and a Strong
Baseline for Deep Person Re-Identification", CVPRW 2019, as implemented in
Torchreid, with early stopping.

| Backbone | Method | Rank-1 | Rank-5 | mAP |
| --- | --- | ---: | ---: | ---: |
| OSNet | CE/triplet baseline | 85.23 | 91.03 | 79.15 |
| OSNet | WICV-Net (3 seeds) | 89.69 ± 0.49 | 94.71 ± 0.39 | 83.34 ± 0.21 |
| OSNet-AIN | CE/triplet baseline | 87.16 | 92.03 | 80.53 |
| OSNet-AIN | WICV-Net | 89.62 | 95.36 | 81.60 |
| ResNet-50 | CE/triplet baseline | 75.35 | 84.30 | 66.67 |
| ResNet-50 | WICV-Net | 90.07 | 95.04 | 84.37 |
| Swin-T | WICV-Net | 90.65 | 94.35 | 84.87 |

**Table 12 — Cross-condition generalization on OSNet (mAP, %).** All three
configurations are trained within the same codebase on the same
condition-restricted splits; `ce_only` is a seed-matched internal control
(identity-CE only, all proposed components disabled) trained in this same
codebase, not the separately-implemented Torchreid CE/triplet baseline used
in the table above.

| Protocol (train → test) | `ce_only` (control) | WICV-Net (proposed) | + FCA (w_adv=0.1) |
| --- | ---: | ---: | ---: |
| no-rain → rain | 33.64 | **42.74** | 42.69 |
| rain → no-rain | 30.81 | **39.60** | 39.43 |
| morning → evening | 9.51 | **13.12** | 12.03 |
| evening → morning | 23.77 | **31.21** | 30.48 |

**Table 13 — k-reciprocal re-ranking add-on (OSNet, %).** Applied to the
seed-42 checkpoint of the final objective.

| Setting | Rank-1 | Rank-5 | mAP |
| --- | ---: | ---: | ---: |
| WICV-Net (seed 42) | 89.22 | 94.28 | 83.13 |
| + re-ranking | 91.24 | 94.98 | 87.02 |

## Citation

<!-- year/volume/pages/doi are placeholders until IEEE Access assigns them on publication. -->

```bibtex
@article{tran2026wicvnet,
  title   = {{WICV-Net}: A Cross-View Alignment Framework for Multi-Weather Traffic Vehicle Re-Identification},
  author  = {Tran, Thi Kim Ngan and Le, Trong Hoang Dung and Phan, Huy Kien and Do, Trong-Hop},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## License

### Code License

The source code in this repository is released under the MIT License — see
[`LICENSE`](LICENSE).

### Dataset License

The VN2V-Weather dataset distributed on Hugging Face is released under
**CC BY-NC 4.0** (non-commercial use, with attribution). See the
[Hugging Face dataset page](https://huggingface.co/datasets/vehicle-research/multi-weather_traffic_data)
for the authoritative license terms.
