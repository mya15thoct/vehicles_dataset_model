# WICV-Net: Weather-Invariant Cross-View Vehicle Re-Identification

WICV-Net is the proposed method for the journal paper. It keeps the same
backbones as the benchmark baselines (OSNet, ResNet, MobileNet from Torchreid)
and changes only the *training framework*, so every improvement is directly
attributable to the proposed objective, not to a bigger architecture.

## Motivation

The benchmark has two properties that generic Re-ID training ignores:

1. **Extreme cross-view gap.** Query crops come from the `after` (rear-side)
   camera and gallery crops from the `before` (front-side) camera, so the model
   must match a vehicle's rear appearance to its front appearance. Standard
   softmax / triplet training treats all positives equally and mostly exploits
   easy same-view pairs.
2. **Free condition labels.** Every crop carries a condition name that factors
   into *time* (morning/evening) and *weather* (norain/rain). These labels are
   available at zero annotation cost but are unused by the baselines.

WICV-Net exploits both with three components on top of a BNNeck identity head:

## Method

```text
                          +--> BNNeck --> ID classifier ----------> L_id (CE + label smoothing)
crop --> backbone --> f --+--> cross-view batch-hard triplet -----> L_cv-tri
                          +--> cross-view prototype memory -------> L_cvpa
                          +--> GRL --> time head (morning/evening)-+
                          +--> GRL --> weather head (norain/rain)--+-> L_adv
```

Total objective:

```text
L = L_id + w_tri * L_cv-tri + w_cvpa * L_cvpa + w_adv * L_adv
```

### 1. Cross-View Batch-Hard Triplet (CV-Tri)

Batch-hard triplet mining where the hardest **positive** for each anchor is
restricted to samples of the same identity from the *opposite* camera view
(falling back to any positive when the identity appears in only one view).
This aligns the training objective with the actual retrieval protocol
(after-view query vs. before-view gallery) instead of letting the loss be
satisfied by trivially easy same-view positives.

### 2. Cross-View Prototype Alignment (CVPA)

An EMA memory stores one L2-normalized prototype per `(identity, view)` pair.
Each embedding is classified with an InfoNCE loss against the prototypes of
the **opposite view**, i.e. a rear-view crop is pulled toward its own
front-view prototype and pushed from all other vehicles' front-view
prototypes. Unlike batch-level losses, the memory provides a stable,
dataset-wide cross-view anchor for every identity in every batch.

### 3. Factorized Condition-Adversarial Learning (FCA)

The condition label is factorized into two binary nuisance factors — time
(morning/evening) and weather (norain/rain) — and two small classifier heads
are attached behind a gradient reversal layer with the standard sigmoid warmup
schedule. The backbone is thus explicitly penalized for encoding time-of-day
or weather information, producing weather-invariant identity features. The
factorized design matches the dataset's 2x2 condition grid and lets the paper
ablate time-invariance and weather-invariance separately if desired.

### Cross-view balanced PK sampling

A PK sampler draws P identities x K instances per batch and splits each
identity's K instances evenly between the two views whenever both exist, so
CV-Tri and CVPA always receive cross-view positives.

## Novelty positioning (checked July 2026)

- CLIP/prompt-based approaches (CLIP-ReID; CLIP-driven view-aware prompt
  learning, AAAI 2025; DW-ReID for weather-degraded person Re-ID) rely on
  large vision-language models and prompts; WICV-Net is a lightweight,
  label-driven framework that works with any Re-ID backbone.
- Disentanglement works for vehicles (e.g. DualDis, WWW 2026) decouple
  component/attribute features but do not use *free scene-level condition
  labels adversarially*, and none target the synchronized front/rear two-view
  setting.
- Viewpoint-aware vehicle Re-ID handles orientation within one camera; our
  cross-view prototype memory explicitly bridges two fixed, opposing views,
  which is the defining challenge of this benchmark.
- To our knowledge no prior vehicle Re-ID method combines (a) cross-view
  positive mining, (b) opposite-view prototype alignment, and (c) factorized
  time/weather adversarial invariance in a single objective.

Contribution claims for the paper:

1. A cross-view-aware objective (CV-Tri + CVPA) that aligns training with the
   two-camera retrieval protocol.
2. Factorized condition-adversarial learning that uses free multi-weather
   labels to learn weather/time-invariant identity features.
3. Consistent gains over six baseline backbones on the multi-weather
   benchmark, with per-condition and cross-condition generalization analysis
   and a full component ablation.

## Files

```text
dataset.py                CSV dataset, condition factorization, cross-view PK sampler
model.py                  Torchreid/torchvision backbone + BNNeck + GRL time/weather heads
losses.py                 CV-Tri loss and CVPA prototype memory
metrics.py                Feature extraction and Rank-1/Rank-5/mAP (same protocol as baselines)
rerank.py                 K-reciprocal re-ranking (Zhong et al., CVPR 2017)
train.py                  Training with validation-mAP model selection and early stopping
evaluate.py               Test evaluation: overall + per-condition (+ --rerank), writes eval.json
run_ablation.py           Trains/evaluates all ablation variants, writes summary.csv
run_sensitivity.py        One-at-a-time loss-weight sweeps (w_adv, w_cvpa, w_tri, temperature)
run_seeds.py              Multi-seed runs with mean +/- std aggregation
run_cross_condition.py    Cross-condition generalization protocols (needs scripts/build_cross_condition_splits.py)
make_retrieval_figures.py Qualitative success/failure retrieval strips per condition
```

Backbones: any Torchreid model name (osnet_x1_0, resnet50, ...) plus
torchvision transformers/CNNs: `tv_swin_t`, `tv_swin_s`, `tv_vit_b_16`
(requires --height 224 --width 224), `tv_convnext_tiny`. Use `--lr 1e-4`
for transformer fine-tuning.

The full experiment playbook for the journal submission lives in
`docs/journal_experiment_plan.md`.

## Usage

Requirements are identical to the baselines (torch, torchvision, torchreid,
pillow). Paths below follow the repository README conventions.

Train the full model:

```bash
python -u methods/wicv/train.py \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --model-name osnet_x1_0 \
  --output-dir results/wicv/osnet_x1_0_full \
  --epochs 60 --eval-every 5 --patience 4
```

Evaluate on the test split (overall + per condition):

```bash
python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_full/model_best.pth \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv"
```

Run the complete ablation study (paper Table: component analysis):

```bash
python -u methods/wicv/run_ablation.py \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --model-name osnet_x1_0 \
  --results-root results/wicv_ablation \
  --skip-existing
```

To show backbone-agnostic gains, repeat with `--model-name resnet50` (and
optionally `mobilenetv2_x1_0`), and compare against the corresponding rows of
`results/baselines_final/summary.csv`.

## Paper tables produced by this folder

| Table | Source |
| --- | --- |
| Main comparison: baseline CE vs. WICV-Net per backbone | `train.py` + `evaluate.py` vs. `baselines/torchreid` results |
| Component ablation (ce_only / plain_triplet / no_adv / no_cvpa / no_cvtri / full) | `run_ablation.py` summary.csv |
| Per-condition robustness of the full model | `eval.json` per_condition section |

## Default hyperparameters

| Parameter | Value |
| --- | --- |
| Batch | 16 identities x 4 instances (view-balanced) |
| Optimizer | Adam, lr 3.5e-4, weight decay 5e-4, cosine schedule |
| Input size | 256x128 (matches baselines) |
| w_tri / w_cvpa / w_adv | 1.0 / 0.5 / 0.5 |
| Triplet margin | 0.3 |
| Prototype momentum / temperature | 0.9 / 0.07 |
| Label smoothing | 0.1 |
| GRL warmup | 2/(1+e^(-10p)) - 1 over training progress p |
