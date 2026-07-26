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

## WICV-Net v2: two structural modules (raises novelty beyond loss combination)

The v1 objective above is three *training-time metric constraints*. All three
are symmetric, and none of them exists at inference -- at test time the model
is just backbone + BNNeck. That is exactly what invites the reviewer comment
"this is a combination of existing techniques". v2 adds two modules that
exploit structure specific to this benchmark instead.

### 4. Cross-View Transition (CVT) -- directional, and alive at inference

The before->after view change here is **not arbitrary**: the same vehicle
passes a fixed point and is observed front-first, rear-second. The view gap is
therefore a *systematic, learnable function*, not merely a distance to be
minimized. CVT learns two residual maps, `T_b2a` and `T_a2b`, trained on the
in-batch cross-view positive pairs the balanced PK sampler guarantees. The
target of each direction is **detached**: CVT's job is to learn the
transformation, not to drag the two subspaces together -- that is CVPA's job,
and letting both gradients flow would let the objectives collapse the view
distinction to satisfy each other.

At inference the gallery (before view) is pushed through `T_b2a` so retrieval
happens inside a single view subspace. This makes CVT the only component that
survives into test time (`--cvt-mode gallery|query|off`).

### 5. Condition-Adaptive Normalization (CAN) -- replaces FCA

The loss-weight sweep produced a clear negative result: adversarially erasing
the condition signal does not help (best mAP was at `w_adv` ~ 0). We read that
as evidence the condition is **not pure nuisance noise but a known covariate**.
CAN therefore keeps one normalization branch per condition (the 2x2
time x weather grid), so condition-specific first- and second-order feature
shifts are removed by construction instead of being fought with a reversed
gradient. A shared branch is always maintained and is used whenever the
condition label is unavailable -- which is also what the cross-condition
protocol must use, since there the test condition is unseen
(`--no-condition-routing`).

Honest caveat to state in the paper: CAN reads the condition label at test
time. That label is scene metadata (timestamp plus weather), not a per-vehicle
annotation, so it leaks no identity information -- but the shared-branch
fallback number should also be reported for readers who do not accept that
assumption.

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
modules.py                v2: CrossViewTransition (CVT) and ConditionAdaptiveBNNeck (CAN)
model.py                  Backbone + (condition-adaptive) BNNeck + GRL heads + CVT
losses.py                 CV-Tri loss, CVPA prototype memory, CVT transition loss
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

Train the v2 model (both structural modules, FCA dropped):

```bash
python -u methods/wicv/train.py \
  --model-name osnet_x1_0 \
  --use-cvt --use-can --no-adv \
  --output-dir results/wicv/osnet_x1_0_v2 \
  --epochs 60 --eval-every 5 --patience 4

python -u methods/wicv/evaluate.py \
  --checkpoint results/wicv/osnet_x1_0_v2/model_best.pth \
  --cvt-mode gallery
```

For the cross-condition protocol the test condition is unseen, so CAN must
fall back to its shared branch:

```bash
python -u methods/wicv/evaluate.py \
  --checkpoint <v2 checkpoint> --no-condition-routing
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
