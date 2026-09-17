# Conference-Subset Evaluation Pipeline

This folder builds a condition-balanced subset of the full VN2V-Weather
benchmark and runs baseline training/evaluation on it.

## Files

| File | Purpose |
| --- | --- |
| `build_subset.py` | Build a condition-balanced subset from the full crop manifest |
| `evaluate_breakdowns.py` | Evaluate a trained checkpoint on condition- and class-specific subsets |
| `make_retrieval_figure.py` | Generate a qualitative top-k retrieval figure |
| `make_result_chart.py` | Generate a condition-wise Rank-1/mAP chart from a breakdown summary |

## Subset Definition

A condition-balanced subset is sampled from the full crop manifest:

| Condition | Selected shared IDs |
| --- | ---: |
| `morning_norain` | 300 |
| `evening_norain` | 300 |
| `morning_rain` | 300 |
| `evening_rain` | 300 |
| **Total** | **1,200** |

Split:

| Split | IDs per condition | Total IDs |
| --- | ---: | ---: |
| Train | 210 | 840 |
| Validation | 30 | 120 |
| Test | 60 | 240 |

Query/gallery rule:

```text
query   = after-view crops
gallery = before-view crops
```

## Setup

Set these paths for your machine (see the root `README.md` for how
`CROP_ROOT` is produced by `scripts/export_reid_crops.py`):

```bash
CROP_ROOT=/path/to/reid_crops
CONF_SPLIT_ROOT=/path/to/reid_benchmark_conference_50
RESULT_ROOT=results/conference_50_e100
```

## Build Conference Subset

```bash
python conference/build_subset.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --output-root "$CONF_SPLIT_ROOT" \
  --ids-per-condition 300 \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
```

Expected outputs:

```text
$CONF_SPLIT_ROOT/train.csv
$CONF_SPLIT_ROOT/val_query.csv
$CONF_SPLIT_ROOT/val_gallery.csv
$CONF_SPLIT_ROOT/query.csv
$CONF_SPLIT_ROOT/gallery.csv
$CONF_SPLIT_ROOT/selected_identities.csv
$CONF_SPLIT_ROOT/stats.json
```

## Audit Conference Split

```bash
python scripts/audit_reid_splits.py \
  --train "$CONF_SPLIT_ROOT/train.csv" \
  --val-query "$CONF_SPLIT_ROOT/val_query.csv" \
  --val-gallery "$CONF_SPLIT_ROOT/val_gallery.csv" \
  --query "$CONF_SPLIT_ROOT/query.csv" \
  --gallery "$CONF_SPLIT_ROOT/gallery.csv" \
  --output "$CONF_SPLIT_ROOT/audit.json"
```

Expected: `"passed": true`

## Train Conference Baselines

Use a separate results directory so conference results do not overwrite
full-dataset results:

```bash
python -u baselines/torchreid/run_all.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --train-csv "$CONF_SPLIT_ROOT/train.csv" \
  --val-query "$CONF_SPLIT_ROOT/val_query.csv" \
  --val-gallery "$CONF_SPLIT_ROOT/val_gallery.csv" \
  --query "$CONF_SPLIT_ROOT/query.csv" \
  --gallery "$CONF_SPLIT_ROOT/gallery.csv" \
  --results-root "$RESULT_ROOT" \
  --epochs 100 \
  --eval-every 5 \
  --patience 4 \
  --batch-size 32 \
  --num-workers 4 \
  --no-auto-split
```

Final result table: `$RESULT_ROOT/summary.csv`

## Evaluate Condition And Class Breakdowns

After at least one checkpoint finishes training, evaluate the same checkpoint
on condition-specific and class-specific test subsets. The breakdown script
reuses the trained checkpoint; it does not train separate models per
condition or class.

```bash
python -u conference/evaluate_breakdowns.py \
  --query "$CONF_SPLIT_ROOT/query.csv" \
  --gallery "$CONF_SPLIT_ROOT/gallery.csv" \
  --results-root "$RESULT_ROOT" \
  --output-root results/conference_50_breakdowns \
  --models osnet_ain_x1_0 \
  --batch-size 64 \
  --num-workers 4
```

Outputs:

```text
results/conference_50_breakdowns/breakdown_summary.csv
results/conference_50_breakdowns/breakdown_summary.json
results/conference_50_breakdowns/subsets/
```

## Generate Qualitative Retrieval Figure

Uses the best mAP checkpoint to generate a qualitative top-k retrieval
figure: after-view crops as queries, before-view crops as ranked gallery
results, green borders for correct matches and red borders for incorrect
matches.

```bash
python -u conference/make_retrieval_figure.py \
  --query "$CONF_SPLIT_ROOT/query.csv" \
  --gallery "$CONF_SPLIT_ROOT/gallery.csv" \
  --weights "$RESULT_ROOT/osnet_ain_x1_0/model_best.pth" \
  --model-name osnet_ain_x1_0 \
  --output-root docs/figures/retrieval_examples \
  --top-k 3 \
  --batch-size 64 \
  --num-workers 4
```

## Generate Result Chart

```bash
python conference/make_result_chart.py \
  --summary results/conference_50_breakdowns/breakdown_summary.csv \
  --model osnet_ain_x1_0 \
  --output-root docs/figures/result_charts
```

Output: `docs/figures/result_charts/condition_performance_osnet_ain_x1_0.png`
