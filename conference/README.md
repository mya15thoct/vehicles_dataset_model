# Conference Experiment Pipeline

This folder records the experiment setup for the shorter Scopus-indexed conference version.

## Files

| File | Purpose |
| --- | --- |
| `README.md` | Pipeline commands and contribution wording |
| `build_subset.py` | Build the condition-balanced 50% conference subset |
| `paper_writing_notes.md` | Paper-ready wording for title, abstract, contributions, sections |
| `conference_statistics.md` | Split statistics, leakage audit, and tables for the paper |
| `result_tables.md` | Tables to fill after baseline training finishes |
| `reviewer_positioning.md` | What to claim and what to avoid for conference vs journal |

## Conference Scope

The conference paper should focus on:

1. Practical cross-view vehicle re-identification formulation.
2. Identity-disjoint and leakage-audited evaluation protocol.
3. Baseline comparison and weather/time analysis on a condition-balanced subset.

The conference paper should not position the dataset release as the main contribution.

## Subset Definition

Use a condition-balanced subset sampled from the full crop manifest:

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

## Build Conference Subset

Run on the server after full crops are exported:

```bash
python conference/build_subset.py \
  --manifest /mnt/ngan/vehicles/reid_crops_full/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark_conference_50 \
  --ids-per-condition 300 \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
```

Expected outputs:

```text
/mnt/ngan/vehicles/reid_benchmark_conference_50/train.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/val_query.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/val_gallery.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/query.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/gallery.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/selected_identities.csv
/mnt/ngan/vehicles/reid_benchmark_conference_50/stats.json
```

## Audit Conference Split

```bash
python scripts/audit_reid_splits.py \
  --train /mnt/ngan/vehicles/reid_benchmark_conference_50/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_conference_50/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_conference_50/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_conference_50/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_conference_50/gallery.csv \
  --output /mnt/ngan/vehicles/reid_benchmark_conference_50/audit.json
```

Expected:

```json
"passed": true
```

## Train Conference Baselines

Use a separate results directory so conference results do not overwrite full-dataset results:

```bash
nohup python -u baselines/torchreid/run_all.py \
  --manifest /mnt/ngan/vehicles/reid_crops_full/manifest.csv \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_conference_50/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_conference_50/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_conference_50/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_conference_50/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_conference_50/gallery.csv \
  --results-root results/conference_50_e100 \
  --epochs 100 \
  --eval-every 5 \
  --patience 4 \
  --batch-size 32 \
  --num-workers 4 \
  --no-auto-split \
  > run_conference_50_e100.log 2>&1 &
```

Monitor:

```bash
tail -f run_conference_50_e100.log
```

Final result table:

```bash
cat results/conference_50_e100/summary.csv
```

## Evaluate Condition And Class Breakdowns

After at least one checkpoint finishes training, evaluate the same checkpoint on condition-specific and class-specific test subsets. For the paper, run this first on the best overall model, for example `osnet_ain_x1_0`:

```bash
nohup python -u conference/evaluate_breakdowns.py \
  --query /mnt/recover/ngan/vehicles/reid_benchmark_conference_50/query.csv \
  --gallery /mnt/recover/ngan/vehicles/reid_benchmark_conference_50/gallery.csv \
  --results-root results/conference_50_e100 \
  --output-root results/conference_50_breakdowns \
  --models osnet_ain_x1_0 \
  --batch-size 64 \
  --num-workers 4 \
  > run_conference_breakdowns.log 2>&1 &
```

Monitor:

```bash
tail -f run_conference_breakdowns.log
```

Outputs:

```text
results/conference_50_breakdowns/breakdown_summary.csv
results/conference_50_breakdowns/breakdown_summary.json
results/conference_50_breakdowns/subsets/
```

The breakdown script reuses the trained checkpoint. It does not train separate models for each condition or class.

## Conference Contribution Wording

Recommended contribution statement:

```text
The main contributions of this work are threefold:

1. We formulate a practical cross-view vehicle re-identification problem using synchronized front/rear traffic camera views under varying weather and lighting conditions.

2. We define an identity-disjoint and leakage-audited evaluation protocol for this cross-view setting, where an automated audit verifies that there is no identity or crop overlap among the training, validation, and test splits.

3. We benchmark representative deep Re-ID models on a condition-balanced subset of 1,200 cross-view vehicle identities and analyze the impact of rain and time-of-day variation on matching performance.
```

Do not list dataset release as a conference contribution. The collected/annotated data should be described in the experimental setup section only.
