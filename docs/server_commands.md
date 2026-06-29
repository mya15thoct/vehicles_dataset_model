# Server Commands To Collect Missing Paper Results

Use this file as a checklist for commands that should be run on the server.

## Activate Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate recognition
```

If the server uses Anaconda:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate recognition
```

## Pull Latest Code And Annotations

```bash
cd ~/mya/vehicles_dataset_model
git pull
```

## Check Image Counts

```bash
for d in morning_norain_before morning_norain_after morning_rain_before morning_rain_after evening_norain_before evening_norain_after evening_rain_before evening_rain_after; do
  echo "== $d =="
  find /mnt/ngan/vehicles/multi-weather_traffic_data/$d -maxdepth 1 -name "*.jpg" | wc -l
done
```

Expected frame counts:

| Folder | Expected frames |
| --- | ---: |
| `morning_norain_before` | 4,923 |
| `morning_norain_after` | 5,207 |
| `morning_rain_before` | 6,671 |
| `morning_rain_after` | 6,700 |
| `evening_norain_before` | 4,597 |
| `evening_norain_after` | 5,037 |
| `evening_rain_before` | 4,561 |
| `evening_rain_after` | 4,558 |

## Validate Annotations

```bash
python scripts/validate_annotations.py --config configs/dataset.json
```

Expected:

```text
missing_id=0
invalid_boxes=0
label_mismatch=0
```

## Export Full Re-ID Crops

```bash
rm -rf /mnt/ngan/vehicles/reid_crops_full

nohup python -u scripts/export_reid_crops.py \
  --config configs/dataset.json \
  --completed-only \
  --output-root /mnt/ngan/vehicles/reid_crops_full \
  > export_reid_crops_full.log 2>&1 &
```

Monitor:

```bash
tail -f export_reid_crops_full.log
```

Expected:

```text
Total crops: 100952
Skipped boxes/images: 0
```

## Build Identity-Disjoint Splits

```bash
rm -rf /mnt/ngan/vehicles/reid_benchmark_identity_full

python -u scripts/build_train_test_split.py \
  --manifest /mnt/ngan/vehicles/reid_crops_full/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark_identity_full \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42
```

## Audit Splits

```bash
python scripts/audit_reid_splits.py \
  --train /mnt/ngan/vehicles/reid_benchmark_identity_full/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_identity_full/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_identity_full/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity_full/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity_full/gallery.csv \
  --output /mnt/ngan/vehicles/reid_benchmark_identity_full/audit.json
```

Expected:

```json
"passed": true
```

## Train And Evaluate Re-ID Baselines

```bash
nohup python -u baselines/torchreid/run_all.py \
  --manifest /mnt/ngan/vehicles/reid_crops_full/manifest.csv \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity_full/train.csv \
  --val-query /mnt/ngan/vehicles/reid_benchmark_identity_full/val_query.csv \
  --val-gallery /mnt/ngan/vehicles/reid_benchmark_identity_full/val_gallery.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity_full/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity_full/gallery.csv \
  --results-root results/baselines_full_e100 \
  --epochs 100 \
  --eval-every 5 \
  --patience 4 \
  --batch-size 64 \
  --num-workers 4 \
  --no-auto-split \
  > run_all_baselines_full_e100.log 2>&1 &
```

Monitor:

```bash
tail -f run_all_baselines_full_e100.log
```

After completion:

```bash
cat results/baselines_full_e100/summary.csv
cat results/baselines_full_e100/summary.json
```

## Generate Paper Figures

The following command creates dataset overview, annotation examples, statistics charts, and cross-view positive Re-ID pairs:

```bash
python scripts/make_paper_figures.py \
  --config configs/dataset.json \
  --image-root /mnt/ngan/vehicles/multi-weather_traffic_data \
  --annotation-root annotation \
  --output-root docs/figures
```

Expected outputs:

```text
docs/figures/figure_01_dataset_overview.jpg
docs/figures/figure_02_annotation_examples.jpg
docs/figures/figure_03_dataset_statistics.png
docs/figures/figure_03a_class_distribution.png
docs/figures/figure_03b_condition_distribution.png
docs/figures/figure_03c_view_distribution.png
docs/figures/figure_03d_shared_identities.png
docs/figures/figure_04_cross_view_positive_pairs.jpg
docs/figures/figure_metadata.json
docs/figures/figure_statistics.json
```

## Information To Send Back To The Writer

Please send these outputs back for documentation:

1. `export_reid_crops_full.log` final lines.
2. `/mnt/ngan/vehicles/reid_benchmark_identity_full/audit.json`.
3. `results/baselines_full_e100/summary.csv`.
4. Best model `eval.json` files if available.
5. Any detection baseline results.
6. Generated files in `docs/figures/`.
