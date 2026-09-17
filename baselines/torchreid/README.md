# Torchreid Baselines

Generic training and evaluation scripts for Torchreid models. Paths below
follow the `$CROP_ROOT` / `$SPLIT_ROOT` / `$RESULT_ROOT` convention set in the
repository root `README.md`.

## Prepare Identity Split

```bash
nohup python -u scripts/build_train_test_split.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --output-root "$SPLIT_ROOT" \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --seed 42 \
  > build_train_test_split.log 2>&1 &
```

This creates:

```text
train.csv
val_query.csv
val_gallery.csv
query.csv
gallery.csv
```

## OSNet

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --model-name osnet_x1_0 \
  --output-root results/osnet_finetuned \
  --epochs 20 \
  --eval-every 5 \
  --patience 3 \
  --min-delta 0.001 \
  --batch-size 64 \
  > osnet_train.log 2>&1 &
```

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --model-name osnet_x1_0 \
  --weights results/osnet_finetuned/model_best.pth \
  --output results/osnet_finetuned/eval.json \
  > osnet_eval_finetuned.log 2>&1 &
```

## ResNet50

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --model-name resnet50 \
  --output-root results/resnet50_finetuned \
  --epochs 20 \
  --eval-every 5 \
  --patience 3 \
  --min-delta 0.001 \
  --batch-size 64 \
  > resnet50_train.log 2>&1 &
```

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --model-name resnet50 \
  --weights results/resnet50_finetuned/model_best.pth \
  --output results/resnet50_finetuned/eval.json \
  > resnet50_eval_finetuned.log 2>&1 &
```

## Run Multiple Baselines

Run the default baseline set sequentially, then aggregate the metrics:

```text
osnet_x1_0
osnet_ain_x1_0
osnet_ibn_x1_0
resnet50
resnet101
mobilenetv2_x1_0
```

```bash
nohup python -u baselines/torchreid/run_all.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --results-root results/baselines \
  --epochs 20 \
  --eval-every 5 \
  --patience 3 \
  --min-delta 0.001 \
  --batch-size 64 \
  > run_all_baselines.log 2>&1 &
```

To run a smaller custom set:

```bash
nohup python -u baselines/torchreid/run_all.py \
  --manifest "$CROP_ROOT/manifest.csv" \
  --train-csv "$SPLIT_ROOT/train.csv" \
  --val-query "$SPLIT_ROOT/val_query.csv" \
  --val-gallery "$SPLIT_ROOT/val_gallery.csv" \
  --query "$SPLIT_ROOT/query.csv" \
  --gallery "$SPLIT_ROOT/gallery.csv" \
  --results-root results/baselines \
  --models osnet_x1_0 resnet50 mobilenetv2_x1_0 \
  --epochs 20 \
  --eval-every 5 \
  --patience 3 \
  --min-delta 0.001 \
  --batch-size 64 \
  > run_custom_baselines.log 2>&1 &
```

Outputs:

```text
results/baselines/summary.csv
results/baselines/summary.json
results/baselines/<model_name>/train.log
results/baselines/<model_name>/eval.log
results/baselines/<model_name>/eval.json
```

If any split file is missing, `run_all.py` automatically builds the identity split from the manifest before training. The final test evaluation uses `model_best.pth`, selected by validation mAP, when available.
