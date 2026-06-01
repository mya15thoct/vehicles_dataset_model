# Torchreid Baselines

Generic training and evaluation scripts for Torchreid models.

## Prepare Identity Split

```bash
nohup python -u scripts/build_train_test_split.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --output-root /mnt/ngan/vehicles/reid_benchmark_identity \
  --train-ratio 0.7 \
  --seed 42 \
  > build_train_test_split.log 2>&1 &
```

## OSNet

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --model-name osnet_x1_0 \
  --output-dir results/osnet_finetuned \
  --epochs 20 \
  --batch-size 64 \
  > osnet_train.log 2>&1 &
```

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --model-name osnet_x1_0 \
  --weights results/osnet_finetuned/model_last.pth \
  --output results/osnet_finetuned/eval.json \
  > osnet_eval_finetuned.log 2>&1 &
```

## ResNet50

```bash
nohup python -u baselines/torchreid/train.py \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --model-name resnet50 \
  --output-dir results/resnet50_finetuned \
  --epochs 20 \
  --batch-size 64 \
  > resnet50_train.log 2>&1 &
```

```bash
nohup python -u baselines/torchreid/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --model-name resnet50 \
  --weights results/resnet50_finetuned/model_last.pth \
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
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --results-root results/baselines \
  --epochs 20 \
  --batch-size 64 \
  > run_all_baselines.log 2>&1 &
```

To run a smaller custom set:

```bash
nohup python -u baselines/torchreid/run_all.py \
  --manifest /mnt/ngan/vehicles/reid_crops/manifest.csv \
  --train-csv /mnt/ngan/vehicles/reid_benchmark_identity/train.csv \
  --query /mnt/ngan/vehicles/reid_benchmark_identity/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv \
  --results-root results/baselines \
  --models osnet_x1_0 resnet50 mobilenetv2_x1_0 \
  --epochs 20 \
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

If `train.csv`, `query.csv`, or `gallery.csv` are missing, `run_all.py` automatically builds them from the manifest before training.
