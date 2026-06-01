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
