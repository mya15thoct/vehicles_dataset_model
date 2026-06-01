# OSNet Baseline

This baseline evaluates a pretrained OSNet model on the exported vehicle Re-ID split.

## Inputs

Expected split files:

```text
/mnt/ngan/vehicles/reid_benchmark/query.csv
/mnt/ngan/vehicles/reid_benchmark/gallery.csv
```

Each CSV row must contain:

```text
condition, view, vehicle_id, label, frame_id, frame_name, crop_path, source_image
```

The matching identity is defined as:

```text
condition + vehicle_id
```

## Install

Use an environment with PyTorch and Torchreid:

```bash
pip install torch torchvision torchreid
```

If `torchreid` is not available from pip in the environment, install it from the official repository.

## Run

```bash
nohup python -u baselines/osnet/evaluate.py \
  --query /mnt/ngan/vehicles/reid_benchmark/query.csv \
  --gallery /mnt/ngan/vehicles/reid_benchmark/gallery.csv \
  --output results/osnet_pretrained.json \
  > osnet_eval.log 2>&1 &
```

Monitor:

```bash
tail -f osnet_eval.log
```

