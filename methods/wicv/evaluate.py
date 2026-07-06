#!/usr/bin/env python3
"""Evaluate a trained WICV-Net checkpoint: overall and per-condition retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dataset import read_csv
from metrics import compute_metrics, extract_features
from dataset import identity
from model import WICVNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/wicv/osnet_x1_0_full/model_best.pth")
    parser.add_argument("--query", default="/mnt/ngan/vehicles/reid_benchmark_identity/query.csv")
    parser.add_argument("--gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to eval.json next to the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Missing checkpoint: {checkpoint_path}", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    height = checkpoint.get("height", 256)
    width = checkpoint.get("width", 128)
    model = WICVNet(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        height=height,
        width=width,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    query_rows = read_csv(Path(args.query))
    gallery_rows = read_csv(Path(args.gallery))
    print(f"query images: {len(query_rows)}, gallery images: {len(gallery_rows)}")

    query_features = extract_features(
        model, query_rows, args.batch_size, args.num_workers, device, height, width
    )
    gallery_features = extract_features(
        model, gallery_rows, args.batch_size, args.num_workers, device, height, width
    )
    query_ids = [identity(row) for row in query_rows]
    gallery_ids = [identity(row) for row in gallery_rows]

    overall = compute_metrics(query_features, gallery_features, query_ids, gallery_ids)
    print(
        f"overall rank1={overall['rank1']:.4f} rank5={overall['rank5']:.4f} mAP={overall['mAP']:.4f}"
    )

    per_condition = {}
    conditions = sorted({row["condition"] for row in query_rows})
    for condition in conditions:
        q_index = [i for i, row in enumerate(query_rows) if row["condition"] == condition]
        g_index = [i for i, row in enumerate(gallery_rows) if row["condition"] == condition]
        if not q_index or not g_index:
            continue
        metrics = compute_metrics(
            query_features[q_index],
            gallery_features[g_index],
            [query_ids[i] for i in q_index],
            [gallery_ids[i] for i in g_index],
        )
        per_condition[condition] = metrics
        print(
            f"{condition}: rank1={metrics['rank1']:.4f} "
            f"rank5={metrics['rank5']:.4f} mAP={metrics['mAP']:.4f}"
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "query": args.query,
        "gallery": args.gallery,
        "overall": overall,
        "per_condition": per_condition,
    }
    output_path = Path(args.output) if args.output else checkpoint_path.parent / "eval.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
