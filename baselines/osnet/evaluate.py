#!/usr/bin/env python3
"""Evaluate a pretrained OSNet baseline on query/gallery vehicle crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from reid_common.csv_schema import identity, read_csv
from reid_common.reid_eval import compute_metrics, extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--output", default="results/osnet_pretrained.json")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-query",
        type=int,
        default=0,
        help="Optional debug limit. 0 means use all query images.",
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=0,
        help="Optional debug limit. 0 means use all gallery images.",
    )
    return parser.parse_args()


def build_model(model_name: str, device: torch.device):
    try:
        import torchreid
    except ImportError as exc:
        raise SystemExit(
            "Missing torchreid. Install it with `pip install torchreid` "
            "or from the official torchreid repository."
        ) from exc

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1000,
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def main() -> int:
    args = parse_args()
    query_rows = read_csv(Path(args.query))
    gallery_rows = read_csv(Path(args.gallery))
    if args.max_query > 0:
        query_rows = query_rows[: args.max_query]
    if args.max_gallery > 0:
        gallery_rows = gallery_rows[: args.max_gallery]

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Query images: {len(query_rows)}", flush=True)
    print(f"Gallery images: {len(gallery_rows)}", flush=True)

    model = build_model(args.model_name, device)

    print("Extracting query features...", flush=True)
    query_features = extract_features(
        model,
        query_rows,
        args.batch_size,
        args.num_workers,
        device,
    )
    print("Extracting gallery features...", flush=True)
    gallery_features = extract_features(
        model,
        gallery_rows,
        args.batch_size,
        args.num_workers,
        device,
    )

    query_ids = [identity(row) for row in query_rows]
    gallery_ids = [identity(row) for row in gallery_rows]
    metrics = compute_metrics(query_features, gallery_features, query_ids, gallery_ids)

    result = {
        "model": args.model_name,
        "query_csv": args.query,
        "gallery_csv": args.gallery,
        "num_query_images": len(query_rows),
        "num_gallery_images": len(gallery_rows),
        **metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
