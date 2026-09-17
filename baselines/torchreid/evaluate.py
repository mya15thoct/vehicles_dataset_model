#!/usr/bin/env python3
"""Evaluate a Torchreid model checkpoint on query/gallery vehicle crops."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from reid_common.csv_schema import identity, read_csv
from reid_common.reid_eval import compute_metrics, extract_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--output", default="results/torchreid_eval.json")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--weights", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--max-query", type=int, default=0)
    parser.add_argument("--max-gallery", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(weights: str, device: torch.device) -> dict | None:
    if not weights:
        return None
    try:
        return torch.load(weights, map_location=device)
    except (OSError, RuntimeError, EOFError) as exc:
        raise SystemExit(f"Failed to load checkpoint {weights}: {exc}") from exc


def build_model(model_name: str, num_classes: int, device: torch.device, pretrained: bool):
    try:
        import torchreid
    except ImportError as exc:
        raise SystemExit("Missing torchreid. Install torchreid and its dependencies first.") from exc

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        loss="softmax",
    )
    model.to(device)
    return model


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.weights, device)
    num_classes = checkpoint.get("num_classes", 1000) if checkpoint else 1000
    model_name = checkpoint.get("model_name", args.model_name) if checkpoint else args.model_name

    query_rows = read_csv(Path(args.query))
    gallery_rows = read_csv(Path(args.gallery))
    if args.max_query > 0:
        query_rows = query_rows[: args.max_query]
    if args.max_gallery > 0:
        gallery_rows = gallery_rows[: args.max_gallery]

    logger.info("Device: %s", device)
    logger.info("Model: %s", model_name)
    logger.info("Weights: %s", args.weights or "pretrained only")
    logger.info("Query images: %d", len(query_rows))
    logger.info("Gallery images: %d", len(gallery_rows))

    model = build_model(model_name, num_classes, device, args.pretrained and checkpoint is None)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Extracting query features...")
    query_features = extract_features(model, query_rows, args.batch_size, args.num_workers, device)
    logger.info("Extracting gallery features...")
    gallery_features = extract_features(model, gallery_rows, args.batch_size, args.num_workers, device)

    metrics = compute_metrics(
        query_features,
        gallery_features,
        [identity(row) for row in query_rows],
        [identity(row) for row in gallery_rows],
    )
    result = {
        "model": model_name,
        "weights": args.weights or None,
        "query_csv": args.query,
        "gallery_csv": args.gallery,
        "num_query_images": len(query_rows),
        "num_gallery_images": len(gallery_rows),
        **metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(json.dumps(result, indent=2))
    logger.info("Saved: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
