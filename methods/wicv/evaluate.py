#!/usr/bin/env python3
"""Evaluate a trained WICV-Net checkpoint: overall and per-condition retrieval.

Optionally applies k-reciprocal re-ranking (--rerank) and reports both plain
and re-ranked metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from dataset import identity, read_csv
from metrics import compute_metrics, compute_metrics_from_dist, extract_features
from model import WICVNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/wicv/osnet_x1_0_full/model_best.pth")
    parser.add_argument("--query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/query.csv")
    parser.add_argument("--gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/gallery.csv")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to eval.json next to the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rerank", action="store_true", help="Also report k-reciprocal re-ranked metrics.")
    parser.add_argument("--k1", type=int, default=20)
    parser.add_argument("--k2", type=int, default=6)
    parser.add_argument("--lambda-value", type=float, default=0.3)
    return parser.parse_args()


def reranked_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
    k1: int,
    k2: int,
    lambda_value: float,
) -> dict:
    from rerank import k_reciprocal_rerank

    q_g = torch.cdist(query_features, gallery_features).numpy()
    q_q = torch.cdist(query_features, query_features).numpy()
    g_g = torch.cdist(gallery_features, gallery_features).numpy()
    final_dist = k_reciprocal_rerank(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lambda_value)
    return compute_metrics_from_dist(torch.from_numpy(final_dist), query_ids, gallery_ids)


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

    result = {
        "checkpoint": str(checkpoint_path),
        "model_name": checkpoint["model_name"],
        "query": args.query,
        "gallery": args.gallery,
        "overall": overall,
        "per_condition": {},
    }

    if args.rerank:
        print("Computing k-reciprocal re-ranking (overall)...", flush=True)
        overall_rerank = reranked_metrics(
            query_features, gallery_features, query_ids, gallery_ids,
            args.k1, args.k2, args.lambda_value,
        )
        result["overall_rerank"] = overall_rerank
        print(
            f"overall+rerank rank1={overall_rerank['rank1']:.4f} "
            f"rank5={overall_rerank['rank5']:.4f} mAP={overall_rerank['mAP']:.4f}"
        )

    conditions = sorted({row["condition"] for row in query_rows})
    for condition in conditions:
        q_index = [i for i, row in enumerate(query_rows) if row["condition"] == condition]
        g_index = [i for i, row in enumerate(gallery_rows) if row["condition"] == condition]
        if not q_index or not g_index:
            continue
        q_feats = query_features[q_index]
        g_feats = gallery_features[g_index]
        q_ids = [query_ids[i] for i in q_index]
        g_ids = [gallery_ids[i] for i in g_index]
        metrics = compute_metrics(q_feats, g_feats, q_ids, g_ids)
        result["per_condition"][condition] = metrics
        print(
            f"{condition}: rank1={metrics['rank1']:.4f} "
            f"rank5={metrics['rank5']:.4f} mAP={metrics['mAP']:.4f}"
        )
        if args.rerank:
            rerank_metrics = reranked_metrics(
                q_feats, g_feats, q_ids, g_ids, args.k1, args.k2, args.lambda_value
            )
            result.setdefault("per_condition_rerank", {})[condition] = rerank_metrics
            print(
                f"{condition}+rerank: rank1={rerank_metrics['rank1']:.4f} "
                f"rank5={rerank_metrics['rank5']:.4f} mAP={rerank_metrics['mAP']:.4f}"
            )

    output_path = Path(args.output) if args.output else checkpoint_path.parent / "eval.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
