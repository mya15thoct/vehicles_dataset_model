#!/usr/bin/env python3
"""Evaluate a Torchreid model checkpoint on query/gallery vehicle crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="/mnt/ngan/vehicles/reid_benchmark_identity/query.csv")
    parser.add_argument("--gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv")
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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


class CropDataset(Dataset):
    def __init__(self, rows: list[dict], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


def load_checkpoint(weights: str, device: torch.device) -> dict | None:
    if not weights:
        return None
    return torch.load(weights, map_location=device)


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


def extract_features(model, rows: list[dict], batch_size: int, num_workers: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = CropDataset(rows, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = [None] * len(rows)
    with torch.no_grad():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            images = images.to(device)
            embeddings = model(images)
            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[-1]
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
            for offset, row_index in enumerate(indices.tolist()):
                features[row_index] = embeddings[offset]
            print(
                f"  batch {batch_index}/{math.ceil(len(dataset) / batch_size)} "
                f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    return torch.stack(features, dim=0)


def compute_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> dict:
    rank1 = 0
    rank5 = 0
    ap_sum = 0.0
    valid_queries = 0

    for index in range(query_features.shape[0]):
        qid = query_ids[index]
        positives = [gid == qid for gid in gallery_ids]
        num_positives = sum(positives)
        if num_positives == 0:
            continue

        scores = torch.mv(gallery_features, query_features[index])
        order = torch.argsort(scores, descending=True).tolist()
        ordered_matches = [positives[i] for i in order]

        valid_queries += 1
        rank1 += int(ordered_matches[0])
        rank5 += int(any(ordered_matches[:5]))

        hits = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(ordered_matches, start=1):
            if is_match:
                hits += 1
                precision_sum += hits / rank
                if hits == num_positives:
                    break
        ap_sum += precision_sum / num_positives

    if valid_queries == 0:
        return {"valid_queries": 0, "rank1": 0.0, "rank5": 0.0, "mAP": 0.0}
    return {
        "valid_queries": valid_queries,
        "rank1": rank1 / valid_queries,
        "rank5": rank5 / valid_queries,
        "mAP": ap_sum / valid_queries,
    }


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

    print(f"Device: {device}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Weights: {args.weights or 'pretrained only'}", flush=True)
    print(f"Query images: {len(query_rows)}", flush=True)
    print(f"Gallery images: {len(gallery_rows)}", flush=True)

    model = build_model(model_name, num_classes, device, args.pretrained and checkpoint is None)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Extracting query features...", flush=True)
    query_features = extract_features(model, query_rows, args.batch_size, args.num_workers, device)
    print("Extracting gallery features...", flush=True)
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
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
