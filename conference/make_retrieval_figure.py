#!/usr/bin/env python3
"""Generate qualitative cross-view retrieval figures for the conference paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CONDITION_ORDER = [
    "morning_norain",
    "evening_norain",
    "morning_rain",
    "evening_rain",
]

CONDITION_LABELS = {
    "morning_norain": "Morning / No rain",
    "evening_norain": "Evening / No rain",
    "morning_rain": "Morning / Rain",
    "evening_rain": "Evening / Rain",
}

INK = (28, 34, 42)
MUTED = (91, 99, 112)
BG = (255, 255, 255)
QUERY_BLUE = (44, 109, 178)
CORRECT_GREEN = (39, 147, 83)
WRONG_RED = (199, 64, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Test query CSV.")
    parser.add_argument("--gallery", required=True, help="Test gallery CSV.")
    parser.add_argument("--weights", required=True, help="Model checkpoint path.")
    parser.add_argument("--model-name", default="osnet_ain_x1_0")
    parser.add_argument("--output-root", default="docs/figures/retrieval_examples")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=168)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--selection",
        choices=["mixed", "success", "failure"],
        default="mixed",
        help="Example selection style. Mixed prefers successes for morning and failures for evening.",
    )
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


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_model(model_name: str, num_classes: int, device: torch.device):
    try:
        import torchreid
    except ImportError as exc:
        raise SystemExit("Missing torchreid. Install torchreid and its dependencies first.") from exc

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=num_classes,
        pretrained=False,
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
    model.eval()
    with torch.no_grad():
        for batch_index, (images, indices) in enumerate(loader, start=1):
            images = images.to(device)
            embeddings = model(images)
            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[-1]
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
            for offset, row_index in enumerate(indices.tolist()):
                features[row_index] = embeddings[offset]
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"  batch {batch_index}/{math.ceil(len(dataset) / batch_size)} "
                    f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    return torch.stack(features, dim=0)


def compute_rankings(query_features: torch.Tensor, gallery_features: torch.Tensor, top_k: int) -> list[list[int]]:
    rankings = []
    for index in range(query_features.shape[0]):
        scores = torch.mv(gallery_features, query_features[index])
        rankings.append(torch.argsort(scores, descending=True)[:top_k].tolist())
    return rankings


def first_correct_rank(query_id: str, gallery_ids: list[str], ranking: list[int]) -> int | None:
    for rank, gallery_index in enumerate(ranking, start=1):
        if gallery_ids[gallery_index] == query_id:
            return rank
    return None


def choose_examples(query_rows: list[dict], gallery_rows: list[dict], rankings: list[list[int]], selection: str) -> list[dict]:
    gallery_ids = [identity(row) for row in gallery_rows]
    examples = []
    for condition in CONDITION_ORDER:
        candidates = []
        for query_index, row in enumerate(query_rows):
            if row["condition"] != condition:
                continue
            qid = identity(row)
            rank = first_correct_rank(qid, gallery_ids, rankings[query_index])
            top1_correct = rank == 1
            has_correct_in_topk = rank is not None
            candidates.append(
                {
                    "query_index": query_index,
                    "condition": condition,
                    "identity": qid,
                    "label": row["label"],
                    "top1_correct": top1_correct,
                    "first_correct_rank": rank,
                    "has_correct_in_topk": has_correct_in_topk,
                }
            )
        if not candidates:
            continue

        if selection == "success":
            preference = "success"
        elif selection == "failure":
            preference = "failure"
        else:
            preference = "failure" if condition.startswith("evening") else "success"

        if preference == "success":
            pool = [item for item in candidates if item["top1_correct"]]
            if not pool:
                pool = [item for item in candidates if item["has_correct_in_topk"]]
            if not pool:
                pool = candidates
            chosen = pool[len(pool) // 2]
        else:
            pool = [item for item in candidates if not item["top1_correct"] and item["has_correct_in_topk"]]
            if not pool:
                pool = [item for item in candidates if not item["top1_correct"]]
            if not pool:
                pool = candidates
            chosen = pool[len(pool) // 2]
        examples.append(chosen)
    return examples


def fit_crop(path: str, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = min(size / image.width, size / image.height)
        resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BG)
    x = (size - resized.width) // 2
    y = (size - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def draw_panel(image: Image.Image, title: str, subtitle: str, color: tuple[int, int, int], size: int) -> Image.Image:
    title_font = load_font(15, bold=True)
    subtitle_font = load_font(12)
    panel_h = size + 48
    panel = Image.new("RGB", (size + 12, panel_h), BG)
    draw = ImageDraw.Draw(panel)
    panel.paste(image, (6, 6))
    draw.rectangle((4, 4, size + 7, size + 7), outline=color, width=4)
    draw.text((6, size + 13), title, fill=color, font=title_font)
    draw.text((6, size + 31), subtitle, fill=MUTED, font=subtitle_font)
    return panel


def make_figure(
    examples: list[dict],
    query_rows: list[dict],
    gallery_rows: list[dict],
    rankings: list[list[int]],
    output_root: Path,
    top_k: int,
    crop_size: int,
) -> list[dict]:
    label_font = load_font(18, bold=True)
    small_font = load_font(13)
    row_label_w = 178
    gap = 14
    cols = top_k + 1
    panel_w = crop_size + 12
    panel_h = crop_size + 48
    figure_w = row_label_w + cols * panel_w + (cols + 1) * gap
    figure_h = len(examples) * panel_h + (len(examples) + 1) * gap
    figure = Image.new("RGB", (figure_w, figure_h), BG)
    draw = ImageDraw.Draw(figure)
    gallery_ids = [identity(row) for row in gallery_rows]
    metadata = []

    for row_index, example in enumerate(examples):
        y = gap + row_index * (panel_h + gap)
        query = query_rows[example["query_index"]]
        qid = identity(query)
        condition_label = CONDITION_LABELS.get(query["condition"], query["condition"])
        draw.text((12, y + 16), condition_label, fill=INK, font=label_font)
        draw.text((12, y + 42), f"ID {query['vehicle_id']} | {query['label']}", fill=MUTED, font=small_font)
        rank_text = "Top-1 correct" if example["top1_correct"] else f"First correct: {example['first_correct_rank'] or '>top-k'}"
        draw.text((12, y + 62), rank_text, fill=MUTED, font=small_font)

        query_panel = draw_panel(
            fit_crop(query["crop_path"], crop_size),
            "Query",
            "after view",
            QUERY_BLUE,
            crop_size,
        )
        figure.paste(query_panel, (row_label_w + gap, y))

        top_items = []
        for rank, gallery_index in enumerate(rankings[example["query_index"]], start=1):
            gallery = gallery_rows[gallery_index]
            is_match = gallery_ids[gallery_index] == qid
            color = CORRECT_GREEN if is_match else WRONG_RED
            panel = draw_panel(
                fit_crop(gallery["crop_path"], crop_size),
                f"Rank {rank}",
                "correct" if is_match else "wrong",
                color,
                crop_size,
            )
            x = row_label_w + gap + rank * (panel_w + gap)
            figure.paste(panel, (x, y))
            top_items.append(
                {
                    "rank": rank,
                    "gallery_condition": gallery["condition"],
                    "gallery_vehicle_id": gallery["vehicle_id"],
                    "gallery_label": gallery["label"],
                    "gallery_crop_path": gallery["crop_path"],
                    "correct": is_match,
                }
            )

        metadata.append(
            {
                "condition": query["condition"],
                "query_vehicle_id": query["vehicle_id"],
                "query_label": query["label"],
                "query_crop_path": query["crop_path"],
                "top1_correct": example["top1_correct"],
                "first_correct_rank": example["first_correct_rank"],
                "top_results": top_items,
            }
        )

        row_image = figure.crop((0, y, figure_w, y + panel_h))
        row_path = output_root / f"retrieval_{query['condition']}.jpg"
        row_image.save(row_path, quality=95, subsampling=0)

    figure_path = output_root / f"qualitative_retrieval_top{top_k}.jpg"
    figure.save(figure_path, quality=95, subsampling=0)
    print(f"Saved {figure_path}")
    return metadata


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    query_rows = read_csv(Path(args.query))
    gallery_rows = read_csv(Path(args.gallery))
    checkpoint = torch.load(args.weights, map_location=device)
    model_name = checkpoint.get("model_name", args.model_name)
    num_classes = checkpoint.get("num_classes", 1000)

    print(f"Device: {device}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Query images: {len(query_rows)}", flush=True)
    print(f"Gallery images: {len(gallery_rows)}", flush=True)

    model = build_model(model_name, num_classes, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Extracting query features...", flush=True)
    query_features = extract_features(model, query_rows, args.batch_size, args.num_workers, device)
    print("Extracting gallery features...", flush=True)
    gallery_features = extract_features(model, gallery_rows, args.batch_size, args.num_workers, device)
    rankings = compute_rankings(query_features, gallery_features, args.top_k)
    examples = choose_examples(query_rows, gallery_rows, rankings, args.selection)
    metadata = make_figure(examples, query_rows, gallery_rows, rankings, output_root, args.top_k, args.crop_size)

    metadata_path = output_root / "qualitative_retrieval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
