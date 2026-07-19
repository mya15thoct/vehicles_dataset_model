#!/usr/bin/env python3
"""Render qualitative retrieval figures: query + top-k gallery matches.

For every condition, samples success cases (rank-1 correct) and failure cases
(rank-1 wrong) and composes strips of [query | top-k gallery crops] with green
borders on correct matches and red on wrong ones. Output goes to one JPG per
condition plus a metadata JSON listing the crops used.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps

from dataset import identity, read_csv
from metrics import extract_features
from model import WICVNet

CELL_WIDTH = 128
CELL_HEIGHT = 256
BORDER = 6
GAP = 10
QUERY_GAP = 26
GREEN = (46, 160, 67)
RED = (207, 34, 46)
BLUE = (9, 105, 218)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/wicv/osnet_x1_0_full/model_best.pth")
    parser.add_argument("--query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/query.csv")
    parser.add_argument("--gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/gallery.csv")
    parser.add_argument("--output-root", default="docs/figures/retrieval")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--success-per-condition", type=int, default=3)
    parser.add_argument("--failure-per-condition", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_cell(crop_path: str, border_color: tuple[int, int, int]) -> Image.Image:
    with Image.open(crop_path) as image:
        image = image.convert("RGB")
        image.thumbnail((CELL_WIDTH, CELL_HEIGHT))
        canvas = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), (24, 24, 24))
        canvas.paste(image, ((CELL_WIDTH - image.width) // 2, (CELL_HEIGHT - image.height) // 2))
    return ImageOps.expand(canvas, border=BORDER, fill=border_color)


def compose_strip(query_row: dict, ranked_gallery: list[tuple[dict, bool]]) -> Image.Image:
    cells = [load_cell(query_row["crop_path"], BLUE)]
    for gallery_row, is_match in ranked_gallery:
        cells.append(load_cell(gallery_row["crop_path"], GREEN if is_match else RED))

    cell_w = CELL_WIDTH + 2 * BORDER
    cell_h = CELL_HEIGHT + 2 * BORDER
    width = cell_w * len(cells) + QUERY_GAP + GAP * (len(cells) - 2)
    strip = Image.new("RGB", (width, cell_h + 22), (255, 255, 255))
    x = 0
    for index, cell in enumerate(cells):
        strip.paste(cell, (x, 0))
        label = "query" if index == 0 else f"rank {index}"
        ImageDraw.Draw(strip).text((x + 4, cell_h + 4), label, fill=(60, 60, 60))
        x += cell_w + (QUERY_GAP if index == 0 else GAP)
    return strip


def stack_strips(strips: list[Image.Image], title: str) -> Image.Image:
    width = max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips) + GAP * (len(strips) - 1) + 30
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 6), title, fill=(0, 0, 0))
    y = 30
    for strip in strips:
        sheet.paste(strip, (0, y))
        y += strip.height + GAP
    return sheet


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
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
    query_features = extract_features(
        model, query_rows, args.batch_size, args.num_workers, device, height, width
    )
    gallery_features = extract_features(
        model, gallery_rows, args.batch_size, args.num_workers, device, height, width
    )
    gallery_ids = [identity(row) for row in gallery_rows]
    gallery_id_set = set(gallery_ids)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {}

    conditions = sorted({row["condition"] for row in query_rows})
    for condition in conditions:
        successes = []
        failures = []
        candidate_indices = [i for i, row in enumerate(query_rows) if row["condition"] == condition]
        rng.shuffle(candidate_indices)
        for q_index in candidate_indices:
            qid = identity(query_rows[q_index])
            if qid not in gallery_id_set:
                continue
            scores = torch.mv(gallery_features, query_features[q_index])
            order = torch.argsort(scores, descending=True)[: args.topk].tolist()
            ranked = [(gallery_rows[g], gallery_ids[g] == qid) for g in order]
            record = {
                "query": query_rows[q_index]["crop_path"],
                "query_id": qid,
                "topk": [
                    {"crop_path": row["crop_path"], "id": identity(row), "match": match}
                    for row, match in ranked
                ],
            }
            if ranked[0][1] and len(successes) < args.success_per_condition:
                successes.append((query_rows[q_index], ranked, record))
            elif not ranked[0][1] and len(failures) < args.failure_per_condition:
                failures.append((query_rows[q_index], ranked, record))
            if (
                len(successes) >= args.success_per_condition
                and len(failures) >= args.failure_per_condition
            ):
                break

        metadata[condition] = {
            "success": [record for _, _, record in successes],
            "failure": [record for _, _, record in failures],
        }
        for kind, cases in (("success", successes), ("failure", failures)):
            if not cases:
                print(f"{condition}: no {kind} cases found", flush=True)
                continue
            strips = [compose_strip(query_row, ranked) for query_row, ranked, _ in cases]
            sheet = stack_strips(strips, f"{condition} - {kind} cases (query blue, correct green, wrong red)")
            out_path = output_root / f"retrieval_{condition}_{kind}.jpg"
            sheet.save(out_path, quality=92)
            print(f"Saved: {out_path}", flush=True)

    (output_root / "retrieval_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output_root / 'retrieval_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
