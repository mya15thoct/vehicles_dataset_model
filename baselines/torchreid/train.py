#!/usr/bin/env python3
"""Fine-tune a Torchreid image Re-ID model on vehicle crop identities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/mnt/ngan/vehicles/reid_benchmark_identity/train.csv")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--output-dir", default="results/osnet_finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


class TrainDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_index: dict[str, int], transform) -> None:
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = self.label_to_index[identity(row)]
        return tensor, label


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


def save_checkpoint(path: Path, model, args: argparse.Namespace, label_to_index: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "num_classes": len(label_to_index),
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    train_rows = read_csv(Path(args.train_csv))
    if not train_rows:
        print("Empty training CSV", file=sys.stderr)
        return 1

    identities = sorted({identity(row) for row in train_rows})
    label_to_index = {name: index for index, name in enumerate(identities)}
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = TrainDataset(train_rows, label_to_index, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args.model_name, len(label_to_index), device, not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metadata = {
        "model_name": args.model_name,
        "train_csv": args.train_csv,
        "num_train_images": len(train_rows),
        "num_train_identities": len(label_to_index),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained": not args.no_pretrained,
    }
    (output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_index}/{len(loader)} "
                    f"loss={total_loss / total:.4f} "
                    f"acc={correct / total:.4f}",
                    flush=True,
                )

        scheduler.step()
        save_checkpoint(output_dir / "model_last.pth", model, args, label_to_index)
        if epoch == args.epochs or epoch % 5 == 0:
            save_checkpoint(output_dir / f"model_epoch_{epoch:03d}.pth", model, args, label_to_index)

    print(f"Saved: {output_dir / 'model_last.pth'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
