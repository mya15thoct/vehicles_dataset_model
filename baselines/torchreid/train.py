#!/usr/bin/env python3
"""Fine-tune a Torchreid image Re-ID model on vehicle crop identities."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from reid_common.csv_schema import identity, read_csv
from reid_common.reid_eval import compute_metrics, extract_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-query", required=True)
    parser.add_argument("--val-gallery", required=True)
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--output-root", default="results/osnet_finetuned")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience in validation checks. 0 disables early stopping.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum validation mAP improvement required to reset patience.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


class TrainDataset(Dataset):
    def __init__(self, rows: list[dict], label_to_index: dict[str, int], transform) -> None:
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        try:
            with Image.open(row["crop_path"]) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unreadable crop image: {row['crop_path']}") from exc
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


def evaluate_validation(
    model,
    val_query_rows: list[dict],
    val_gallery_rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    query_features = extract_features(model, val_query_rows, args.batch_size, args.num_workers, device)
    gallery_features = extract_features(model, val_gallery_rows, args.batch_size, args.num_workers, device)
    return compute_metrics(
        query_features,
        gallery_features,
        [identity(row) for row in val_query_rows],
        [identity(row) for row in val_gallery_rows],
    )


def main() -> int:
    args = parse_args()
    train_rows = read_csv(Path(args.train_csv))
    if not train_rows:
        logger.error("Empty training CSV")
        return 1

    identities = sorted({identity(row) for row in train_rows})
    val_query_path = Path(args.val_query)
    val_gallery_path = Path(args.val_gallery)
    val_query_rows = read_csv(val_query_path) if val_query_path.exists() else []
    val_gallery_rows = read_csv(val_gallery_path) if val_gallery_path.exists() else []
    use_validation = bool(val_query_rows and val_gallery_rows and args.eval_every > 0)
    label_to_index = {name: index for index, name in enumerate(identities)}
    device = torch.device(args.device)
    output_dir = Path(args.output_root)
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
        "eval_every": args.eval_every,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained": not args.no_pretrained,
        "val_query": args.val_query if use_validation else None,
        "val_gallery": args.val_gallery if use_validation else None,
        "num_val_query_images": len(val_query_rows),
        "num_val_gallery_images": len(val_gallery_rows),
    }
    (output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(json.dumps(metadata, indent=2))

    best_map = -1.0
    best_epoch = None
    stale_checks = 0
    val_history = []
    should_stop = False
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
                logger.info(
                    "epoch=%d/%d batch=%d/%d loss=%.4f acc=%.4f",
                    epoch, args.epochs, batch_index, len(loader),
                    total_loss / total, correct / total,
                )

        scheduler.step()
        save_checkpoint(output_dir / "model_last.pth", model, args, label_to_index)
        if epoch == args.epochs or epoch % 5 == 0:
            save_checkpoint(output_dir / f"model_epoch_{epoch:03d}.pth", model, args, label_to_index)

        if use_validation and (epoch % args.eval_every == 0 or epoch == args.epochs):
            logger.info("Evaluating validation at epoch %d...", epoch)
            metrics = evaluate_validation(model, val_query_rows, val_gallery_rows, args, device)
            metrics["epoch"] = epoch
            val_history.append(metrics)
            (output_dir / "val_history.json").write_text(json.dumps(val_history, indent=2), encoding="utf-8")
            logger.info(
                "val epoch=%d rank1=%.4f rank5=%.4f mAP=%.4f",
                epoch, metrics["rank1"], metrics["rank5"], metrics["mAP"],
            )
            if metrics["mAP"] > best_map + args.min_delta:
                best_map = metrics["mAP"]
                best_epoch = epoch
                stale_checks = 0
                save_checkpoint(output_dir / "model_best.pth", model, args, label_to_index)
                (output_dir / "best_val.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                logger.info("Saved new best checkpoint at epoch %d", epoch)
            else:
                stale_checks += 1
                logger.info(
                    "No validation mAP improvement >= %s. stale_checks=%d/%d",
                    args.min_delta, stale_checks, args.patience,
                )
                if args.patience > 0 and stale_checks >= args.patience:
                    should_stop = True
                    logger.info(
                        "Early stopping at epoch %d. Best epoch=%s, best mAP=%.4f",
                        epoch, best_epoch, best_map,
                    )

        if should_stop:
            break

    logger.info("Saved: %s", output_dir / "model_last.pth")
    if use_validation:
        logger.info("Best validation epoch: %s, mAP=%.4f", best_epoch, best_map)
        logger.info("Saved: %s", output_dir / "model_best.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
