#!/usr/bin/env python3
"""Train WICV-Net: weather-invariant cross-view vehicle Re-ID.

Total objective:
    L = L_id + w_tri * L_cv-triplet + w_cvpa * L_cvpa + w_adv * (L_time + L_weather)

where L_id is label-smoothed identity cross-entropy on the BNNeck feature,
L_cv-triplet is the cross-view batch-hard triplet, L_cvpa is the cross-view
prototype alignment InfoNCE, and the adversarial terms are gradient-reversed
time (morning/evening) and weather (norain/rain) classifiers.

Ablation flags: --no-adv --no-cvpa --no-triplet --plain-triplet
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CrossViewIdentitySampler, ReidTrainDataset, identity, read_csv
from losses import CrossViewPrototypeMemory, cross_view_batch_hard_triplet
from metrics import evaluate_retrieval
from model import WICVNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/train.csv")
    parser.add_argument("--val-query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_gallery.csv")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--output-dir", default="results/wicv/osnet_x1_0_full")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--num-ids-per-batch", type=int, default=16)
    parser.add_argument("--instances-per-id", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--w-tri", type=float, default=1.0)
    parser.add_argument("--w-adv", type=float, default=0.5)
    parser.add_argument("--w-cvpa", type=float, default=0.5)
    parser.add_argument("--proto-momentum", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-adv", action="store_true", help="Disable condition-adversarial heads.")
    parser.add_argument("--no-cvpa", action="store_true", help="Disable cross-view prototype alignment.")
    parser.add_argument("--no-triplet", action="store_true", help="Disable the triplet loss entirely.")
    parser.add_argument(
        "--plain-triplet",
        action="store_true",
        help="Use standard batch-hard triplet instead of cross-view mining.",
    )
    return parser.parse_args()


def build_train_transform(height: int, width: int):
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ]
    )


def save_checkpoint(path: Path, model: WICVNet, args: argparse.Namespace, label_to_index: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "num_classes": len(label_to_index),
            "height": args.height,
            "width": args.width,
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_rows = read_csv(Path(args.train_csv))
    if not train_rows:
        print("Empty training CSV", file=sys.stderr)
        return 1

    identities = sorted({identity(row) for row in train_rows})
    label_to_index = {name: index for index, name in enumerate(identities)}
    val_query_path = Path(args.val_query)
    val_gallery_path = Path(args.val_gallery)
    val_query_rows = read_csv(val_query_path) if val_query_path.exists() else []
    val_gallery_rows = read_csv(val_gallery_path) if val_gallery_path.exists() else []
    use_validation = bool(val_query_rows and val_gallery_rows and args.eval_every > 0)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReidTrainDataset(train_rows, label_to_index, build_train_transform(args.height, args.width))
    sampler = CrossViewIdentitySampler(
        train_rows,
        label_to_index,
        num_ids=args.num_ids_per_batch,
        num_instances=args.instances_per_id,
        seed=args.seed,
    )
    batch_size = args.num_ids_per_batch * args.instances_per_id
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = WICVNet(
        args.model_name,
        num_classes=len(label_to_index),
        pretrained=not args.no_pretrained,
        height=args.height,
        width=args.width,
    ).to(device)
    memory = CrossViewPrototypeMemory(
        num_ids=len(label_to_index),
        feat_dim=model.feat_dim,
        momentum=args.proto_momentum,
        temperature=args.temperature,
    ).to(device)

    id_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    adv_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metadata = {
        "method": "wicv",
        "model_name": args.model_name,
        "feat_dim": model.feat_dim,
        "num_train_images": len(train_rows),
        "num_train_identities": len(label_to_index),
        "batch_size": batch_size,
        "ablation": {
            "adv": not args.no_adv,
            "cvpa": not args.no_cvpa,
            "triplet": not args.no_triplet,
            "cross_view_mining": not args.plain_triplet,
        },
        "args": vars(args),
    }
    (output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    best_map = -1.0
    best_epoch = None
    stale_checks = 0
    val_history = []
    should_stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = (epoch - 1) / max(1, args.epochs - 1)
        grl_weight = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        totals = {"loss": 0.0, "id": 0.0, "tri": 0.0, "cvpa": 0.0, "adv": 0.0}
        correct = 0
        total = 0

        for batch_index, (images, labels, views, times, weathers) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            views = views.to(device)
            times = times.to(device)
            weathers = weathers.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, grl_weight=grl_weight)

            id_loss = id_criterion(outputs["id_logits"], labels)
            loss = id_loss

            tri_loss = images.new_zeros(())
            if not args.no_triplet:
                if args.plain_triplet:
                    # same views tensor for every sample disables cross-view mining
                    tri_views = torch.zeros_like(views)
                else:
                    tri_views = views
                tri_loss = cross_view_batch_hard_triplet(
                    outputs["features"], labels, tri_views, margin=args.margin
                )
                loss = loss + args.w_tri * tri_loss

            cvpa_loss = images.new_zeros(())
            if not args.no_cvpa:
                cvpa_loss = memory.loss(outputs["features"], labels, views)
                loss = loss + args.w_cvpa * cvpa_loss

            adv_loss = images.new_zeros(())
            if not args.no_adv:
                adv_loss = adv_criterion(outputs["time_logits"], times) + adv_criterion(
                    outputs["weather_logits"], weathers
                )
                loss = loss + args.w_adv * adv_loss

            loss.backward()
            optimizer.step()

            if not args.no_cvpa:
                memory.update(outputs["features"].detach(), labels, views)

            count = labels.size(0)
            totals["loss"] += loss.item() * count
            totals["id"] += id_loss.item() * count
            totals["tri"] += float(tri_loss) * count
            totals["cvpa"] += float(cvpa_loss) * count
            totals["adv"] += float(adv_loss) * count
            predictions = torch.argmax(outputs["id_logits"], dim=1)
            correct += (predictions == labels).sum().item()
            total += count

            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_index}/{len(loader)} "
                    f"loss={totals['loss'] / total:.4f} id={totals['id'] / total:.4f} "
                    f"tri={totals['tri'] / total:.4f} cvpa={totals['cvpa'] / total:.4f} "
                    f"adv={totals['adv'] / total:.4f} acc={correct / total:.4f} "
                    f"grl={grl_weight:.3f}",
                    flush=True,
                )

        scheduler.step()
        save_checkpoint(output_dir / "model_last.pth", model, args, label_to_index)

        if use_validation and (epoch % args.eval_every == 0 or epoch == args.epochs):
            print(f"Evaluating validation at epoch {epoch}...", flush=True)
            metrics = evaluate_retrieval(
                model,
                val_query_rows,
                val_gallery_rows,
                batch_size,
                args.num_workers,
                device,
                args.height,
                args.width,
            )
            metrics["epoch"] = epoch
            val_history.append(metrics)
            (output_dir / "val_history.json").write_text(json.dumps(val_history, indent=2), encoding="utf-8")
            print(
                f"val epoch={epoch} rank1={metrics['rank1']:.4f} "
                f"rank5={metrics['rank5']:.4f} mAP={metrics['mAP']:.4f}",
                flush=True,
            )
            if metrics["mAP"] > best_map + args.min_delta:
                best_map = metrics["mAP"]
                best_epoch = epoch
                stale_checks = 0
                save_checkpoint(output_dir / "model_best.pth", model, args, label_to_index)
                (output_dir / "best_val.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                print(f"Saved new best checkpoint at epoch {epoch}", flush=True)
            else:
                stale_checks += 1
                print(
                    f"No validation mAP improvement >= {args.min_delta}. "
                    f"stale_checks={stale_checks}/{args.patience}",
                    flush=True,
                )
                if args.patience > 0 and stale_checks >= args.patience:
                    should_stop = True
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch={best_epoch}, best mAP={best_map:.4f}",
                        flush=True,
                    )

        if should_stop:
            break

    print(f"Saved: {output_dir / 'model_last.pth'}", flush=True)
    if use_validation:
        print(f"Best validation epoch: {best_epoch}, mAP={best_map:.4f}", flush=True)
        print(f"Saved: {output_dir / 'model_best.pth'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
