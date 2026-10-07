"""Patient-independent five-fold training for the ZCHSound QC model."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from torch.utils.data import DataLoader

from qc_ordinal import PCGQualityDataset, ResNet18CORAL, ResNet18Softmax, coral_loss, ordinal_probabilities, quality_score


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, head="coral"):
    model.eval(); labels = []; probabilities = []; scores = []; names = []
    for features, target, filename in loader:
        logits = model(features.to(device, non_blocking=True))
        if head == "coral":
            probabilities.append(ordinal_probabilities(logits).cpu().numpy())
            scores.append(quality_score(logits).cpu().numpy())
        else:
            probability = torch.softmax(logits, dim=1)
            probabilities.append(probability.cpu().numpy())
            levels = torch.arange(4, device=device, dtype=probability.dtype)
            scores.append((probability * levels).sum(dim=1).div(3).cpu().numpy())
        labels.extend(target.numpy().tolist()); names.extend(filename)
    return np.asarray(labels), np.concatenate(probabilities), np.concatenate(scores), names


def ordinal_metrics(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    expected = probabilities @ np.arange(4)
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(labels, predicted, weights="quadratic")),
        "mae_expected": float(np.mean(np.abs(labels - expected))),
    }


def optimal_threshold(labels, scores):
    candidates = np.linspace(0.05, 0.95, 181)
    values = [f1_score(labels, scores >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def binary_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zch-manifest", type=Path, required=True)
    parser.add_argument("--cdhs-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crop-seconds", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head", choices=("coral", "softmax"), default="coral")
    parser.add_argument("--balanced-coral", action="store_true",
                        help="Balance positive/negative examples separately at each CORAL threshold.")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    zch_rows = read_rows(args.zch_manifest)
    fold = str(args.fold); val_fold = str((args.fold + 1) % 5)
    train_rows = [row for row in zch_rows if row["cv_fold"] not in {fold, val_fold}]
    val_rows = [row for row in zch_rows if row["cv_fold"] == val_fold]
    test_rows = [row for row in zch_rows if row["cv_fold"] == fold]
    loaders = {
        "train": DataLoader(PCGQualityDataset(train_rows, True, args.crop_seconds), args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
        "val": DataLoader(PCGQualityDataset(val_rows, False, args.crop_seconds), args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
        "test": DataLoader(PCGQualityDataset(test_rows, False, args.crop_seconds), args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0),
    }
    model = (ResNet18CORAL() if args.head == "coral" else ResNet18Softmax()).to(device)
    coral_pos_weight = None
    if args.balanced_coral:
        labels = np.asarray([int(row["quality_ordinal"]) for row in train_rows])
        positives = np.asarray([(labels > threshold).sum() for threshold in range(3)], dtype=np.float32)
        coral_pos_weight = torch.as_tensor((len(labels) - positives) / positives, device=device)
        print(json.dumps({"coral_pos_weight": coral_pos_weight.tolist()}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = {"macro_f1": -1.0}

    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for features, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device, non_blocking=True))
            target = target.to(device, non_blocking=True)
            loss = (coral_loss(logits, target, coral_pos_weight) if args.head == "coral"
                    else torch.nn.functional.cross_entropy(logits, target))
            loss.backward(); optimizer.step(); losses.append(loss.item())
        scheduler.step()
        labels, probabilities, _, _ = predict(model, loaders["val"], device, args.head)
        metrics = ordinal_metrics(labels, probabilities)
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), **metrics}), flush=True)
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, **metrics}
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "val": best}, args.output_dir / "best.pt")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_labels, test_probabilities, _, test_names = predict(model, loaders["test"], device, args.head)
    test_metrics = ordinal_metrics(test_labels, test_probabilities)
    val_labels, _, val_scores, _ = predict(model, loaders["val"], device, args.head)
    val_usable = (val_labels >= 2).astype(int)
    threshold = optimal_threshold(val_usable, val_scores)
    cdhs_rows = read_rows(args.cdhs_manifest)
    cdhs_loader = DataLoader(PCGQualityDataset(cdhs_rows, False, args.crop_seconds), args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    _, _, cdhs_scores, cdhs_names = predict(model, cdhs_loader, device, args.head)
    cdhs_labels = np.asarray([int(row["usable_high_quality"]) for row in cdhs_rows])
    report = {"fold": args.fold, "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}, "best_val": best, "internal_test": test_metrics, "external_cdhs": binary_metrics(cdhs_labels, cdhs_scores, threshold)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "cdhs_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["filename", "quality_score_q", "usable_high_quality"])
        writer.writerows(zip(cdhs_names, cdhs_scores, cdhs_labels))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
