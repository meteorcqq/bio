"""Audio-only, patient-level Challenge 2022 Outcome training and QC ablation."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pcg_audio import load_mono_segment
from train_ssl_byola2 import AudioNTT2022


METADATA_COLUMNS = (
    "sex_male", "pregnancy", "age_neonate", "age_infant", "age_child",
    "age_adolescent", "bmi_underweight", "bmi_healthy", "bmi_overweight",
    "bmi_obese",
)


class OutcomeDataset(Dataset):
    def __init__(self, rows, training: bool, multimodal: bool = False, segment_seconds: float = 4.0,
                 window_fraction: float | None = None):
        self.rows, self.training = list(rows), training
        self.segment_samples = int(segment_seconds * 4000)
        self.multimodal = multimodal
        self.window_fraction = window_fraction

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        metadata = ([float(row[column]) for column in METADATA_COLUMNS] if self.multimodal
                    else [])
        return (load_mono_segment(row["server_source_path"], self.segment_samples,
                                  random_start=self.training, start_fraction=self.window_fraction),
                int(row["outcome_binary"]), row["patient_id"], float(row.get("quality_score_q", 1.0)),
                torch.tensor(metadata, dtype=torch.float32))


class OutcomeModel(nn.Module):
    def __init__(self, multimodal: bool = False):
        super().__init__()
        self.multimodal = multimodal
        self.encoder = AudioNTT2022()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=4000, n_fft=512, win_length=512, hop_length=160,
                                                         n_mels=64, f_min=60, f_max=2000, power=2.0)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3072 + (len(METADATA_COLUMNS) if multimodal else 0), 2))

    def forward(self, waveform, metadata=None):
        mel = torch.log(self.mel(waveform).clamp_min(1e-8)).unsqueeze(1)
        mel = (mel - mel.mean(dim=(2, 3), keepdim=True)) / mel.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        features = self.encoder(mel)
        if self.multimodal:
            if metadata is None or metadata.shape[1] != len(METADATA_COLUMNS):
                raise ValueError("Multimodal OutcomeModel requires all 10 clinical metadata covariates.")
            features = torch.cat((features, metadata), dim=1)
        return self.head(features)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@torch.no_grad()
def patient_predictions(model, loader, device, infer_q: bool, gamma: float,
                        hard_q: bool = False, quality_threshold: float = 0.0):
    grouped = defaultdict(list)
    model.eval()
    for waveform, label, patient, q, metadata in loader:
        probability = torch.softmax(model(waveform.to(device, non_blocking=True),
                                          metadata.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy()
        for pid, target, score, quality in zip(patient, label.numpy(), probability, q.numpy()):
            grouped[pid].append((int(target), float(score), float(quality)))
    labels, scores = [], []
    for records in grouped.values():
        target = records[0][0]
        retained = [value for value in records if value[2] >= quality_threshold] if hard_q else records
        # Preserve every patient in the evaluation cohort.  A patient with no
        # above-threshold recording falls back to its most reliable recording.
        if not retained:
            retained = [max(records, key=lambda value: value[2])]
        weights = (np.asarray([max(1e-3, value[2]) ** gamma for value in retained])
                   if infer_q else np.ones(len(retained)))
        labels.append(target); scores.append(float(np.average([value[1] for value in retained], weights=weights)))
    return np.asarray(labels), np.asarray(scores)


def outcome_cost(labels, scores, threshold):
    """Official Challenge 2022 mean clinical-outcome cost for one cohort."""
    predicted = scores >= threshold
    tp = int(np.sum((predicted == 1) & (labels == 1)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    patients = len(labels); referred = tp + fp; ratio = referred / patients
    algorithm = 10 * patients
    expert = (25 + 397 * ratio - 1718 * ratio**2 + 11296 * ratio**4) * patients
    treatment = 10000 * tp
    error = 50000 * fn
    return float((algorithm + expert + treatment + error) / patients)


def threshold_from_validation(labels, scores):
    options = np.r_[np.unique(scores), np.nextafter(np.max(scores), np.inf)]
    return float(options[np.argmin([outcome_cost(labels, scores, item) for item in options])])


def metrics(labels, scores, threshold):
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {"auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)),
            "f1": float(f1_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "sensitivity": float(np.sum(predicted & positive) / np.sum(positive)),
            "specificity": float(np.sum(~predicted & negative) / np.sum(negative)),
            "challenge_cost": outcome_cost(labels, scores, threshold), "threshold": threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ssl-encoder", default="")
    parser.add_argument("--multimodal", action="store_true",
                        help="Concatenate the reference project's 10 clinical metadata covariates to the audio embedding.")
    parser.add_argument("--quality-mode", choices=("none", "hard", "train", "infer", "both"), default="none")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Power applied to q; the prespecified primary continuous-QC analysis uses 1.0.")
    parser.add_argument("--quality-threshold", type=float, default=0.5170241594314575,
                        help="Fixed QC hard-filter threshold, selected only on held-out ZCH validation folds.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--skip-test", action="store_true",
                        help="Train and select checkpoints from train/validation data only.")
    args = parser.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_rows(args.manifest)
    if args.multimodal and any(any(row.get(column, "") == "" for column in METADATA_COLUMNS) for row in rows):
        raise ValueError("A multimodal run requires all clinical metadata columns in the manifest.")
    if args.quality_mode != "none" and any(row.get("quality_score_q", "") == "" for row in rows):
        raise ValueError("A quality-aware run requires quality_score_q for every recording.")
    split_names = ("train", "val") if args.skip_test else ("train", "val", "test")
    split = {name: [row for row in rows if row["split"] == name] for name in split_names}
    if args.quality_mode == "hard":
        split["train"] = [row for row in split["train"]
                          if float(row["quality_score_q"]) >= args.quality_threshold]
        if not split["train"]:
            raise ValueError("Hard QC removed all training recordings.")
    loaders = {name: DataLoader(OutcomeDataset(part, name == "train", args.multimodal), batch_size=args.batch_size, shuffle=name == "train",
                                num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
               for name, part in split.items()}
    model = OutcomeModel(args.multimodal).to(device)
    if args.ssl_encoder:
        model.encoder.load_state_dict(torch.load(args.ssl_encoder, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = float("inf")
    train_q = args.quality_mode in {"train", "both"}; infer_q = args.quality_mode in {"infer", "both"}
    hard_q = args.quality_mode == "hard"
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for waveform, label, _, q, metadata in loaders["train"]:
            logits = model(waveform.to(device, non_blocking=True), metadata.to(device, non_blocking=True)); target = label.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(logits, target, reduction="none")
            if train_q:
                weights = 1e-3 + q.to(device, non_blocking=True).pow(args.gamma)
                loss = (loss * weights).sum() / weights.sum()
            else: loss = loss.mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        labels, scores = patient_predictions(model, loaders["val"], device, infer_q, args.gamma,
                                             hard_q, args.quality_threshold)
        threshold = threshold_from_validation(labels, scores)
        value = outcome_cost(labels, scores, threshold)
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": float(roc_auc_score(labels, scores)), "val_cost": value}), flush=True)
        if value < best:
            best = value; torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
    val_labels, val_scores = patient_predictions(model, loaders["val"], device, infer_q, args.gamma,
                                                 hard_q, args.quality_threshold)
    report = {"multimodal": args.multimodal, "metadata_columns": list(METADATA_COLUMNS) if args.multimodal else [],
              "quality_mode": args.quality_mode, "quality_threshold": args.quality_threshold if hard_q else None,
              "train_recordings": len(split["train"]), "best_val_cost": best,
              "validation": metrics(val_labels, val_scores, threshold_from_validation(val_labels, val_scores))}
    if not args.skip_test:
        test_labels, test_scores = patient_predictions(model, loaders["test"], device, infer_q, args.gamma,
                                                       hard_q, args.quality_threshold)
        report["test"] = metrics(test_labels, test_scores, threshold_from_validation(val_labels, val_scores))
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
