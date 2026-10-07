"""Re-evaluate a saved Outcome checkpoint and export patient-level scores."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_outcome import (OutcomeDataset, OutcomeModel, metrics, read_rows,
                           threshold_from_validation)


@torch.no_grad()
def predict_patients(model, loader, device, infer_q, gamma, hard_q, threshold):
    grouped = defaultdict(list)
    model.eval()
    for waveform, label, patient, q, metadata in loader:
        probability = torch.softmax(model(waveform.to(device, non_blocking=True),
                                          metadata.to(device, non_blocking=True)), 1)[:, 1].cpu().numpy()
        for pid, target, score, quality in zip(patient, label.numpy(), probability, q.numpy()):
            grouped[pid].append((int(target), float(score), float(quality)))
    patient_ids, labels, scores = [], [], []
    for patient_id, records in grouped.items():
        retained = [item for item in records if item[2] >= threshold] if hard_q else records
        if not retained:
            retained = [max(records, key=lambda item: item[2])]
        weights = (np.asarray([max(1e-3, item[2]) ** gamma for item in retained])
                   if infer_q else np.ones(len(retained)))
        patient_ids.append(patient_id); labels.append(records[0][0])
        scores.append(float(np.average([item[1] for item in retained], weights=weights)))
    return patient_ids, np.asarray(labels), np.asarray(scores)


def write_predictions(path, patient_ids, labels, scores):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "outcome_binary", "outcome_score"])
        writer.writerows(zip(patient_ids, labels, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-mode", choices=("none", "hard", "train", "infer", "both"), default="none")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--quality-threshold", type=float, default=0.5170241594314575)
    parser.add_argument("--windows", choices=(1, 3), type=int, default=1,
                        help="Average start/middle/end 4-s windows when set to 3.")
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val",),
                        help="Evaluate validation only by default; test is requested explicitly after selection.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    split = {name: [row for row in rows if row["split"] == name] for name in args.splits}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OutcomeModel(args.multimodal).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True)["model"])
    infer_q, hard_q = args.quality_mode in {"infer", "both"}, args.quality_mode == "hard"
    results = {}
    fractions = (0.5,) if args.windows == 1 else (0.0, 0.5, 1.0)
    for name, part in split.items():
        window_predictions = []
        for fraction in fractions:
            loader = DataLoader(OutcomeDataset(part, False, args.multimodal, window_fraction=fraction),
                                batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                                persistent_workers=args.workers > 0)
            window_predictions.append(predict_patients(model, loader, device, infer_q, args.gamma,
                                                       hard_q, args.quality_threshold))
        patient_ids, labels, scores = window_predictions[0]
        if any(not np.array_equal(labels, candidate[1]) or patient_ids != candidate[0]
               for candidate in window_predictions[1:]):
            raise RuntimeError("Patient order changed between temporal windows")
        scores = np.mean([candidate[2] for candidate in window_predictions], axis=0)
        write_predictions(args.output_dir / f"{name}_patient_predictions.csv", patient_ids, labels, scores)
        results[name] = (labels, scores)
    cut = threshold_from_validation(*results["val"])
    report = {"multimodal": args.multimodal, "quality_mode": args.quality_mode, "windows": args.windows,
              "validation": metrics(*results["val"], cut)}
    if "test" in results:
        report["test"] = metrics(*results["test"], cut)
    (args.output_dir / "evaluation_report.json").write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
