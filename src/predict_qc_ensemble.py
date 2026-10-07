"""Score recording manifests with a frozen segment-level QC ensemble.

The QC models are trained on Label Studio segments.  For a recording-level
downstream dataset, this script evaluates deterministic start/middle/end
windows, averages the five CV models per window, and then averages windows.
This avoids assigning a whole-recording label from an arbitrary centre crop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from qc_ordinal import (PCGQualityDataset, ResNet18CORAL, ResNet18Softmax,
                        quality_score)


def read_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = row.get("server_source_path") or row.get("source_path")
        if not source:
            raise ValueError(f"Missing source path for row: {row}")
        row.setdefault("filename", Path(source).name)
    return rows


def recording_windows(rows, crop_seconds: float, windows: int):
    if windows not in (1, 3):
        raise ValueError("windows must be 1 or 3")
    expanded = []
    for recording_index, row in enumerate(rows):
        source = row.get("server_source_path") or row["source_path"]
        info = torchaudio.info(source)
        duration = info.num_frames / info.sample_rate
        latest_start = max(0.0, duration - crop_seconds)
        starts = [latest_start / 2] if windows == 1 else sorted({0.0, latest_start / 2, latest_start})
        for window_index, start in enumerate(starts):
            window = dict(row)
            window.update({
                "start_s": start,
                "end_s": min(duration, start + crop_seconds),
                "_recording_index": recording_index,
                "_window_index": window_index,
            })
            expanded.append(window)
    return expanded


@torch.no_grad()
def predict(model, rows, device, batch_size, workers, head, crop_seconds):
    loader = DataLoader(PCGQualityDataset(rows, False, crop_seconds), batch_size=batch_size,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    model.eval(); values = []
    for features, _, _ in loader:
        logits = model(features.to(device, non_blocking=True))
        if head == "coral":
            values.append(quality_score(logits).cpu().numpy())
        else:
            probability = torch.softmax(logits, dim=1)
            levels = torch.arange(4, device=device, dtype=probability.dtype)
            values.append((probability * levels).sum(dim=1).div(3).cpu().numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qc-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", choices=("softmax", "coral"), default="softmax")
    parser.add_argument("--crop-seconds", type=float, default=4.208)
    parser.add_argument("--windows", type=int, choices=(1, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    window_rows = recording_windows(rows, args.crop_seconds, args.windows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_scores = []
    for fold in range(5):
        checkpoint = torch.load(args.qc_run_root / f"fold{fold}" / "best.pt",
                                map_location=device, weights_only=False)
        model = (ResNet18Softmax() if args.head == "softmax" else ResNet18CORAL()).to(device)
        model.load_state_dict(checkpoint["model"])
        fold_scores.append(predict(model, window_rows, device, args.batch_size, args.workers,
                                   args.head, args.crop_seconds))
    ensemble_window_scores = np.stack(fold_scores)
    window_mean = ensemble_window_scores.mean(axis=0)
    grouped = [[] for _ in rows]
    for window, score in zip(window_rows, window_mean):
        grouped[window["_recording_index"]].append(score)
    recording_mean = np.asarray([np.mean(scores) for scores in grouped])
    recording_std = np.asarray([np.std(scores) for scores in grouped])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) + ["quality_score_q", "quality_score_q_window_std",
                                  "quality_windows", "quality_model_head"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, mean, std, scores in zip(rows, recording_mean, recording_std, grouped):
            writer.writerow({**row, "quality_score_q": f"{mean:.8f}",
                             "quality_score_q_window_std": f"{std:.8f}",
                             "quality_windows": len(scores), "quality_model_head": args.head})


if __name__ == "__main__":
    main()
