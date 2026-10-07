"""Manifest-driven BYOL-A2 pretraining for the 16-source PCG corpus.

The script intentionally starts from random initialization.  It uses the
shared 4 kHz loader so WAV, MP3 and WFDB records enter the same pipeline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pcg_audio import load_mono_segment


class AudioNTT2022(nn.Module):
    """The AudioNTT encoder used by BYOL-A2, with time-invariant pooling."""
    def __init__(self, n_mels: int = 64, feature_dim: int = 3072):
        super().__init__()
        channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2),
        )
        conv_dim = channels * (n_mels // 4)
        self.frame_mlp = nn.Sequential(
            nn.Linear(conv_dim, 2048), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, feature_dim - conv_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).permute(0, 3, 2, 1)
        batch, frames, mel, channels = x.shape
        x = x.reshape(batch, frames, mel * channels)
        learned = self.frame_mlp(x)
        x = torch.cat((x, learned), dim=-1)
        return x.mean(dim=1) + x.max(dim=1).values


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BYOLA2(nn.Module):
    def __init__(self, feature_dim: int = 3072, projection_dim: int = 256, ema: float = 0.99):
        super().__init__()
        self.online_encoder = AudioNTT2022(feature_dim=feature_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.online_projector = MLP(feature_dim, projection_dim)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = MLP(projection_dim, projection_dim)
        self.ema = ema
        for parameter in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            parameter.requires_grad_(False)

    @staticmethod
    def _loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2 - 2 * (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(dim=-1)

    def forward(self, view_one: torch.Tensor, view_two: torch.Tensor) -> torch.Tensor:
        online_one = self.predictor(self.online_projector(self.online_encoder(view_one)))
        online_two = self.predictor(self.online_projector(self.online_encoder(view_two)))
        with torch.no_grad():
            target_one = self.target_projector(self.target_encoder(view_one))
            target_two = self.target_projector(self.target_encoder(view_two))
        return (self._loss(online_one, target_two) + self._loss(online_two, target_one)).mean()

    @torch.no_grad()
    def update_target(self) -> None:
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(self.ema).add_(online.data, alpha=1 - self.ema)
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(self.ema).add_(online.data, alpha=1 - self.ema)


class PCGCorpus(Dataset):
    def __init__(self, rows: list[dict[str, str]], segment_seconds: float, sample_rate: int = 4000):
        self.rows = rows
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return load_mono_segment(self.rows[index]["server_source_path"], self.segment_samples, self.sample_rate)


class SourceAwareViews(nn.Module):
    """Clinical-source staged augmentation: A0 crop, then A1–A4 in sequence."""
    def __init__(self, stage: int, sample_rate: int = 4000):
        super().__init__()
        self.stage = stage
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=512, hop_length=160,
            n_mels=64, f_min=60, f_max=2000, power=2.0,
        )

    def _device_effects(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, length = waveform.shape
        gain = torch.empty(batch, 1, device=waveform.device).uniform_(0.7, 1.3)
        waveform = waveform * gain
        if torch.rand((), device=waveform.device) < 0.5:
            temporary_rate = int(torch.randint(3000, 3901, (), device=waveform.device))
            waveform = torchaudio.functional.resample(waveform, self.sample_rate, temporary_rate)
            waveform = torchaudio.functional.resample(waveform, temporary_rate, self.sample_rate)[..., :length]
            if waveform.shape[-1] < length:
                waveform = F.pad(waveform, (0, length - waveform.shape[-1]))
        quantize = torch.rand(batch, 1, device=waveform.device) < 0.3
        bits = torch.randint(7, 11, (batch, 1), device=waveform.device)
        levels = (2 ** bits - 1).float()
        quantized = torch.round(waveform.clamp(-1, 1) * levels) / levels
        return torch.where(quantize, quantized, waveform)

    def _time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """A0: independent temporal alignment changes for the two BYOL views."""
        maximum = max(1, int(0.5 * self.sample_rate))
        return torch.stack([
            torch.roll(item, int(torch.randint(-maximum, maximum + 1, (), device=item.device)))
            for item in waveform
        ])

    def _artifact_effects(self, waveform: torch.Tensor) -> torch.Tensor:
        batch, length = waveform.shape
        scale = torch.empty(batch, 1, device=waveform.device).uniform_(0.003, 0.04)
        white = torch.randn_like(waveform)
        coloured = F.avg_pool1d(white.unsqueeze(1), 65, 1, 32).squeeze(1)
        waveform = waveform + scale * (0.5 * white + coloured / coloured.std(dim=1, keepdim=True).clamp_min(1e-6))
        for index in range(batch):
            if torch.rand((), device=waveform.device) < 0.2:
                width = int(torch.randint(max(8, length // 200), max(9, length // 40), (), device=waveform.device))
                start = int(torch.randint(0, max(1, length - width), (), device=waveform.device))
                waveform[index, start:start + width] += torch.randn(width, device=waveform.device) * 0.12
        return waveform

    def _log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        feature = torch.log(self.mel(waveform).clamp_min(1e-8))
        return (feature - feature.mean(dim=(1, 2), keepdim=True)) / feature.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _smooth_eq(feature: torch.Tensor) -> torch.Tensor:
        batch, bands, frames = feature.shape
        anchors = torch.empty(batch, 1, 5, device=feature.device).uniform_(-0.35, 0.35)
        curve = F.interpolate(anchors, size=bands, mode="linear", align_corners=True).squeeze(1)
        return feature + curve.unsqueeze(-1)

    @staticmethod
    def _time_stretch(feature: torch.Tensor) -> torch.Tensor:
        batch, bands, frames = feature.shape
        output = []
        for index in range(batch):
            rate = float(torch.empty((), device=feature.device).uniform_(0.9, 1.1))
            stretched = F.interpolate(feature[index:index + 1].unsqueeze(1), size=(bands, max(2, int(frames / rate))), mode="bilinear", align_corners=False).squeeze(1)
            if stretched.shape[-1] >= frames:
                start = (stretched.shape[-1] - frames) // 2
                output.append(stretched[..., start:start + frames])
            else:
                output.append(F.pad(stretched, (0, frames - stretched.shape[-1])))
        return torch.cat(output, dim=0)

    @staticmethod
    def _spec_mask(feature: torch.Tensor) -> torch.Tensor:
        batch, bands, frames = feature.shape
        for index in range(batch):
            if torch.rand((), device=feature.device) < 0.5:
                width = int(torch.randint(1, 9, (), device=feature.device))
                start = int(torch.randint(0, max(1, bands - width), (), device=feature.device))
                feature[index, start:start + width] = 0
            if torch.rand((), device=feature.device) < 0.5:
                width = int(torch.randint(1, max(2, frames // 12), (), device=feature.device))
                start = int(torch.randint(0, max(1, frames - width), (), device=feature.device))
                feature[index, :, start:start + width] = 0
        return feature

    @staticmethod
    def _pcg_mix(feature: torch.Tensor, permutation: torch.Tensor, alpha: torch.Tensor,
                 apply_mask: torch.Tensor) -> torch.Tensor:
        mixed = (1 - alpha) * feature + alpha * feature[permutation]
        return torch.where(apply_mask, mixed, feature)

    def one_view(self, waveform: torch.Tensor, pcg_mix_params=None) -> torch.Tensor:
        waveform = self._time_shift(waveform)
        if self.stage >= 1:
            waveform = self._device_effects(waveform)
        if self.stage >= 2:
            waveform = self._artifact_effects(waveform)
        feature = self._log_mel(waveform)
        if self.stage >= 1:
            feature = self._smooth_eq(feature)
        if self.stage >= 3:
            feature = self._time_stretch(feature)
        if self.stage >= 4:
            feature = self._spec_mask(feature)
            feature = self._pcg_mix(feature, *pcg_mix_params)
        return feature.unsqueeze(1)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Both BYOL views must preserve the same cross-patient pairing. Only
        # the view-specific recording, artifact, temporal and mask transforms
        # differ; otherwise PCGMix would form different positive-pair targets.
        batch = waveform.shape[0]
        pcg_mix_params = None
        if self.stage >= 4:
            pcg_mix_params = (
                torch.randperm(batch, device=waveform.device),
                torch.empty(batch, 1, 1, device=waveform.device).uniform_(0.03, 0.10),
                (torch.rand(batch, 1, 1, device=waveform.device) < 0.20),
            )
        return (self.one_view(waveform.clone(), pcg_mix_params),
                self.one_view(waveform.clone(), pcg_mix_params))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(5), default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--source-sampling", choices=("uniform", "sqrt", "quarter", "none"), default="sqrt",
                        help="Source-balanced weights n^-1, n^-0.5, n^-0.25, or unweighted recording shuffle.")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    if not rows or not all(row.get("server_source_path") for row in rows):
        raise ValueError("Manifest is empty or has unresolved server_source_path values")
    source_count = Counter(row["dataset_id"] for row in rows)
    samples = args.samples_per_epoch or len(rows)
    if args.source_sampling == "none":
        if args.samples_per_epoch:
            raise ValueError("Unweighted sampling traverses each recording once; omit --samples-per-epoch")
        sampler, shuffle = None, True
    else:
        exponent = {"uniform": 1.0, "sqrt": 0.5, "quarter": 0.25}[args.source_sampling]
        weights = torch.DoubleTensor([source_count[row["dataset_id"]] ** -exponent for row in rows])
        sampler, shuffle = WeightedRandomSampler(weights, num_samples=samples, replacement=True), False
    loader = DataLoader(PCGCorpus(rows, args.segment_seconds), batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, drop_last=True)
    model = BYOLA2().to(device)
    views = SourceAwareViews(args.stage).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # FP32 is deliberate here: a long run exposed occasional AMP overflow in
    # the BYOL projection branch, while the 49-GB GPU has ample memory.
    scaler = GradScaler(enabled=False)
    metadata = {"args": vars(args), "records": len(rows), "source_records": source_count}
    (args.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train(); start = time.time(); losses = []
            for waveform in loader:
                waveform = waveform.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=False):
                    view_one, view_two = views(waveform)
                    loss = model(view_one, view_two)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}")
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); model.update_target()
                losses.append(float(loss.detach()))
            scheduler.step()
            result = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.time() - start, "lr": optimizer.param_groups[0]["lr"]}
            log.write(json.dumps(result) + "\n"); log.flush(); print(json.dumps(result), flush=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}, args.run_dir / "last.pt")
    torch.save(model.online_encoder.state_dict(), args.run_dir / "encoder_final.pt")


if __name__ == "__main__":
    main()
