"""Four-level CORAL quality model and Log-Mel PCG dataset."""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio

from pcg_audio import load_mono_audio


class PCGQualityDataset(Dataset):
    def __init__(self, rows, training: bool, crop_seconds: float = 8.0, sample_rate: int = 4000):
        self.rows = list(rows)
        self.training = training
        self.crop_samples = int(crop_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=256, win_length=256, hop_length=64,
            f_min=20, f_max=2000, n_mels=64, power=2.0,
        )

    def __len__(self):
        return len(self.rows)

    def _crop(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.numel() < self.crop_samples:
            repeats = math.ceil(self.crop_samples / waveform.numel())
            waveform = waveform.repeat(repeats)
        if waveform.numel() > self.crop_samples:
            start = (random.randint(0, waveform.numel() - self.crop_samples)
                     if self.training else (waveform.numel() - self.crop_samples) // 2)
            waveform = waveform[start:start + self.crop_samples]
        return waveform

    def __getitem__(self, index):
        row = self.rows[index]
        source_path = row.get("server_source_path", row.get("source_path"))
        waveform = load_mono_audio(source_path, self.sample_rate)
        # Segment-level Label Studio annotations are stored as time boundaries
        # in the source waveform. Crop only inside the labelled interval.
        if row.get("start_s", "") != "" and row.get("end_s", "") != "":
            start = max(0, int(round(float(row["start_s"]) * self.sample_rate)))
            end = min(waveform.numel(), int(round(float(row["end_s"]) * self.sample_rate)))
            waveform = waveform[start:end]
        waveform = self._crop(waveform)
        feature = torch.log(self.mel(waveform).clamp_min(1e-8))
        feature = (feature - feature.mean()) / feature.std().clamp_min(1e-6)
        # CDHS is used only for binary external evaluation and therefore has no
        # four-level ZCH ordinal target; prediction still needs a placeholder.
        return feature.unsqueeze(0), int(row.get("quality_ordinal", -1)), row["filename"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels)
        ))

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class ResNet18CORAL(nn.Module):
    """ResNet-18 encoder with a monotonic shared-weight CORAL ordinal head."""
    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_channels = 64
        self.layer1 = self._layer(64, 2)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score = nn.Linear(512, 1)
        # Positive increments make theta_0 < theta_1 < theta_2, hence p(y>k)
        # is monotonic in k. The first threshold remains freely learnable.
        self.first_threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(num_levels - 2))
        self.num_levels = num_levels

    def _layer(self, channels, blocks, stride=1):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(self.in_channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        score = self.score(self.pool(x).flatten(1))
        if self.num_levels == 2:
            thresholds = self.first_threshold[None]
        else:
            increments = torch.nn.functional.softplus(self.threshold_deltas)
            thresholds = torch.cat((self.first_threshold[None], self.first_threshold + torch.cumsum(increments, 0)))
        return score - thresholds[None, :]


class ResNet18Softmax(ResNet18CORAL):
    """Architecture-matched four-class Softmax baseline for CORAL."""
    def __init__(self, num_levels: int = 4):
        super().__init__(num_levels=num_levels)
        self.classifier = nn.Linear(512, num_levels)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(self.pool(x).flatten(1))


def coral_targets(labels: torch.Tensor, num_levels: int = 4) -> torch.Tensor:
    thresholds = torch.arange(num_levels - 1, device=labels.device)
    return (labels[:, None] > thresholds[None, :]).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, coral_targets(labels, logits.shape[1] + 1), pos_weight=pos_weight
    )


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    higher = torch.sigmoid(logits)
    classes = [1.0 - higher[:, 0]]
    classes.extend(higher[:, idx - 1] - higher[:, idx] for idx in range(1, higher.shape[1]))
    classes.append(higher[:, -1])
    return torch.stack(classes, dim=1).clamp_min(0.0)


def quality_score(logits: torch.Tensor) -> torch.Tensor:
    probabilities = ordinal_probabilities(logits)
    levels = torch.arange(probabilities.shape[1], device=logits.device, dtype=probabilities.dtype)
    return (probabilities * levels).sum(dim=1) / (probabilities.shape[1] - 1)
