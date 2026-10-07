"""Unified, source-aware audio loading for the PCG experiments.

The pretraining corpus contains conventional audio files and two WFDB sources.
This module deliberately keeps decoding separate from task-specific datasets so
that the exact same signal handling can be used for SSL, QC and Outcome tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import random

import numpy as np
import torch
import torchaudio
import wfdb


def _load_wfdb(path: Path) -> Tuple[torch.Tensor, int]:
    """Read a WFDB record and select its primary PCG channel when available."""
    record_path = str(path.with_suffix(""))
    signal, fields = wfdb.rdsamp(record_path)
    names = [str(name).upper() for name in fields.get("sig_name", [])]

    # EPHNOGRAM records contain ECG, PCG and PCG2; the exact PCG channel is the
    # clinically intended primary phonocardiogram. Fetal records are single-PCG.
    channel = next((i for i, name in enumerate(names) if name == "PCG"), None)
    if channel is None:
        channel = next((i for i, name in enumerate(names) if "PCG" in name), 0)

    waveform = torch.from_numpy(np.asarray(signal[:, channel], dtype=np.float32))
    return waveform, int(fields["fs"])


def _load_standard_audio(path: Path) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] == 0:
        raise ValueError(f"Unsupported audio shape {tuple(waveform.shape)}: {path}")
    return waveform.mean(dim=0).to(torch.float32), int(sample_rate)


def _normalize_and_resample(waveform: torch.Tensor, sample_rate: int, target_sample_rate: int, path: Path) -> torch.Tensor:
    waveform = waveform.flatten().contiguous()
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Empty or non-finite signal: {path}")
    waveform -= waveform.mean()
    peak = waveform.abs().max()
    if peak <= 0:
        raise ValueError(f"Constant signal: {path}")
    waveform /= peak
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
    return waveform.to(torch.float32)


def load_mono_audio(path: str | Path, target_sample_rate: int = 4000) -> torch.Tensor:
    """Decode an audio/WFDB PCG record, mono-convert and resample to target SR.

    Raises a descriptive exception for corrupt, silent or non-finite recordings;
    callers can record the failure without silently changing a cohort.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == ".hea":
            waveform, sample_rate = _load_wfdb(path)
        else:
            waveform, sample_rate = _load_standard_audio(path)
    except Exception as exc:
        raise RuntimeError(f"Decode failed: {path}") from exc

    return _normalize_and_resample(waveform, sample_rate, target_sample_rate, path)


def load_mono_segment(path: str | Path, segment_samples: int, target_sample_rate: int = 4000,
                      random_start: bool = True, start_fraction: float | None = None) -> torch.Tensor:
    """Load a random fixed-length segment without materializing long WFDB records."""
    if start_fraction is not None and not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be between 0 and 1")
    path = Path(path)
    if path.suffix.lower() != ".hea":
        waveform = load_mono_audio(path, target_sample_rate)
    else:
        record_path = str(path.with_suffix(""))
        try:
            header = wfdb.rdheader(record_path)
            names = [str(name).upper() for name in header.sig_name]
            channel = next((i for i, name in enumerate(names) if name == "PCG"), None)
            channel = channel if channel is not None else next((i for i, name in enumerate(names) if "PCG" in name), 0)
            native_samples = max(2, int(np.ceil(segment_samples * header.fs / target_sample_rate)))
            maximum_start = max(0, header.sig_len - native_samples)
            if start_fraction is not None:
                start = round(maximum_start * start_fraction)
            else:
                start = random.randint(0, maximum_start) if random_start else maximum_start // 2
            signal, fields = wfdb.rdsamp(record_path, sampfrom=start, sampto=min(header.sig_len, start + native_samples), channels=[channel])
            waveform = _normalize_and_resample(torch.from_numpy(np.asarray(signal[:, 0], dtype=np.float32)), int(fields["fs"]), target_sample_rate, path)
        except Exception as exc:
            raise RuntimeError(f"Segment decode failed: {path}") from exc
    if waveform.numel() < segment_samples:
        waveform = waveform.repeat(int(np.ceil(segment_samples / waveform.numel())))
    maximum_start = waveform.numel() - segment_samples
    if path.suffix.lower() == ".hea" and start_fraction is not None:
        # The requested temporal position was already applied while decoding.
        start = 0
    elif start_fraction is not None:
        start = round(maximum_start * start_fraction)
    else:
        start = random.randint(0, maximum_start) if random_start else maximum_start // 2
    return waveform[start:start + segment_samples]
