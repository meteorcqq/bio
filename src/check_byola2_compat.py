"""Report parameter-shape compatibility between an AudioNTT2022 weight and this encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_ssl_byola2 import AudioNTT2022


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=3072)
    args = parser.parse_args()
    weight = torch.load(args.weight, map_location="cpu", weights_only=True)
    model = AudioNTT2022(feature_dim=args.feature_dim)
    own = model.state_dict()
    matching = [key for key in weight if key in own and weight[key].shape == own[key].shape]
    mismatched = {key: {"weight": list(weight[key].shape), "model": list(own[key].shape)}
                  for key in weight if key in own and weight[key].shape != own[key].shape}
    print(json.dumps({"weight_tensors": len(weight), "model_tensors": len(own),
                      "matching_tensors": len(matching), "mismatched": mismatched,
                      "weight_only_keys": sorted(set(weight) - set(own)),
                      "model_only_keys": sorted(set(own) - set(weight))}, indent=2))


if __name__ == "__main__":
    main()
