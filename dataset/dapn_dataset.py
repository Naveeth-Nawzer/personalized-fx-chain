"""
dataset/dapn_dataset.py
=======================
PyTorch Dataset and DataLoader for synthetic DAPN audio FX chain training.

Loads:
  - dry_audio: (1, T) float32
  - wet_audio: (1, T) float32
  - presence_multihot: (7,) float32 binary label
  - param_targets: (29,) float32 continuous normalized targets in [0, 1]
  - param_mask: (29,) float32 binary mask indicating which parameters are active
"""

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader

from audio_io import load_audio

ROOT = Path(__file__).resolve().parent.parent

# Ordered list of 7 effects
SUPPORTED_EFFECTS = [
    "EQ",
    "Compressor",
    "Delay",
    "Reverb",
    "Distortion",
    "Chorus",
    "Flanger",
]

# Canonical 29-parameter schema across all 7 DAPN effects
PARAM_SCHEMA: dict[str, list[str]] = {
    "EQ": ["frequency", "gain", "q"],
    "Compressor": ["threshold", "ratio", "attack", "release", "makeup_gain"],
    "Delay": ["delay_time", "feedback", "wet_mix", "dry_mix"],
    "Reverb": ["room_size", "damping", "wet_mix", "dry_mix", "decay"],
    "Distortion": ["drive", "tone", "mix", "output_gain"],
    "Chorus": ["rate", "depth", "delay", "mix"],
    "Flanger": ["rate", "depth", "feedback", "mix"],
}

# Pre-compute parameter start/end offsets in the 29-dim vector
PARAM_OFFSETS: dict[str, tuple[int, int]] = {}
_offset = 0
for fx, params in PARAM_SCHEMA.items():
    PARAM_OFFSETS[fx] = (_offset, _offset + len(params))
    _offset += len(params)
TOTAL_NUM_PARAMS = _offset  # 29


class DAPNSyntheticDataset(Dataset):
    """PyTorch Dataset loading synthetic dry/wet audio pairs with effect labels."""

    def __init__(self, metadata_path: str | Path, max_samples: int | None = None) -> None:
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.records: list[dict[str, Any]] = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    if max_samples and len(self.records) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.records[idx]
        dry_file = ROOT / item["dry_path"]
        wet_file = ROOT / item["wet_path"]

        dry_audio, _ = load_audio(dry_file)
        wet_audio, _ = load_audio(wet_file)

        # 1. Multi-hot effect presence: shape (7,)
        presence = torch.tensor(item["ml_targets"]["presence_multihot"], dtype=torch.float32)

        # 2. Continuous parameter vector: shape (29,)
        # Mask: shape (29,) indicating which parameters were present in this sample
        param_vector = torch.zeros(TOTAL_NUM_PARAMS, dtype=torch.float32)
        param_mask = torch.zeros(TOTAL_NUM_PARAMS, dtype=torch.float32)

        for step in item.get("effects", []):
            fx_name = step["effect"]
            if fx_name in PARAM_SCHEMA:
                start_idx, end_idx = PARAM_OFFSETS[fx_name]
                param_names = PARAM_SCHEMA[fx_name]
                norm_dict = step.get("parameters_normalized", {})

                for i, p_name in enumerate(param_names):
                    if p_name in norm_dict:
                        param_vector[start_idx + i] = float(norm_dict[p_name])
                        param_mask[start_idx + i] = 1.0

        return {
            "sample_id": item["sample_id"],
            "dry_audio": dry_audio,
            "wet_audio": wet_audio,
            "presence_multihot": presence,
            "param_targets": param_vector,
            "param_mask": param_mask,
            "chain_sequence": item["chain_sequence"],
            "chain_length": item["chain_length"],
        }


def dapn_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collator that stacks tensors and keeps variable-length sequences as lists."""
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "dry_audio": torch.stack([item["dry_audio"] for item in batch]),
        "wet_audio": torch.stack([item["wet_audio"] for item in batch]),
        "presence_multihot": torch.stack([item["presence_multihot"] for item in batch]),
        "param_targets": torch.stack([item["param_targets"] for item in batch]),
        "param_mask": torch.stack([item["param_mask"] for item in batch]),
        "chain_sequence": [item["chain_sequence"] for item in batch],
        "chain_length": [item["chain_length"] for item in batch],
    }


def get_dataloader(
    metadata_path: str | Path,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Helper to instantiate a PyTorch DataLoader."""
    dataset = DAPNSyntheticDataset(metadata_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dapn_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
