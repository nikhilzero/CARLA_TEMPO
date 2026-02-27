"""
temporal_dataset.py — Dataset wrapper that yields temporal windows of frames.

Wraps any single-frame CARLA/LMDrive dataset and groups consecutive frames
into windows of size T for temporal training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class TemporalWindowDataset(Dataset):
    """
    Wraps a single-frame dataset and returns windows of T consecutive frames.

    The underlying dataset must:
      - Support integer indexing (dataset[i])
      - Return a dict with at least: 'front', 'left', 'right', 'rear', 'waypoints'
      - Items from the same route/episode must be stored consecutively.

    Args:
        base_dataset: Single-frame dataset (e.g., LMDrive dataset).
        num_frames (int): Temporal window size T.
        frame_stride (int): Step between sampled frames (1 = every frame).
        episode_lengths (List[int]): Number of frames per episode.
            If None, assumes all frames form a single episode.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        num_frames: int = 4,
        frame_stride: int = 1,
        episode_lengths: List[int] = None,
    ):
        self.base_dataset = base_dataset
        self.num_frames = num_frames
        self.frame_stride = frame_stride

        total = len(base_dataset)
        if episode_lengths is None:
            episode_lengths = [total]

        # Build list of valid window start indices
        # A window [i, i+stride, ..., i+(T-1)*stride] is valid if all indices
        # fall within the same episode.
        self.windows: List[List[int]] = []
        episode_start = 0
        for ep_len in episode_lengths:
            ep_end = episode_start + ep_len
            window_end = ep_end - (num_frames - 1) * frame_stride
            for start in range(episode_start, window_end):
                indices = [start + t * frame_stride for t in range(num_frames)]
                self.windows.append(indices)
            episode_start = ep_end

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        indices = self.windows[idx]

        frames = []
        for i in indices:
            item = self.base_dataset[i]
            frames.append({
                "front": item["front"],
                "left":  item["left"],
                "right": item["right"],
                "rear":  item["rear"],
            })

        # Labels come from the LAST (most current) frame
        last_item = self.base_dataset[indices[-1]]
        return {
            "frames": frames,                          # list of T dicts
            "waypoints": last_item["waypoints"],       # supervision target
            "command": last_item.get("command", None),
            "measurements": last_item.get("measurements", None),
        }


def collate_temporal(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate for temporal windows.
    Returns:
        frames: list of T dicts, each with stacked tensors (B, C, H, W)
        waypoints: (B, N_wp, 2)
        command: (B,) or None
        measurements: (B, M) or None
    """
    T = len(batch[0]["frames"])
    frames = []
    for t in range(T):
        frame_t = {
            key: torch.stack([b["frames"][t][key] for b in batch])
            for key in batch[0]["frames"][0].keys()
        }
        frames.append(frame_t)

    waypoints = torch.stack([b["waypoints"] for b in batch])

    command = None
    if batch[0]["command"] is not None:
        command = torch.stack([b["command"] for b in batch])

    measurements = None
    if batch[0]["measurements"] is not None:
        measurements = torch.stack([b["measurements"] for b in batch])

    return {
        "frames": frames,
        "waypoints": waypoints,
        "command": command,
        "measurements": measurements,
    }
