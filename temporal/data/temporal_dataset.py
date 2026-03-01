"""
temporal_dataset.py — Temporal window wrapper around CarlaMVDetDataset.

Wraps the baseline CarlaMVDetDataset and groups consecutive frames into
windows of size T without crossing route boundaries.

CarlaMVDetDataset.__getitem__ returns (data_dict, target_tuple):
  data_dict:    {"rgb", "rgb_left", "rgb_right", "rgb_center", "lidar",
                  "measurements", "command", "target_point", ...}
  target_tuple: 7-element tuple of tensors used by the baseline loss functions

TemporalWindowDataset returns (frames, target) where:
  frames: list of T data_dicts (oldest → most recent)
  target: target_tuple from the LAST (most recent) frame

collate_temporal handles batching for the DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Any


def episode_lengths_from_carla_dataset(ds) -> List[int]:
    """
    Derive per-episode frame counts from CarlaMVDetDataset.route_frames.

    route_frames is a flat list of (route_dir, frame_id) pairs where
    all frames for one route are stored consecutively.

    Returns a list of frame counts, one per route.
    """
    if not hasattr(ds, "route_frames") or not ds.route_frames:
        return [len(ds)]

    lengths = []
    current_route = ds.route_frames[0][0]
    count = 0
    for route_dir, _ in ds.route_frames:
        if route_dir == current_route:
            count += 1
        else:
            lengths.append(count)
            current_route = route_dir
            count = 1
    lengths.append(count)
    return lengths


class TemporalWindowDataset(Dataset):
    """
    Wraps CarlaMVDetDataset and returns temporal windows of T consecutive frames.

    Windows are built so they never cross route (episode) boundaries.

    Args:
        base_dataset:     CarlaMVDetDataset instance (with transforms already set).
        num_frames:       Temporal window size T.
        frame_stride:     Step between sampled frames within a window.
        episode_lengths:  Frame counts per episode. If None, inferred from
                          base_dataset.route_frames (if available), else single
                          episode assumed.
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

        if episode_lengths is None:
            episode_lengths = episode_lengths_from_carla_dataset(base_dataset)

        # Build list of valid window start indices.
        # Window [start, start+stride, ..., start+(T-1)*stride] is valid when
        # all indices fall within the same episode.
        self.windows: List[List[int]] = []
        episode_start = 0
        for ep_len in episode_lengths:
            ep_end = episode_start + ep_len
            # Last valid start ensures all T frames stay within this episode
            window_end = ep_end - (num_frames - 1) * frame_stride
            for start in range(episode_start, window_end):
                indices = [start + t * frame_stride for t in range(num_frames)]
                self.windows.append(indices)
            episode_start = ep_end

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[List[Any], Any]:
        """
        Returns:
            frames: list of T data_dicts (oldest → most recent)
            target: target_tuple from the last frame (for loss computation)
        """
        indices = self.windows[idx]

        frames = []
        last_target = None
        for i in indices:
            data, target = self.base_dataset[i]
            frames.append(data)
            last_target = target  # overwritten each iter; final value = last frame

        return frames, last_target


def collate_temporal(batch: List[Tuple]) -> Tuple:
    """
    Collate a batch of (frames_list, target_tuple) items.

    Args:
        batch: list of B items, each item is (frames_list, target_tuple)
               frames_list: list of T data_dicts
               target_tuple: 7-element tuple of tensors (from CarlaMVDetDataset)

    Returns:
        inputs: list of T dicts, each dict maps key → (B, ...) tensor
        target: tuple of (B, ...) tensors, identical structure to baseline target
    """
    frames_batch = [item[0] for item in batch]   # B × [T × dict]
    targets_batch = [item[1] for item in batch]  # B × tuple

    T = len(frames_batch[0])
    B = len(batch)

    # Collate T frames across the batch dimension
    inputs = []
    for t in range(T):
        frame_t_items = [frames_batch[b][t] for b in range(B)]
        collated_frame = {}
        for key in frame_t_items[0]:
            vals = [item[key] for item in frame_t_items]
            if isinstance(vals[0], torch.Tensor):
                collated_frame[key] = torch.stack(vals)
            elif isinstance(vals[0], np.ndarray):
                collated_frame[key] = torch.stack([torch.from_numpy(v) for v in vals])
            elif isinstance(vals[0], (int, float, bool)):
                collated_frame[key] = torch.tensor(vals)
            else:
                collated_frame[key] = vals  # metadata: keep as list
        inputs.append(collated_frame)

    # Collate the target tuple element-wise across the batch
    n_targets = len(targets_batch[0])
    target = tuple(
        torch.stack([targets_batch[b][i] for b in range(B)])
        if isinstance(targets_batch[0][i], torch.Tensor)
        else torch.stack([torch.from_numpy(targets_batch[b][i]) for b in range(B)])
        if isinstance(targets_batch[0][i], np.ndarray)
        else [targets_batch[b][i] for b in range(B)]
        for i in range(n_targets)
    )

    return inputs, target
