"""
dataset.py
----------
PyTorch IterableDataset that streams fire-risk temporal sequences directly
from Google Earth Engine — no intermediate file storage required.

Each dataset item is:
    inputs : FloatTensor (T, C, H, W)   — T bi-weekly composites, C=10 bands
    label  : FloatTensor (1, H, W)      — binary fire mask for period T+1

The dataset supports:
  - Sliding-window sequences so each (point, year) pair yields multiple samples
  - Multi-worker DataLoader via worker_init_fn (each worker gets a shard)
  - Optional in-memory cache for a small repeated-use subset (dev / tuning)
  - Band-wise normalisation via pre-computed statistics

Usage
-----
    import ee
    from sampling import build_sample_points
    from dataset  import FireRiskDataset, get_dataloader

    ee.Initialize()

    years      = list(range(2015, 2026))
    train_pts, val_pts = build_sample_points(years)

    train_ds = FireRiskDataset(train_pts, sequence_length=6, patch_size=64)
    val_ds   = FireRiskDataset(val_pts,   sequence_length=6, patch_size=64)

    train_loader = get_dataloader(train_ds, num_workers=4, prefetch_factor=2)
    val_loader   = get_dataloader(val_ds,   num_workers=2, prefetch_factor=2)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import ee

from gee_utils import (
    BANDS, N_BANDS,
    build_composite,
    build_label,
    fetch_patch,
    get_biweekly_date_ranges,
)

# ── Types ─────────────────────────────────────────────────────────────────────

Point = Tuple[float, float, int]   # (lon, lat, year)
Sample = Tuple[torch.Tensor, torch.Tensor]   # (inputs, label)


# ── Default normalisation statistics ─────────────────────────────────────────
# Per-band (mean, std) computed over a representative Portugal sample.
# Replace with your own values once you have a first data pass.
# Order matches BANDS: Red, Green, Blue, SWIR1, SWIR2, NDVI, NDWI, LST, Elevation, Slope

DEFAULT_BAND_STATS: Dict[str, Tuple[float, float]] = {
    "Red":       (1200.0, 600.0),
    "Green":     (1100.0, 550.0),
    "Blue":      ( 900.0, 500.0),
    "SWIR1":     (1800.0, 700.0),
    "SWIR2":     (1400.0, 650.0),
    "NDVI":      (   0.4,   0.3),
    "NDWI":      (  -0.1,   0.3),
    "LST":       (  28.0,  10.0),
    "Elevation": ( 300.0, 250.0),
    "Slope":     (   8.0,   8.0),
}


def _make_norm_tensors(
    stats: Dict[str, Tuple[float, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) tensors of shape (C, 1, 1) for broadcasting."""
    means = torch.tensor([stats[b][0] for b in BANDS], dtype=torch.float32).view(-1, 1, 1)
    stds  = torch.tensor([stats[b][1] for b in BANDS], dtype=torch.float32).view(-1, 1, 1)
    return means, stds


# ── Dataset ───────────────────────────────────────────────────────────────────

class FireRiskDataset(IterableDataset):
    """
    Streaming dataset for fire-risk temporal prediction.

    Each (point, year) pair can yield multiple samples via a sliding window
    over the bi-weekly composites of that fire season.

    The sequence-to-one framing is:
        Input:  composites[i : i + sequence_length]         → (T, C, H, W)
        Label:  burned_mask[i + sequence_length]            → (1, H, W)

    GEE is called lazily inside __iter__; initialise ee before creating a
    DataLoader with num_workers > 0 by passing worker_init_fn=worker_init_fn.

    Args:
        sample_points:    List of (lon, lat, year) from sampling.py.
        sequence_length:  Number of bi-weekly composites per input (default 6).
        patch_size:       Spatial edge length in pixels (default 64).
        band_stats:       Per-band (mean, std) dict for normalisation.
                          Pass None to skip normalisation.
        shuffle:          Shuffle sample_points at the start of each epoch.
        cache_size:       If > 0, cache this many fetched samples in RAM for
                          fast reuse (useful for small debug datasets).
    """

    def __init__(
        self,
        sample_points: List[Point],
        sequence_length: int = 6,
        patch_size: int = 64,
        band_stats: Optional[Dict[str, Tuple[float, float]]] = DEFAULT_BAND_STATS,
        shuffle: bool = True,
        cache_size: int = 0,
    ) -> None:
        super().__init__()
        self.sample_points   = sample_points
        self.sequence_length = sequence_length
        self.patch_size      = patch_size
        self.shuffle         = shuffle
        self.cache_size      = cache_size

        # Normalisation
        if band_stats is not None:
            self._norm_mean, self._norm_std = _make_norm_tensors(band_stats)
        else:
            self._norm_mean = self._norm_std = None

        # In-memory cache
        self._cache: Dict[int, Sample] = {}

        # Pre-build date ranges for all unique years to avoid repeated GEE calls
        unique_years = {yr for _, _, yr in sample_points}
        self._date_ranges: Dict[int, List[Tuple[str, str]]] = {
            yr: get_biweekly_date_ranges(yr) for yr in unique_years
        }

    # Normalisation

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """z-score each band; x shape (T, C, H, W)."""
        if self._norm_mean is None:
            return x
        # mean/std broadcast over (T, C, H, W) — unsqueeze for T dim
        mean = self._norm_mean.unsqueeze(0)
        std  = self._norm_std.unsqueeze(0)
        return (x - mean) / (std + 1e-6)

    # Sliding-window windows for one (point, year) pair

    def _windows(
        self,
        lon: float,
        lat: float,
        year: int,
    ) -> List[Tuple[List[Tuple[str, str]], Tuple[str, str]]]:
        """
        Return all valid (input_ranges, label_range) sliding windows for the
        given year's fire season.

        A valid window requires sequence_length input periods plus 1 label
        period, so we need at least sequence_length + 1 date ranges.
        """
        ranges = self._date_ranges[year]
        n      = len(ranges)
        if n < self.sequence_length + 1:
            return []

        return [
            (ranges[i : i + self.sequence_length], ranges[i + self.sequence_length])
            for i in range(n - self.sequence_length)
        ]

    # Fetch one (inputs, label) pair from GEE

    def _fetch(
        self,
        lon: float,
        lat: float,
        input_ranges: List[Tuple[str, str]],
        label_range:  Tuple[str, str],
    ) -> Optional[Sample]:
        """
        Stream a temporal sequence and its label directly from GEE.

        Returns None on any GEE error so __iter__ can skip gracefully.
        """
        try:
            frames = []
            for start, end in input_ranges:
                composite = build_composite(start, end)
                patch     = fetch_patch(composite, lon, lat, self.patch_size)  # (H, W, C)
                frames.append(patch)

            label_img   = build_label(*label_range)
            label_patch = fetch_patch(label_img, lon, lat, self.patch_size)    # (H, W, 1)

            # Stack frames: (T, H, W, C) and  permute  (T, C, H, W)
            inputs_np = np.stack(frames, axis=0)                # (T, H, W, C)
            inputs    = torch.from_numpy(inputs_np).permute(0, 3, 1, 2).float()
            label     = torch.from_numpy(label_patch).permute(2, 0, 1).float()

            inputs = self._normalise(inputs)
            return inputs, label

        except Exception as exc:
            print(f"[dataset] Fetch failed ({lon:.4f}, {lat:.4f}, {year}): {exc}")
            return None

    # __iter__

    def __iter__(self):
        """
        Yield (inputs, label) tensors.

        When used with DataLoader(num_workers>0), each worker receives a
        non-overlapping shard of sample_points via worker_init_fn below.
        """
        points = list(self.sample_points)
        if self.shuffle:
            random.shuffle(points)

        cache_hits = 0

        for idx, (lon, lat, year) in enumerate(points):
            windows = self._windows(lon, lat, year)
            if not windows:
                continue

            # Pick one random window per point per epoch (sliding variety)
            input_ranges, label_range = random.choice(windows)

            # Try cache first
            if idx in self._cache:
                cache_hits += 1
                yield self._cache[idx]
                continue

            result = self._fetch(lon, lat, input_ranges, label_range)
            if result is None:
                continue

            # Populate cache if space remains
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[idx] = result

            yield result


# Worker initialiser

def worker_init_fn(worker_id: int) -> None:
    """
    Initialise GEE and shard the dataset inside each DataLoader worker.

    Pass this to DataLoader(worker_init_fn=worker_init_fn).

    Each worker calls ee.Initialize() independently (required — GEE sessions
    are not fork-safe) and receives a 1/num_workers slice of sample_points
    to avoid duplicate fetches.
    """
    ee.Initialize()

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    ds          = worker_info.dataset
    num_workers = worker_info.num_workers
    wid         = worker_info.id

    # Assign a contiguous shard of points to this worker
    total   = len(ds.sample_points)
    shard   = math.ceil(total / num_workers)
    start   = wid * shard
    end     = min(start + shard, total)

    ds.sample_points = ds.sample_points[start:end]


# DataLoader factory

def get_dataloader(
    dataset: FireRiskDataset,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    **kwargs,
) -> DataLoader:
    """
    Convenience wrapper that creates a DataLoader with the correct settings
    for GEE streaming.

    Multi-worker loading with prefetch_factor=2 keeps GPU utilisation high
    despite GEE latency (~1–5 s per patch fetch).

    Args:
        dataset:         A FireRiskDataset instance.
        num_workers:     Parallel GEE fetch threads (4 is a good default).
        prefetch_factor: Batches to prefetch per worker.
        **kwargs:        Additional arguments forwarded to DataLoader
                         (e.g. batch_size, pin_memory).

    Returns:
        Configured DataLoader.

    Example:
        >>> loader = get_dataloader(train_ds, num_workers=4, batch_size=8)
    """
    return DataLoader(
        dataset,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        **kwargs,
    )
