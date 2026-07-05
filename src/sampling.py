"""
sampling.py
-----------
Stratified spatial sampling of fire / non-fire points across Portugal.

Strategy
--------
1.  Fire points   — sampled directly from MCD64A1 burned pixels, so every
                    positive sample genuinely experienced a fire that year.
2.  Non-fire pts  — sampled from unburned pixels within Portugal, at a
                    configurable ratio relative to fire points (default 1:3).
3.  Spatial split — Portugal is divided into a regular grid of ~0.5° blocks
                    (~50 km per side). Blocks are randomly assigned to train
                    or validation (80/20), so fire-prone regions (central and
                    northern interior) are proportionally represented in both
                    splits rather than concentrated in one.

Each point is stored as (lon, lat, year) so the dataset can build the
correct temporal sequence for that location and year.
"""

import ee
import numpy as np
from typing import Dict, List, Set, Tuple

# Grid cell size in decimal degrees for the spatial block split.
# ~0.5° ≈ 55 km N–S and ~40 km E–W at Portugal's latitude.
# Produces ~60–70 occupied cells within Portugal; ~12–14 go to validation.
BLOCK_SIZE_DEG = 0.5

# Fraction of grid blocks held out for validation.
VAL_BLOCK_FRACTION = 0.20

Point = Tuple[float, float, int]   # (lon, lat, year)


# ROI

def get_portugal_geometry() -> ee.Geometry:
    """Return mainland Portugal boundary from FAO GAUL 2015."""
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    return countries.filter(ee.Filter.eq("ADM0_NAME", "Portugal")).geometry()


# Fire points

def sample_fire_points(
    years: List[int],
    n_per_year: int = 200,
    seed_offset: int = 0,
) -> List[Point]:
    """
    For each year, sample n_per_year points from pixels confirmed burned
    by MCD64A1 during the fire season (April–October).

    Uses stratifiedSample with explicit class targeting (class=1, i.e. burned)
    which is more reliable than .sample() on a selfMask()-ed image — the latter
    can silently return zero results when the masked image computation fails.

    Sampling at 500 m scale is much faster than 20 m and sufficient for
    choosing patch centres — the actual 20 m data is fetched later.

    Args:
        years:        List of years to sample (e.g. list(range(2015, 2026))).
        n_per_year:   Target number of fire sample points per year.
        seed_offset:  Added to the year to set GEE's random seed (reproducibility).

    Returns:
        List of (lon, lat, year) tuples.
    """
    portugal = get_portugal_geometry()
    points: List[Point] = []

    for year in years:
        # Binary fire mask: 1 = burned, 0 = unburned. unmask(0) ensures no
        # masked pixels — stratifiedSample requires a complete image.
        burned_binary = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filterDate(f"{year}-04-01", f"{year}-10-31")
            .select("BurnDate")
            .max()
            .gt(0)
            .unmask(0)
            .rename("fire")
            .toByte()
        )

        samples = burned_binary.stratifiedSample(
            numPoints=0,               # default: 0 from any unspecified class
            classBand="fire",
            region=portugal,
            scale=500,
            geometries=True,
            seed=year + seed_offset,
            classValues=[0, 1],
            classPoints=[0, n_per_year],
        )

        try:
            feats = samples.getInfo()["features"]
            # keep only confirmed fire pixels
            fire_feats = [f for f in feats if f["properties"].get("fire") == 1]
            if not fire_feats:
                print(f"[sampling] Warning — 0 fire features returned for {year}. "
                      f"Check MCD64A1 coverage for April–October {year}.")
            for feat in fire_feats:
                lon, lat = feat["geometry"]["coordinates"]
                points.append((lon, lat, year))
            print(f"  {year}: {len(fire_feats)} fire points sampled")
        except Exception as e:
            print(f"[sampling] Error — fire points for {year}: {e}")

    return points


# Non-fire points

def sample_nonfire_points(
    years: List[int],
    n_per_year: int = 600,
    seed_offset: int = 9999,
) -> List[Point]:
    """
    For each year, sample n_per_year points from pixels that were NOT burned
    during the fire season, drawn uniformly within Portugal.

    Burned pixels are excluded via masking to ensure clean negatives.

    Args:
        years:        List of years to sample.
        n_per_year:   Target number of non-fire sample points per year.
                      Default 600 gives a ~1:3 fire:non-fire ratio when
                      fire_per_year=200.
        seed_offset:  Offset added to year for the GEE random seed.

    Returns:
        List of (lon, lat, year) tuples.
    """
    portugal = get_portugal_geometry()
    points: List[Point] = []

    for year in years:
        burned_mask = (
            ee.ImageCollection("MODIS/061/MCD64A1")
            .filterDate(f"{year}-04-01", f"{year}-10-31")
            .select("BurnDate")
            .max()
            .gt(0)
            .unmask(0)
        )

        nonfire = burned_mask.eq(0).selfMask()

        samples = nonfire.sample(
            region=portugal,
            scale=500,
            numPixels=n_per_year,
            geometries=True,
            seed=year + seed_offset,
        )

        try:
            feats = samples.getInfo()["features"]
            for feat in feats:
                lon, lat = feat["geometry"]["coordinates"]
                points.append((lon, lat, year))
        except Exception as e:
            print(f"[sampling] Warning — non-fire points for {year} failed: {e}")

    return points


# Spatial block split

def _point_to_block(lon: float, lat: float, block_size: float) -> Tuple[int, int]:
    """Map a (lon, lat) coordinate to a grid block index (col, row)."""
    col = int(np.floor(lon / block_size))
    row = int(np.floor(lat / block_size))
    return (col, row)


def spatial_block_split(
    points: List[Point],
    val_fraction: float = VAL_BLOCK_FRACTION,
    block_size: float   = BLOCK_SIZE_DEG,
    seed: int           = 42,
) -> Tuple[List[Point], List[Point]]:
    """
    Split points into train and validation sets using a random spatial block
    assignment.

    Portugal is covered by a regular grid of `block_size` × `block_size`
    degree cells. Each occupied cell is randomly assigned to train or val at
    the ratio (1 - val_fraction) : val_fraction. All points within a cell
    go to the same split, which enforces spatial separation and prevents the
    leakage that a per-point random split would cause.

    Because blocks are chosen randomly rather than by a geographic axis, fire-
    prone regions (central and northern interior) appear in both splits
    proportionally to their coverage, avoiding the systematic imbalance that
    a latitude-threshold split would create.

    Args:
        points:       Combined list of (lon, lat, year) points.
        val_fraction: Proportion of grid blocks assigned to validation (~0.20).
        block_size:   Grid cell size in decimal degrees (default 0.5° ≈ 50 km).
        seed:         Random seed for reproducible block assignment.

    Returns:
        (train_points, val_points)
    """
    rng = np.random.default_rng(seed)

    # Identify all occupied grid blocks
    occupied_blocks: Set[Tuple[int, int]] = {
        _point_to_block(lon, lat, block_size) for lon, lat, _ in points
    }
    block_list = sorted(occupied_blocks)   # sort for determinism before shuffle

    # Randomly designate val blocks
    n_val = max(1, round(len(block_list) * val_fraction))
    val_indices   = set(rng.choice(len(block_list), size=n_val, replace=False))
    val_blocks: Set[Tuple[int, int]] = {block_list[i] for i in val_indices}

    train, val = [], []
    for point in points:
        lon, lat, _ = point
        block = _point_to_block(lon, lat, block_size)
        (val if block in val_blocks else train).append(point)

    return train, val


def build_sample_points(
    years: List[int],
    fire_per_year: int    = 200,
    nonfire_ratio: float  = 3.0,
    val_fraction: float   = VAL_BLOCK_FRACTION,
    block_size: float     = BLOCK_SIZE_DEG,
    split_seed: int       = 42,
    verbose: bool         = True,
) -> Tuple[List[Point], List[Point]]:
    """
    Build stratified train / val point lists for the given years.

    Calls GEE twice per year (once for fire pixels, once for non-fire pixels).
    Expect ~30–90 s per year depending on GEE load.

    Args:
        years:          Years to sample, e.g. list(range(2015, 2026)).
        fire_per_year:  Target fire-positive samples per year.
        nonfire_ratio:  non-fire : fire ratio (default 3.0 → 1:3).
        val_fraction:   Fraction of spatial blocks held out for validation.
        block_size:     Grid cell size in degrees for the spatial block split.
        split_seed:     Seed for reproducible block assignment.
        verbose:        Print summary statistics when True.

    Returns:
        (train_points, val_points) — each a list of (lon, lat, year) tuples.

    Example:
        >>> import ee
        >>> ee.Initialize()
        >>> years = list(range(2015, 2026))
        >>> train_pts, val_pts = build_sample_points(years)
    """
    nonfire_per_year = int(fire_per_year * nonfire_ratio)

    fire_pts    = sample_fire_points(years, fire_per_year)
    nonfire_pts = sample_nonfire_points(years, nonfire_per_year)
    all_pts     = fire_pts + nonfire_pts

    train, val = spatial_block_split(
        all_pts,
        val_fraction=val_fraction,
        block_size=block_size,
        seed=split_seed,
    )

    if verbose:
        n_blocks_total = len({_point_to_block(p[0], p[1], block_size) for p in all_pts})
        n_val_blocks   = len({_point_to_block(p[0], p[1], block_size) for p in val})
        print(f"Sampled  {len(fire_pts):>5} fire points    across {len(years)} years")
        print(f"Sampled  {len(nonfire_pts):>5} non-fire points across {len(years)} years")
        print(f"Grid     {n_blocks_total} occupied {block_size}° blocks  →  "
              f"{n_val_blocks} val blocks ({n_val_blocks/n_blocks_total:.0%})")
        print(f"Split →  {len(train):>5} train  |  {len(val):>4} val")

    return train, val