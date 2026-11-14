"""Grid: aggregate native backscatter to 100 m cells, convert to dB, build masks, tiles and folds.

The cube written here is the single join point between the synthetic and real paths:
everything downstream reads only cube.npz and cube_meta.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .log import get_logger, stage

log = get_logger()


@dataclass
class Cube:
    hh_db: np.ndarray      # (T,H,W) float32, nan if invalid
    hv_db: np.ndarray      # (T,H,W) float32
    terrain: np.ndarray    # (H,W) float32
    valid: np.ndarray      # (H,W) bool
    tile_id: np.ndarray    # (H,W) int32
    fold: np.ndarray       # (H,W) int8
    meta: dict

    @property
    def dates(self) -> list[str]:
        return list(self.meta["dates"])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.valid.shape)


def aggregate_linear(x: np.ndarray, k: int, min_valid: float) -> np.ndarray:
    """Mean of k x k blocks in linear power. Non-finite and non-positive pixels are missing;
    a cell with fewer than min_valid * k^2 usable pixels becomes nan. Edges that do not fill a
    whole block are dropped."""
    if k < 1:
        raise ValueError(f"aggregation factor k must be >= 1, got {k}")
    x = np.asarray(x, dtype=np.float64)
    H, W = x.shape[0] // k, x.shape[1] // k
    if H == 0 or W == 0:
        raise ValueError(f"array {x.shape} is smaller than one {k}x{k} block")
    b = x[: H * k, : W * k].reshape(H, k, W, k)
    ok = np.isfinite(b) & (b > 0)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / n
    m[n < min_valid * k * k] = np.nan
    return m.astype(np.float32)


def to_db(x: np.ndarray) -> np.ndarray:
    """10*log10(x); nan where x is non-finite or <= 0."""
    x = np.asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    pos = np.isfinite(x) & (x > 0)
    out[pos] = 10.0 * np.log10(x[pos])
    return out


def assign_folds(H: int, W: int, tile_cells: int, n_folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Square tiles of tile_cells; each tile gets one fold from a seeded permutation of tile ids."""
    if tile_cells < 1:
        raise ValueError("tile_cells must be >= 1")
    ntc = -(-W // tile_cells)
    tile = ((np.arange(H) // tile_cells)[:, None] * ntc + (np.arange(W) // tile_cells)[None, :]).astype(np.int32)
    n_tiles = int(tile.max()) + 1
    if n_tiles < n_folds:
        raise ValueError(f"{n_tiles} tiles cannot fill {n_folds} folds; reduce grid.tile_km")
    perm = np.random.default_rng(seed).permutation(n_tiles)
    fold_of_tile = (perm % n_folds).astype(np.int8)
    return tile, fold_of_tile[tile]


def compute_masks(hh_db: np.ndarray, hv_db: np.ndarray, terrain: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict]:
    missing = ~(np.isfinite(hh_db).all(0) & np.isfinite(hv_db).all(0) & np.isfinite(terrain))
    with np.errstate(invalid="ignore"):
        water = (hv_db <= cfg.mask.water_hv_db_max).all(0) & ~missing
    lo, hi = np.nanpercentile(np.where(missing, np.nan, terrain), cfg.mask.terrain_pct)
    terr = ((terrain < lo) | (terrain > hi)) & ~missing & ~water
    valid = ~missing & ~water & ~terr
    counts = {"water": int(water.sum()), "terrain": int(terr.sum()),
              "missing": int(missing.sum()), "valid": int(valid.sum())}
    return valid, counts


def finalize_cube(cfg: Config, hh_db: np.ndarray, hv_db: np.ndarray, terrain: np.ndarray, dates: list[str],
                  crs_epsg: int, transform: tuple, native_spacing_m: float, k: int, scenes: list[str]) -> list[Path]:
    """Mask, tile, assert and write cube.npz + cube_meta.json. Shared by synthetic and real paths."""
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError(f"dates must be sorted and unique: {dates}")
    T, H, W = hh_db.shape
    if hv_db.shape != (T, H, W) or terrain.shape != (H, W) or len(dates) != T:
        raise ValueError("cube arrays disagree in shape")
    finite = np.concatenate([hh_db[np.isfinite(hh_db)], hv_db[np.isfinite(hv_db)]])
    in_range = np.mean((finite >= -40) & (finite <= 10)) if finite.size else 0.0
    if in_range < 0.99:
        raise ValueError(f"only {in_range:.1%} of dB values lie in [-40, 10]; expected >= 99%. "
                         "Are the inputs linear power?")

    valid, counts = compute_masks(hh_db, hv_db, terrain, cfg)
    frac = valid.mean()
    if frac < cfg.mask.min_valid_area_fraction:
        raise ValueError(f"valid fraction {frac:.2f} < mask.min_valid_area_fraction "
                         f"{cfg.mask.min_valid_area_fraction}; mask counts {counts}")
    tile_cells = int(round(cfg.grid.tile_km * 1000 / cfg.grid.target_cell_m))
    tile, fold = assign_folds(H, W, tile_cells, cfg.grid.n_folds, cfg.seed)

    out = cfg.interim
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cube.npz", hh_db=hh_db.astype(np.float32), hv_db=hv_db.astype(np.float32),
                        terrain=terrain.astype(np.float32), valid=valid, tile_id=tile, fold=fold)
    meta = {"dates": dates, "crs_epsg": int(crs_epsg), "transform": [float(v) for v in transform],
            "shape": [H, W], "cell_m": cfg.grid.target_cell_m, "native_spacing_m": float(native_spacing_m),
            "k": int(k), "scenes": scenes, "mask_counts": counts}
    (out / "cube_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("cube %dx%d, %d dates, valid %.1f%%, mask counts %s", H, W, T, 100 * frac, counts)
    return [out / "cube.npz", out / "cube_meta.json"]


def load_cube(cfg: Config) -> Cube:
    z = np.load(cfg.interim / "cube.npz")
    meta = json.loads((cfg.interim / "cube_meta.json").read_text())
    return Cube(z["hh_db"], z["hv_db"], z["terrain"], z["valid"], z["tile_id"], z["fold"], meta)


def build_cube(cfg: Config) -> None:
    """Real-site cube from GCOV HDF5 (Track B). Reads one date at a time, aggregates, discards."""
    cfg.require_scene_selection()
    from .io_gcov import build_real_cube  # imported lazily: needs h5py and scene manifest
    with stage("grid") as outputs:
        outputs += build_real_cube(cfg)
