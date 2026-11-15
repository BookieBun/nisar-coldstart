"""Places to check: connected patches of flagged cells on the display grid.

Used by the app's "Places to check first" list. It reads only the stored display arrays; sizes and
positions are approximate because the display grid is a reprojection of the 100 m analysis grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label

COLUMNS = ["area_km2", "lat", "lon", "cells", "mean_score", "overlaps_perimeter", "other_fire"]


def flagged_places(flags: np.ndarray, labels: np.ndarray, score: np.ndarray,
                   bounds: list, shape: list, other_ids: np.ndarray | None = None,
                   other_names: dict[int, str] | None = None) -> pd.DataFrame:
    """One row per 8-connected patch of flagged cells, largest first.

    flags: bool (h, w); labels: int (h, w), 1 inside the reference perimeter; score: float (h, w);
    bounds: [[south, west], [north, east]] of the display grid; shape: [h, w].
    other_ids: optional uint8 (h, w), k > 0 where another mapped fire lies (scripts/other_fires.py);
    other_names: {k: fire name}. A patch's other_fire lists the fires it touches ("" if none).
    """
    comp, n = label(np.asarray(flags, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)
    flat = comp.ravel()
    cells = np.bincount(flat, minlength=n + 1)[1:].astype(float)
    rows, cols = np.indices(comp.shape)
    r_mean = np.bincount(flat, weights=rows.ravel(), minlength=n + 1)[1:] / cells
    c_mean = np.bincount(flat, weights=cols.ravel(), minlength=n + 1)[1:] / cells
    sc = np.where(np.isfinite(score), score, 0.0).astype(float)
    mean_score = np.bincount(flat, weights=sc.ravel(), minlength=n + 1)[1:] / cells
    inside = np.bincount(flat, weights=(np.asarray(labels) == 1).ravel().astype(float), minlength=n + 1)[1:] > 0
    (s, w), (nn, e) = bounds
    h, wd = shape
    lat = nn - (r_mean + 0.5) / h * (nn - s)
    lon = w + (c_mean + 0.5) / wd * (e - w)
    cell_km2 = ((nn - s) / h * 111.32) * ((e - w) / wd * 111.32 * np.cos(np.radians(lat)))
    other = [""] * n
    if other_ids is not None:
        ids = np.asarray(other_ids).ravel()
        hit = (flat > 0) & (ids > 0)
        names = other_names or {}
        found: dict[int, set] = {}
        for p, k in set(zip(flat[hit].tolist(), ids[hit].tolist())):
            found.setdefault(p, set()).add(names.get(k, f"fire {k}"))
        for p, v in found.items():
            other[p - 1] = "; ".join(sorted(v))
    df = pd.DataFrame({"area_km2": cells * cell_km2, "lat": lat, "lon": lon, "cells": cells.astype(int),
                       "mean_score": mean_score, "overlaps_perimeter": inside, "other_fire": other})
    return df.sort_values(["area_km2", "mean_score"], ascending=[False, False]).reset_index(drop=True)
