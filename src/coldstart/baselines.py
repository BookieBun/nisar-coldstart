"""Baselines B1-B3. All score the same cells as the model, on the same grid.

B1  log-ratio:            max_p |d_p|
B2  short-baseline z:     max_p |x_b - mean(hist)| / max(std(hist), floor), fixed-length same-regime history
B3  stratified shift:     max_p |d_p - median_c| / max((q95_c - q05_c)/2, floor)
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .features import FeatureSet, to_map
from .grid import Cube
from .log import get_logger
from .pairs import Pair

log = get_logger()


def score_b1(fs: FeatureSet, shape) -> np.ndarray:
    return to_map(np.maximum(np.abs(fs.y_hh), np.abs(fs.y_hv)), fs.idx, shape)


def score_b2(cube: Cube, pair: Pair, valid: np.ndarray, cfg: Config) -> np.ndarray:
    """All-nan when the pair has too little same-regime history; such pairs are left out of
    B2's calibration so every B2 score uses the same estimator (decision C4)."""
    hist = pair.b2_history_dates
    if not hist:
        log.info("B2: pair %s has fewer than %d same-regime history dates; not scored",
                 pair.id, cfg.baselines.b2_history_dates)
        return np.full(cube.shape, np.nan, dtype=np.float32)
    ih = [cube.dates.index(d) for d in hist]
    ib = cube.dates.index(pair.date_b)
    floor = cfg.baselines.b2_std_floor_db
    score = np.zeros(cube.shape, dtype=np.float64)
    for arr in (cube.hh_db, cube.hv_db):
        h = arr[ih].astype(np.float64)
        sd = np.maximum(h.std(axis=0, ddof=1), floor)
        score = np.maximum(score, np.abs(arr[ib] - h.mean(axis=0)) / sd)
    out = score.astype(np.float32)
    ok = valid & np.isfinite(out)
    out[~ok] = np.nan
    return out


def score_b3(fs: FeatureSet, shape, cfg: Config) -> np.ndarray:
    cls = fs.X[:, fs.feature_names.index("cls")].astype(int)
    floor = cfg.baselines.b3_spread_floor_db
    lo_q, _, hi_q = cfg.model.quantiles
    score = np.zeros(len(cls), dtype=np.float64)
    for y in (fs.y_hh, fs.y_hv):
        r = np.empty(len(cls), dtype=np.float64)
        for c in np.unique(cls):
            m = cls == c
            q_lo, med, q_hi = np.quantile(y[m], [lo_q, 0.5, hi_q])
            r[m] = np.abs(y[m] - med) / max((q_hi - q_lo) / 2.0, floor)
        score = np.maximum(score, r)
    return to_map(score.astype(np.float32), fs.idx, shape)
