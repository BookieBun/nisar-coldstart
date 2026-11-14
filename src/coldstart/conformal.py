"""Split-conformal thresholds from no-event pairs, and the leave-one-pair-out H3 check.

Threshold for target false-alarm rate alpha from n calibration scores sorted ascending:
    t_alpha = S_(ceil((n+1)(1-alpha)))      (1-based; +inf if the index exceeds n)
A cell is flagged when its score is strictly greater than t_alpha. If unchanged cells of the
test pair are exchangeable with the calibration cells, the flag rate among them is <= alpha.

H3 is judged on the RAW realized rate, which is what the guarantee covers (decision C3).
The rate after the minimum-patch filter is reported beside it as the operational number.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from .config import Config
from .log import get_logger, require_inputs, stage
from .pairs import load_pairs
from .postprocess import remove_small_patches

log = get_logger()

DETECTORS = ["B1", "B2", "B3", "M"]


def threshold(cal_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64)[np.isfinite(cal_scores)])
    n = len(s)
    i = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if i > n else float(s[i - 1])


def pvalues(scores: np.ndarray, cal_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value (1 + #{cal >= s}) / (n + 1)."""
    cal = np.sort(np.asarray(cal_scores, dtype=np.float64)[np.isfinite(cal_scores)])
    n = len(cal)
    ge = n - np.searchsorted(cal, np.asarray(scores, dtype=np.float64), side="left")
    return (1.0 + ge) / (n + 1.0)


def calibration_sample(score_map: np.ndarray, max_cells: int, seed: int, salt: int) -> np.ndarray:
    v = score_map[np.isfinite(score_map)]
    if len(v) > max_cells:
        rng = np.random.default_rng([seed, salt])
        v = v[np.sort(rng.choice(len(v), size=max_cells, replace=False))]
    return v.astype(np.float64)


def realized_rates(score_map: np.ndarray, thr: float, min_patch: int) -> tuple[float, float, int]:
    ok = np.isfinite(score_map)
    flag = np.zeros(score_map.shape, dtype=bool)
    flag[ok] = score_map[ok] > thr
    n = int(ok.sum())
    raw = flag[ok].mean() if n else float("nan")
    filt = remove_small_patches(flag, min_patch)[ok].mean() if n else float("nan")
    return float(raw), float(filt), n


def h3_pass(realized: float, alpha: float, cfg: Config) -> bool:
    lo, hi = cfg.calibration.h3_band
    return bool(lo * alpha <= realized <= hi * alpha)


def leave_one_pair_out(scores_by_pair: dict[str, np.ndarray], alphas: list[float], cfg: Config) -> pd.DataFrame:
    """For each pair j: thresholds from the other pairs, realized rates on j."""
    ids = sorted(scores_by_pair)
    samples = {pid: calibration_sample(scores_by_pair[pid], cfg.calibration.max_cal_cells_per_pair, cfg.seed, i)
               for i, pid in enumerate(ids)}
    rows = []
    for j in ids:
        cal = np.concatenate([samples[i] for i in ids if i != j])
        for a in alphas:
            t = threshold(cal, a)
            raw, filt, n = realized_rates(scores_by_pair[j], t, cfg.postprocess.min_patch_cells)
            rows.append({"left_out": j, "alpha": a, "threshold": t, "realized": raw,
                         "realized_filtered": filt, "n_cells": n,
                         "pass": h3_pass(raw, a, cfg) if a in cfg.calibration.h3_alphas else None})
    return pd.DataFrame(rows)


def _json_num(x: float):
    return None if not np.isfinite(x) else float(x)


def run_calibration(cfg: Config) -> None:
    inter = cfg.interim
    require_inputs("calibrate", inter / "pairs.json", inter / "scores")
    with stage("calibrate") as outputs:
        pairs = load_pairs(inter / "pairs.json")
        no_event = [p for p in pairs if not p.is_event]
        alphas = list(cfg.calibration.alphas)
        out = {"mode": None, "alphas": alphas, "thresholds": {}, "lopo": [], "pairs_used": {}}
        modes = set()
        for det in DETECTORS:
            maps = {}
            for p in no_event:
                f = inter / "scores" / det / f"{p.id}.npy"
                require_inputs("calibrate", f)
                m = np.load(f)
                if np.isfinite(m).any():
                    maps[p.id] = m
            if len(maps) >= 2:
                modes.add("pairs")
                ids = sorted(maps)
                pooled = np.concatenate([calibration_sample(maps[pid], cfg.calibration.max_cal_cells_per_pair,
                                                            cfg.seed, i) for i, pid in enumerate(ids)])
                out["thresholds"][det] = {str(a): _json_num(threshold(pooled, a)) for a in alphas}
                lopo = leave_one_pair_out(maps, alphas, cfg)
                lopo.insert(0, "detector", det)
                out["lopo"] += [{k: (_json_num(v) if isinstance(v, float) else v) for k, v in r.items()}
                                for r in lopo.to_dict("records")]
                out["pairs_used"][det] = ids
            else:
                modes.add("spatial")
                out["thresholds"][det], rows = _spatial_fallback(cfg, det, pairs, alphas)
                out["lopo"] += rows
                out["pairs_used"][det] = []
        out["mode"] = "pairs" if modes == {"pairs"} else ("spatial" if modes == {"spatial"} else "mixed")
        (inter / "calibration.json").write_text(json.dumps(out, indent=2))
        outputs.append(inter / "calibration.json")
        for det in DETECTORS:
            log.info("%s thresholds: %s", det, out["thresholds"][det])


def _spatial_fallback(cfg: Config, det: str, pairs, alphas):
    """Handoff 6.6 fallback when fewer than two no-event pairs exist: label-0 cells of the event
    pair, thresholds from folds 0-2, realized rate on folds 3-4. NOTE: this uses the negative
    labels, so the perimeter reaches the threshold; it is reported as mode 'spatial' and the
    README must say so."""
    log.warning("%s: fewer than two usable no-event pairs; SPATIAL calibration fallback (uses labels)", det)
    from .pairs import event_pair
    inter = cfg.interim
    require_inputs("calibrate", inter / "labels.npy", inter / "cube.npz")
    ev = event_pair(pairs)
    score = np.load(inter / "scores" / det / f"{ev.id}.npy")
    labels = np.load(inter / "labels.npy")
    fold = np.load(inter / "cube.npz")["fold"]
    neg = (labels == 0) & np.isfinite(score)
    cal = score[neg & (fold <= 2)]
    hold = np.where(neg & (fold >= 3), score, np.nan)
    thr, rows = {}, []
    for a in alphas:
        t = threshold(cal, a)
        thr[str(a)] = _json_num(t)
        raw, filt, n = realized_rates(hold, t, cfg.postprocess.min_patch_cells)
        rows.append({"detector": det, "left_out": "spatial_folds_3_4", "alpha": a, "threshold": _json_num(t),
                     "realized": raw, "realized_filtered": filt, "n_cells": n,
                     "pass": h3_pass(raw, a, cfg) if a in cfg.calibration.h3_alphas else None})
    return thr, rows


def load_thresholds(path) -> dict[str, dict[float, float]]:
    d = json.loads(open(path).read())
    return {det: {float(a): (float("inf") if t is None else t) for a, t in v.items()}
            for det, v in d["thresholds"].items()}
