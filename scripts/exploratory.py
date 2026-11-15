"""Exploratory analyses on a scored site. Run AFTER the frozen test; none of this is part of H1/H3.

1. Reproducibility: recompute M on the event pair and compare with the stored score.
2. Ablations of M on the event pair: single polarization, no land class, no texture,
   no cross-fitting, and a model trained on a no-event pair (the "burn learned as normal" check).
3. Sensitivity: tile size and number of folds.
4. Robustness: added noise and randomly missing cells, for all four detectors.
5. Failure analysis: the largest false-alarm patches of M, and what each detector does inside
   the other 2026 fire perimeters in the window (excluded from the labels).

Usage:
    python scripts/exploratory.py config/site.yaml
Writes artifacts/<site>/exploratory.json and artifacts/<site>/false_alarm_patches.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label
from shapely.geometry import mapping, shape

from coldstart.baselines import score_b1, score_b2, score_b3
from coldstart.config import load_config
from coldstart.conformal import load_thresholds
from coldstart.evaluate import _Sorted, block_bootstrap, flags_at
from coldstart.features import FeatureSet, build_features, load_features, to_map
from coldstart.grid import Cube, assign_folds, load_cube
from coldstart.model import _regressor, crossfit_quantiles
from coldstart.pairs import event_pair, load_pairs

T0 = time.perf_counter()


def say(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:6.0f}s] {msg}", flush=True)


def score_from_preds(fs: FeatureSet, preds: dict, cfg, pols=("hh", "hv")) -> np.ndarray:
    s = np.zeros(len(fs.idx))
    for p in pols:
        y = fs.y_hh if p == "hh" else fs.y_hv
        q = preds[p]
        s = np.maximum(s, np.abs((y - q[:, 1]) / np.maximum((q[:, 2] - q[:, 0]) / 2.0, cfg.model.spread_floor_db)))
    return s.astype(np.float32)


def metrics(score_map: np.ndarray, labels: np.ndarray, cfg) -> dict:
    ok = np.isin(labels, (0, 1)) & np.isfinite(score_map)
    st = _Sorted(score_map[ok], labels[ok] == 1).stats(None, cfg.evaluate.recall_target, cfg.evaluate.far_target)
    st.pop("_curve", None)
    return {k: round(float(v), 5) for k, v in st.items()}


def drop_feature(fs: FeatureSet, name: str) -> FeatureSet:
    keep = [i for i, n in enumerate(fs.feature_names) if n != name]
    return replace(fs, X=fs.X[:, keep], feature_names=[fs.feature_names[i] for i in keep], cat_mask=fs.cat_mask[keep])


def fit_predict(train: FeatureSet, test: FeatureSet, cfg, salt: int) -> dict:
    """One model per polarization and quantile, fitted on a sample of `train`, applied to `test`."""
    rng = np.random.default_rng([cfg.seed, salt])
    rows = np.arange(len(train.idx))
    if len(rows) > cfg.model.max_train_cells:
        rows = np.sort(rng.choice(rows, size=cfg.model.max_train_cells, replace=False))
    out = {}
    for p, y in (("hh", train.y_hh), ("hv", train.y_hv)):
        q = np.column_stack([_regressor(t, train.cat_mask, cfg).fit(train.X[rows], y[rows]).predict(test.X)
                             for t in cfg.model.quantiles])
        q.sort(axis=1)
        out[p] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    inter = cfg.interim
    cube = load_cube(cfg)
    shape_ = cube.shape
    pairs = load_pairs(inter / "pairs.json")
    ev = event_pair(pairs)
    labels = np.load(inter / "labels.npy")
    cls = np.load(inter / "cls.npy")
    fs = load_features(inter / "features" / f"{ev.id}.npz")
    stored = {d: np.load(inter / "scores" / d / f"{ev.id}.npy") for d in ("B1", "B2", "B3", "M")}
    out: dict = {"site": cfg.site, "event_pair": ev.id, "note": "exploratory; run after the frozen test"}

    # ---- 1. reproducibility of M on real data
    preds = crossfit_quantiles(fs, cfg)
    m_map = to_map(score_from_preds(fs, preds, cfg), fs.idx, shape_)
    same = bool(np.array_equal(m_map, stored["M"], equal_nan=True))
    out["reproducibility"] = {"m_score_identical_to_stored": same,
                              "max_abs_diff": float(np.nanmax(np.abs(m_map - stored["M"])))}
    say(f"recomputed M identical to stored: {same}")

    # ---- 2. ablations and 3. sensitivity (all on the frozen event pair and labels)
    variants: dict[str, np.ndarray] = {"M": m_map}
    variants["M, HV only"] = to_map(score_from_preds(fs, preds, cfg, ("hv",)), fs.idx, shape_)
    variants["M, HH only"] = to_map(score_from_preds(fs, preds, cfg, ("hh",)), fs.idx, shape_)
    for name, feat in (("M, no land class", "cls"), ("M, no texture", "tex0")):
        f2 = drop_feature(fs, feat)
        variants[name] = to_map(score_from_preds(f2, crossfit_quantiles(f2, cfg), cfg), fs.idx, shape_)
        say(name)
    variants["M, no cross-fitting"] = to_map(score_from_preds(fs, fit_predict(fs, fs, cfg, 1), cfg), fs.idx, shape_)
    say("M, no cross-fitting")
    pre = [p for p in pairs if p.kind == "no_event_pre"][-1]
    fs_pre = load_features(inter / "features" / f"{pre.id}.npz")
    variants[f"M, trained on no-event pair {pre.id}"] = to_map(
        score_from_preds(fs, fit_predict(fs_pre, fs, cfg, 2), cfg), fs.idx, shape_)
    say("M, trained on a no-event pair")
    for name, tile_km, n_folds in (("M, 5 km tiles", 5, cfg.grid.n_folds), ("M, 20 km tiles", 20, cfg.grid.n_folds),
                                   ("M, 3 folds", cfg.grid.tile_km, 3)):
        _, fold = assign_folds(*shape_, int(round(tile_km * 1000 / cfg.grid.target_cell_m)), n_folds, cfg.seed)
        f2 = replace(fs, fold=fold.ravel()[fs.idx].astype(np.int8))
        variants[name] = to_map(score_from_preds(f2, crossfit_quantiles(f2, cfg), cfg), fs.idx, shape_)
        say(name)
    variants["B3"] = stored["B3"]

    dummy = {k: np.zeros(shape_, bool) for k in variants}
    boot = block_bootstrap(variants, labels, cfg, dummy)
    wide = boot.pivot_table(index="rep", columns=["detector", "metric"], values="value", dropna=False)
    abl = {}
    for name, sc in variants.items():
        row = metrics(sc, labels, cfg)
        if name != "M":
            for met in ("far_at_recall50", "ap"):
                v = (wide[(name, met)] - wide[("M", met)]).to_numpy(float)
                v = v[np.isfinite(v)]
                row[f"{met}_minus_M_ci95"] = [round(float(np.percentile(v, 2.5)), 5), round(float(np.percentile(v, 97.5)), 5)]
        abl[name] = row
    out["ablations"] = abl
    say("ablation bootstrap done")

    # ---- 4. robustness: perturb the cube, rebuild features, score all four detectors
    def score_all(cube_p: Cube, lab: np.ndarray) -> dict:
        f = build_features(cube_p, ev, cls, cfg)
        cells = np.zeros(shape_[0] * shape_[1], bool)
        cells[f.idx] = True
        sc = {"B1": score_b1(f, shape_), "B2": score_b2(cube_p, ev, cells.reshape(shape_), cfg),
              "B3": score_b3(f, shape_, cfg),
              "M": to_map(score_from_preds(f, crossfit_quantiles(f, cfg), cfg), f.idx, shape_)}
        return {d: metrics(s, lab, cfg) for d, s in sc.items()}

    rob = {"unperturbed": {d: metrics(stored[d], labels, cfg) for d in stored}}
    for sd in (0.5, 1.0):
        rng = np.random.default_rng([cfg.seed, int(sd * 10)])
        cp = replace(cube, hh_db=cube.hh_db + rng.normal(0, sd, cube.hh_db.shape).astype(np.float32),
                     hv_db=cube.hv_db + rng.normal(0, sd, cube.hv_db.shape).astype(np.float32))
        rob[f"noise {sd} dB on every date"] = score_all(cp, labels)
        say(f"noise {sd} dB")
    rng = np.random.default_rng([cfg.seed, 77])
    gone = cube.valid & (rng.random(shape_) < 0.2)
    hh, hv = cube.hh_db.copy(), cube.hv_db.copy()
    hh[:, gone] = np.nan
    hv[:, gone] = np.nan
    lab_m = labels.copy()
    lab_m[gone] = -1
    rob["20% of cells missing at random"] = score_all(replace(cube, hh_db=hh, hv_db=hv, valid=cube.valid & ~gone), lab_m)
    say("missing 20%")
    out["robustness"] = rob

    # ---- 5a. largest false-alarm patches of M at the reported alpha
    thr = load_thresholds(inter / "calibration.json")
    ra = float(cfg.calibration.report_alpha)
    flags = {d: flags_at(stored[d], thr[d][ra], cfg.postprocess.min_patch_cells) for d in stored}
    fa = flags["M"] & (labels == 0)
    lab_cc, n_cc = cc_label(fa, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab_cc.ravel(), minlength=n_cc + 1)
    sizes[0] = 0
    order = np.argsort(-sizes)[:20]
    tr = Affine(*cube.meta["transform"])
    to_ll = Transformer.from_crs(cube.meta["crs_epsg"], 4326, always_xy=True)
    dist_km = distance_transform_edt(labels != 1) * cube.meta["cell_m"] / 1000.0
    ia, ib = cube.dates.index(ev.date_a), cube.dates.index(ev.date_b)
    rows = []
    for rank, k in enumerate(order, 1):
        if sizes[k] == 0:
            break
        m = lab_cc == k
        rr, cc = np.nonzero(m)
        x, y = tr * (cc.mean() + 0.5, rr.mean() + 0.5)
        lon, lat = to_ll.transform(x, y)
        cl = np.bincount(cls[m][cls[m] >= 0], minlength=cfg.features.n_classes)
        rows.append({"rank": rank, "cells": int(sizes[k]), "area_km2": round(sizes[k] * (cube.meta["cell_m"] / 1000) ** 2, 2),
                     "lat": round(lat, 4), "lon": round(lon, 4), "dominant_class": int(cl.argmax()),
                     "mean_hv_before_db": round(float(np.nanmean(cube.hv_db[ia][m])), 2),
                     "mean_d_hv_db": round(float(np.nanmean((cube.hv_db[ib] - cube.hv_db[ia])[m])), 2),
                     "mean_d_hh_db": round(float(np.nanmean((cube.hh_db[ib] - cube.hh_db[ia])[m])), 2),
                     "km_from_fire": round(float(dist_km[m].min()), 1)})
    art = cfg.artifacts
    pd.DataFrame(rows).to_csv(art / "false_alarm_patches.csv", index=False)
    out["false_alarms"] = {"alpha": ra, "flagged_control_cells": int(fa.sum()), "patches": int(n_cc),
                           "share_in_20_largest": round(float(sizes[order].sum() / max(fa.sum(), 1)), 3),
                           "largest": rows[:8]}

    # ---- 5b. other 2026 perimeters in the window (excluded from labels): an independent look
    gj = json.loads(cfg.perimeter_path.read_text())
    others = []
    for f in gj["features"]:
        p = f["properties"]
        name = str(p.get(cfg.perimeter.name_field) or "").strip()
        if name.lower() == cfg.perimeter.incident_name.lower():
            continue
        g = shape(transform_geom("EPSG:4326", f"EPSG:{cube.meta['crs_epsg']}", f["geometry"]))
        inside = rasterize([(mapping(g), 1)], out_shape=shape_, transform=tr, fill=0, dtype="uint8").astype(bool)
        inside &= np.isfinite(stored["M"])
        if inside.sum() < 20:
            continue
        disc = p.get("attr_FireDiscoveryDateTime")
        disc = pd.to_datetime(disc, unit="ms").strftime("%Y-%m-%d") if isinstance(disc, (int, float)) else str(disc)
        row = {"fire": name, "discovered": disc, "cells": int(inside.sum())}
        for d in ("M", "B3", "B1"):
            row[f"flagged_{d}"] = round(float(flags[d][inside].mean()), 3)
        others.append(row)
    out["other_perimeters"] = {"alpha": ra, "rows": others,
                               "control_flag_rate": {d: round(float(flags[d][labels == 0].mean()), 4) for d in ("M", "B3", "B1")},
                               "burn_flag_rate": {d: round(float(flags[d][labels == 1].mean()), 4) for d in ("M", "B3", "B1")}}

    (art / "exploratory.json").write_text(json.dumps(out, indent=2))
    say(f"wrote {art / 'exploratory.json'}")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
