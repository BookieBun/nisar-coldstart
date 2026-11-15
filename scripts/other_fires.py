"""Other agency-mapped fires in a site's window, for the app's cross-check (exploratory).

Added after the results; not one of the pre-registered tests. The pipeline already reads every
2026 WFIGS perimeter in the window and leaves the cells of fires other than the studied one out of
all scores (truth.py sets them to -1). This script records which fire each of those cells belongs
to, so the app can say whether a flagged place lies inside another mapped fire, and measures how
often each method flags those fires compared with ordinary land.

It uses the same perimeter selection and rasterization as truth.py, the same display
reprojection as export.py (nearest neighbour), and the same thresholds and flag rule as the
evaluation. It fits nothing and changes no stored result.

Writes, from local data only:
  artifacts/<site>/other_fires.npz   ids: uint8 on the display grid, 0 = no other fire, k = fires[k-1]
  artifacts/<site>/other_fires.json  fire names, discovery dates and cell counts; shares flagged

Usage:
    python scripts/other_fires.py config/site.yaml
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, transform_geom
from shapely.geometry import box, mapping, shape

from coldstart.config import Config, load_config
from coldstart.conformal import DETECTORS, load_thresholds
from coldstart.evaluate import flags_at
from coldstart.export import _reproj, _target
from coldstart.pairs import event_pair, load_pairs
from coldstart.truth import _rasterize

DATE_FIELD = "attr_FireDiscoveryDateTime"   # WFIGS, milliseconds since 1970


def other_fires(cfg: Config) -> tuple[np.ndarray, list[dict]]:
    """Native-grid id raster and fire list, selected exactly as truth.build_labels selects 'others'."""
    d = cfg.interim
    meta = json.loads((d / "cube_meta.json").read_text())
    H, W = meta["shape"]
    tr = Affine(*meta["transform"])
    dst = CRS.from_epsg(meta["crs_epsg"])
    gj = json.loads(cfg.perimeter_path.read_text())
    field, name = cfg.perimeter.name_field, cfg.perimeter.incident_name.strip().lower()
    window = box(tr.c, tr.f + tr.e * H, tr.c + tr.a * W, tr.f)
    groups: dict[str, dict] = {}
    for f in gj.get("features", []):
        props = f.get("properties") or {}
        g = shape(transform_geom("EPSG:4326", dst, f["geometry"]))
        fire = str(props.get(field) or "").strip()
        if fire.lower() == name or not g.intersects(window):
            continue
        grp = groups.setdefault(fire, {"geoms": [], "discovered": None})
        grp["geoms"].append(g)
        ms = props.get(DATE_FIELD)
        if ms:
            day = datetime.fromtimestamp(int(ms) / 1000, timezone.utc).date().isoformat()
            grp["discovered"] = min(filter(None, [grp["discovered"], day]))
    ids = np.zeros((H, W), dtype=np.uint8)
    fires = []
    for fire, grp in sorted(groups.items()):
        mask = _rasterize([mapping(g) for g in grp["geoms"]], (H, W), tr)
        if not mask.any():
            continue
        if len(fires) == 255:
            raise ValueError("more than 255 other fires in the window")
        fires.append({"id": len(fires) + 1, "name": fire, "discovered": grp["discovered"]})
        ids[mask & (ids == 0)] = fires[-1]["id"]
    return ids, fires


def summarize(cfg: Config) -> tuple[np.ndarray, dict]:
    d = cfg.interim
    meta = json.loads((d / "cube_meta.json").read_text())
    valid = np.load(d / "cube.npz")["valid"]
    labels = np.load(d / "labels.npy")
    ev = event_pair(load_pairs(d / "pairs.json"))
    thr = load_thresholds(d / "calibration.json")
    ids, fires = other_fires(cfg)
    if np.any((ids > 0) & (labels != -1)):
        raise AssertionError("other-fire cells must be excluded from the labels (truth.py sets them to -1)")
    other = (ids > 0) & valid
    control = labels == 0
    for f in fires:
        f["cells"] = int(((ids == f["id"]) & valid).sum())
    rates: dict = {}
    for det in DETECTORS:
        score = np.load(d / "scores" / det / f"{ev.id}.npy")
        rates[det] = {}
        for a in cfg.calibration.alphas:
            fl = flags_at(score, thr[det][float(a)], cfg.postprocess.min_patch_cells)
            rates[det][str(a)] = {
                "other_fires": float(fl[other].mean()) if other.any() else None,
                "control": float(fl[control].mean()),
                "per_fire": {str(f["id"]): (float(fl[(ids == f["id"]) & valid].mean()) if f["cells"] else None)
                             for f in fires},
            }
    shape_d, t_d, _ = _target(meta, cfg.display.max_display_cells)
    disp = _reproj(ids.astype(np.float32), meta, shape_d, t_d, Resampling.nearest)
    disp = np.where(np.isfinite(disp), disp, 0).astype(np.uint8)
    out = {"site": cfg.site, "event_pair": ev.id, "exploratory": True,
           "source": "WFIGS Interagency Fire Perimeters (NIFC), the same file as the labels: "
                     + str(cfg.perimeter.path),
           "note": "Fires other than the studied one whose perimeter intersects the window. Their cells are "
                   "left out of every score. Shares are on the native 100 m grid, after the minimum-patch "
                   "filter, as mapped; 'control' is land more than labels.negative_buffer_km outside the "
                   "studied perimeter.",
           "n_other_cells": int(other.sum()), "n_control": int(control.sum()),
           "fires": fires, "rates": rates}
    return disp, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    disp, out = summarize(cfg)
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.artifacts / "other_fires.npz", ids=disp)
    (cfg.artifacts / "other_fires.json").write_text(json.dumps(out, indent=2))
    ra = str(cfg.calibration.report_alpha)
    print(f"{cfg.site}: {len(out['fires'])} other fires, {out['n_other_cells']} valid cells")
    for det in DETECTORS:
        r = out["rates"][det][ra]
        if r["other_fires"] is not None:
            print(f"  {det} at alpha {ra}: other fires {r['other_fires']:.1%}, control {r['control']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
