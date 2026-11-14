"""Synthetic scenes with shared (class-wide) shifts and one injected local change.

The generator writes the same cube format as the real grid stage plus a perimeter
GeoJSON, so every later stage runs unchanged on it.

Backscatter for cell i of class c at date t and polarization p:
    x = base_p + offset_{c,p} + static_sd * z_{i,p}            (stable per cell)
        + shift_{c,t,p} + beta_{t,p} * z_{i,p}                  (shared change)
        + noise_sd_c * e_{i,t,p}                                (speckle-like noise)
        + drop_p * 1[i in disc, t >= event_index]               (the event)
The beta term makes the shared shift depend on how bright a cell is, which is the
dependence the pooled model conditions on and class stratification cannot see.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

from .config import Config
from .grid import finalize_cube
from .log import get_logger, stage

log = get_logger()


def _smooth(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return (f - f.mean()) / f.std()


def synthetic_dates(cfg: Config) -> list[str]:
    s = cfg.synthetic
    d0 = date.fromisoformat(s.start_date)
    return [(d0 + timedelta(days=i * s.interval_days)).isoformat() for i in range(s.n_dates)]


def make_synthetic(cfg: Config) -> None:
    s = cfg.synthetic
    with stage("synth") as outputs:
        rng = np.random.default_rng(cfg.seed)
        H, W = s.shape
        T = s.n_dates
        dates = synthetic_dates(cfg)
        ev = s.event_index
        if not (0 < ev < T):
            raise ValueError("synthetic.event_index must leave at least one date on each side")
        if not (dates[ev - 1] < cfg.event.ignition <= cfg.event.end == dates[ev]):
            raise ValueError(f"event.ignition/end must fall after {dates[ev - 1]} with event.end == {dates[ev]}")

        nc = s.n_classes
        cls = np.argmax(np.stack([_smooth(rng, (H, W), s.blob_sigma) for _ in range(nc)]), 0)
        z = {"hh": rng.standard_normal((H, W)), "hv": rng.standard_normal((H, W))}
        base = {"hh": s.hh_base_db + np.asarray(s.hh_class_offset_db)[cls] + s.static_sd_db * z["hh"],
                "hv": s.hv_base_db + np.asarray(s.hv_class_offset_db)[cls] + s.static_sd_db * z["hv"]}
        noise_sd = np.asarray(s.class_noise_sd_db)[cls]
        drop = {"hh": s.hh_drop_db, "hv": s.hv_drop_db}

        rr, cc = np.mgrid[0:H, 0:W]
        r0, c0, rad = H / 2, W / 2, s.disc_radius_cells
        disc = (rr + 0.5 - r0) ** 2 + (cc + 0.5 - c0) ** 2 <= rad ** 2

        cube = {p: np.empty((T, H, W), np.float32) for p in ("hh", "hv")}
        for t in range(T):
            for p in ("hh", "hv"):
                shift = rng.normal(0.0, s.shared_shift_sd_db, nc)[cls]
                beta = rng.normal(0.0, s.brightness_coupling_sd)
                x = base[p] + shift + beta * z[p] + noise_sd * rng.standard_normal((H, W))
                if t >= ev:
                    x = np.where(disc, x + drop[p], x)
                cube[p][t] = x
        mt, mr, mc, ms = s.missing_block
        for p in ("hh", "hv"):
            cube[p][mt, mr:mr + ms, mc:mc + ms] = np.nan
        terrain = (1.0 + s.terrain_sd * _smooth(rng, (H, W), s.blob_sigma)).astype(np.float32)

        cell = float(cfg.grid.target_cell_m)
        x0, y0 = s.origin_xy
        transform = (cell, 0.0, x0, 0.0, -cell, y0)
        outputs += finalize_cube(cfg, cube["hh"], cube["hv"], terrain, dates, s.crs_epsg, transform,
                                 native_spacing_m=cell, k=1, scenes=[f"synthetic_{d}" for d in dates])

        # Perimeter of the disc, in WGS84, like a WFIGS feature.
        th = np.linspace(0, 2 * np.pi, 257)
        px = x0 + (c0 + rad * np.cos(th)) * cell
        py = y0 - (r0 + rad * np.sin(th)) * cell
        lon, lat = Transformer.from_crs(s.crs_epsg, 4326, always_xy=True).transform(px, py)
        ring = [[float(a), float(b)] for a, b in zip(lon, lat)]
        ring[-1] = ring[0]
        gj = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {cfg.perimeter.name_field: cfg.perimeter.incident_name},
            "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
        pp = cfg.perimeter_path
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(json.dumps(gj))
        outputs.append(pp)
        np.save(cfg.interim / "synthetic_truth_classes.npy", cls.astype(np.int8))
        log.info("synthetic: %d true classes, disc of %d cells", nc, int(disc.sum()))
