"""Truth raster from the fire perimeter. Read only here and in evaluate (rule 2).

 1  inside the incident perimeter after eroding by labels.shrink_cells
 0  farther than labels.negative_buffer_km outside it
-1  everything else: the ring between, masked cells, and other perimeters in the window
"""
from __future__ import annotations

import json

import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from scipy.ndimage import binary_erosion, distance_transform_edt
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from .config import Config
from .log import get_logger, require_inputs, stage

log = get_logger()


def _rasterize(geoms, out_shape, transform) -> np.ndarray:
    if not geoms:
        return np.zeros(out_shape, dtype=bool)
    return rasterize([(g, 1) for g in geoms], out_shape=out_shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


def build_labels(cfg: Config) -> np.ndarray:
    inter = cfg.interim
    meta = json.loads((inter / "cube_meta.json").read_text())
    valid = np.load(inter / "cube.npz")["valid"]
    H, W = meta["shape"]
    tr = Affine(*meta["transform"])
    dst = CRS.from_epsg(meta["crs_epsg"])
    gj = json.loads(cfg.perimeter_path.read_text())
    field, name = cfg.perimeter.name_field, cfg.perimeter.incident_name.strip().lower()

    window = box(tr.c, tr.f + tr.e * H, tr.c + tr.a * W, tr.f)
    target, others = [], []
    for f in gj.get("features", []):
        props = f.get("properties") or {}
        if field not in props:
            raise KeyError(f"perimeter feature lacks property '{field}' (perimeter.name_field); "
                           f"available: {sorted(props)}")
        g = shape(transform_geom("EPSG:4326", dst, f["geometry"]))
        if str(props[field] or "").strip().lower() == name:
            target.append(g)
        elif g.intersects(window):
            others.append(g)
    if not target:
        raise ValueError(f"no perimeter feature with {field} == {cfg.perimeter.incident_name!r}")
    inside = _rasterize([mapping(unary_union(target))], (H, W), tr)
    other = _rasterize([mapping(g) for g in others], (H, W), tr)

    pos = binary_erosion(inside, iterations=cfg.labels.shrink_cells) if cfg.labels.shrink_cells else inside
    dist_m = distance_transform_edt(~inside) * meta["cell_m"]
    neg = dist_m > cfg.labels.negative_buffer_km * 1000

    labels = np.full((H, W), -1, dtype=np.int8)
    labels[neg] = 0
    labels[pos] = 1
    labels[other] = -1
    labels[~valid] = -1
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"labels have {n_pos} positive and {n_neg} negative cells; check the "
                         "perimeter and window (an axis-order or CRS bug shows up here)")
    log.info("labels: %d positive, %d negative, %d other-perimeter cells", n_pos, n_neg, int(other.sum()))
    return labels


def run_truth(cfg: Config) -> None:
    require_inputs("truth", cfg.interim / "cube.npz", cfg.interim / "cube_meta.json", cfg.perimeter_path)
    with stage("truth") as outputs:
        labels = build_labels(cfg)
        np.save(cfg.interim / "labels.npy", labels)
        outputs.append(cfg.interim / "labels.npy")
