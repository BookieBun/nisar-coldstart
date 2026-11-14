import json

import numpy as np
from pyproj import Transformer

from coldstart.truth import build_labels


def _square(tr, x0, y0, cell, r0, c0, side):
    xs = [x0 + c0 * cell, x0 + (c0 + side) * cell]
    ys = [y0 - r0 * cell, y0 - (r0 + side) * cell]
    ring = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]), (xs[0], ys[0])]
    lon, lat = tr.transform([p[0] for p in ring], [p[1] for p in ring])
    return {"type": "Polygon", "coordinates": [[[a, b] for a, b in zip(lon, lat)]]}


def test_square_perimeter_labels(cfg_small):
    H = W = 80
    cell, x0, y0 = 100.0, 500000.0, 4500000.0
    inter = cfg_small.interim
    inter.mkdir(parents=True, exist_ok=True)
    valid = np.ones((H, W), bool)
    np.savez(inter / "cube.npz", valid=valid)
    (inter / "cube_meta.json").write_text(json.dumps(
        {"shape": [H, W], "transform": [cell, 0, x0, 0, -cell, y0], "crs_epsg": 32610, "cell_m": cell}))
    tr = Transformer.from_crs(32610, 4326, always_xy=True)
    name = cfg_small.perimeter.name_field
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {name: cfg_small.perimeter.incident_name},
         "geometry": _square(tr, x0, y0, cell, 20, 20, 10)},
        {"type": "Feature", "properties": {name: "Other"},
         "geometry": _square(tr, x0, y0, cell, 60, 60, 6)}]}
    cfg_small.perimeter_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_small.perimeter_path.write_text(json.dumps(gj))

    lab = build_labels(cfg_small)
    assert (lab == 1).sum() == 8 * 8                      # 10x10 square eroded by one cell
    assert (lab[60:66, 60:66] == -1).all()                # other perimeter excluded
    buf = cfg_small.labels.negative_buffer_km * 1000 / cell
    assert lab[25, 30 + int(buf)] == -1 or lab[25, 30 + int(buf)] == 0
    assert lab[25, 29 + int(np.ceil(buf)) + 2] == 0       # beyond the buffer -> negative
    assert lab[25, 30] == -1                              # just outside the edge -> ring
