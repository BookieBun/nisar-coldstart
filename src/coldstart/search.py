"""Search ASF for NISAR GCOV scenes over the configured window and write manifest.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .log import get_logger, stage

log = get_logger()

# NISAR_L2_PR_GCOV_<cycle>_<track>_<dir>_<frame>_<mode>_<pol>_<S>_<start>_...
NAME_IDX = {"track": 5, "direction": 6, "frame": 7, "mode": 8, "polarization": 9, "start": 11}
COLUMNS = ["scene_name", "date", "track", "direction", "frame", "mode", "polarization", "bytes", "url", "selected"]


def parse_scene_name(name: str) -> dict:
    parts = Path(name).stem.split("_")
    if len(parts) <= NAME_IDX["start"] or parts[0] != "NISAR" or parts[3] != "GCOV":
        raise ValueError(f"unexpected GCOV scene name: {name!r}")
    return {k: parts[i] for k, i in NAME_IDX.items()}


def _bytes(p: dict):
    b = p.get("bytes")
    if isinstance(b, dict):
        tot = sum(int(v.get("bytes", 0) if isinstance(v, dict) else v) for v in b.values())
        return tot or None
    return int(b) if isinstance(b, (int, float)) else None


def search_scenes(cfg: Config) -> pd.DataFrame:
    import asf_search as asf

    with stage("search") as outputs:
        w, s, e, n = cfg.search.bbox_wgs84
        wkt = f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
        res = asf.search(dataset=cfg.search.dataset, processingLevel="GCOV", intersectsWith=wkt,
                         start=cfg.search.start, end=cfg.search.end)
        rows = []
        for r in res:
            p = r.properties
            meta = parse_scene_name(p["sceneName"])
            rows.append({"scene_name": p["sceneName"], "date": meta["start"][:8], "track": meta["track"],
                         "direction": meta["direction"], "frame": meta["frame"], "mode": meta["mode"],
                         "polarization": meta["polarization"], "bytes": _bytes(p), "url": p.get("url", "")})
        df = pd.DataFrame(rows, columns=COLUMNS[:-1]).drop_duplicates("scene_name").sort_values(["date", "scene_name"])
        sel = {k: getattr(cfg.search, k) for k in ("track", "frame", "direction", "mode", "polarization")}
        mask = pd.Series(True, index=df.index)
        for k, v in sel.items():
            mask &= df[k].astype(str) == (str(v) if v is not None else "__none__")
        df["selected"] = mask
        cfg.interim.mkdir(parents=True, exist_ok=True)
        out = cfg.interim / "manifest.csv"
        df.to_csv(out, index=False)
        outputs.append(out)
        log.info("%d scenes found, %d selected (%s)", len(df), int(df.selected.sum()), sel)
        return df
