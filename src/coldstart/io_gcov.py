"""Read NISAR L2 GCOV HDF5, from a local file or streamed over HTTPS with byte-range requests.

Streaming matters: a GCOV scene is about 7.5 GB, but the window around one fire needs a few
hundred MB of it. h5py reads only the chunks that intersect the requested slice.

Authentication uses the Earthdata credentials in ~/.netrc through asf_search's session,
which follows the Earthdata Login redirect. The final (pre-signed) data URL is cached and
re-resolved when it expires.
"""
from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .log import get_logger

log = get_logger()

GRID_GROUP = "/science/LSAR/GCOV/grids/frequency{freq}"
LAYERS = ["HHHH", "HVHV", "mask", "numberOfLooks", "rtcGammaToSigmaFactor"]


# --------------------------------------------------------------------------- streaming file
class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTPS byte ranges, with an LRU cache of fixed-size blocks."""

    def __init__(self, url: str, session=None, block_size: int = 4 * 2**20, max_blocks: int = 256):
        super().__init__()
        if session is None:
            import asf_search as asf
            session = asf.ASFSession()
        self.url, self.session = url, session
        self.block_size, self.max_blocks = block_size, max_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._ranges: dict[int, bytes] = {}   # exact byte ranges fetched ahead (HDF5 chunks)
        self._pos = 0
        self.bytes_fetched = 0
        self._resolve()

    def prefetch(self, ranges: list[tuple[int, int]], workers: int) -> None:
        """Fetch (offset, size) ranges concurrently. HDF5 then reads each chunk from memory with
        one call at its offset. Threads only do HTTP I/O; h5py itself stays single-threaded."""
        from concurrent.futures import ThreadPoolExecutor
        todo = sorted({(int(o), int(s)) for o, s in ranges if s > 0 and int(o) not in self._ranges})
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for (o, _), data in zip(todo, ex.map(lambda r: self._get(r[0], r[0] + r[1] - 1), todo)):
                self._ranges[o] = data

    def drop_prefetched(self) -> None:
        self._ranges.clear()

    def _resolve(self) -> None:
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=120)
        if r.status_code not in (200, 206):
            raise PermissionError(f"HTTP {r.status_code} opening {self.url}: check ~/.netrc Earthdata credentials "
                                  "and that the NISAR data use agreement is accepted for your account")
        self._final = r.url
        cr = r.headers.get("Content-Range", "")
        if "/" not in cr:
            raise OSError(f"server did not honour a Range request for {self.url}")
        self.size = int(cr.rsplit("/", 1)[1])

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(3):
            r = self.session.get(self._final, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206:
                self.bytes_fetched += len(r.content)
                return r.content
            if r.status_code in (401, 403) and attempt < 2:  # pre-signed URL expired
                self._resolve()
                continue
            raise OSError(f"HTTP {r.status_code} reading bytes {start}-{end} of {self.url}")
        raise OSError("unreachable")

    def _block(self, i: int) -> bytes:
        b = self._cache.get(i)
        if b is not None:
            self._cache.move_to_end(i)
            return b
        start = i * self.block_size
        b = self._get(start, min(start + self.block_size, self.size) - 1)
        self._cache[i] = b
        if len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
        return b

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), max(0, self.size - self._pos))
        if n == 0:
            return 0
        mv = memoryview(buf)
        pre = self._ranges.get(self._pos)
        if pre is not None and n <= len(pre):
            mv[:n] = pre[:n]
            self._pos += n
            return n
        got = 0
        while got < n:
            i, off = divmod(self._pos + got, self.block_size)
            b = self._block(i)
            take = min(n - got, len(b) - off)
            mv[got:got + take] = b[off:off + take]
            got += take
        self._pos += n
        return n


class _GCOVFile(h5py.File):
    """h5py.File that remembers the streaming file object so read_window can prefetch."""
    _coldstart_raw = None


def open_gcov(src: str | Path) -> h5py.File:
    s = str(src)
    if s.startswith(("http://", "https://")):
        raw = HTTPRangeFile(s)
        f = _GCOVFile(raw, "r")
        f._coldstart_raw = raw
        return f
    return h5py.File(s, "r")


# --------------------------------------------------------------------------- structure
def list_layers(path: str | Path) -> dict[str, tuple]:
    """HDF5 dataset path -> shape, for scripts/inspect_gcov.py."""
    out: dict[str, tuple] = {}
    with open_gcov(path) as f:
        f.visititems(lambda n, o: out.__setitem__("/" + n, o.shape) if isinstance(o, h5py.Dataset) else None)
    return out


@dataclass
class GridInfo:
    crs_epsg: int
    transform: tuple       # affine (a, b, c, d, e, f) of the native grid, pixel-corner convention
    shape: tuple           # (rows, cols)
    spacing_m: float


def _find(group: h5py.Group, names: list[str]) -> h5py.Dataset:
    for n in names:
        if n in group:
            return group[n]
    raise KeyError(f"none of {names} in {group.name}; group contains {sorted(group.keys())}")


def read_grid_info(f: h5py.File, freq: str = "A") -> GridInfo:
    g = f[GRID_GROUP.format(freq=freq)]
    x = _find(g, ["xCoordinates"])[()]
    y = _find(g, ["yCoordinates"])[()]
    proj = _find(g, ["projection"])
    epsg = None
    for key in ("epsg_code", "epsg"):
        if key in proj.attrs:
            epsg = int(np.asarray(proj.attrs[key]).ravel()[0])
            break
    if epsg is None:
        v = proj[()]
        epsg = int(np.asarray(v).ravel()[0])
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # Coordinates are pixel centres [VERIFY]; the transform uses the outer corner.
    transform = (dx, 0.0, float(x[0]) - dx / 2, 0.0, dy, float(y[0]) - dy / 2)
    return GridInfo(crs_epsg=epsg, transform=transform, shape=(len(y), len(x)), spacing_m=abs(dx))


def chunk_ranges(ds: h5py.Dataset, window: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """(byte offset, size) of every stored chunk intersecting the window."""
    if ds.chunks is None:
        return []
    r0, r1, c0, c1 = window
    cr, cc = ds.chunks
    out = []
    for r in range((r0 // cr) * cr, r1, cr):
        for c in range((c0 // cc) * cc, c1, cc):
            info = ds.id.get_chunk_info_by_coord((r, c))
            if info.byte_offset is not None and info.size:
                out.append((info.byte_offset, info.size))
    return out


def read_window(f: h5py.File, window: tuple[int, int, int, int], freq: str = "A",
                layers: list[str] | None = None, workers: int = 1) -> dict[str, np.ndarray]:
    """window = (row0, row1, col0, col1) on the native grid. Returns float32 for the power layers.
    When the file is streamed and workers > 1, each layer's chunks are prefetched in parallel."""
    g = f[GRID_GROUP.format(freq=freq)]
    r0, r1, c0, c1 = window
    raw = getattr(f, "_coldstart_raw", None)
    out = {}
    for name in layers or LAYERS:
        if name not in g:
            raise KeyError(f"layer {name} missing from {g.name}; available: {sorted(g.keys())}")
        if isinstance(raw, HTTPRangeFile) and workers > 1:
            raw.prefetch(chunk_ranges(g[name], window), workers)
        out[name] = g[name][r0:r1, c0:c1]
        if isinstance(raw, HTTPRangeFile):
            raw.drop_prefetched()
    for p in ("HHHH", "HVHV"):
        if p in out:
            out[p] = out[p].astype(np.float32)
    return out


def describe(f: h5py.File, freq: str = "A") -> dict:
    """Facts needed to resolve the [VERIFY] items, printed by scripts/inspect_gcov.py."""
    g = f[GRID_GROUP.format(freq=freq)]
    info = {"group": g.name, "members": {}}
    for k in sorted(g.keys()):
        o = g[k]
        if isinstance(o, h5py.Dataset):
            info["members"][k] = {"shape": list(o.shape), "dtype": str(o.dtype), "chunks": o.chunks,
                                  "compression": o.compression,
                                  "attrs": {a: _attr(o.attrs[a]) for a in list(o.attrs)[:12]}}
    return info


def _attr(v):
    v = np.asarray(v)
    if v.dtype.kind in "SO":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in v.ravel()][:6]
    return v.ravel()[:6].tolist()


# --------------------------------------------------------------------------- real cube
def _scenes(cfg) -> list[dict]:
    import pandas as pd
    man = cfg.interim / "manifest.csv"
    if not man.exists():
        raise FileNotFoundError(f"{man} missing; run `python -m coldstart search {cfg.path}` first")
    df = pd.read_csv(man, dtype=str)
    # Same date, several products: names differ only in the trailing product counter (_001, _002, ...);
    # sorting by name and keeping the last picks the latest reprocessing.
    df = df[df.selected.str.lower() == "true"].sort_values(["date", "scene_name"])
    if df.empty:
        raise ValueError("manifest has no selected scenes for the configured track/frame/direction/mode/pol")
    for k in ("track", "frame", "direction", "mode", "polarization"):
        if df[k].nunique() != 1:
            raise ValueError(f"selected scenes disagree in {k}: {sorted(df[k].unique())}")
    dup = df.date.duplicated(keep="last")
    if dup.any():
        log.warning("dropping %d older products of the same date: %s", int(dup.sum()), df[dup].scene_name.tolist())
        df = df[~dup]
    out = []
    for _, r in df.iterrows():
        src = r.url if cfg.io.stream else str(cfg.raw_dir / Path(r.url).name)
        if not cfg.io.stream and not Path(src).exists():
            raise FileNotFoundError(f"{src} missing; run download or set io.stream: true")
        out.append({"date": f"{r.date[:4]}-{r.date[4:6]}-{r.date[6:8]}", "scene": r.scene_name, "src": src})
    return out


def _plan(cfg, infos: dict[str, GridInfo]) -> tuple[dict[str, tuple], tuple, int, int]:
    """Common window on the native grid of every date, snapped to whole 100 m cells.
    Scenes on a different CRS, spacing or pixel alignment are dropped, never resampled."""
    from rasterio.warp import transform_bounds
    dates = sorted(infos)
    ref = infos[dates[0]]
    dx, dy = ref.transform[0], ref.transform[4]
    if dy >= 0:
        raise ValueError("expected a north-up grid (negative y spacing)")
    k = int(round(cfg.grid.target_cell_m / ref.spacing_m))
    if k < 1 or abs(k * ref.spacing_m - cfg.grid.target_cell_m) > 1e-6 * cfg.grid.target_cell_m:
        raise ValueError(f"target_cell_m {cfg.grid.target_cell_m} is not a whole multiple of spacing {ref.spacing_m}")
    rx0, ry0 = ref.transform[2], ref.transform[5]
    keep = {}
    for d in dates:
        gi = infos[d]
        ox, oy = (gi.transform[2] - rx0) / dx, (gi.transform[5] - ry0) / dy
        if (gi.crs_epsg != ref.crs_epsg or abs(gi.spacing_m - ref.spacing_m) > 1e-9
                or abs(ox - round(ox)) > 1e-6 or abs(oy - round(oy)) > 1e-6):
            log.warning("dropping %s: grid differs from the first date (crs %s, spacing %s, offset %.3f,%.3f px)",
                        d, gi.crs_epsg, gi.spacing_m, ox, oy)
            continue
        keep[d] = gi
    w, s, e, n = cfg.search.bbox_wgs84
    fx0, fy0, fx1, fy1 = transform_bounds("EPSG:4326", f"EPSG:{ref.crs_epsg}", w, s, e, n, densify_pts=21)
    b = cfg.grid.buffer_km * 1000
    xmin, ymin, xmax, ymax = fx0 - b, fy0 - b, fx1 + b, fy1 + b
    for gi in keep.values():
        x0, y0 = gi.transform[2], gi.transform[5]
        xmin, xmax = max(xmin, x0), min(xmax, x0 + gi.shape[1] * dx)
        ymax, ymin = min(ymax, y0), max(ymin, y0 + gi.shape[0] * dy)
    C = k * dx
    cx0 = rx0 + np.ceil((xmin - rx0) / C) * C
    cx1 = rx0 + np.floor((xmax - rx0) / C) * C
    cy1 = ry0 - np.ceil((ry0 - ymax) / C) * C
    cy0 = ry0 - np.floor((ry0 - ymin) / C) * C
    Wc, Hc = int(round((cx1 - cx0) / C)), int(round((cy1 - cy0) / C))
    if Wc <= 0 or Hc <= 0:
        raise ValueError("the scenes have no common area around the fire")
    if not (cx0 <= fx0 and fx1 <= cx1 and cy0 <= fy0 and fy1 <= cy1):
        raise ValueError("the common scene area does not contain the whole fire bbox; choose another frame")
    if cx0 > fx0 - b + 1 or cx1 < fx1 + b - 1 or cy0 > fy0 - b + 1 or cy1 < fy1 + b - 1:
        log.warning("window clipped by the frame edge: buffer is less than %.0f km on some side", cfg.grid.buffer_km)
    windows = {}
    for d, gi in keep.items():
        c0 = int(round((cx0 - gi.transform[2]) / dx))
        r0 = int(round((gi.transform[5] - cy1) / -dy))
        windows[d] = (r0, r0 + Hc * k, c0, c0 + Wc * k)
    out_transform = (float(C), 0.0, float(cx0), 0.0, float(-C), float(cy1))
    return windows, out_transform, k, ref.crs_epsg


def _aggregate_strips(x: np.ndarray, k: int, min_valid: float, rows_per_strip: int = 200) -> np.ndarray:
    from .grid import aggregate_linear
    step = k * rows_per_strip
    return np.vstack([aggregate_linear(x[r:r + step], k, min_valid) for r in range(0, x.shape[0], step)])


def build_real_cube(cfg) -> list[Path]:
    import time
    from .grid import finalize_cube, to_db

    scenes = _scenes(cfg)
    infos, names = {}, {}
    for sc in scenes:
        with open_gcov(sc["src"]) as f:
            infos[sc["date"]] = read_grid_info(f, cfg.grid.freq)
        names[sc["date"]] = sc
    windows, transform, k, epsg = _plan(cfg, infos)
    dates = sorted(windows)
    r0, r1, c0, c1 = windows[dates[0]]
    log.info("window %d x %d native px (%d x %d cells of %d m), k=%d, %d dates",
             r1 - r0, c1 - c0, (r1 - r0) // k, (c1 - c0) // k, cfg.grid.target_cell_m, k, len(dates))

    cache = cfg.interim / "dates"
    cache.mkdir(parents=True, exist_ok=True)
    mv = cfg.grid.min_valid_fraction
    hh, hv, terr, mask_vals = [], [], [], {}
    for d in dates:
        cf = cache / f"{d}.npz"
        key = json.dumps({"window": windows[d], "k": k, "mv": mv, "scene": names[d]["scene"]})
        if cf.exists() and str(np.load(cf)["key"]) == key:
            z = np.load(cf)
        else:
            t0 = time.perf_counter()
            with open_gcov(names[d]["src"]) as f:
                lay = read_window(f, windows[d], cfg.grid.freq,
                                  layers=["HHHH", "HVHV", "rtcGammaToSigmaFactor", "mask"], workers=cfg.io.workers)
                fetched = getattr(getattr(f, "_coldstart_raw", None), "bytes_fetched", 0)
            u, c = np.unique(lay["mask"], return_counts=True)
            np.savez_compressed(cf, key=key, hh=_aggregate_strips(lay["HHHH"], k, mv),
                                hv=_aggregate_strips(lay["HVHV"], k, mv),
                                terr=_aggregate_strips(lay["rtcGammaToSigmaFactor"], k, mv),
                                mask_u=u, mask_c=c)
            del lay
            z = np.load(cf)
            log.info("date %s read in %.0f s (%.0f MB fetched)", d, time.perf_counter() - t0, fetched / 1e6)
        hh.append(z["hh"]); hv.append(z["hv"]); terr.append(z["terr"])
        mask_vals[d] = {int(a): int(b) for a, b in zip(z["mask_u"], z["mask_c"])}
    log.info("GCOV mask values per date (semantics [VERIFY]; validity uses power only): %s", mask_vals)
    with np.errstate(all="ignore"):
        terrain = np.nanmean(np.stack(terr), axis=0).astype(np.float32)
    outputs = finalize_cube(cfg, to_db(np.stack(hh)), to_db(np.stack(hv)), terrain, dates, epsg, transform,
                            native_spacing_m=infos[dates[0]].spacing_m, k=k,
                            scenes=[names[d]["scene"] for d in dates])
    meta_p = cfg.interim / "cube_meta.json"
    meta = json.loads(meta_p.read_text())
    meta["gcov_mask_values"] = mask_vals
    meta_p.write_text(json.dumps(meta, indent=2))
    return outputs
