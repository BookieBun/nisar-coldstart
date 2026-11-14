"""Static land classes and per-pair feature tables.

Every feature describes the earlier date or something fixed. No coordinates, row/col
indices, tile ids or fold ids are ever features (rule 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.cluster import KMeans

from .config import Config
from .grid import Cube, load_cube
from .log import get_logger, require_inputs, stage
from .pairs import Pair, define_pairs, save_pairs

log = get_logger()

FEATURE_NAMES = ["hh0", "hv0", "ratio0", "tex0", "terrain", "cls"]
CATEGORICAL = {"cls"}


@dataclass
class FeatureSet:
    X: np.ndarray            # (N,F) float32
    feature_names: list[str]
    cat_mask: np.ndarray     # (F,) bool
    y_hh: np.ndarray         # (N,) change in dB
    y_hv: np.ndarray
    idx: np.ndarray          # (N,) flat cell indices into (H,W)
    fold: np.ndarray         # (N,) int8


def to_map(values: np.ndarray, idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    out[idx] = values
    return out.reshape(shape)


def nan_std_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Std of finite values in a size x size window. Uses moving sums of x and x^2 in float64;
    the variance is clipped at zero because E[x^2] - E[x]^2 can round slightly negative."""
    ok = np.isfinite(x)
    xv = np.where(ok, x, 0.0).astype(np.float64)
    n = uniform_filter(ok.astype(np.float64), size, mode="constant")
    s1 = uniform_filter(xv, size, mode="constant")
    s2 = uniform_filter(xv * xv, size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1 / n
        var = np.clip(s2 / n - m * m, 0.0, None)
    out = np.sqrt(var).astype(np.float32)
    out[n * size * size < 0.5] = np.nan
    return out


def fit_classes(cube: Cube, cfg: Config) -> np.ndarray:
    """K-means on per-cell medians of pre-ignition HH and HV. Never sees post-ignition dates.
    Class ids are ordered by HV centre so they are stable and interpretable."""
    pre = [i for i, d in enumerate(cube.dates) if d < cfg.event.ignition]
    if not pre:
        raise ValueError("no pre-ignition dates to fit land classes")
    with np.errstate(all="ignore"), _quiet():
        hh = np.nanmedian(cube.hh_db[pre], axis=0)
        hv = np.nanmedian(cube.hv_db[pre], axis=0)
    ok = cube.valid & np.isfinite(hh) & np.isfinite(hv)
    Z = np.column_stack([hh[ok], hv[ok]]).astype(np.float64)
    rng = np.random.default_rng(cfg.seed)
    fit_rows = rng.choice(len(Z), size=min(len(Z), cfg.features.kmeans_max_cells), replace=False)
    fit_rows.sort()
    n = cfg.features.n_classes
    km = KMeans(n_clusters=n, n_init=10, random_state=cfg.seed).fit(Z[fit_rows])
    remap = np.empty(n, dtype=np.int8)
    remap[np.argsort(km.cluster_centers_[:, 1], kind="stable")] = np.arange(n, dtype=np.int8)
    cls = np.full(cube.shape, -1, dtype=np.int8)
    cls[ok] = remap[km.predict(Z)]
    return cls


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def build_features(cube: Cube, pair: Pair, cls: np.ndarray, cfg: Config) -> FeatureSet:
    if cfg.features.use_rain or cfg.features.use_worldcover:
        raise NotImplementedError("features.use_rain / use_worldcover (O2, O5) are not implemented")
    ia, ib = cube.dates.index(pair.date_a), cube.dates.index(pair.date_b)
    hh_a, hv_a, hh_b, hv_b = cube.hh_db[ia], cube.hv_db[ia], cube.hh_db[ib], cube.hv_db[ib]
    cell = cube.valid & (cls >= 0) & np.isfinite(hh_a) & np.isfinite(hv_a) & np.isfinite(hh_b) & np.isfinite(hv_b)
    tex = nan_std_filter(hv_a, cfg.features.tex_window)
    idx = np.flatnonzero(cell)
    cols = {"hh0": hh_a[cell], "hv0": hv_a[cell], "ratio0": hv_a[cell] - hh_a[cell],
            "tex0": tex[cell], "terrain": cube.terrain[cell], "cls": cls[cell].astype(np.float32)}
    X = np.column_stack([cols[n] for n in FEATURE_NAMES]).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"non-finite feature values in pair {pair.id}")
    return FeatureSet(X=X, feature_names=list(FEATURE_NAMES),
                      cat_mask=np.array([n in CATEGORICAL for n in FEATURE_NAMES]),
                      y_hh=(hh_b - hh_a)[cell].astype(np.float32), y_hv=(hv_b - hv_a)[cell].astype(np.float32),
                      idx=idx.astype(np.int64), fold=cube.fold[cell].astype(np.int8))


def save_features(fs: FeatureSet, path) -> None:
    np.savez_compressed(path, X=fs.X, feature_names=np.array(fs.feature_names), cat_mask=fs.cat_mask,
                        y_hh=fs.y_hh, y_hv=fs.y_hv, idx=fs.idx, fold=fs.fold)


def load_features(path) -> FeatureSet:
    z = np.load(path)
    return FeatureSet(X=z["X"], feature_names=[str(s) for s in z["feature_names"]], cat_mask=z["cat_mask"],
                      y_hh=z["y_hh"], y_hv=z["y_hv"], idx=z["idx"], fold=z["fold"])


def run_features(cfg: Config) -> None:
    require_inputs("features", cfg.interim / "cube.npz", cfg.interim / "cube_meta.json")
    with stage("features") as outputs:
        cube = load_cube(cfg)
        pairs = define_pairs(cube.dates, cfg)
        save_pairs(pairs, cfg.interim / "pairs.json")
        cls = fit_classes(cube, cfg)
        np.save(cfg.interim / "cls.npy", cls)
        fdir = cfg.interim / "features"
        fdir.mkdir(parents=True, exist_ok=True)
        for p in pairs:
            fs = build_features(cube, p, cls, cfg)
            save_features(fs, fdir / f"{p.id}.npz")
            log.info("pair %s (%s): %d cells", p.id, p.kind, len(fs.idx))
        outputs += [cfg.interim / "pairs.json", cfg.interim / "cls.npy", fdir]
