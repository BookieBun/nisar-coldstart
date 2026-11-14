"""Display artifacts for the read-only app, plus README figures.

Everything is reprojected once, offline, to a WGS84 grid whose longest side is at most
display.max_display_cells: nearest-neighbour for scores, labels and flags, bilinear for
backscatter. Flags are computed on the native 100 m grid (same rule as the metrics) and
stored as a bitmask, bit i = flagged at calibration.alphas[i].
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

from .config import Config
from .conformal import DETECTORS, load_thresholds
from .evaluate import flags_at
from .grid import load_cube
from .log import get_logger, require_inputs, stage
from .pairs import event_pair, load_pairs

log = get_logger()
MAX_BYTES = 100 * 2**20


def _target(meta: dict, max_cells: int):
    H, W = meta["shape"]
    src_t = Affine(*meta["transform"])
    w, s, e, n = transform_bounds(CRS.from_epsg(meta["crs_epsg"]), "EPSG:4326", *array_bounds(H, W, src_t))
    f = min(1.0, max_cells / max(H, W))
    h, wd = max(1, int(round(H * f))), max(1, int(round(W * f)))
    return (h, wd), from_bounds(w, s, e, n, wd, h), (w, s, e, n)


def _reproj(a: np.ndarray, meta: dict, dst_shape, dst_t, method) -> np.ndarray:
    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(a.astype(np.float32), out, src_transform=Affine(*meta["transform"]),
              src_crs=CRS.from_epsg(meta["crs_epsg"]), dst_transform=dst_t, dst_crs="EPSG:4326",
              resampling=method, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def export_artifacts(cfg: Config) -> None:
    inter, art = cfg.interim, cfg.artifacts
    require_inputs("export", inter / "metrics.json", inter / "calibration.json", inter / "labels.npy",
                   inter / "pairs.json", inter / "curves.csv")
    with stage("export") as outputs:
        cube = load_cube(cfg)
        meta = cube.meta
        pairs = load_pairs(inter / "pairs.json")
        ev = event_pair(pairs)
        labels = np.load(inter / "labels.npy")
        cls = np.load(inter / "cls.npy")
        thr = load_thresholds(inter / "calibration.json")
        alphas = list(cfg.calibration.alphas)
        if len(alphas) > 8:
            raise ValueError("at most 8 alphas fit the uint8 flag bitmask")
        (dst_shape, dst_t, (w, s, e, n)) = _target(meta, cfg.display.max_display_cells)
        NN, BL = Resampling.nearest, Resampling.bilinear
        ia, ib = cube.dates.index(ev.date_a), cube.dates.index(ev.date_b)

        arrs: dict[str, np.ndarray] = {}
        for d in DETECTORS:
            sc = np.load(inter / "scores" / d / f"{ev.id}.npy")
            arrs[f"score_{d}"] = _reproj(sc, meta, dst_shape, dst_t, NN)
            bits = np.zeros(cube.shape, dtype=np.uint8)
            if np.isfinite(sc).any():
                for i, a in enumerate(alphas):
                    bits |= (flags_at(sc, thr[d][float(a)], cfg.postprocess.min_patch_cells).astype(np.uint8) << i)
            arrs[f"flags_{d}"] = _reproj(bits.astype(np.float32), meta, dst_shape, dst_t, NN)
        arrs["labels"] = _reproj(labels.astype(np.float32), meta, dst_shape, dst_t, NN)
        arrs["cls"] = _reproj(cls.astype(np.float32), meta, dst_shape, dst_t, NN)
        arrs["valid"] = _reproj(cube.valid.astype(np.float32), meta, dst_shape, dst_t, NN)
        arrs["hv_before"] = _reproj(cube.hv_db[ia], meta, dst_shape, dst_t, BL)
        arrs["hv_after"] = _reproj(cube.hv_db[ib], meta, dst_shape, dst_t, BL)
        arrs["hh_ts"] = np.stack([_reproj(cube.hh_db[t], meta, dst_shape, dst_t, BL) for t in range(len(cube.dates))])
        arrs["hv_ts"] = np.stack([_reproj(cube.hv_db[t], meta, dst_shape, dst_t, BL) for t in range(len(cube.dates))])
        q = np.load(inter / "quantiles" / f"{ev.id}.npz")
        for k in ("q05", "q50", "q95"):
            # predicted after-date HV = before-date HV + predicted change
            pred = cube.hv_db[ia] + q[f"hv_{k}"].astype(np.float32)
            arrs[f"hv_{k}"] = _reproj(pred, meta, dst_shape, dst_t, NN)

        art.mkdir(parents=True, exist_ok=True)
        packed = {}
        for k, v in arrs.items():
            if k.startswith("flags_") or k in ("labels", "cls", "valid"):
                packed[k] = np.where(np.isfinite(v), v, -1 if k in ("labels", "cls") else 0).astype(np.int8 if k in ("labels", "cls") else np.uint8)
            else:
                packed[k] = v.astype(np.float16)
        np.savez_compressed(art / "display.npz", **packed)
        dmeta = {"site": cfg.site, "bounds": [[s, w], [n, e]], "shape": list(dst_shape), "dates": cube.dates,
                 "event_pair": {"id": ev.id, "date_a": ev.date_a, "date_b": ev.date_b},
                 "alphas": alphas, "report_alpha": cfg.calibration.report_alpha, "detectors": DETECTORS,
                 "thresholds": {d: {str(a): (None if not np.isfinite(t) else t) for a, t in v.items()}
                                for d, v in thr.items()},
                 "crs_native": meta["crs_epsg"], "cell_m": meta["cell_m"]}
        (art / "display_meta.json").write_text(json.dumps(dmeta, indent=2))
        for name in ("metrics.json", "curves.csv", "calibration.json", "failure_table.csv", "pairs.json"):
            shutil.copyfile(inter / name, art / name)
        figs = art / "figures"
        figs.mkdir(exist_ok=True)
        _fig_pr(pd.read_csv(inter / "curves.csv"), figs / "pr_curves.png")
        _fig_h3(json.loads((inter / "calibration.json").read_text()), cfg, figs / "h3_realized.png")
        _fig_maps(cube, ev, labels, inter, thr, cfg, figs / "maps.png")

        total = sum(p.stat().st_size for p in art.rglob("*") if p.is_file())
        if total >= MAX_BYTES:
            raise RuntimeError(f"artifacts total {total / 2**20:.1f} MB >= 100 MB; lower display.max_display_cells")
        log.info("artifacts: %.1f MB in %s", total / 2**20, art)
        outputs += [art / "display.npz", art / "display_meta.json", figs]


COLORS = {"B1": "#8c8c8c", "B2": "#c08a2b", "B3": "#3b7dbf", "M": "#c23b3b"}


def _fig_pr(curves: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for d, g in curves.groupby("detector"):
        ax[0].plot(g.recall, g.precision, label=d, color=COLORS.get(d), lw=1.6)
        ax[1].plot(g.far, g.recall, label=d, color=COLORS.get(d), lw=1.6)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, event pair", xlim=(0, 1), ylim=(0, 1))
    ax[1].set(xlabel="False-alarm rate", ylabel="Recall", title="Recall vs false alarms", xscale="log",
              xlim=(1e-4, 1), ylim=(0, 1))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_h3(calib: dict, cfg: Config, path) -> None:
    rows = pd.DataFrame(calib["lopo"])
    a3 = list(cfg.calibration.h3_alphas)
    fig, axes = plt.subplots(1, len(a3), figsize=(5 * len(a3), 3.8), squeeze=False)
    lo, hi = cfg.calibration.h3_band
    for ax, a in zip(axes[0], a3):
        sub = rows[np.isclose(rows.alpha.astype(float), a)] if len(rows) else rows
        dets = [d for d in DETECTORS if len(sub) and (sub.detector == d).any()]
        for i, d in enumerate(dets):
            v = sub[sub.detector == d].realized.astype(float).to_numpy()
            ax.scatter(np.full(len(v), i), v, color=COLORS.get(d), zorder=3)
        ax.axhspan(lo * a, hi * a, color="#2e8b57", alpha=0.12, label="pass band")
        ax.axhline(a, color="#2e8b57", lw=1, ls="--", label=f"target {a}")
        ax.set_xticks(range(len(dets)), dets)
        ax.set(title=f"Left-out no-event pairs, alpha = {a}", ylabel="Realized false-alarm rate (raw)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_maps(cube, ev, labels, inter, thr, cfg, path) -> None:
    ia, ib = cube.dates.index(ev.date_a), cube.dates.index(ev.date_b)
    sc = np.load(inter / "scores" / "M" / f"{ev.id}.npy")
    fl = flags_at(sc, thr["M"][float(cfg.calibration.report_alpha)], cfg.postprocess.min_patch_cells)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, img, t, kw in [(ax[0], cube.hv_db[ia], f"HV {ev.date_a} (dB)", dict(vmin=-25, vmax=-5, cmap="gray")),
                          (ax[1], cube.hv_db[ib], f"HV {ev.date_b} (dB)", dict(vmin=-25, vmax=-5, cmap="gray")),
                          (ax[2], sc, "M score", dict(vmin=0, vmax=6, cmap="magma")),
                          (ax[3], np.where(np.isfinite(sc), fl, np.nan), f"M flags, alpha={cfg.calibration.report_alpha}",
                           dict(vmin=0, vmax=1, cmap="Reds"))]:
        im = a.imshow(img, interpolation="nearest", **kw)
        a.contour(labels == 1, levels=[0.5], colors="#00c8ff", linewidths=0.8)
        a.set_title(t, fontsize=10)
        a.set_axis_off()
        if a is not ax[3]:
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
