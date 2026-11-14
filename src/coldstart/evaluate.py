"""Evaluation on the event pair: PR curves, point metrics, block bootstrap, H1/H3 verdicts.

All threshold-free metrics share one weighted implementation so that the block bootstrap
can reuse a single sort per detector: resampling blocks with replacement is equivalent to
weighting each cell by how many times its block was drawn.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Config
from .conformal import DETECTORS, load_thresholds
from .log import get_logger, require_inputs, stage
from .pairs import event_pair, load_pairs
from .postprocess import remove_small_patches

log = get_logger()

METRICS = ["ap", "far_at_recall50", "recall_at_far1", "f1_at_report_alpha", "iou_at_report_alpha"]


# ---------------------------------------------------------------- core weighted statistics
class _Sorted:
    """One detector's scores on the evaluation cells, sorted descending with tie groups."""

    def __init__(self, score: np.ndarray, y: np.ndarray):
        self.order = np.argsort(-score, kind="mergesort")
        self.s = score[self.order]
        self.y = y[self.order].astype(np.float64)
        self.ends = np.flatnonzero(np.r_[self.s[1:] != self.s[:-1], True])

    def stats(self, w: np.ndarray | None, recall_target: float, far_target: float) -> dict:
        ws = np.ones_like(self.y) if w is None else w[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        fp = np.cumsum(ws * (1 - self.y))[self.ends]
        P, N = tp[-1], fp[-1]
        if P <= 0 or N <= 0:
            return {"ap": np.nan, "far_at_recall50": np.nan, "recall_at_far1": np.nan}
        rec, far = tp / P, fp / N
        # Leading groups can carry zero weight in a bootstrap resample (0/0). Recall does not
        # move there, so their precision never enters AP; define it as 1 to avoid nan.
        prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
        ap = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))
        k = min(int(np.searchsorted(rec, recall_target - 1e-12, side="left")), len(rec) - 1)
        j = int(np.searchsorted(far, far_target + 1e-12, side="right")) - 1
        return {"ap": ap, "far_at_recall50": float(far[k]), "recall_at_far1": float(rec[j]) if j >= 0 else 0.0,
                "_curve": (self.s[self.ends], prec, rec, far)}


def _flag_stats(flag: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> dict:
    w = np.ones_like(y, dtype=np.float64) if w is None else w
    tp = np.sum(w * (flag & (y == 1)))
    fp = np.sum(w * (flag & (y == 0)))
    fn = np.sum(w * (~flag & (y == 1)))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    return {"f1_at_report_alpha": float(f1), "iou_at_report_alpha": float(iou)}


# ---------------------------------------------------------------- public API (handoff 4.2)
def pr_curve(score: np.ndarray, labels: np.ndarray, max_points: int = 2000) -> pd.DataFrame:
    ok = np.isin(labels, (0, 1)) & np.isfinite(score)
    st = _Sorted(score[ok], (labels[ok] == 1)).stats(None, 0.5, 0.01)
    thr, prec, rec, far = st["_curve"]
    keep = np.unique(np.linspace(0, len(thr) - 1, min(max_points, len(thr))).round().astype(int))
    return pd.DataFrame({"threshold": thr[keep], "precision": prec[keep], "recall": rec[keep], "far": far[keep]})


def flags_at(score: np.ndarray, thr: float, min_patch: int) -> np.ndarray:
    raw = np.zeros(score.shape, dtype=bool)
    ok = np.isfinite(score)
    raw[ok] = score[ok] > thr
    return remove_small_patches(raw, min_patch)


def point_metrics(score: np.ndarray, labels: np.ndarray, thr: float, cfg: Config) -> dict:
    ok = np.isin(labels, (0, 1)) & np.isfinite(score)
    y = labels[ok] == 1
    st = _Sorted(score[ok], y).stats(None, cfg.evaluate.recall_target, cfg.evaluate.far_target)
    st.pop("_curve", None)
    st.update(_flag_stats(flags_at(score, thr, cfg.postprocess.min_patch_cells)[ok], y.astype(int), None))
    return st


def block_ids(shape: tuple[int, int], block_cells: int) -> np.ndarray:
    H, W = shape
    nbc = -(-W // block_cells)
    return ((np.arange(H) // block_cells)[:, None] * nbc + (np.arange(W) // block_cells)[None, :]).ravel()


def block_bootstrap(scores: dict[str, np.ndarray], labels: np.ndarray, cfg: Config,
                    flags: dict[str, np.ndarray]) -> pd.DataFrame:
    """Resample blocks of bootstrap_block_km with replacement; every detector is evaluated on the
    same resample, so differences between detectors are paired."""
    dets = list(scores)
    ok = np.isin(labels, (0, 1))
    for d in dets:
        ok &= np.isfinite(scores[d])
    okf = ok.ravel()
    y = (labels.ravel()[okf] == 1).astype(int)
    bcells = int(round(cfg.evaluate.bootstrap_block_km * 1000 / cfg.grid.target_cell_m))
    blk = block_ids(labels.shape, bcells)[okf]
    uniq, blk_local = np.unique(blk, return_inverse=True)
    sorted_ = {d: _Sorted(scores[d].ravel()[okf], y) for d in dets}
    fl = {d: flags[d].ravel()[okf] for d in dets}
    rng = np.random.default_rng([cfg.seed, 7])
    rows = []
    for rep in range(cfg.evaluate.bootstrap_n):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(np.float64)
        w = counts[blk_local]
        for d in dets:
            st = sorted_[d].stats(w, cfg.evaluate.recall_target, cfg.evaluate.far_target)
            st.pop("_curve", None)
            st.update(_flag_stats(fl[d], y, w))
            rows += [{"rep": rep, "detector": d, "metric": m, "value": st[m]} for m in METRICS]
    return pd.DataFrame(rows)


def failure_table(flags: dict[str, np.ndarray], labels: np.ndarray, cls: np.ndarray) -> pd.DataFrame:
    rows = []
    for d, f in flags.items():
        for c in sorted(int(v) for v in np.unique(cls) if v >= 0):
            m = cls == c
            pos, neg = m & (labels == 1), m & (labels == 0)
            rows.append({"detector": d, "class": c, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                         "recall": float(f[pos].mean()) if pos.any() else None,
                         "far": float(f[neg].mean()) if neg.any() else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- provenance
def _versions() -> dict:
    out = {}
    for p in ["numpy", "scipy", "pandas", "scikit-learn", "h5py", "rasterio", "shapely", "pyproj"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _commit(cfg: Config) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


# ---------------------------------------------------------------- stage
def run_evaluate(cfg: Config) -> None:
    inter = cfg.interim
    require_inputs("evaluate", inter / "pairs.json", inter / "labels.npy", inter / "calibration.json",
                   inter / "cls.npy")
    with stage("evaluate") as outputs:
        ev = event_pair(load_pairs(inter / "pairs.json"))
        labels = np.load(inter / "labels.npy")
        cls = np.load(inter / "cls.npy")
        thr = load_thresholds(inter / "calibration.json")
        calib = json.loads((inter / "calibration.json").read_text())
        ra = float(cfg.calibration.report_alpha)

        scores, excluded = {}, []
        for d in DETECTORS:
            s = np.load(inter / "scores" / d / f"{ev.id}.npy")
            if np.isfinite(s[np.isin(labels, (0, 1))]).mean() < 0.99:
                excluded.append(d)
                log.warning("%s scores fewer than 99%% of labelled cells on the event pair; excluded", d)
                continue
            scores[d] = s
        if "M" not in scores:
            raise RuntimeError("model M has no usable event-pair scores")
        flags = {d: flags_at(scores[d], thr[d][ra], cfg.postprocess.min_patch_cells) for d in scores}

        point = {d: point_metrics(scores[d], labels, thr[d][ra], cfg) for d in scores}
        curves = pd.concat([pr_curve(scores[d], labels).assign(detector=d) for d in scores])
        curves = curves[["detector", "threshold", "precision", "recall", "far"]]
        boot = block_bootstrap(scores, labels, cfg, flags)
        fail = failure_table(flags, labels, cls)

        def ci(det, metric):
            v = boot[(boot.detector == det) & (boot.metric == metric)].value.to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            return [_f(np.percentile(v, 2.5)), _f(np.percentile(v, 97.5))] if len(v) else [None, None]

        wide = boot.pivot_table(index="rep", columns=["detector", "metric"], values="value", dropna=False)
        diffs = {}
        for b in [d for d in scores if d != "M"]:
            diffs[f"M_minus_{b}"] = {}
            for m in ("far_at_recall50", "ap", "recall_at_far1"):
                v = (wide[("M", m)] - wide[(b, m)]).to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                diffs[f"M_minus_{b}"][m] = {"est": _f(point["M"][m] - point[b][m]),
                                           "ci95": [_f(np.percentile(v, 2.5)), _f(np.percentile(v, 97.5))]}

        baselines = [d for d in scores if d != "M"]
        h1_parts = []
        for b in baselines:
            lower = point["M"]["far_at_recall50"] < point[b]["far_at_recall50"]
            up = diffs[f"M_minus_{b}"]["far_at_recall50"]["ci95"][1]
            h1_parts.append((b, lower and up is not None and up < 0))
        h1 = all(ok for _, ok in h1_parts)
        h1_detail = "; ".join(f"vs {b}: {'pass' if ok else 'fail'} (FAR@R50 M={point['M']['far_at_recall50']:.4f}, "
                              f"{b}={point[b]['far_at_recall50']:.4f}, diff CI95 upper="
                              f"{diffs[f'M_minus_{b}']['far_at_recall50']['ci95'][1]})" for b, ok in h1_parts)
        if excluded:
            h1_detail += f"; excluded detectors: {excluded}"

        lopo_m = [r for r in calib["lopo"] if r["detector"] == "M" and r["pass"] is not None]
        h3 = bool(lopo_m) and all(r["pass"] for r in lopo_m)
        h3_detail = (f"calibration mode {calib['mode']}; " +
                     "; ".join(f"{r['left_out']} a={r['alpha']}: realized {r['realized']:.4f} "
                               f"(filtered {r['realized_filtered']:.4f}) {'pass' if r['pass'] else 'fail'}"
                               for r in lopo_m)) if lopo_m else "no leave-one-out rows for M"

        metrics = {
            "site": cfg.site, "commit": _commit(cfg), "config_sha256": cfg.sha256, "versions": _versions(),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_pair": ev.id, "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
            "calibration_mode": calib["mode"], "report_alpha": ra, "excluded_detectors": excluded,
            "detectors": {d: {**{m: _f(point[d][m]) for m in METRICS},
                              "ci95": {m: ci(d, m) for m in METRICS}} for d in scores},
            "differences": diffs,
            "hypotheses": {"H1": {"pass": h1, "detail": h1_detail},
                           "H3": {"pass": h3, "detail": h3_detail}, "H2": None},
        }
        (inter / "metrics.json").write_text(json.dumps(metrics, indent=2))
        curves.to_csv(inter / "curves.csv", index=False, float_format="%.6g")
        boot.to_csv(inter / "bootstrap.csv", index=False, float_format="%.6g")
        fail.to_csv(inter / "failure_table.csv", index=False, float_format="%.6g")
        outputs += [inter / n for n in ("metrics.json", "curves.csv", "bootstrap.csv", "failure_table.csv")]
        for d in scores:
            log.info("%-2s AP %.3f  FAR@R50 %.4f  R@FAR1 %.3f  F1 %.3f", d, point[d]["ap"],
                     point[d]["far_at_recall50"], point[d]["recall_at_far1"], point[d]["f1_at_report_alpha"])
        log.info("H1 %s | H3 %s", "PASS" if h1 else "FAIL", "PASS" if h3 else "FAIL")
