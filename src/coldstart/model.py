"""Pooled conditional model M: cross-fitted gradient-boosted quantile regression.

For each fold k, polarization p and quantile tau, a model trained on cells outside fold k
predicts the change d_p for cells in fold k. Each cell is therefore scored by models that
never saw its tile. Score: r_p = (d_p - q50) / max((q95 - q05)/2, floor); s = max_p |r_p|.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Config
from .features import FeatureSet, to_map
from .log import get_logger

log = get_logger()


def qname(tau: float) -> str:
    return f"q{int(round(tau * 100)):02d}"


def _regressor(tau: float, cat_mask: np.ndarray, cfg: Config) -> HistGradientBoostingRegressor:
    m = cfg.model
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=tau, learning_rate=m.learning_rate, max_iter=m.max_iter,
        max_leaf_nodes=m.max_leaf_nodes, min_samples_leaf=m.min_samples_leaf,
        l2_regularization=m.l2_regularization, categorical_features=cat_mask,
        early_stopping=False, random_state=cfg.seed)


def crossfit_quantiles(fs: FeatureSet, cfg: Config) -> dict[str, np.ndarray]:
    """Return {'hh': (N,3), 'hv': (N,3)} sorted quantile predictions per cell."""
    taus = list(cfg.model.quantiles)
    preds = {p: np.full((len(fs.idx), len(taus)), np.nan) for p in ("hh", "hv")}
    t_pair = time.perf_counter()
    for k in np.unique(fs.fold):
        test = fs.fold == k
        train = np.flatnonzero(~test)
        if len(train) == 0:
            raise ValueError(f"fold {k} holds every cell; need more tiles")
        rng = np.random.default_rng([cfg.seed, int(k)])
        if len(train) > cfg.model.max_train_cells:
            train = np.sort(rng.choice(train, size=cfg.model.max_train_cells, replace=False))
        for p, y in (("hh", fs.y_hh), ("hv", fs.y_hv)):
            for j, tau in enumerate(taus):
                t0 = time.perf_counter()
                reg = _regressor(tau, fs.cat_mask, cfg).fit(fs.X[train], y[train])
                preds[p][test, j] = reg.predict(fs.X[test])
                log.debug("fold %d %s tau=%.2f fit %.1fs", k, p, tau, time.perf_counter() - t0)
    for p in preds:
        preds[p].sort(axis=1)  # repair quantile crossing
        if not np.isfinite(preds[p]).all():
            raise RuntimeError("some cells received no cross-fitted prediction")
    dt = time.perf_counter() - t_pair
    if dt > 600:
        log.warning("model took %.0f s for one pair; consider model.max_train_cells 150000", dt)
    return preds


def score_m(fs: FeatureSet, shape, cfg: Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    preds = crossfit_quantiles(fs, cfg)
    floor = cfg.model.spread_floor_db
    names = [qname(t) for t in cfg.model.quantiles]
    score = np.zeros(len(fs.idx))
    maps: dict[str, np.ndarray] = {}
    for p, y in (("hh", fs.y_hh), ("hv", fs.y_hv)):
        q = preds[p]
        r = (y - q[:, 1]) / np.maximum((q[:, 2] - q[:, 0]) / 2.0, floor)
        score = np.maximum(score, np.abs(r))
        for j, n in enumerate(names):
            maps[f"{p}_{n}"] = to_map(q[:, j].astype(np.float32), fs.idx, shape)
        if p == "hv":
            maps["hv_resid_signed"] = to_map(r.astype(np.float32), fs.idx, shape)
    return to_map(score.astype(np.float32), fs.idx, shape), maps
