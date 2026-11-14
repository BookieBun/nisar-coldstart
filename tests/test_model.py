import numpy as np

from coldstart.features import load_features
from coldstart.model import crossfit_quantiles
from coldstart.pairs import event_pair, load_pairs


def _event_fs(cfg):
    ev = event_pair(load_pairs(cfg.interim / "pairs.json"))
    return load_features(cfg.interim / "features" / f"{ev.id}.npz")


def test_quantiles_sorted_and_crossfit_and_deterministic(synth_cfg):
    fs = _event_fs(synth_cfg)
    a = crossfit_quantiles(fs, synth_cfg)
    for q in a.values():
        assert (np.diff(q, axis=1) >= 0).all()

    # Altering the targets of fold k must not change fold k's predictions.
    k = int(np.unique(fs.fold)[0])
    fs.y_hh = fs.y_hh.copy()
    fs.y_hv = fs.y_hv.copy()
    fs.y_hh[fs.fold == k] += 9.0
    fs.y_hv[fs.fold == k] -= 9.0
    b = crossfit_quantiles(fs, synth_cfg)
    for p in ("hh", "hv"):
        assert np.array_equal(a[p][fs.fold == k], b[p][fs.fold == k])
        assert not np.array_equal(a[p][fs.fold != k], b[p][fs.fold != k])

    # Deterministic under the seed.
    c = crossfit_quantiles(_event_fs(synth_cfg), synth_cfg)
    for p in ("hh", "hv"):
        assert np.array_equal(a[p], c[p])
