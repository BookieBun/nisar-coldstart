import numpy as np

from coldstart.baselines import score_b1, score_b3
from coldstart.features import FEATURE_NAMES, FeatureSet


def _fs(y_hh, y_hv, cls):
    n = len(cls)
    X = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    X[:, FEATURE_NAMES.index("cls")] = cls
    return FeatureSet(X=X, feature_names=list(FEATURE_NAMES), cat_mask=np.array([False] * 5 + [True]),
                      y_hh=np.asarray(y_hh, np.float32), y_hv=np.asarray(y_hv, np.float32),
                      idx=np.arange(n), fold=np.zeros(n, np.int8))


def test_b1_is_max_abs_change():
    fs = _fs([1.0, -3.0, 0.5, 0.0], [-2.0, 1.0, 0.25, 0.0], [0, 0, 1, 1])
    assert np.allclose(score_b1(fs, (2, 2)).ravel(), [2.0, 3.0, 0.5, 0.0])


def test_b3_zero_at_class_median(cfg_small):
    y = np.array([0.0, 1.0, 1.0, 1.0, 5.0, -4.0, -2.0, -2.0, -2.0, 3.0])
    cls = np.array([0] * 5 + [1] * 5)
    s = score_b3(_fs(y, y, cls), (2, 5), cfg_small).ravel()
    assert np.allclose(s[[1, 2, 3, 6, 7, 8]], 0.0)
    assert (s[[0, 4, 5, 9]] > 0).all()


def test_b3_removes_class_wide_shift(cfg_small):
    rng = np.random.default_rng(1)
    cls = np.repeat([0, 1], 500)
    base = rng.normal(0, 0.3, 1000)
    shifted = base + np.where(cls == 1, 4.0, 0.0)
    a = score_b3(_fs(base, base, cls), (10, 100), cfg_small)
    b = score_b3(_fs(shifted, shifted, cls), (10, 100), cfg_small)
    assert np.allclose(a, b, atol=1e-5)
