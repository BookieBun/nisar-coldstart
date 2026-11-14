import numpy as np
from sklearn.metrics import average_precision_score

from coldstart.evaluate import _Sorted, block_bootstrap, flags_at, point_metrics


def _labels(H=60, W=60):
    lab = np.zeros((H, W), np.int8)
    lab[20:35, 20:35] = 1
    lab[18:20, :] = -1
    return lab


def test_perfect_scores(cfg_small):
    lab = _labels()
    s = (lab == 1).astype(np.float32)
    m = point_metrics(s, lab, 0.5, cfg_small)
    assert m["ap"] == 1.0 and m["far_at_recall50"] == 0.0 and m["recall_at_far1"] == 1.0


def test_random_scores_ap_near_positive_rate(cfg_small):
    rng = np.random.default_rng(0)
    lab = np.zeros((300, 300), np.int8)
    lab[rng.random(lab.shape) < 0.1] = 1
    m = point_metrics(rng.random(lab.shape).astype(np.float32), lab, 0.9, cfg_small)
    assert abs(m["ap"] - 0.1) < 0.01


def test_weighted_ap_matches_sklearn():
    rng = np.random.default_rng(3)
    y = (rng.random(5000) < 0.2).astype(int)
    s = np.round(rng.normal(y, 1.0), 1)                # many ties on purpose
    w = rng.integers(0, 4, 5000).astype(float)
    st = _Sorted(s, y)
    assert np.isclose(st.stats(None, 0.5, 0.01)["ap"], average_precision_score(y, s))
    assert np.isclose(st.stats(w, 0.5, 0.01)["ap"], average_precision_score(y, s, sample_weight=w))


def test_bootstrap_ci_contains_point(cfg_small):
    rng = np.random.default_rng(1)
    lab = _labels()
    s = (lab == 1) + rng.normal(0, 0.6, lab.shape)
    s = s.astype(np.float32)
    thr = 0.8
    point = point_metrics(s, lab, thr, cfg_small)
    boot = block_bootstrap({"M": s}, lab, cfg_small, {"M": flags_at(s, thr, cfg_small.postprocess.min_patch_cells)})
    for metric in ("ap", "far_at_recall50", "recall_at_far1"):
        v = boot[boot.metric == metric].value.to_numpy(float)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        assert lo <= point[metric] <= hi
