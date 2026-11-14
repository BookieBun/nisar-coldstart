import numpy as np

from coldstart.features import FEATURE_NAMES, fit_classes, load_features, nan_std_filter
from coldstart.grid import load_cube
from coldstart.pairs import load_pairs

FORBIDDEN = {"x", "y", "row", "col", "lat", "lon", "tile", "tile_id", "fold", "idx"}


def test_feature_table(synth_cfg):
    for p in load_pairs(synth_cfg.interim / "pairs.json"):
        fs = load_features(synth_cfg.interim / "features" / f"{p.id}.npz")
        assert fs.feature_names == ["hh0", "hv0", "ratio0", "tex0", "terrain", "cls"] == FEATURE_NAMES
        assert not FORBIDDEN & set(fs.feature_names)
        assert np.isfinite(fs.X).all()
        assert fs.cat_mask.tolist() == [False] * 5 + [True]


def test_classes_ignore_post_ignition_dates(synth_cfg):
    cube = load_cube(synth_cfg)
    a = fit_classes(cube, synth_cfg)
    post = [i for i, d in enumerate(cube.dates) if d >= synth_cfg.event.ignition]
    cube.hv_db[post] += 7.0
    cube.hh_db[post] -= 5.0
    assert np.array_equal(a, fit_classes(cube, synth_cfg))


def test_nan_std_filter_matches_direct():
    rng = np.random.default_rng(0)
    x = rng.normal(-15, 2, (30, 30)).astype(np.float32)
    x[5, 5] = np.nan
    s = nan_std_filter(x, 3)
    w = x[9:12, 9:12]
    assert np.isclose(s[10, 10], np.nanstd(w), atol=1e-4)
    w2 = x[4:7, 4:7]
    assert np.isclose(s[5, 5], np.nanstd(w2), atol=1e-4)
    assert (s[np.isfinite(s)] >= 0).all()
