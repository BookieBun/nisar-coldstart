import json

from coldstart.cli import run_all

INTERIM = ["cube.npz", "cube_meta.json", "labels.npy", "pairs.json", "cls.npy", "calibration.json",
           "metrics.json", "curves.csv", "bootstrap.csv", "failure_table.csv"]
ARTIFACTS = ["display.npz", "display_meta.json", "metrics.json", "curves.csv", "calibration.json",
             "failure_table.csv", "figures/pr_curves.png", "figures/h3_realized.png", "figures/maps.png"]


def test_run_synthetic_end_to_end(cfg_small):
    run_all(cfg_small)
    for f in INTERIM:
        assert (cfg_small.interim / f).exists(), f
    for f in ARTIFACTS:
        assert (cfg_small.artifacts / f).exists(), f
    pairs = json.loads((cfg_small.interim / "pairs.json").read_text())
    for p in pairs:
        assert (cfg_small.interim / "features" / f"{p['id']}.npz").exists()
        for d in ("B1", "B2", "B3", "M"):
            assert (cfg_small.interim / "scores" / d / f"{p['id']}.npy").exists()

    m1 = json.loads((cfg_small.interim / "metrics.json").read_text())
    assert m1["detectors"]["M"]["ap"] > m1["detectors"]["B1"]["ap"]
    assert set(m1["hypotheses"]) == {"H1", "H2", "H3"}

    run_all(cfg_small)
    m2 = json.loads((cfg_small.interim / "metrics.json").read_text())
    m1.pop("created_utc"), m2.pop("created_utc")
    assert m1 == m2
