"""scripts/flag_summary.py reports exactly what the stored outputs say."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from coldstart.cli import run_all
from coldstart.conformal import load_thresholds
from coldstart.evaluate import flags_at
from coldstart.pairs import event_pair, load_pairs

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("flag_summary", ROOT / "scripts" / "flag_summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_matches_stored_outputs(cfg_small):
    run_all(cfg_small)
    s = _load_script().summarize(cfg_small)
    d = cfg_small.interim
    labels = np.load(d / "labels.npy")
    ev = event_pair(load_pairs(d / "pairs.json"))
    thr = load_thresholds(d / "calibration.json")
    cal = json.loads((d / "calibration.json").read_text())
    assert s["n_inside"] == int((labels == 1).sum()) and s["n_outside"] == int((labels == 0).sum())
    for det, by_alpha in s["detectors"].items():
        score = np.load(d / "scores" / det / f"{ev.id}.npy")
        for a, x in by_alpha.items():
            f = flags_at(score, thr[det][float(a)], cfg_small.postprocess.min_patch_cells)
            assert np.isclose(x["inside_flagged"], f[labels == 1].mean())
            assert np.isclose(x["outside_flagged"], f[labels == 0].mean())
            held = [r["realized"] for r in cal["lopo"] if r["detector"] == det and abs(r["alpha"] - float(a)) < 1e-12]
            assert x["heldout_pairs"] == len(held)
            if held:
                assert np.isclose(x["heldout_min"], min(held)) and np.isclose(x["heldout_max"], max(held))
