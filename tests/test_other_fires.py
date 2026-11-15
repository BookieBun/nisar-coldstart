"""Stored other-fire artifacts (scripts/other_fires.py) agree with the stored labels and with themselves."""
import json

import numpy as np
import pytest

from conftest import ROOT

SITES = [p.parent.name for p in sorted((ROOT / "artifacts").glob("*/other_fires.json"))]


@pytest.mark.parametrize("site", SITES)
def test_other_fire_cells_were_left_out_of_the_scores(site):
    art = ROOT / "artifacts" / site
    ids = np.load(art / "other_fires.npz")["ids"]
    disp = np.load(art / "display.npz")
    assert ids.shape == disp["labels"].shape
    # truth.py sets other-perimeter cells to -1, so none of them can be a scored inside or control cell
    assert not np.any((ids > 0) & (disp["labels"] != -1))


@pytest.mark.parametrize("site", SITES)
def test_other_fire_summary_is_consistent(site):
    art = ROOT / "artifacts" / site
    j = json.loads((art / "other_fires.json").read_text())
    ids = np.load(art / "other_fires.npz")["ids"]
    known = {f["id"] for f in j["fires"]}
    assert set(np.unique(ids[ids > 0]).tolist()) <= known
    assert j["exploratory"] is True
    meta = json.loads((art / "display_meta.json").read_text())
    for det in meta["detectors"]:
        for a in meta["alphas"]:
            r = j["rates"][det][str(a)]
            assert 0.0 <= r["control"] <= 1.0
            assert set(r["per_fire"]) == {str(k) for k in known}
