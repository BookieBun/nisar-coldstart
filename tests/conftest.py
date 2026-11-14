"""Shared fixtures: a small synthetic config written into a temp project, fixed seed."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def small_synthetic_dict(tmp_path: Path) -> dict:
    d = yaml.safe_load((ROOT / "config" / "synthetic.yaml").read_text())
    d = copy.deepcopy(d)
    d["paths"] = {"data_root": str(tmp_path / "data"), "artifacts_root": str(tmp_path / "artifacts")}
    d["perimeter"]["path"] = str(tmp_path / "data" / "perimeter" / "synthetic.geojson")
    s = d["synthetic"]
    s.update(shape=[120, 120], disc_radius_cells=12, missing_block=[2, 10, 10, 5], blob_sigma=5.0)
    d["grid"].update(tile_km=2, n_folds=3)
    d["features"]["kmeans_max_cells"] = 20000
    d["model"].update(max_iter=30, max_train_cells=6000, min_samples_leaf=50)
    d["labels"]["negative_buffer_km"] = 0.5
    d["evaluate"].update(bootstrap_block_km=1, bootstrap_n=40)
    d["display"]["max_display_cells"] = 120
    return d


def write_config(tmp_path: Path, d: dict, name: str = "synthetic.yaml") -> Path:
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "config" / name
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return p


@pytest.fixture
def cfg_small(tmp_path):
    from coldstart.config import load_config
    return load_config(write_config(tmp_path, small_synthetic_dict(tmp_path)))


@pytest.fixture
def synth_cfg(cfg_small):
    """Small synthetic site with cube and features already built."""
    from coldstart.features import run_features
    from coldstart.synthetic import make_synthetic
    make_synthetic(cfg_small)
    run_features(cfg_small)
    return cfg_small
