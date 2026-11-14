import pytest
import yaml

from coldstart.config import ConfigError, load_config
from conftest import ROOT, small_synthetic_dict, write_config


def test_shipped_configs_load():
    for name in ("synthetic.yaml", "site.yaml"):
        cfg = load_config(ROOT / "config" / name)
        assert cfg.seed == 20261114


def test_unknown_key_raises(tmp_path):
    d = small_synthetic_dict(tmp_path)
    d["grid"]["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write_config(tmp_path, d))


def test_missing_key_raises(tmp_path):
    d = small_synthetic_dict(tmp_path)
    del d["model"]["max_iter"]
    with pytest.raises(ConfigError, match="missing"):
        load_config(write_config(tmp_path, d))


def test_unsorted_alphas_raise(tmp_path):
    d = small_synthetic_dict(tmp_path)
    d["calibration"]["alphas"] = [0.05, 0.01]
    with pytest.raises(ConfigError, match="alphas"):
        load_config(write_config(tmp_path, d))


def test_real_site_grid_with_null_track_raises(tmp_path):
    d = yaml.safe_load((ROOT / "config" / "site.yaml").read_text())
    d["search"]["track"] = None
    cfg = load_config(write_config(tmp_path, d, "site.yaml"))
    with pytest.raises(ConfigError, match="search.track"):
        cfg.require_scene_selection()
