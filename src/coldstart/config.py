"""Load and validate the YAML config.

Every key is required (values may be null only where noted), and unknown keys
raise. There are no defaults in code: a run is fully defined by its config file.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
from pathlib import Path
from typing import Any

import yaml

REQ = "__required__"        # key must be present and non-null
NULLABLE = "__nullable__"   # key must be present; null allowed

SCHEMA: dict[str, Any] = {
    "site": REQ, "mode": REQ, "seed": REQ,
    "paths": {"data_root": REQ, "artifacts_root": REQ},
    "search": {"dataset": REQ, "bbox_wgs84": REQ, "start": REQ, "end": REQ,
               "track": NULLABLE, "frame": NULLABLE, "direction": NULLABLE,
               "mode": NULLABLE, "polarization": NULLABLE},
    "perimeter": {"path": REQ, "incident_name": REQ, "name_field": REQ},
    "event": {"ignition": REQ, "end": NULLABLE},
    "grid": {"freq": REQ, "buffer_km": REQ, "target_cell_m": REQ, "min_valid_fraction": REQ,
             "tile_km": REQ, "n_folds": REQ},
    "mask": {"water_hv_db_max": REQ, "terrain_pct": REQ, "min_valid_area_fraction": REQ},
    "features": {"n_classes": REQ, "tex_window": REQ, "kmeans_max_cells": REQ,
                 "use_worldcover": REQ, "use_rain": REQ},
    "model": {"quantiles": REQ, "learning_rate": REQ, "max_iter": REQ, "max_leaf_nodes": REQ,
              "min_samples_leaf": REQ, "l2_regularization": REQ, "max_train_cells": REQ,
              "spread_floor_db": REQ},
    "baselines": {"b2_std_floor_db": REQ, "b3_spread_floor_db": REQ, "b2_history_dates": REQ},
    "postprocess": {"min_patch_cells": REQ},
    "calibration": {"alphas": REQ, "max_cal_cells_per_pair": REQ, "report_alpha": REQ,
                    "h3_alphas": REQ, "h3_band": REQ},
    "labels": {"shrink_cells": REQ, "negative_buffer_km": REQ},
    "evaluate": {"recall_target": REQ, "far_target": REQ, "bootstrap_block_km": REQ, "bootstrap_n": REQ},
    "display": {"max_display_cells": REQ},
}

# Sections required only in one mode.
MODE_SECTIONS: dict[str, dict[str, Any]] = {
    "synthetic": {"synthetic": {
        "shape": REQ, "n_dates": REQ, "start_date": REQ, "interval_days": REQ, "n_classes": REQ,
        "blob_sigma": REQ, "hh_base_db": REQ, "hv_base_db": REQ, "hh_class_offset_db": REQ,
        "hv_class_offset_db": REQ, "static_sd_db": REQ, "class_noise_sd_db": REQ,
        "shared_shift_sd_db": REQ, "brightness_coupling_sd": REQ, "terrain_sd": REQ,
        "event_index": REQ, "disc_radius_cells": REQ, "hh_drop_db": REQ, "hv_drop_db": REQ,
        "missing_block": REQ, "origin_xy": REQ, "crs_epsg": REQ}},
    "real": {"io": {"stream": REQ, "max_download_gb": REQ, "min_free_gb": REQ, "workers": REQ}},
}

DATE_KEYS = [("search", "start"), ("search", "end"), ("event", "ignition"), ("event", "end"),
             ("synthetic", "start_date")]


class ConfigError(ValueError):
    pass


class Section:
    """Read-only attribute access over a nested dict."""

    def __init__(self, d: dict, where: str = ""):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_where", where)

    def __getattr__(self, k: str) -> Any:
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(f"config has no key '{self._where}{k}'") from None
        return Section(v, f"{self._where}{k}.") if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        raise AttributeError("config is read-only")

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def to_dict(self) -> dict:
        return copy.deepcopy(self._d)


class Config(Section):
    def __init__(self, d: dict, path: Path, sha256: str):
        super().__init__(d)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", sha256)
        # Project root: the folder that holds config/. Relative paths resolve against it.
        object.__setattr__(self, "root", path.resolve().parent.parent)

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.root / p

    @property
    def interim(self) -> Path:
        return self.resolve(self.paths.data_root) / "interim" / self.site

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.paths.data_root) / "raw" / self.site

    @property
    def artifacts(self) -> Path:
        return self.resolve(self.paths.artifacts_root) / self.site

    @property
    def perimeter_path(self) -> Path:
        return self.resolve(self.perimeter.path)

    def require_scene_selection(self) -> None:
        """Real-site `grid` needs the scene group chosen with scripts/find_scenes.py."""
        if self.mode != "real":
            return
        missing = [k for k in ("track", "frame", "direction", "mode", "polarization")
                   if getattr(self.search, k) is None]
        if missing:
            raise ConfigError(f"search.{', search.'.join(missing)} must be set before `grid` "
                              "on a real site (run scripts/find_scenes.py)")
        if self.event.end is None:
            raise ConfigError("event.end must be set before `grid` on a real site")


def _check(d: Any, schema: dict, where: str) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"'{where or 'root'}' must be a mapping")
    unknown = sorted(set(d) - set(schema))
    if unknown:
        raise ConfigError(f"unknown config key(s) under '{where or 'root'}': {unknown}")
    for k, rule in schema.items():
        key = f"{where}{k}"
        if k not in d:
            raise ConfigError(f"missing config key '{key}'")
        if isinstance(rule, dict):
            _check(d[k], rule, f"{key}.")
        elif rule == REQ and d[k] is None:
            raise ConfigError(f"config key '{key}' must not be null")


def _norm_dates(d: dict) -> None:
    for sec, key in DATE_KEYS:
        if sec in d and isinstance(d[sec], dict) and isinstance(d[sec].get(key), (_dt.date, _dt.datetime)):
            d[sec][key] = d[sec][key].isoformat()[:10]
        v = d.get(sec, {}).get(key) if isinstance(d.get(sec), dict) else None
        if v is not None:
            try:
                _dt.date.fromisoformat(str(v))
            except ValueError:
                raise ConfigError(f"{sec}.{key} must be an ISO date YYYY-MM-DD, got {v!r}") from None


def validate(d: dict) -> None:
    mode = d.get("mode")
    if mode not in MODE_SECTIONS:
        raise ConfigError(f"mode must be one of {sorted(MODE_SECTIONS)}, got {mode!r}")
    _check(d, {**SCHEMA, **MODE_SECTIONS[mode]}, "")
    _norm_dates(d)

    a = d["calibration"]["alphas"]
    if not (isinstance(a, list) and a and all(0 < x < 0.5 for x in a) and a == sorted(a) and len(set(a)) == len(a)):
        raise ConfigError(f"calibration.alphas must be strictly increasing values in (0, 0.5), got {a}")
    if d["calibration"]["report_alpha"] not in a:
        raise ConfigError("calibration.report_alpha must be one of calibration.alphas")
    if not set(d["calibration"]["h3_alphas"]) <= set(a):
        raise ConfigError("calibration.h3_alphas must be a subset of calibration.alphas")
    q = d["model"]["quantiles"]
    if not (isinstance(q, list) and len(q) == 3 and q == sorted(q) and all(0 < x < 1 for x in q)):
        raise ConfigError(f"model.quantiles must be exactly three sorted values in (0, 1), got {q}")
    if d["baselines"]["b2_history_dates"] < 2:
        raise ConfigError("baselines.b2_history_dates must be >= 2 (one date gives no spread)")
    if d["event"]["end"] is not None and d["event"]["end"] < d["event"]["ignition"]:
        raise ConfigError("event.end must not precede event.ignition")


def load_config(path: Path | str) -> Config:
    path = Path(path)
    raw = path.read_bytes()
    d = yaml.safe_load(raw)
    validate(d)
    return Config(d, path, hashlib.sha256(raw).hexdigest())
