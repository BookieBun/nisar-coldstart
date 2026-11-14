"""Command line: one subcommand per stage, plus `run`. Every command takes the config path."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .log import get_logger, require_inputs, stage

log = get_logger()

HELP = {
    "synth": "write a synthetic cube and perimeter (synthetic mode)",
    "search": "search ASF for GCOV scenes; write manifest.csv",
    "download": "download the selected scenes (checks free disk first)",
    "grid": "read GCOV windows; write cube.npz and cube_meta.json",
    "truth": "rasterize the fire perimeter to labels.npy",
    "features": "define pairs, fit land classes, write one feature table per pair",
    "detect": "score every pair with B1, B2, B3 and M",
    "calibrate": "conformal thresholds and the leave-one-pair-out H3 table",
    "evaluate": "PR curves, metrics, block bootstrap, H1/H3 verdicts",
    "export": "write display artifacts and figures to artifacts/<site>/",
    "run": "run every stage from synth/grid through export",
}


def run_detect(cfg: Config) -> None:
    from .baselines import score_b1, score_b2, score_b3
    from .features import load_features
    from .grid import load_cube
    from .model import score_m
    from .pairs import load_pairs

    inter = cfg.interim
    require_inputs("detect", inter / "pairs.json", inter / "features", inter / "cube.npz")
    with stage("detect") as outputs:
        cube = load_cube(cfg)
        pairs = load_pairs(inter / "pairs.json")
        for d in ("B1", "B2", "B3", "M"):
            (inter / "scores" / d).mkdir(parents=True, exist_ok=True)
        (inter / "quantiles").mkdir(parents=True, exist_ok=True)
        for p in pairs:
            f = inter / "features" / f"{p.id}.npz"
            require_inputs("detect", f)
            fs = load_features(f)
            cells = np.zeros(cube.shape[0] * cube.shape[1], dtype=bool)
            cells[fs.idx] = True
            cells = cells.reshape(cube.shape)
            sc = {"B1": score_b1(fs, cube.shape), "B2": score_b2(cube, p, cells, cfg),
                  "B3": score_b3(fs, cube.shape, cfg)}
            sc["M"], qmaps = score_m(fs, cube.shape, cfg)
            for d, m in sc.items():
                np.save(inter / "scores" / d / f"{p.id}.npy", m.astype(np.float32))
            np.savez_compressed(inter / "quantiles" / f"{p.id}.npz", **{k: v.astype(np.float16) for k, v in qmaps.items()})
            log.info("scored pair %s (%s)", p.id, p.kind)
        outputs += [inter / "scores", inter / "quantiles"]


def _stages() -> dict:
    from .conformal import run_calibration
    from .download import download_scenes
    from .evaluate import run_evaluate
    from .export import export_artifacts
    from .features import run_features
    from .grid import build_cube
    from .search import search_scenes
    from .synthetic import make_synthetic
    from .truth import run_truth
    return {"synth": make_synthetic, "search": search_scenes, "download": download_scenes, "grid": build_cube,
            "truth": run_truth, "features": run_features, "detect": run_detect, "calibrate": run_calibration,
            "evaluate": run_evaluate, "export": export_artifacts}


def run_all(cfg: Config) -> None:
    st = _stages()
    first = "synth" if cfg.mode == "synthetic" else "grid"
    for name in (first, "truth", "features", "detect", "calibrate", "evaluate", "export"):
        st[name](cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="coldstart", description="Cold-start change detection for NISAR.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    for name, h in HELP.items():
        sp = sub.add_parser(name, help=h, description=h)
        sp.add_argument("config", type=Path, help="path to a YAML config, e.g. config/synthetic.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "synth" and cfg.mode != "synthetic":
        ap.error("synth needs a config with mode: synthetic")
    if args.cmd in ("grid", "search", "download") and cfg.mode != "real":
        ap.error(f"{args.cmd} needs a config with mode: real")
    if args.cmd == "run":
        run_all(cfg)
    else:
        _stages()[args.cmd](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
