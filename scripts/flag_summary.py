"""Per-setting summary for the app's "What this setting means" panel.

For every detector and calibrated setting (alpha) it records, from stored outputs only:
  - the realized false-alarm rates on the held-out no-event pairs (from calibration.json; raw, as tested);
  - on the event pair, after the minimum-patch filter (as mapped): the share of cells inside the
    reference perimeter that are flagged, and the share of control cells (more than
    labels.negative_buffer_km outside it) that are flagged.

It uses the same thresholds and flag rule as the evaluation; it fits nothing and changes no result.

Usage:
    python scripts/flag_summary.py config/site.yaml     # writes artifacts/<site>/flag_summary.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from coldstart.config import Config, load_config
from coldstart.conformal import DETECTORS, load_thresholds
from coldstart.evaluate import flags_at
from coldstart.pairs import event_pair, load_pairs


def summarize(cfg: Config) -> dict:
    d = cfg.interim
    labels = np.load(d / "labels.npy")
    ev = event_pair(load_pairs(d / "pairs.json"))
    thr = load_thresholds(d / "calibration.json")
    cal = json.loads((d / "calibration.json").read_text())
    inside, outside = labels == 1, labels == 0
    out = {"site": cfg.site, "event_pair": ev.id, "n_inside": int(inside.sum()), "n_outside": int(outside.sum()),
           "negative_buffer_km": cfg.labels.negative_buffer_km, "detectors": {}}
    for det in DETECTORS:
        score = np.load(d / "scores" / det / f"{ev.id}.npy")
        out["detectors"][det] = {}
        for a in cfg.calibration.alphas:
            f = flags_at(score, thr[det][float(a)], cfg.postprocess.min_patch_cells)
            held = [r["realized"] for r in cal["lopo"] if r["detector"] == det and abs(r["alpha"] - a) < 1e-12]
            out["detectors"][det][str(a)] = {
                "inside_flagged": float(f[inside].mean()),
                "outside_flagged": float(f[outside].mean()),
                "heldout_min": float(min(held)) if held else None,
                "heldout_max": float(max(held)) if held else None,
                "heldout_pairs": len(held),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    s = summarize(cfg)
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts / "flag_summary.json").write_text(json.dumps(s, indent=2))
    m = s["detectors"]["M"][str(cfg.calibration.report_alpha)]
    print(f"{cfg.site}: M at alpha {cfg.calibration.report_alpha}: inside {m['inside_flagged']:.1%}, "
          f"outside {m['outside_flagged']:.2%}, held-out {m['heldout_min']:.3f}-{m['heldout_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
