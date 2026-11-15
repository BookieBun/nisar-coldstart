"""Side-by-side figure: the standard before/after difference (B1) and the pooled model (M), both at the
same calibrated false-alarm setting, zoomed on the fire. Reads only the committed display artifacts and
flag_summary.json; computes nothing new.

Usage:
    python scripts/compare_figure.py config/site.yaml   # writes artifacts/<site>/figures/compare_methods.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

from coldstart.config import load_config

def pct(v: float) -> str:
    return f"{v:.0%}" if v >= 0.1 else f"{v:.1%}"


PANELS = (("B1", "Standard before/after difference"), ("M", "Our model (learns from the whole scene)"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = load_config(ap.parse_args().config)
    art = cfg.artifacts
    z = np.load(art / "display.npz")
    meta = json.loads((art / "display_meta.json").read_text())
    summ = json.loads((art / "flag_summary.json").read_text())
    alpha = cfg.calibration.report_alpha
    bit = meta["alphas"].index(alpha)
    lab = z["labels"] == 1
    rr, cc = np.nonzero(lab)
    pad_r, pad_c = max(25, (rr.max() - rr.min()) // 2), max(25, (cc.max() - cc.min()) // 2)
    r0, r1 = max(0, rr.min() - pad_r), min(lab.shape[0], rr.max() + pad_r)
    c0, c1 = max(0, cc.min() - pad_c), min(lab.shape[1], cc.max() + pad_c)
    sl = (slice(r0, r1), slice(c0, c1))
    edge = (lab & ~binary_erosion(lab))[sl]
    hv = z["hv_after"].astype(np.float32)[sl]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (det, title) in zip(axes, PANELS):
        flags = ((z[f"flags_{det}"].astype(np.uint8) >> bit) & 1).astype(bool)[sl]
        ax.imshow(hv, cmap="gray", vmin=-25, vmax=-5, interpolation="nearest")
        rgba = np.zeros(flags.shape + (4,))
        rgba[flags] = (0.86, 0.12, 0.12, 0.95)
        rgba[edge] = (0.0, 0.78, 1.0, 1.0)
        ax.imshow(rgba, interpolation="nearest")
        s = summ["detectors"][det][str(alpha)]
        ax.set_title(f"{title}\nflags {pct(s['inside_flagged'])} of the cells inside the perimeter, "
                     f"{s['outside_flagged']:.1%} of the land outside", fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"Same false-alarm setting (1 in {round(1 / alpha)}), same radar dates "
                 f"({meta['event_pair']['date_a']} to {meta['event_pair']['date_b']}). "
                 "Red: flagged. Cyan: mapped fire perimeter.", fontsize=11)
    fig.text(0.5, 0.015, "Inside and outside refer to the agency-mapped perimeter, not verified ground truth. "
             "Outside shares are for land more than 2 km beyond it.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = art / "figures" / "compare_methods.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
