"""Print the grid group of one GCOV file and sample statistics that resolve the [VERIFY] items.

Usage:
    python scripts/inspect_gcov.py PATH_OR_URL [--freq A] [--sample 512]

PATH_OR_URL may be a local .h5 file or an https:// URL (streamed with byte ranges; needs
Earthdata credentials in ~/.netrc).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from coldstart.io_gcov import GRID_GROUP, describe, open_gcov, read_grid_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--freq", default="A")
    ap.add_argument("--sample", type=int, default=512, help="side of a central sample window, pixels")
    a = ap.parse_args()
    t0 = time.perf_counter()
    with open_gcov(a.src) as f:
        print("top-level groups:", list(f.keys()))
        print(json.dumps(describe(f, a.freq), indent=1, default=str))
        gi = read_grid_info(f, a.freq)
        print("\nGridInfo:", gi)
        g = f[GRID_GROUP.format(freq=a.freq)]
        H, W = g["HHHH"].shape
        r0, c0 = H // 2 - a.sample // 2, W // 2 - a.sample // 2
        for name in ("HHHH", "HVHV", "mask", "numberOfLooks", "rtcGammaToSigmaFactor"):
            if name not in g:
                print(f"{name}: MISSING")
                continue
            v = g[name][r0:r0 + a.sample, c0:c0 + a.sample]
            fin = np.isfinite(v) if v.dtype.kind == "f" else np.ones(v.shape, bool)
            vals = v[fin]
            msg = f"{name:24s} dtype={v.dtype} finite={fin.mean():.3f}"
            if vals.size:
                msg += (f" min={vals.min():.4g} p50={np.median(vals):.4g} max={vals.max():.4g}"
                        f" n<=0={(vals <= 0).mean():.3f}")
                if name == "mask":
                    u, c = np.unique(vals, return_counts=True)
                    msg += f" unique={dict(zip(u.tolist()[:12], c.tolist()[:12]))}"
            print(msg)
    print(f"\nelapsed {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
