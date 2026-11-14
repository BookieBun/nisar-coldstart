"""Find NISAR L2 GCOV scenes over the candidate fires and rank them for site selection.

Standalone: needs only asf_search. Searching ASF needs no credentials.

For each candidate fire it searches every NISAR dataset name asf_search knows,
groups results by (track, direction, frame, mode, polarization), and counts
acquisition dates before and after ignition. The fire whose best single group
has the most pre-ignition dates is the preferred site.

Usage:
    python scripts/find_scenes.py                      # all candidates
    python scripts/find_scenes.py --fire McCully       # one fire
    python scripts/find_scenes.py --out data/interim/scene_search.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import asf_search as asf

NISAR_START = "2026-06-17"  # first calibrated GCOV acquisitions

# Handoff section 11 (WFIGS "Current" perimeters). bbox = (W, S, E, N), WGS84.
CANDIDATES = {
    "Timber": {"bbox": (-121.791, 36.192, -121.622, 36.310), "discovered": "2026-08-09", "acres": 25350},
    "McCully": {"bbox": (-118.353, 44.803, -118.187, 44.914), "discovered": "2026-08-22", "acres": 13600},
    "Austin": {"bbox": (-122.064, 44.995, -121.862, 45.056), "discovered": "2026-08-09", "acres": 12890},
    "Grasshopper": {"bbox": (-121.600, 45.237, -121.199, 45.403), "discovered": "2026-07-24", "acres": 93970},
}

# Scene name layout (handoff 5.2):
# NISAR_L2_PR_GCOV_<cycle>_<track>_<dir>_<frame>_<mode>_<pol>_<S>_<start>_...
IDX = {"cycle": 4, "track": 5, "direction": 6, "frame": 7, "mode": 8, "polarization": 9, "start": 11}


def bbox_wkt(b: tuple[float, float, float, float]) -> str:
    w, s, e, n = b
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def nisar_datasets() -> list[str]:
    """Every dataset name asf_search registers for NISAR."""
    try:
        from asf_search.CMR.datasets import dataset_collections
        names = [k for k in dataset_collections if "NISAR" in k.upper()]
        if names:
            return names
    except Exception:
        pass
    return ["NISAR"]


def parse_name(name: str) -> dict | None:
    parts = Path(name).stem.split("_")
    if len(parts) <= IDX["start"] or parts[0] != "NISAR" or "GCOV" not in parts:
        return None
    out = {k: parts[i] for k, i in IDX.items()}
    try:
        out["date"] = datetime.strptime(out["start"][:8], "%Y%m%d").date()
    except ValueError:
        return None
    return out


def total_bytes(props: dict) -> int | None:
    """'bytes' may be an int or a {filename: {'bytes': n}} dict depending on dataset."""
    b = props.get("bytes")
    if isinstance(b, (int, float)):
        return int(b)
    if isinstance(b, dict):
        tot = 0
        for v in b.values():
            n = v.get("bytes") if isinstance(v, dict) else v
            if isinstance(n, (int, float)):
                tot += int(n)
        return tot or None
    return None


def search_fire(name: str, info: dict, end: str, show_props: bool) -> list[dict]:
    rows: dict[str, dict] = {}
    for ds in nisar_datasets():
        try:
            res = asf.search(dataset=ds, processingLevel="GCOV",
                             intersectsWith=bbox_wkt(info["bbox"]),
                             start=NISAR_START, end=end)
        except Exception as exc:
            print(f"  [{name}] dataset={ds!r}: search failed: {exc}", file=sys.stderr)
            continue
        print(f"  [{name}] dataset={ds!r}: {len(res)} results")
        for r in res:
            p = r.properties
            if show_props:
                print("\n  ---- [VERIFY] properties of first result ----")
                for k in sorted(p):
                    print(f"    {k:28s} {type(p[k]).__name__:6s} {repr(p[k])[:110]}")
                print("  ---------------------------------------------\n")
                show_props = False
            scene = p.get("sceneName") or p.get("fileID") or p.get("fileName") or ""
            parsed = parse_name(scene)
            if parsed is None:
                print(f"  [{name}] unparsed scene name, skipped: {scene!r}", file=sys.stderr)
                continue
            rows[scene] = {
                "fire": name, "dataset": ds, "scene_name": scene,
                "date": parsed["date"].isoformat(), "cycle": parsed["cycle"],
                "track": parsed["track"], "direction": parsed["direction"],
                "frame": parsed["frame"], "mode": parsed["mode"],
                "polarization": parsed["polarization"],
                "bytes": total_bytes(p), "url": p.get("url", ""),
            }
    return list(rows.values())


def summarize(name: str, info: dict, rows: list[dict]) -> list[dict]:
    ign = date.fromisoformat(info["discovered"])
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["track"], r["direction"], r["frame"], r["mode"], r["polarization"])].append(r)
    out = []
    for key, rs in groups.items():
        dates = sorted({r["date"] for r in rs})
        pre = [d for d in dates if date.fromisoformat(d) < ign]
        post = [d for d in dates if date.fromisoformat(d) >= ign]
        sizes = [r["bytes"] for r in rs if r["bytes"]]
        out.append({
            "fire": name, "track": key[0], "direction": key[1], "frame": key[2],
            "mode": key[3], "polarization": key[4], "n_dates": len(dates),
            "n_pre": len(pre), "n_post": len(post),
            "pre_dates": " ".join(pre), "post_dates": " ".join(post),
            "total_gb": round(sum(sizes) / 1e9, 2) if sizes else None,
            "gb_per_scene": round(sum(sizes) / len(sizes) / 1e9, 2) if sizes else None,
        })
    out.sort(key=lambda g: (g["n_pre"], g["n_dates"]), reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fire", choices=sorted(CANDIDATES), help="search one fire only")
    ap.add_argument("--end", default=date.today().isoformat(), help="search end date (default today)")
    ap.add_argument("--out", type=Path, default=Path("data/interim/scene_search.csv"))
    ap.add_argument("--min-pre", type=int, default=3,
                    help="pre-ignition dates needed for two no-event pairs (default 3)")
    args = ap.parse_args()

    fires = {args.fire: CANDIDATES[args.fire]} if args.fire else CANDIDATES
    print(f"asf_search {asf.__version__}; datasets tried: {nisar_datasets()}")
    all_rows, all_groups, first = [], [], True
    for name, info in fires.items():
        print(f"\n== {name}: discovered {info['discovered']}, ~{info['acres']:,} acres ==")
        rows = search_fire(name, info, args.end, show_props=first)
        first = first and not rows
        all_rows += rows
        all_groups += summarize(name, info, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nAll scenes written to {args.out}")

    print("\n== Same-track groups, best first ==")
    cols = ("fire", "track", "direction", "frame", "mode", "polarization", "n_pre", "n_post", "gb_per_scene")
    print(" | ".join(cols))
    for g in sorted(all_groups, key=lambda g: (g["n_pre"], g["n_dates"]), reverse=True):
        print(" | ".join(str(g[c]) for c in cols))
        print(f"    pre:  {g['pre_dates']}")
        print(f"    post: {g['post_dates']}")

    ok = [g for g in all_groups if g["n_pre"] >= args.min_pre]
    print()
    if ok:
        b = max(ok, key=lambda g: (g["n_pre"], g["n_dates"]))
        print(f"RECOMMENDED: {b['fire']} track={b['track']} dir={b['direction']} frame={b['frame']} "
              f"mode={b['mode']} pol={b['polarization']} ({b['n_pre']} pre, {b['n_post']} post)")
        return 0
    print(f"No group has >= {args.min_pre} pre-ignition dates. "
          "With exactly 2, H3 falls back to the spatial hold-out (handoff 5.5).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
