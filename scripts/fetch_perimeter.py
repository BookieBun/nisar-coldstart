"""Fetch the fire perimeter (and other perimeters in the window) from the NIFC WFIGS service.

Writes a GeoJSON FeatureCollection in WGS84 to the config's perimeter.path and prints the
incident's dates, so event.ignition and event.end can be checked against the source.

Usage:
    python scripts/fetch_perimeter.py config/site.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone

import requests

from coldstart.config import load_config

SERVICE = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
           "WFIGS_Interagency_Perimeters/FeatureServer/0/query")
DATE_FIELDS = ["attr_FireDiscoveryDateTime", "attr_ContainmentDateTime", "attr_ControlDateTime",
               "attr_FireOutDateTime", "poly_DateCurrent", "poly_PolygonDateTime", "poly_CreateDate"]


def _iso(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    w, s, e, n = cfg.search.bbox_wgs84
    lat = (s + n) / 2
    dlat = cfg.grid.buffer_km / 111.0
    dlon = cfg.grid.buffer_km / (111.0 * math.cos(math.radians(lat)))
    env = f"{w - dlon},{s - dlat},{e + dlon},{n + dlat}"
    params = {"where": "1=1", "geometry": env, "geometryType": "esriGeometryEnvelope", "inSR": 4326,
              "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": 4326, "f": "geojson"}
    r = requests.get(SERVICE, params=params, timeout=120)
    r.raise_for_status()
    gj = r.json()
    if "features" not in gj:
        raise RuntimeError(f"unexpected response: {str(gj)[:300]}")
    year = cfg.event.ignition[:4]
    keep = []
    for f in gj["features"]:
        p = f.get("properties") or {}
        disc = _iso(p.get("attr_FireDiscoveryDateTime")) or ""
        if disc.startswith(year) or str(p.get(cfg.perimeter.name_field, "")).strip().lower() == cfg.perimeter.incident_name.lower():
            keep.append(f)
    out = {"type": "FeatureCollection", "features": keep}
    cfg.perimeter_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.perimeter_path.write_text(json.dumps(out))
    print(f"{len(gj['features'])} perimeters intersect the window; {len(keep)} from {year} kept -> {cfg.perimeter_path}")
    for f in keep:
        p = f["properties"]
        name = p.get(cfg.perimeter.name_field)
        tag = "TARGET" if str(name).strip().lower() == cfg.perimeter.incident_name.lower() else "other "
        acres = p.get("poly_GISAcres") or p.get("attr_IncidentSize")
        print(f"  {tag} {name!s:24s} acres={acres}")
        for k in DATE_FIELDS:
            if p.get(k) is not None:
                print(f"         {k:28s} {_iso(p[k])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
