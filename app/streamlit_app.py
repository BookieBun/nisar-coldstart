"""Read-only viewer for the cold-start change-detection results.

Reads only artifacts/<site>/ (and docs/LIMITATIONS.md). Runs no models.
    streamlit run app/streamlit_app.py

Wording is written for readers without a radar or machine-learning background; the technical
term for each plain-language label is kept in a tooltip or caption beside it.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from scipy.ndimage import binary_erosion
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent))   # app/places.py, under any launcher
import places  # noqa: E402

# A running server re-executes this script after a code update but keeps imported modules; reload so an
# updated places.py is never paired with a newer caller.
importlib.reload(places)
flagged_places = places.flagged_places
# Cached tables are keyed on this, so an updated places.py never returns a table built by the old one.
PLACES_VERSION = hashlib.sha1(Path(places.__file__).read_bytes()).hexdigest()[:12]

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
# Plain-language names; the technical names used in the docs are kept alongside.
DET_NAMES = {"M": "Our model (whole scene)", "B3": "Baseline: land-type average",
             "B2": "Baseline: own short history", "B1": "Baseline: before/after difference"}
DET_SHORT = {"M": "our model", "B3": "land-type average", "B2": "short history", "B1": "before/after difference"}
DET_TECH = {"M": "M, pooled conditional quantile model", "B3": "B3, stratified class shift",
            "B2": "B2, short-baseline z-score", "B1": "B1, log-ratio"}
SITE_NAMES = {"mccully": "McCully Fire, Oregon (2026)", "austin": "Austin Fire, Oregon (2026)",
              "synthetic": "Simulated test scene (not real data)"}
# Where each site's tests were written down before scoring.
SITE_PREREG = {"mccully": "HYPOTHESES.md, committed before the fire was scored",
               "austin": "HYPOTHESES_REPLICATION.md, pushed to GitHub at 20:21 UTC on 15 November 2026, before "
                         "any radar data for this site were processed"}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
SITE_ROLE = {"mccully": "the original test", "austin": "the pre-registered replication"}
COLORS = {"B1": "#8c8c8c", "B2": "#c08a2b", "B3": "#3b7dbf", "M": "#c23b3b"}

st.set_page_config(page_title="Cold Start: change detection with NISAR", layout="wide")


def odds(alpha: float) -> str:
    """0.05 -> '1 in 20'."""
    return f"1 in {round(1 / alpha)}"


def setting_label(alpha: float) -> str:
    return f"{odds(alpha)} cells ({alpha:.0%})"


# ----------------------------------------------------------------------------- data
def sites() -> list[str]:
    return sorted(p.name for p in ART.iterdir() if (p / "display_meta.json").exists()) if ART.exists() else []


@st.cache_data(show_spinner=False)
def load(site: str) -> dict:
    d = ART / site
    z = np.load(d / "display.npz")
    out = {"arr": {k: z[k] for k in z.files},
           "meta": json.loads((d / "display_meta.json").read_text()),
           "metrics": json.loads((d / "metrics.json").read_text()),
           "calib": json.loads((d / "calibration.json").read_text()),
           "curves": pd.read_csv(d / "curves.csv"),
           "failure": pd.read_csv(d / "failure_table.csv") if (d / "failure_table.csv").exists() else None,
           "summary": json.loads((d / "flag_summary.json").read_text()) if (d / "flag_summary.json").exists() else None,
           "compare_png": (d / "figures" / "compare_methods.png") if (d / "figures" / "compare_methods.png").exists() else None}
    # Other agency-mapped fires in the window (scripts/other_fires.py; exploratory, optional).
    if (d / "other_fires.json").exists() and (d / "other_fires.npz").exists():
        out["other"] = {**json.loads((d / "other_fires.json").read_text()),
                        "ids": np.load(d / "other_fires.npz")["ids"]}
    else:
        out["other"] = None
    return out


def pct(v: float) -> str:
    """Shares under 10% keep one decimal so 0.045 reads 4.5%, not 5%."""
    return f"{v:.0%}" if v >= 0.1 else f"{v:.1%}"


def real_sites() -> list[str]:
    return sorted((x for x in sites() if x != "synthetic"), key=lambda x: (x != "mccully", x))


def site_facts(D: dict) -> dict:
    """Numbers the plain-language text quotes, read from the stored summaries for this site."""
    meta, cal, met = D["meta"], D["calib"], D["metrics"]
    ra = str(meta["report_alpha"])
    summ = (D["summary"] or {}).get("detectors", {})
    m = summ.get("M", {}).get(ra)
    h3 = [r for r in cal["lopo"] if r.get("pass") is not None]
    count = lambda d: (sum(bool(r["pass"]) for r in h3 if r["detector"] == d), sum(r["detector"] == d for r in h3))
    out = {"m": m, "m_h3": count("M"), "b3_h3": count("B3"), "h3_pass": bool(met["hypotheses"]["H3"]["pass"]),
           "all_under_half": all(v.get(ra, {}).get("inside_flagged", 1) < 0.5 for v in summ.values()) if summ else None}
    if m and D["summary"]:
        n_in = m["inside_flagged"] * D["summary"]["n_inside"]
        n_out = m["outside_flagged"] * D["summary"]["n_outside"]
        out["most_outside"] = n_out > n_in
    return out


def setting_meaning(D: dict, det: str, alpha: float) -> None:
    """What the chosen setting did, from stored outputs (scripts/flag_summary.py)."""
    x = (D["summary"] or {}).get("detectors", {}).get(det, {}).get(str(alpha))
    if not x:
        return
    st.markdown("**What this setting did**")
    if x["heldout_pairs"]:
        st.markdown(f"On {x['heldout_pairs']} pairs of dates with no fire, held out from calibration: "
                    f"{x['heldout_min']:.1%} to {x['heldout_max']:.1%} of the land flagged, against a "
                    f"target of {alpha:.0%}.")
    st.markdown(f"On the fire dates, as mapped: **{pct(x['inside_flagged'])}** of the cells inside the fire "
                f"perimeter and **{pct(x['outside_flagged'])}** of the land more than 2 km outside it.")
    st.caption("Held-out shares are before small-patch removal, as tested. Inside and outside refer to the "
               "agency-mapped perimeter, not verified ground truth. Switch methods to compare.")


@st.cache_data(show_spinner=False)
def rgba(site: str, key: str, vmin: float, vmax: float, cmap: str, bit: int | None = None) -> np.ndarray:
    a = load(site)["arr"][key].astype(np.float32)
    if bit is not None:
        on = ((a.astype(np.uint8) >> bit) & 1).astype(bool)
        img = np.zeros(a.shape + (4,), np.uint8)
        img[on] = (220, 30, 30, 230)
        return img
    ok = np.isfinite(a)
    norm = np.clip((np.where(ok, a, vmin) - vmin) / (vmax - vmin), 0, 1)
    img = (colormaps[cmap](norm) * 255).astype(np.uint8)
    img[~ok, 3] = 0
    return img


@st.cache_data(show_spinner=False)
def outline(site: str) -> np.ndarray:
    lab = load(site)["arr"]["labels"] == 1
    edge = lab & ~binary_erosion(lab)
    img = np.zeros(lab.shape + (4,), np.uint8)
    img[edge] = (0, 200, 255, 255)
    return img


@st.cache_data(show_spinner=False)
def other_outline(site: str) -> np.ndarray | None:
    oth = load(site)["other"]
    if not oth or not oth["fires"]:
        return None
    on = oth["ids"] > 0
    edge = on & ~binary_erosion(on)
    img = np.zeros(on.shape + (4,), np.uint8)
    img[edge] = (255, 210, 0, 255)
    return img


def to_mercator_rows(img: np.ndarray, south: float, north: float) -> np.ndarray:
    """Resample rows of a lat/lon (equal-degree) image to equal Web-Mercator spacing, nearest
    neighbour. Leaflet stretches overlays linearly in Mercator; folium's own mercator_project
    mixes the RGBA channels, so it is not used."""
    def y(lat):
        return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    h = img.shape[0]
    ys = y(north) - (np.arange(h) + 0.5) / h * (y(north) - y(south))
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
    src = np.clip(((north - lats) / (north - south) * h).astype(int), 0, h - 1)
    return img[src]


def cell_at(meta: dict, lat: float, lon: float):
    (s, w), (n, e) = meta["bounds"]
    h, wd = meta["shape"]
    if not (s <= lat <= n and w <= lon <= e):
        return None
    return min(h - 1, int((n - lat) / (n - s) * h)), min(wd - 1, int((lon - w) / (e - w) * wd))


# ----------------------------------------------------------------------------- views
def how_to_read(det: str, alpha: float) -> None:
    with st.expander("How to read this map", expanded=True):
        st.markdown(
            "- **Grey picture:** the radar image. Brighter means the ground sends more of the radar signal "
            "back, as forest does; darker is open ground or water.\n"
            f"- **Red cells:** places that *{DET_SHORT[det]}* flags as having changed between the two dates. "
            "Each cell is 100 m across.\n"
            "- **Cyan outline:** the edge of the fire as mapped by fire agencies. It is the reference used to "
            "test the methods. No method sees it.\n"
            "- **Red inside the outline** is change found where the fire burned. **Red outside it** is either a "
            "false alarm or a real change of another kind, such as another fire or farm work. The map cannot "
            "tell these apart.\n"
            "- **A red cell is a detected change, not confirmed fire damage.** The outline is not a damage map "
            "either: it encloses patches that did not burn.")
        st.caption("Switch layers with the box at the top right of the map. Click any place to see its radar "
                   "signal over time. Technical terms: the grey picture is NISAR L-band cross-polarized (HV) "
                   f"backscatter in decibels; red cells have a change score above the threshold calibrated for a "
                   f"target false-alarm rate (alpha) of {alpha}.")


def keep_in_mind(D: dict) -> None:
    f = site_facts(D)
    n = len(real_sites())
    lines = ["- **One fire is one case study.** Results may differ at other places and for other kinds of change."
             if n == 1 else
             f"- **{NUM_WORDS.get(n, n).capitalize()} fires are still a small sample.** Results may differ at other "
             "places and for other kinds of change."]
    if f["m"]:
        where = ("most red cells on the map lie outside it" if f.get("most_outside")
                 else "most red cells on the map lie inside it")
        lines.append(f"- **At the default setting our model flags {pct(f['m']['inside_flagged'])} of the cells inside "
                     f"this fire's perimeter**, and {where}. Some flags outside may be real changes of another kind; "
                     "none were checked on the ground. The Evidence tab has the numbers.")
    lines.append("- **A flag means \"changed more than expected\", not \"burned\".**")
    mk, mn = f["m_h3"]
    bk, bn = f["b3_h3"]
    held = ("stayed within half to double the setting on every check" if f["h3_pass"]
            else f"stayed within half to double the setting on {mk} of {mn} checks")
    note = (" so most of this comes from comparing each place with similar land, not from the learning."
            if bn and bk >= mk - 2 else ".")
    lines.append(f"- **The false-alarm setting is a tested target, not a guarantee.** On dates the method had not "
                 f"seen at this site, the share of land our model flagged {held}. A simpler method that compares "
                 f"each place with the average for its land type stayed in range on {bk} of {bn} checks,{note}"
                 if bn else
                 f"- **The false-alarm setting is a tested target, not a guarantee.** On dates the method had not "
                 f"seen at this site, the share of land our model flagged {held}.")
    lines.append("- The Limits tab has the full list.")
    with st.expander("Keep in mind"):
        st.markdown("\n".join(lines))


PLACES_TOP = 15
ALL_PLACES = "All flagged places"


def places_away(site: str) -> bool:
    """True when the places list is filtered to places away from mapped fires."""
    return (load(site)["other"] is not None
            and st.session_state.get(f"places_filter_{site}", ALL_PLACES) != ALL_PLACES)


def places_view(site: str, det: str, alpha: float, away: bool) -> pd.DataFrame:
    """The rows the places table shows; the index + 1 is each place's rank among all flagged places."""
    df = places_table(site, det, alpha, PLACES_VERSION)
    if away and not df.empty:
        df = df[~(df["overlaps_perimeter"] | (df["other_fire"] != ""))]
    return df.head(PLACES_TOP)


def places_key(site: str, det: str, alpha: float, away: bool) -> str:
    # A new key (other setting, filter, or "Back to the fire") starts the table with nothing selected.
    return f"places_{site}_{det}_{alpha}_{int(away)}_{st.session_state.get('places_nonce', 0)}"


def focused_place(site: str, det: str, alpha: float) -> dict | None:
    """The place selected in the table, if any (read before the map is drawn so the map can move to it)."""
    away = places_away(site)
    state = st.session_state.get(places_key(site, det, alpha, away))
    try:
        rows = list(state["selection"]["rows"]) if state is not None else []
    except (KeyError, TypeError):
        rows = []
    view = places_view(site, det, alpha, away)
    if not rows or rows[0] >= len(view):
        return None
    return {**view.iloc[rows[0]].to_dict(), "rank": int(view.index[rows[0]]) + 1}


def _back_to_fire() -> None:
    st.session_state["places_nonce"] = st.session_state.get("places_nonce", 0) + 1


def view_map(site: str, D: dict, det: str, alpha: float) -> None:
    meta, arr = D["meta"], D["arr"]
    bit = meta["alphas"].index(alpha)
    focus = focused_place(site, det, alpha)
    (s, w), (n, e) = meta["bounds"]
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="Satellite imagery").add_to(m)
    b = [[s, w], [n, e]]
    ev = meta["event_pair"]
    # (layer name, image, shown at start, is a radar picture)
    layers = [(f"Radar before ({ev['date_a']})", rgba(site, "hv_before", -25, -5, "gray"), False, True),
              (f"Radar after ({ev['date_b']})", rgba(site, "hv_after", -25, -5, "gray"), True, True),
              ("Change score", rgba(site, f"score_{det}", 0, 6, "magma"), False, False),
              (f"Flagged as changed ({odds(alpha)})", rgba(site, f"flags_{det}", 0, 1, "Reds", bit=bit), True, False),
              ("Fire perimeter (reference)", outline(site), True, False)]
    other = other_outline(site)
    if other is not None:
        layers.append(("Other fires mapped in 2026 (reference)", other, True, False))
    for name, img, show, radar in layers:
        folium.raster_layers.ImageOverlay(image=to_mercator_rows(img, s, n), bounds=b, name=name,
                                          opacity=0.85 if radar else 1.0, show=show).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    # Open the view on the fire (cells inside the reference perimeter), centred left-right and placed in the upper
    # part of the map, so it shows on the first screen of a laptop or phone where only the map's top is visible.
    inside = arr["labels"] == 1
    if focus:
        # Zoom to the selected place, mark its strongest cell and show that cell's history once.
        plat, plon = focus["peak_lat"], focus["peak_lon"]
        r_km = max(2.0, 3 * float(np.sqrt(focus["area_km2"])))
        dlat, dlon = r_km / 111.32, r_km / (111.32 * np.cos(np.radians(plat)))
        m.fit_bounds([[plat - dlat, plon - dlon], [plat + dlat, plon + dlon]])
        folium.CircleMarker([plat, plon], radius=14, color="#00e5ff", weight=3, fill=False,
                            tooltip=f"Place {focus['rank']}").add_to(m)
        fkey = f"{plat:.5f},{plon:.5f}"
        if st.session_state.get("focus_seen") != fkey:
            st.session_state["focus_seen"] = fkey
            st.session_state["click"] = (plat, plon)
    elif inside.any():
        st.session_state["focus_seen"] = None
        rr, cc = np.nonzero(inside)
        h, wd = meta["shape"]
        fire = [[n - (rr.max() + 1) / h * (n - s), w + cc.min() / wd * (e - w)],
                [n - rr.min() / h * (n - s), w + (cc.max() + 1) / wd * (e - w)]]
        m.fit_bounds(fire, padding_top_left=(20, 20), padding_bottom_right=(20, 420))
    st.markdown("**Red** = changed more than the scene predicts. **Cyan** = mapped fire edge. "
                + ("**Yellow** = other fires mapped in 2026. " if other is not None else "")
                + "**Grey** = radar image. Red is a detected change, not confirmed damage.")
    if focus:
        note, back = st.columns([4, 1])
        note.info(f"Showing place {focus['rank']} from the list below (cyan circle). Its radar history is on the "
                  "right.")
        back.button("Back to the fire", on_click=_back_to_fire, key=f"back_{site}")
    map_key = f"map_{site}" + (f"_{focus['peak_lat']:.5f}_{focus['peak_lon']:.5f}" if focus else "")
    left, right = st.columns([3, 2])
    with left:
        out = st_folium(m, height=620, use_container_width=True, returned_objects=["last_clicked"], key=map_key)
        st.caption(f"Showing: {DET_NAMES[det]}, at the {odds(alpha)} setting. Switch layers with the box at the "
                   "top right of the map. Click any place to inspect it.")
    with right:
        click = (out or {}).get("last_clicked")
        if click:
            # Each map instance reports its last click on every rerun; act only on a new one, so a selected
            # place is not overwritten by an older click.
            pt = (click["lat"], click["lng"])
            seen = st.session_state.setdefault("seen_clicks", {})
            if seen.get(map_key) != pt:
                seen[map_key] = pt
                st.session_state["click"] = pt
        view_inspector(D, st.session_state.get("click"))
    view_places(site, det, alpha)
    how_to_read(det, alpha)
    keep_in_mind(D)


@st.cache_data(show_spinner=False)
def places_table(site: str, det: str, alpha: float, places_version: str = "") -> pd.DataFrame:
    D = load(site)
    arr, meta = D["arr"], D["meta"]
    bit = meta["alphas"].index(alpha)
    flags = ((arr[f"flags_{det}"].astype(np.uint8) >> bit) & 1).astype(bool)
    oth = D["other"]
    return flagged_places(flags, arr["labels"], arr[f"score_{det}"].astype(np.float32), meta["bounds"], meta["shape"],
                          other_ids=oth["ids"] if oth else None,
                          other_names={f["id"]: f["name"] for f in oth["fires"]} if oth else None)


def view_places(site: str, det: str, alpha: float) -> None:
    df = places_table(site, det, alpha, PLACES_VERSION)
    st.markdown("#### Places to check first")
    if df.empty:
        st.info("Nothing is flagged at this setting.")
        return
    st.caption(f"{len(df):,} separate patches are flagged by {DET_SHORT[det]} at the {odds(alpha)} setting. "
               "The largest are listed first. Each is a place to look at with other imagery or on the ground, "
               "not confirmed damage. Areas and positions are approximate (display grid).")
    if load(site)["other"] is not None:
        known = df["overlaps_perimeter"] | (df["other_fire"] != "")
        top = min(15, len(df))
        st.caption(f"{int(known.head(top).sum())} of the {top} largest touch a fire mapped by agencies in 2026: this "
                   "one or another (yellow outline on the map). Those match a known event; the rest are unexplained "
                   "change, or false alarms.")
        st.radio("Show", [ALL_PLACES, "Only places away from mapped fires"], horizontal=True,
                          key=f"places_filter_{site}",
                          help="Mapped fires: this fire's perimeter and the other 2026 perimeters in the window "
                               "(NIFC). A flagged place away from them is either a change nobody has recorded or "
                               "a false alarm; those are the ones to check with other imagery.")
    away = places_away(site)
    view = places_view(site, det, alpha, away)
    if view.empty:
        st.info("No flagged place matches this filter.")
    else:
        show = view.copy()
        show.insert(0, "Rank", show.index + 1)
        show["overlaps_perimeter"] = show["overlaps_perimeter"].map({True: "yes", False: "no"})
        show["other_fire"] = show["other_fire"].replace("", "no")
        show = show[["Rank", "area_km2", "overlaps_perimeter", "other_fire", "mean_score", "cells", "lat", "lon"]]
        show = show.rename(columns={"area_km2": "Area (km², approx.)", "lat": "Latitude", "lon": "Longitude",
                                    "cells": "Cells", "mean_score": "Mean change score",
                                    "overlaps_perimeter": "Touches this fire",
                                    "other_fire": "Touches another mapped fire"})
        if load(site)["other"] is None:
            show = show.drop(columns=["Touches another mapped fire"])
        st.dataframe(show.round({"Area (km², approx.)": 2, "Latitude": 4, "Longitude": 4, "Mean change score": 2}),
                     hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                     key=places_key(site, det, alpha, away))
        st.caption("Select a row (the box at its left) to move the map to that place and show its radar history.")
    st.download_button(f"Download all {len(df):,} patches (CSV)", df.round(5).to_csv(index=False),
                       file_name=f"{site}_{det}_1in{round(1 / alpha)}_flagged_places.csv", mime="text/csv")


def view_inspector(D: dict, click) -> None:
    meta, arr = D["meta"], D["arr"]
    st.subheader("What happened at this spot?")
    if not click:
        st.info("Click a place on the map to see its radar signal on every date.")
        return
    rc = cell_at(meta, *click)
    if rc is None:
        st.warning("That point is outside the area that was processed.")
        return
    r, c = rc
    dates = pd.to_datetime(meta["dates"])
    hh, hv = arr["hh_ts"][:, r, c].astype(float), arr["hv_ts"][:, r, c].astype(float)
    ev = meta["event_pair"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(dates, hh, "o-", color="#3b7dbf", label="Radar signal, HH channel")
    ax.plot(dates, hv, "o-", color="#2e8b57", label="Radar signal, HV channel")
    da, db = pd.Timestamp(ev["date_a"]), pd.Timestamp(ev["date_b"])
    ax.axvspan(da, db, color="#f0a030", alpha=0.15, label="Between the before and after dates")
    q05, q50, q95 = (float(arr[k][r, c]) for k in ("hv_q05", "hv_q50", "hv_q95"))
    if np.isfinite(q50):
        ax.errorbar([db], [q50], yerr=[[q50 - q05], [q95 - q50]], fmt="s", color="#c23b3b", capsize=5,
                    label="What our model expected for HV (5-95% range)")
    ax.set_ylabel("Radar signal strength (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)
    st.caption("The lines are measurements. The red marker is a prediction: the range our model expected on the "
               "after date. A measurement well outside that range is what gets flagged. HH and HV are the two "
               "radar channels (polarizations) NISAR records; HV responds mostly to vegetation structure.")
    lab = {1: "inside the mapped fire perimeter", 0: "outside the fire (comparison land)",
           -1: "not used in the test (perimeter edge, another fire, or masked)"}.get(int(arr["labels"][r, c]), "?")
    scores = {d: float(arr[f"score_{d}"][r, c]) for d in meta["detectors"]}
    k = int(arr["cls"][r, c])
    st.markdown(f"**This cell** (row {r}, col {c}, at {click[0]:.4f}, {click[1]:.4f}) is *{lab}*; "
                + (f"land-type group {k}." if k >= 0 else "masked (water, steep terrain or missing data; not scored)."))
    st.dataframe(pd.DataFrame({"Method": [DET_NAMES[d] for d in scores],
                               "Change score": [f"{v:.2f}" if np.isfinite(v) else "not scored" for v in scores.values()]}),
                 hide_index=True, width="stretch")
    st.caption("Each method scores on its own scale, so compare a score with that method's threshold in the "
               "sidebar, not with the other methods' scores. Land-type groups (0 to 5) come from radar "
               "brightness before the fire, darkest to brightest.")


def _fmt(v, ci):
    if v is None:
        return "n/a"
    lo, hi = ci or (None, None)
    return f"{v:.3f}" + (f"  [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "")


def event_dates(met: dict) -> tuple[str, str]:
    """'20260818_20260923' -> ('2026-08-18', '2026-09-23')."""
    a, b = str(met["event_pair"]).split("_")[:2]
    return tuple(f"{x[:4]}-{x[4:6]}-{x[6:8]}" for x in (a, b))


def h3_count(cal: dict, det: str) -> tuple[int, int]:
    """(checks in range, checks) for one method on the left-out pairs that Test 2 scores."""
    rows = [r for r in cal["lopo"] if r.get("pass") is not None and r["detector"] == det]
    return sum(bool(r["pass"]) for r in rows), len(rows)


def pct1(v: float) -> str:
    """One decimal, so close results (27.8% against 27.9%) stay distinguishable."""
    return f"{v:.1%}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def test_reasons(D: dict) -> dict:
    """One plain-language sentence per test result, built from the stored metrics and calibration."""
    met, cal = D["metrics"], D["calib"]
    dets, diffs = met["detectors"], met.get("differences", {})
    out = {}
    if "M" in dets:
        less, same, more = [], [], []
        for b in ("B3", "B2", "B1"):
            d = diffs.get(f"M_minus_{b}", {}).get("far_at_recall50")
            if b not in dets or d is None:
                continue
            item = f"{DET_SHORT[b]} method ({pct1(dets[b]['far_at_recall50'])})"
            lo, hi = d["ci95"]
            (less if hi < 0 else more if lo > 0 else same).append(item)
        s = f"Ours flagged {pct1(dets['M']['far_at_recall50'])} of the land outside"
        if less and not same and not more:
            s += (f", less than the {_join(less)}. Each gap held up in the uncertainty check (its 95% range "
                  "stays below zero).")
        else:
            parts = []
            if less:
                parts.append(f"less than the {_join(less)}")
            if same:
                parts.append(f"about the same as the {_join(same)}: the 95% range of that gap includes zero")
            if more:
                parts.append(f"more than the {_join(more)}")
            s += ", " + ", but ".join(parts) + "."
        out["H1"] = s
    mk, mn = h3_count(cal, "M")
    alphas = sorted({r["alpha"] for r in cal["lopo"] if r.get("pass") is not None})
    others = [f"{DET_SHORT[d]} {k} of {n}" for d in ("B3", "B2", "B1") for k, n in [h3_count(cal, d)] if n]
    if mn:
        out["H3"] = (f"Ours stayed in range on {mk} of {mn} checks: each check leaves one pair of dates out of the "
                     f"calibration, at the {_join([odds(a) for a in alphas])} settings."
                     + (f" The other methods: {', '.join(others)}." if others else ""))
    return out


def headline() -> str | None:
    """One-line result across the real fires, built from the stored metrics, for the first screen."""
    rs = real_sites()
    if len(rs) < 2:
        return None
    k = n = 0
    beat_std = True
    lt: dict[str, list[str]] = {"beat": [], "tied": [], "lost": []}
    for x in rs:
        met, cal = load(x)["metrics"], load(x)["calib"]
        mk, mn = h3_count(cal, "M")
        k, n = k + mk, n + mn
        diffs = met.get("differences", {})
        d1 = diffs.get("M_minus_B1", {}).get("far_at_recall50")
        beat_std = beat_std and bool(d1) and d1["ci95"][1] < 0
        d3 = diffs.get("M_minus_B3", {}).get("far_at_recall50")
        if d3:
            lo, hi = d3["ci95"]
            lt["beat" if hi < 0 else "lost" if lo > 0 else "tied"].append(SITE_NAMES.get(x, x).split(" Fire")[0])
    both = "both" if len(rs) == 2 else "all"
    out = (f"**{NUM_WORDS.get(len(rs), str(len(rs))).capitalize()} fires, tests written down first.** The false-alarm "
           f"setting held on {k} of {n} checks")
    out += (f", and our model beat the standard before/after method on {both}." if beat_std else ".")
    rest = lt["tied"] + lt["lost"]
    if lt["beat"] and not rest:
        out += f" It also beat the simpler land-type average on {both}."
    elif lt["beat"]:
        out += (f" Its edge over the simpler land-type average did not replicate on {_join(rest)}"
                + (" (a tie)." if not lt["lost"] else "."))
    elif rest:
        out += " It did not beat the simpler land-type average."
    return (out + " Places to check are listed below the map. Results are precomputed; the app does not analyse "
            "other places or new dates.")


def view_sites_summary() -> None:
    """Both (or all) real fires side by side, so the replication outcome is visible from any site."""
    rows = []
    for x in real_sites():
        met, cal = load(x)["metrics"], load(x)["calib"]
        dets, h = met["detectors"], met["hypotheses"]
        base = [d for d in ("B3", "B2", "B1") if d in dets]
        if "M" not in dets or not base:
            continue
        best = min(base, key=lambda d: dets[d]["far_at_recall50"])
        a, b = event_dates(met)
        mk, mn = h3_count(cal, "M")
        role = SITE_ROLE.get(x, "").removeprefix("the ")
        rows.append({"Fire": SITE_NAMES.get(x, x).split(",")[0] + (f" ({role})" if role else ""),
                     "Dates compared": f"{a} to {b[5:]}",
                     "Test 1": "PASS" if h["H1"]["pass"] else "FAIL",
                     "Test 2": f"{'PASS' if h['H3']['pass'] else 'FAIL'} ({mk} of {mn})",
                     "Outside land flagged: ours vs best other method":
                         f"{pct1(dets['M']['far_at_recall50'])} vs {pct1(dets[best]['far_at_recall50'])} "
                         f"({DET_SHORT[best]})"})
    if len(rows) < 2:
        return
    st.markdown(f"#### {'Both' if len(rows) == 2 else 'All ' + str(NUM_WORDS.get(len(rows), len(rows)))} fires "
                "at a glance")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Each fire's tests were written down before it was scored. Test 1: our model flags the least land "
               "outside the perimeter. Test 2: the false-alarm setting held on dates left out of calibration. "
               "\"Outside land flagged\" is the share of land more than 2 km outside the perimeter that a method "
               "flags when it finds half of the inside (lower is better). The selected site's details follow.")


def view_other_fires(D: dict) -> None:
    """Exploratory: how often each method flags other mapped fires, which were left out of every score."""
    oth = D.get("other")
    if not oth or not oth["fires"] or not oth.get("n_other_cells"):
        return
    ra = str(D["meta"]["report_alpha"])
    st.markdown("#### Exploratory check: other fires mapped in 2026")
    st.caption("Added after the results; not one of the tests written down in advance. Agencies mapped other "
               "fires in this window in 2026 (yellow outlines on the map). Their cells were left out of every "
               "score on this page, so they give an independent check: a method that responds to burns should "
               "flag them far more often than ordinary land.")
    rows = []
    for d in ("M", "B3", "B2", "B1"):
        r = oth["rates"].get(d, {}).get(ra)
        if not r or r["other_fires"] is None:
            continue
        rows.append({"Method": DET_NAMES[d], "Cells in other mapped fires flagged": pct1(r["other_fires"]),
                     "Ordinary land flagged": pct1(r["control"]),
                     "Times the ordinary rate": f"{r['other_fires'] / r['control']:.1f}" if r["control"] else "n/a"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    big = sorted((f for f in oth["fires"] if f.get("cells", 0) >= 50), key=lambda f: -f["cells"])
    if big:
        st.dataframe(pd.DataFrame([{
            "Fire": f["name"], "Discovered": f.get("discovered") or "unknown", "Cells of 100 m in the window": f["cells"],
            "Flagged by our model": pct1(oth["rates"]["M"][ra]["per_fire"][str(f["id"])]),
            "Flagged by before/after difference": pct1(oth["rates"]["B1"][ra]["per_fire"][str(f["id"])])}
            for f in big]), hide_index=True, width="stretch")
    a, b = event_dates(D["metrics"])
    st.caption(f"At the {odds(float(ra))} setting, as mapped (after small patches are removed). Dates compared: {a} "
               f"to {b}; a fire that stopped changing before the first date should show little. Perimeters are "
               "outlines, not burn-severity maps, and when each cell burned is not known. Ordinary land: more than "
               f"2 km outside this fire's perimeter ({oth['n_control']:,} cells); other fires: {oth['n_other_cells']:,} "
               "cells. Fires with fewer than 50 cells in the window are counted but not listed.")


def view_validation(D: dict) -> None:
    met, cal = D["metrics"], D["calib"]
    h = met["hypotheses"]
    view_sites_summary()
    st.markdown("Two tests were written down, with all settings, before the fire dates were scored. "
                "This page shows how they came out, and every number behind them.")
    where = SITE_PREREG.get(D["meta"].get("site", ""))
    if where:
        st.caption(f"For this site the tests are in {where}.")
    reasons = test_reasons(D)
    c1, c2 = st.columns(2)
    for col, key, name, text in (
            (c1, "H1", "Test 1", "With every method set to flag half of the cells inside the fire perimeter, ours flags the least land outside it."),
            (c2, "H3", "Test 2", "On pairs of dates the calibration had not seen, the share of land flagged stayed within half to double the setting.")):
        ok = h[key]["pass"]
        col.markdown(f"### {name} ({key}): {'PASS' if ok else 'FAIL'}")
        col.caption(text)
        if key in reasons:
            col.markdown(reasons[key])
        with col.expander("Detail"):
            st.write(h[key]["detail"])
    if D.get("compare_png") is not None:
        st.markdown("#### Same false-alarm setting, two methods")
        st.image(str(D["compare_png"]), width="stretch")
        ra = str(D["meta"]["report_alpha"])
        sm = (D["summary"] or {}).get("detectors", {})
        b1, mm = sm.get("B1", {}).get(ra), sm.get("M", {}).get(ra)
        st.caption(f"Both methods are set to the same calibrated false-alarm level ({odds(float(ra))}) on the same "
                   "two radar dates."
                   + (f" Inside the fire perimeter the standard method flags {pct(b1['inside_flagged'])} of the "
                      f"cells and ours {pct(mm['inside_flagged'])}; on land more than 2 km outside, "
                      f"{pct(b1['outside_flagged'])} and {pct(mm['outside_flagged'])}." if b1 and mm else "")
                   + " Flags outside count as false alarms in the scores, though some may be real changes.")
    view_other_fires(D)
    st.caption(f"Before and after dates: {met['event_pair']} (the event pair). {met['n_pos']:,} burned and "
               f"{met['n_neg']:,} comparison (control) cells. Calibration mode: {met['calibration_mode']}. "
               f"Code version: commit {met['commit'][:10]}.")

    at_setting = f"{odds(met['report_alpha'])} setting"
    cols = {"far": "Land outside the perimeter flagged, when half the inside is flagged", "rec": "Inside cells flagged at a 1 in 100 false-alarm setting",
            "ap": "Ranking quality", "f1": f"Map agreement at the {at_setting}"}
    rows = []
    for d in ("M", "B3", "B2", "B1"):
        if d not in met["detectors"]:
            continue
        x = met["detectors"][d]
        rows.append({"Method": DET_NAMES[d],
                     cols["far"]: _fmt(x["far_at_recall50"], x["ci95"]["far_at_recall50"]),
                     cols["rec"]: _fmt(x["recall_at_far1"], x["ci95"]["recall_at_far1"]),
                     cols["ap"]: _fmt(x["ap"], x["ci95"]["ap"]),
                     cols["f1"]: _fmt(x["f1_at_report_alpha"], x["ci95"]["f1_at_report_alpha"])})
    st.markdown("#### How the four methods compare on the fire dates")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Method": st.column_config.TextColumn(help="Technical names: " + "; ".join(DET_TECH[d] for d in ("M", "B3", "B2", "B1"))),
        cols["far"]: st.column_config.TextColumn(help="False-alarm rate (FAR) at 50% recall. Lower is better. This is the measure Test 1 uses."),
        cols["rec"]: st.column_config.TextColumn(help="Recall at a 1% false-alarm rate. Higher is better."),
        cols["ap"]: st.column_config.TextColumn(help="Average precision (AP): area under the precision-recall curve. Higher is better; a random ranking scores the burned share, about 0.01 here."),
        cols["f1"]: st.column_config.TextColumn(help=f"F1 score at the threshold calibrated for alpha = {met['report_alpha']}, after removing small patches. Higher is better."),
    })
    st.caption("All values are fractions between 0 and 1. Brackets give the 95% range from re-drawing 5 km blocks "
               "of the map (block bootstrap). Hover over a column heading for its technical name.")
    st.caption("**Inside** means inside the agency-mapped fire perimeter (shrunk by one cell); **outside** means more "
               "than 2 km beyond it. Neither was checked cell by cell: the perimeter encloses patches that did not "
               "burn, and some flags outside it may be real changes such as other fires or farm work. In these "
               "scores every flag outside counts as a false alarm, so they describe agreement with the perimeter, "
               "not confirmed burn detection.")

    cv = D["curves"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, g in cv.groupby("detector"):
        ax[0].plot(g.recall, g.precision, color=COLORS.get(d), label=DET_SHORT.get(d, d), lw=1.6)
        ax[1].plot(g.far, g.recall, color=COLORS.get(d), label=DET_SHORT.get(d, d), lw=1.6)
    ax[0].set(xlabel="Share of inside cells flagged (recall)", ylabel="Share of flags that are inside (precision)",
              xlim=(0, 1), ylim=(0, 1), title="Precision-recall")
    ax[1].set(xlabel="Share of outside land flagged (false-alarm rate, log scale)",
              ylabel="Share of inside cells flagged (recall)", xscale="log", xlim=(1e-4, 1), ylim=(0, 1),
              title="Inside flagged against outside flagged")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)
    st.caption("Each curve traces one method as its threshold moves. Higher curves are better in both panels.")

    st.markdown("#### Test 2 (H3): did the false-alarm setting hold on dates it had not seen?")
    st.caption("For each pair of dates with no fire, the threshold was set from the other pairs and then applied "
               "to it. The share of land flagged should be close to the setting. Technical name: leave-one-pair-out "
               "conformal calibration. The raw share is what the guarantee covers and decides the test; the share "
               "after small-patch removal is what the map shows.")
    lo = pd.DataFrame(cal["lopo"])
    if len(lo):
        lo = lo[lo["pass"].notna()][["detector", "left_out", "alpha", "realized", "realized_filtered", "pass"]]
        lo["detector"] = lo["detector"].map(lambda d: DET_NAMES.get(d, d))
        lo = lo.rename(columns={"detector": "Method", "left_out": "Dates left out", "alpha": "Setting (alpha)",
                                "realized": "Land flagged (raw)", "realized_filtered": "After small-patch removal",
                                "pass": "Within half to double the setting"})
        st.dataframe(lo, hide_index=True, width="stretch")
    if D["failure"] is not None:
        st.markdown("#### Where each method succeeds and fails, by land type")
        ft = D["failure"].copy()
        ft["detector"] = ft["detector"].map(lambda d: DET_NAMES.get(d, d))
        ft = ft.rename(columns={"detector": "Method", "class": "Land-type group", "n_pos": "Cells inside perimeter",
                                "n_neg": "Cells outside (over 2 km)", "recall": "Share of inside cells flagged (recall)",
                                "far": "Share of outside cells flagged (false-alarm rate)"})
        st.dataframe(ft, hide_index=True, width="stretch")
        st.caption(f"At the {at_setting}. Land-type groups run from the darkest radar return (0, open land) to the "
                   "brightest (5, dense forest); they are clusters of radar brightness before the fire.")


def view_limits(D: dict) -> None:
    n = len(real_sites())
    f = site_facts(D)
    first = ("- This is one fire at one site, seen from one satellite track.\n" if n == 1 else
             f"- These are {NUM_WORDS.get(n, n)} fires, each seen from one satellite track: a small sample.\n")
    third = ("- On this fire every method, including ours, flags under half of the cells inside the perimeter.\n"
             if f["all_under_half"] else
             f"- At the default setting our model flags {pct(f['m']['inside_flagged'])} of the cells inside this "
             "fire's perimeter.\n" if f["m"] else "")
    st.markdown(
        "#### In short\n"
        + first +
        "- The method finds changes; it does not confirm what caused them.\n"
        + third +
        "- The reference outline has its own errors: unburned patches inside it, real changes outside it.\n"
        "- The app shows stored results. It does not analyse new places or dates.\n\n"
        "The full technical list follows.\n\n---")
    p = ROOT / "docs" / "LIMITATIONS.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "docs/LIMITATIONS.md not found.")


# ----------------------------------------------------------------------------- page
def main() -> None:
    avail = sites()
    if not avail:
        st.error("No artifacts found. Run `python -m coldstart run config/synthetic.yaml` first.")
        return
    with st.sidebar:
        st.markdown("## Cold Start")
        st.caption("Finding surface change with NISAR, a radar too new to have a history.")
        avail = sorted(avail, key=lambda x: (x == "synthetic", x != "mccully", x))
        default = avail.index("mccully") if "mccully" in avail else 0
        site = st.selectbox("Study site", avail, index=default, format_func=lambda s: SITE_NAMES.get(s, s),
                            help="Results are precomputed for the sites listed here. The app does not analyse "
                                 "other places.")
        D = load(site)
        meta = D["meta"]
        det = st.selectbox("Detection method", [d for d in ("M", "B3", "B2", "B1") if d in meta["detectors"]],
                           format_func=lambda d: DET_NAMES[d],
                           help="Four ways of deciding which places changed. Our model learns the expected change "
                                "from every pixel in the scene. The baselines compare each place with the average "
                                "for its land type, with its own few earlier dates, or simply before against "
                                "after. Technical names: "
                                + "; ".join(DET_TECH[d] for d in ("M", "B3", "B2", "B1")) + ".")
        st.caption(f"Technical name: {DET_TECH[det]}.")
        alpha = st.select_slider("False alarms you accept", options=meta["alphas"], value=meta["report_alpha"],
                                 format_func=setting_label,
                                 help="How much unchanged land you are willing to see flagged by mistake. A lower "
                                      "setting flags less land and also finds less of the real change. "
                                      "Technically this is the target false-alarm rate (alpha): the threshold is "
                                      "set on pairs of dates with no fire so that about this share of unchanged "
                                      "cells scores above it.")
        setting_meaning(D, det, alpha)
        thr = meta["thresholds"].get(det, {}).get(str(alpha))
        st.caption(f"Score threshold at this setting: {thr:.3f} (calibrated for {det}, alpha = {alpha})."
                   if thr is not None else "No finite threshold at this setting.")
        st.markdown("---")
        st.caption("A cell is flagged when its change score is above the threshold. Flagged patches smaller than "
                   "4 cells are then removed, so the map usually shows fewer false alarms than the setting.")
    st.title("Cold Start")
    st.markdown("Finding surface change with NISAR, a radar too new to have a history. "
                f"**Study site: {SITE_NAMES.get(site, site)}**"
                + (f", {SITE_ROLE[site]}" if site in SITE_ROLE else ""))
    head = headline() if site in real_sites() else None
    if head:
        st.markdown(head)
    else:
        st.markdown("Red cells on the map are places that changed more than the rest of the scene predicts, between "
                    "a radar image taken before the event and one taken after. These are **precomputed results for "
                    f"{NUM_WORDS.get(len(real_sites()), len(real_sites()))} study site{'s' if len(real_sites()) != 1 else ''}**; the app does not analyse other places or new dates.")
    with st.expander("What is this, and how does it work?"):
        st.markdown("NISAR is a NASA-ISRO radar satellite whose calibrated images begin on 17 June 2026. Many methods "
                    "for spotting change in radar images compare each place with years of its own past, which NISAR "
                    "does not have yet. This project asks a different question: **on the same two dates, how did all "
                    "the similar-looking land in the scene change?** Places that changed differently from what that "
                    "predicts are flagged.")
        st.markdown("**Intended use.** A triage aid for an analyst after an event: the map shows where the radar "
                    "changed more than similar land did, to help decide where to look first. Each flag still needs "
                    "checking against other imagery or on the ground; nobody uses this operationally yet.")
        st.markdown("The app shows stored results for "
                    + " and ".join(SITE_NAMES.get(x, x).split(":")[0] for x in real_sites())
                    + ", plus a simulated test scene."
                    + (" The Austin Fire is a replication: its tests were written down and pushed to GitHub before "
                       "any of its radar data were processed." if "austin" in real_sites() else "")
                    + " Running the method elsewhere means running the open-source pipeline in the repository.")
        st.markdown("**For specialists.** A gradient-boosted quantile regression, cross-fitted over 10 km tiles, "
                    "predicts each 100 m cell's change in L-band HH and HV backscatter from its state on the earlier "
                    "date, using all pixels in the scene on the same two dates. Cells are scored by their residual "
                    "scaled by the predicted spread, and flagged at a false-alarm rate set in advance by conformal "
                    "calibration on date pairs with no event. Three baselines run on the same grid: log-ratio, a "
                    "short-baseline z-score and a stratified class shift.")
    t1, t2, t3 = st.tabs(["Map", "Evidence", "Limits"])
    with t1:
        view_map(site, D, det, alpha)
    with t2:
        view_validation(D)
    with t3:
        view_limits(D)


main()
