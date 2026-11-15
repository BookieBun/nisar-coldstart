"""Read-only viewer for the cold-start change-detection results.

Reads only artifacts/<site>/ (and docs/LIMITATIONS.md). Runs no models.
    streamlit run app/streamlit_app.py

Wording is written for readers without a radar or machine-learning background; the technical
term for each plain-language label is kept in a tooltip or caption beside it.
"""
from __future__ import annotations

import json
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

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
# Plain-language names; the technical names used in the docs are kept alongside.
DET_NAMES = {"M": "Our model (whole scene)", "B3": "Baseline: land-type average",
             "B2": "Baseline: own short history", "B1": "Baseline: before/after difference"}
DET_SHORT = {"M": "our model", "B3": "land-type average", "B2": "short history", "B1": "before/after difference"}
DET_TECH = {"M": "M, pooled conditional quantile model", "B3": "B3, stratified class shift",
            "B2": "B2, short-baseline z-score", "B1": "B1, log-ratio"}
SITE_NAMES = {"mccully": "McCully Fire, Oregon (2026)", "synthetic": "Simulated test scene (not real data)"}
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
           "failure": pd.read_csv(d / "failure_table.csv") if (d / "failure_table.csv").exists() else None}
    return out


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


def keep_in_mind() -> None:
    with st.expander("Keep in mind"):
        st.markdown(
            "- **One fire is one case study.** Results may differ at other places and for other kinds of change.\n"
            "- **This burn was hard to see with radar.** Even the best method here misses much of the burned "
            "area and flags some unburned land. The Evidence tab has the numbers.\n"
            "- **A flag means \"changed more than expected\", not \"burned\".**\n"
            "- **The false-alarm setting is a tested promise, not a guarantee.** It held on dates the method had "
            "not seen at this site, and it assumes those dates behave like the fire dates.\n"
            "- The Limits tab has the full list.")


def view_map(site: str, D: dict, det: str, alpha: float) -> None:
    meta, arr = D["meta"], D["arr"]
    bit = meta["alphas"].index(alpha)
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
    for name, img, show, radar in layers:
        folium.raster_layers.ImageOverlay(image=to_mercator_rows(img, s, n), bounds=b, name=name,
                                          opacity=0.85 if radar else 1.0, show=show).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    st.markdown("**Red** = flagged as changed. **Cyan outline** = the mapped edge of the fire. **Grey** = the radar "
                "image. A red cell is a detected change, not confirmed fire damage; the full guide is below the map.")
    left, right = st.columns([3, 2])
    with left:
        out = st_folium(m, height=620, use_container_width=True, returned_objects=["last_clicked"], key=f"map_{site}")
        st.caption(f"Showing: {DET_NAMES[det]}, at the {odds(alpha)} setting. Switch layers with the box at the "
                   "top right of the map. Click any place to inspect it.")
    with right:
        click = (out or {}).get("last_clicked")
        if click:
            st.session_state["click"] = (click["lat"], click["lng"])
        view_inspector(D, st.session_state.get("click"))
    how_to_read(det, alpha)
    keep_in_mind()


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


def view_validation(D: dict) -> None:
    met, cal = D["metrics"], D["calib"]
    h = met["hypotheses"]
    st.markdown("Two tests were written down, with all settings, before the fire dates were scored. "
                "This page shows how they came out, and every number behind them.")
    c1, c2 = st.columns(2)
    for col, key, name, text in (
            (c1, "H1", "Test 1", "With every method set to find half of the burned area, ours flags the least unburned land."),
            (c2, "H3", "Test 2", "The false-alarm setting held on pairs of dates the calibration had not seen.")):
        ok = h[key]["pass"]
        col.markdown(f"### {name} ({key}): {'PASS' if ok else 'FAIL'}")
        col.caption(text)
        with col.expander("Detail"):
            st.write(h[key]["detail"])
    st.caption(f"Before and after dates: {met['event_pair']} (the event pair). {met['n_pos']:,} burned and "
               f"{met['n_neg']:,} comparison (control) cells. Calibration mode: {met['calibration_mode']}. "
               f"Code version: commit {met['commit'][:10]}.")

    pct = f"{odds(met['report_alpha'])} setting"
    cols = {"far": "Unburned land flagged when half the burn is found", "rec": "Burn found at 1 in 100 false alarms",
            "ap": "Ranking quality", "f1": f"Map agreement at the {pct}"}
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

    cv = D["curves"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, g in cv.groupby("detector"):
        ax[0].plot(g.recall, g.precision, color=COLORS.get(d), label=DET_SHORT.get(d, d), lw=1.6)
        ax[1].plot(g.far, g.recall, color=COLORS.get(d), label=DET_SHORT.get(d, d), lw=1.6)
    ax[0].set(xlabel="Share of the burn found (recall)", ylabel="Share of flags inside the burn (precision)",
              xlim=(0, 1), ylim=(0, 1), title="Precision-recall")
    ax[1].set(xlabel="Share of unburned land flagged (false-alarm rate, log scale)",
              ylabel="Share of the burn found (recall)", xscale="log", xlim=(1e-4, 1), ylim=(0, 1),
              title="Burn found against false alarms")
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
        ft = ft.rename(columns={"detector": "Method", "class": "Land-type group", "n_pos": "Burned cells",
                                "n_neg": "Unburned cells", "recall": "Share of burned cells flagged (recall)",
                                "far": "Share of unburned cells flagged (false-alarm rate)"})
        st.dataframe(ft, hide_index=True, width="stretch")
        st.caption(f"At the {pct}. Land-type groups run from the darkest radar return (0, open land) to the "
                   "brightest (5, dense forest); they are clusters of radar brightness before the fire.")


def view_limits() -> None:
    st.markdown(
        "#### In short\n"
        "- This is one fire at one site, seen from one satellite track.\n"
        "- The method finds changes; it does not confirm what caused them.\n"
        "- On this fire every method, including ours, misses much of the burned area.\n"
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
        thr = meta["thresholds"].get(det, {}).get(str(alpha))
        st.caption(f"Score threshold at this setting: {thr:.3f} (calibrated for {det}, alpha = {alpha})."
                   if thr is not None else "No finite threshold at this setting.")
        st.markdown("---")
        st.caption("A cell is flagged when its change score is above the threshold. Flagged patches smaller than "
                   "4 cells are then removed, so the map usually shows fewer false alarms than the setting.")
    st.title("Cold Start")
    st.markdown("Finding surface change with NISAR, a radar too new to have a history. "
                f"**Study site: {SITE_NAMES.get(site, site)}**")
    st.markdown("Red cells on the map are places that changed more than the rest of the scene predicts, between a "
                "radar image taken before the event and one taken after. These are **precomputed results for one "
                "study site**; the app does not analyse other places or new dates.")
    with st.expander("What is this, and how does it work?"):
        st.markdown("NISAR is a NASA-ISRO radar satellite whose calibrated images begin on 17 June 2026. Many methods "
                    "for spotting change in radar images compare each place with years of its own past, which NISAR "
                    "does not have yet. This project asks a different question: **on the same two dates, how did all "
                    "the similar-looking land in the scene change?** Places that changed differently from what that "
                    "predicts are flagged.")
        st.markdown("The app shows stored results for the 2026 McCully Fire in Oregon, plus a simulated test scene. "
                    "Running the method elsewhere means running the open-source pipeline in the repository.")
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
        view_limits()


main()
