"""Read-only viewer for the cold-start change-detection results.

Reads only artifacts/<site>/ (and docs/LIMITATIONS.md). Runs no models.
    streamlit run app/streamlit_app.py
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
DET_NAMES = {"M": "M  pooled conditional model (proposed)", "B3": "B3  stratified class shift",
             "B2": "B2  short-baseline z-score", "B1": "B1  log-ratio"}
COLORS = {"B1": "#8c8c8c", "B2": "#c08a2b", "B3": "#3b7dbf", "M": "#c23b3b"}

st.set_page_config(page_title="NISAR cold-start change detection", layout="wide")


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
def view_map(site: str, D: dict, det: str, alpha: float) -> None:
    meta, arr = D["meta"], D["arr"]
    bit = meta["alphas"].index(alpha)
    (s, w), (n, e) = meta["bounds"]
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=10, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="Satellite imagery").add_to(m)
    b = [[s, w], [n, e]]
    ev = meta["event_pair"]
    layers = [(f"HV before ({ev['date_a']})", rgba(site, "hv_before", -25, -5, "gray"), False),
              (f"HV after ({ev['date_b']})", rgba(site, "hv_after", -25, -5, "gray"), True),
              (f"{det} score", rgba(site, f"score_{det}", 0, 6, "magma"), False),
              (f"{det} flagged, alpha={alpha}", rgba(site, f"flags_{det}", 0, 1, "Reds", bit=bit), True),
              ("Fire perimeter (truth)", outline(site), True)]
    for name, img, show in layers:
        folium.raster_layers.ImageOverlay(image=to_mercator_rows(img, s, n), bounds=b, name=name,
                                          opacity=0.85 if "HV" in name else 1.0, show=show).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    left, right = st.columns([3, 2])
    with left:
        out = st_folium(m, height=620, use_container_width=True, returned_objects=["last_clicked"], key=f"map_{site}")
        st.caption("Layers: toggle in the map's top-right control. Click any cell to inspect it.")
    with right:
        click = (out or {}).get("last_clicked")
        if click:
            st.session_state["click"] = (click["lat"], click["lng"])
        view_inspector(D, st.session_state.get("click"))


def view_inspector(D: dict, click) -> None:
    meta, arr = D["meta"], D["arr"]
    st.subheader("Cell inspector")
    if not click:
        st.info("Click a cell on the map to chart its HH and HV backscatter across every date.")
        return
    rc = cell_at(meta, *click)
    if rc is None:
        st.warning("That point is outside the processed window.")
        return
    r, c = rc
    dates = pd.to_datetime(meta["dates"])
    hh, hv = arr["hh_ts"][:, r, c].astype(float), arr["hv_ts"][:, r, c].astype(float)
    ev = meta["event_pair"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(dates, hh, "o-", color="#3b7dbf", label="HH")
    ax.plot(dates, hv, "o-", color="#2e8b57", label="HV")
    da, db = pd.Timestamp(ev["date_a"]), pd.Timestamp(ev["date_b"])
    ax.axvspan(da, db, color="#f0a030", alpha=0.15, label="event pair")
    q05, q50, q95 = (float(arr[k][r, c]) for k in ("hv_q05", "hv_q50", "hv_q95"))
    if np.isfinite(q50):
        ax.errorbar([db], [q50], yerr=[[q50 - q05], [q95 - q50]], fmt="s", color="#c23b3b", capsize=5,
                    label="M: expected HV (5-95%)")
    ax.set_ylabel("Backscatter (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)
    lab = {1: "inside perimeter", 0: "control land", -1: "excluded"}.get(int(arr["labels"][r, c]), "?")
    scores = {d: float(arr[f"score_{d}"][r, c]) for d in meta["detectors"]}
    k = int(arr["cls"][r, c])
    st.markdown(f"**Cell** row {r}, col {c} at {click[0]:.4f}, {click[1]:.4f}: *{lab}*, "
                + (f"land class {k}" if k >= 0 else "masked (water, steep terrain or missing data; not scored)"))
    st.dataframe(pd.DataFrame({"detector": list(scores), "score": [f"{v:.2f}" if np.isfinite(v) else "not scored"
                                                                      for v in scores.values()]}),
                 hide_index=True, width="stretch")


def _fmt(v, ci):
    if v is None:
        return "n/a"
    lo, hi = ci or (None, None)
    return f"{v:.3f}" + (f"  [{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "")


def view_validation(D: dict) -> None:
    met, cal = D["metrics"], D["calib"]
    h = met["hypotheses"]
    c1, c2 = st.columns(2)
    for col, key, text in ((c1, "H1", "At 50% recall the pooled model raises fewer false alarms than every baseline"),
                           (c2, "H3", "The calibrated threshold delivers its stated false-alarm rate on unseen pairs")):
        ok = h[key]["pass"]
        col.markdown(f"### {key}: {'PASS' if ok else 'FAIL'}")
        col.caption(text)
        with col.expander("Detail"):
            st.write(h[key]["detail"])
    st.caption(f"Event pair {met['event_pair']}; {met['n_pos']:,} burned and {met['n_neg']:,} control cells; "
               f"calibration mode: {met['calibration_mode']}; commit {met['commit'][:10]}.")

    rows = []
    for d in ("M", "B3", "B2", "B1"):
        if d not in met["detectors"]:
            continue
        x = met["detectors"][d]
        rows.append({"detector": DET_NAMES[d],
                     "AP": _fmt(x["ap"], x["ci95"]["ap"]),
                     "FAR at 50% recall": _fmt(x["far_at_recall50"], x["ci95"]["far_at_recall50"]),
                     "recall at 1% FAR": _fmt(x["recall_at_far1"], x["ci95"]["recall_at_far1"]),
                     f"F1 at alpha={met['report_alpha']}": _fmt(x["f1_at_report_alpha"], x["ci95"]["f1_at_report_alpha"])})
    st.markdown("#### Metrics on the event pair (95% block-bootstrap intervals)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    cv = D["curves"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, g in cv.groupby("detector"):
        ax[0].plot(g.recall, g.precision, color=COLORS.get(d), label=d, lw=1.6)
        ax[1].plot(g.far, g.recall, color=COLORS.get(d), label=d, lw=1.6)
    ax[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Precision-recall")
    ax[1].set(xlabel="False-alarm rate (log)", ylabel="Recall", xscale="log", xlim=(1e-4, 1), ylim=(0, 1),
              title="Recall against false alarms")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(frameon=False)
    st.pyplot(fig, clear_figure=True)

    st.markdown("#### H3: leave-one-pair-out calibration")
    st.caption("Thresholds from the other no-event pairs, applied to the left-out pair. "
               "`realized` is the raw rate the conformal guarantee covers and decides H3; "
               "`after patch filter` is what the map shows.")
    lo = pd.DataFrame(cal["lopo"])
    if len(lo):
        lo = lo[lo["pass"].notna()][["detector", "left_out", "alpha", "realized", "realized_filtered", "pass"]]
        lo = lo.rename(columns={"realized_filtered": "after patch filter"})
        st.dataframe(lo, hide_index=True, width="stretch")
    if D["failure"] is not None:
        st.markdown("#### Where each detector succeeds and fails, by land class")
        st.dataframe(D["failure"], hide_index=True, width="stretch")


def view_limits() -> None:
    p = ROOT / "docs" / "LIMITATIONS.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "docs/LIMITATIONS.md not found.")


# ----------------------------------------------------------------------------- page
def main() -> None:
    avail = sites()
    if not avail:
        st.error("No artifacts found. Run `python -m coldstart run config/synthetic.yaml` first.")
        return
    with st.sidebar:
        st.markdown("## NISAR cold-start change detection")
        default = avail.index("mccully") if "mccully" in avail else 0
        site = st.selectbox("Site", avail, index=default)
        D = load(site)
        meta = D["meta"]
        det = st.selectbox("Detector", [d for d in ("M", "B3", "B2", "B1") if d in meta["detectors"]],
                           format_func=lambda d: DET_NAMES[d])
        alpha = st.select_slider("Target false-alarm rate (alpha)", options=meta["alphas"],
                                 value=meta["report_alpha"])
        thr = meta["thresholds"].get(det, {}).get(str(alpha))
        st.caption(f"Calibrated threshold for {det}: {thr:.3f}" if thr is not None else "No finite threshold.")
        st.markdown("---")
        st.caption("Flags are scores above a threshold calibrated on date pairs with no fire, "
                   "then cleaned of patches under 4 cells.")
    st.title(f"Cold-start change detection, {site}")
    st.markdown("NISAR's calibrated record began on 17 June 2026, too recently for methods that compare each pixel "
                "with years of its own history. This model learns the change to expect from **all pixels in the "
                "scene on the same two dates**, then flags cells that depart from it, at a false-alarm rate set "
                "in advance by conformal calibration.")
    t1, t2, t3 = st.tabs(["Map and cell inspector", "Validation", "Limits"])
    with t1:
        view_map(site, D, det, alpha)
    with t2:
        view_validation(D)
    with t3:
        view_limits()


main()
