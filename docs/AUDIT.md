# Technical audit

Date: 2026-11-14, after the results commit `f7c8c3c`. Scope: the repository as committed, the
real McCully run, and the app. Every "verified" line names the evidence; anything without
evidence is listed as unverified.

## Summary

- The implementation matches the frozen methodology, and the reported results reproduce.
- No critical findings. Two high findings, both fixed. Four medium findings: two fixed, two
  left as documented limitations because fixing them would change the frozen test.
- The McCully claims are supported as stated. A pre-registered replication on the Austin Fire
  reproduced the calibration result but not M's advantage over the land-type baseline (H1
  failed there); the README reports both. Nothing broader is supported.

## Findings, by severity

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| H-1 | High | The README's install command did not install Streamlit, so the third reproduction command failed on a clean machine | **Fixed**: app packages moved into the core dependencies; checked from a fresh clone (below) |
| H-2 | High | Map overlays were drawn in wrong colours (flags white instead of red, greys tinted) because the mapping library's Mercator option mixes colour channels | **Fixed** before the results commit: rows are resampled in the app; checked in the browser |
| M-1 | Medium | The GCOV validity `mask` is not applied. 3.3% of valid cells contain partially focused pixels, mostly in the south-west corner | **Open by design** (frozen pipeline). Excluding those cells moves FAR at 50% recall from 0.0846 to 0.0836 (M) and 0.1285 to 0.1298 (B3); conclusions unchanged |
| M-2 | Medium | Streaming had no retry for dropped connections; one transient HTTP 403 on opening a scene was seen during the audit | **Fixed**: retries with backoff for connection errors, timeouts, 429, 5xx and the opening request; five offline tests |
| M-3 | Medium | M's advantage depends on the cross-fitting tile size (FAR at 50% recall 0.076 at 5 km, 0.085 at 10 km, 0.101 at 20 km); cross-fitting itself costs accuracy | **Open**: reported in `EVALUATION.md`; the frozen 10 km result is the one claimed. All variants remain below the best baseline (0.128) |
| M-4 | Medium | Post-fire "no-event" pairs contain real change (the scar keeps evolving, other fires), which raises thresholds for every detector | **Open**: documented; conservative |
| L-1 | Low | No pinned dependency versions | **Fixed**: `requirements.lock` |
| L-2 | Low | A hosted install of the app would pull packages it does not use (rasterio, h5py, asf_search, pytest) | **Fixed**: `app/requirements.txt` lists only what the viewer imports |
| L-3 | Low | The log warned "window clipped by the frame edge" when the window was only snapped to whole cells | **Fixed**: tolerance of one cell |
| L-4 | Low | Masked cells showed "land class -1" in the inspector | **Fixed**: shown as masked |
| L-5 | Low | One dead line in `scripts/inspect_gcov.py` | **Fixed** |
| L-6 | Low | Data-use terms were not documented | **Fixed**: `DATA.md` |
| L-7 | Low | `numberOfLooks` is not read although the specification lists it | **Open**: documented; no quality filter depends on it |
| L-8 | Low | The water threshold (-24 dB) was never compared with a water map | **Open**: unverified |
| L-9 | Low | B3's class medians and the k-means land classes are fitted in-sample | **Open**: label-free and tens of thousands of cells per class; effect judged negligible, not measured |
| L-10 | Low | `pyproject.toml` uses a licence syntax that setuptools will stop accepting in 2027 | **Open**: builds today with a warning |
| L-11 | Low | The spatial calibration fallback reads the labels | **Open**: not triggered on McCully; flagged in the output as `calibration_mode: spatial` |
| L-12 | Low | `config_sha256` in `metrics.json` depended on the checkout's line endings | **Fixed**: `.gitattributes` keeps YAML files LF on every platform |
| L-13 | Low | `scripts/fetch_perimeter.py` imports `requests`, which was only installed indirectly | **Fixed**: declared |

## Scientific integrity

| Question | Answer | Evidence |
| --- | --- | --- |
| Does the implementation match the stated method? | Yes, with three deviations decided before the freeze and recorded in `HYPOTHESES.md` | Code read against `METHODS.md`; the freeze commit `6994eb2` contains all three |
| Were the hypotheses fixed before scoring? | Yes | `6994eb2` (hypotheses and config) precedes `f7c8c3c` (results); `metrics.json` records commit `6994eb2` |
| Is data provenance accurate? | Yes | Scene names in `cube_meta.json`; perimeter from the WFIGS service with dates printed by `scripts/fetch_perimeter.py` |
| Are conclusions supported? | Yes for H1 and H3 on this fire. The README states the weak absolute performance and the single-site limit | `metrics.json`, `calibration.json` |
| Assumptions not verified | Exchangeability between calibration and event cells (tested only through H3); the event end date rests on the last perimeter update, since no containment date is recorded; the cause of the backscatter change inside the burn | |

## Machine-learning integrity

| Check | Result | Evidence |
| --- | --- | --- |
| Truth kept from detectors | Pass | Perimeter read only in `truth.py`, `evaluate.py`, `scripts/exploratory.py` |
| No location features | Pass | `tests/test_features.py` |
| Cross-fitting isolates folds | Pass | `tests/test_model.py` |
| Baselines implemented fairly | Pass, with one stated difference | Same grid, mask, calibration and patch rule; B2 is scored on fewer pairs because it needs history |
| Metrics appropriate | Pass | Precision-recall based, block bootstrap for spatial correlation |
| Results reproducible | Pass | Full real-data rerun from an anonymous clone of the public repository (scene search, streaming, every stage; 1 h 53 min) gave identical metrics, intervals and calibration; only the commit and timestamp differ. Refit of M on the event pair identical to the stored score; synthetic `metrics.json` identical across runs (`tests/test_end_to_end.py`) |
| Failure cases documented | Pass | `EVALUATION.md` sections 5 to 7, `LIMITATIONS.md` |
| Overfitting | No sign of the burn being fitted away | Out-of-pair and in-sample ablations, `EVALUATION.md` section 4 |

## Engineering

| Check | Result | Evidence |
| --- | --- | --- |
| Test suite | 35 tests pass, offline, about a minute | `pytest` on the working tree |
| Clean install | See "Fresh clone" below | |
| Credentials | None in the repository or its history | Scan of tracked files for credential patterns; `data/`, `.netrc`, `_netrc`, `*.h5` ignored; full commit history scanned before the repository was made public |
| App security | Reads local artifact files only; takes no file paths or free text from the user | Code read |
| Dependencies | All imported packages declared; none unused in the pipeline | Import scan against `pyproject.toml` |
| Performance | Real run: about 19 min to stream 11 dates, 16 min to score 8 pairs, 1 min to evaluate, on a laptop | Run log |
| Memory | One date at native resolution at a time | `io_gcov.build_real_cube` |

## Product

| Check | Result | Evidence |
| --- | --- | --- |
| App starts and renders three tabs on both sites, no console errors | Pass | Scripted browser session against the running app |
| Click-to-cell mapping | Pass | A burned cell chosen by coordinates reports "inside perimeter" and its own row and column |
| Flags and thresholds match the pipeline | Pass by construction | Flags are computed in `export.py` with the same function as the metrics |
| Perimeter aligned with the radar image | Pass | `artifacts/mccully/figures/overlay_check.png` |
| Observed data distinguished from predictions | Pass | Inspector plots measured HH and HV as lines and the model's expected 5 to 95% range as a separate marker |
| Public deployment | Pass | https://nisar-coldstart.streamlit.app, Python 3.11; full exercise on both sites from a signed-out browser, desktop and phone (2026-11-14) |

## Fresh clone

Run on 2026-11-14 from commit `63e2d6f`: `git clone`, a new Python 3.11 virtual environment,
`pip install -e ".[test]"`, `pytest`, `python -m coldstart run config/synthetic.yaml`, then the
app started from that environment.

| Step | Result |
| --- | --- |
| Install into an empty environment | Succeeded (one setuptools deprecation warning, finding L-10) |
| Test suite | 35 passed |
| Synthetic pipeline | Completed; H1 and H3 pass |
| Regenerated synthetic artifacts against the committed ones | Byte-identical (maps, curves, calibration, figures); `metrics.json` differs only in commit, config hash and timestamp |
| App from the fresh install | Health endpoint returned 200 |

The config hash differed because the fresh checkout had Windows line endings (finding L-12).

## Final review (2026-11-14, after deployment)

Re-checked on an anonymous clone of the public repository and the live app in a signed-out
browser.

| ID | Severity | Finding | Status |
| --- | --- | --- | --- |
| R-1 | Medium | On a 1366x768 screen the map began below the first screen; a visitor saw only text | **Fixed**: short intro, explanation folded away, guide moved below the map. About half the map is now visible on load |
| R-2 | Medium | The deck said the model "kept its promised false-alarm rate" and that the method "needs two images" | **Fixed**: "stayed within half to double its target"; "needs months of images, not years" (calibration uses the other date pairs) |
| R-3 | Medium | H3 is partly guaranteed by the score's construction (see `EVALUATION.md` section 2) | **Documented**; the claim is worded accordingly |
| R-4 | Low | One of M's 14 H3 checks passes by a margin of 0.00004 (0.01996 against 0.02) | Documented |
| R-5 | Low | Commit timestamps are set locally; the repository was private until 22:48 UTC on 2026-11-14, so the freeze order cannot be proven by an outside party | Documented |
| R-6 | Low | Test coverage 79% overall; `download` 21%, `search` 26%, `io_gcov` 53%. No continuous integration. `ruff` reports 33 minor findings | Open |
| R-7 | Low | `requirements.lock` was generated on Windows (includes `colorama`); other platforms should install from `requirements.txt` | Open |
| R-8 | Low | The stale "Public deployment: Not done" line in this file | **Fixed** |
| R-9 | High | The result rested on one fire | **Addressed by a pre-registered replication** (Austin). Its pre-registration was pushed at 20:21:03 UTC on 2026-11-15 per GitHub's activity log, before processing; H3 replicated, H1 did not. This also gives the replication the server-side timestamp that R-5 notes McCully lacks |

## Unverified

- Every `[V]` literature statement inherited from the planning documents (read as search
  excerpts, not full papers), and therefore any claim of novelty. The README makes none.
- Behaviour on any site other than McCully.
- The physical cause of the change inside the burn.
- The app on a hosted service.
