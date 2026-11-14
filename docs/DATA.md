# Data

## Products

| Dataset | Role | Access |
| --- | --- | --- |
| NISAR L2 GCOV, L-band, provisional (NASA/ISRO, via ASF DAAC) | Backscatter, the signal | `asf_search`, dataset `NISAR`, processing level `GCOV`; free Earthdata Login |
| NIFC WFIGS Interagency Fire Perimeters | Ground truth | Public ArcGIS feature service, no login |

## Access

1. Create an Earthdata Login account and put the credentials in `~/.netrc` (Windows: also
   `%USERPROFILE%\_netrc`) for `urs.earthdata.nasa.gov`.
2. `python scripts/find_scenes.py` searches the candidate fires and ranks same-track groups.
3. `python scripts/fetch_perimeter.py config/site.yaml` writes `data/perimeter/mccully.geojson`.
4. `python -m coldstart search config/site.yaml` writes `data/interim/mccully/manifest.csv`.
5. `python -m coldstart grid config/site.yaml` streams only the window from each scene (no
   7.5 GB downloads) and caches the 100 m result per date in `data/interim/mccully/dates/`.

## Site selection (2026-11-14)

`scripts/find_scenes.py` over the four WFIGS candidates; best same-track group per fire:

| Fire | Track/dir/frame | Pre-ignition dates | Post dates |
| --- | --- | --- | --- |
| **McCully** (chosen) | 077 A 024 | 5 | 6 |
| Timber | 042 D 070 | 4 | 7 |
| Austin | 106 A 024 | 4 | 7 |
| Grasshopper | 106 A 024 | 3 | 8 |

McCully gives the most pre-ignition dates, hence the most no-event pairs for calibration, and
is compact (about 55 km²), so the burn cannot dominate the scene the model learns from.

## Selected scenes

Track 077, ascending, frame 024, mode 4005 (40 + 5 MHz), dual-pol HH/HV, all provisional
(`PR`) products. Dates: 2026-06-19, 07-01, 07-13, 07-25, 08-18 (pre-ignition);
08-30, 09-11 (during the fire, unused); 09-23, 10-05, 10-17, 10-29 (post). June 19 has three
products (`_001` to `_003`); the latest reprocessing is used. The exact list is in
`cube_meta.json` (`scenes`).

## Fire dates

- Ignition: 2026-08-22 (WFIGS `attr_FireDiscoveryDateTime`, lightning).
- `event.end` = 2026-09-23: WFIGS records no containment date for McCully. The final
  perimeter (13,604 acres) was last updated 2026-09-15 (`poly_DateCurrent`) and not after;
  news reports give 87% containment on 2026-09-15. 2026-09-23 is the first acquisition after.
- 13 other 2026 perimeters intersect the window; they are excluded from the labels.

## Resolved [VERIFY] items (from a streamed GCOV file, 2026-11-14)

| Item | Finding |
| --- | --- |
| `asf_search` accepts `dataset="NISAR"` | Yes (v14.1.0); a second dataset `NISAR-SCIENCE-PRODUCTS` returns the same GCOV scenes |
| Result property names | `sceneName`, `bytes` (int), `url`; `polarization` is null, so it is parsed from the name |
| Coordinate datasets | `xCoordinates`, `yCoordinates` (pixel centres), `projection` with attribute `epsg_code` |
| CRS and spacing | EPSG:32611, 10 m (mode 4005) so k = 10; frame 35,568 x 36,000 px |
| Units | Linear gamma-0 power: sample median HH 0.086 (-10.6 dB), HV 0.016 (-18.0 dB) |
| `mask` | uint8; values recorded per date in `cube_meta.json`; validity uses power only |
| Storage | 512 x 512 chunks, gzip; about 7.5 GB per scene |
| Streaming | 2.3 MB/s with sequential range requests; chunk-aware parallel prefetch is used |

## Known issues

- All NISAR products are provisional: calibrated, partly validated.
- Another 2026 fire (Olive Butte, 3,092 acres, discovered 2026-07-08) burned inside the window
  between two pre-ignition dates. It is excluded from labels but its change remains in one
  calibration pair, which makes thresholds slightly conservative.
