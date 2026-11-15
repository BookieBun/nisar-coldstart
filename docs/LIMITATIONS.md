# Limitations

## What the conformal guarantee covers

The threshold for a target false-alarm rate α is set from scores on date pairs with no fire.
If unchanged cells on the event pair are **exchangeable** with those calibration cells, the
expected fraction of unchanged cells flagged is at most α. That condition is an assumption
about the data, not a property of the code:

- Neighbouring 100 m cells are correlated, so the realized rate on any one pair scatters
  around α more than independent cells would. The H3 table shows that scatter.
- Different date pairs have different weather. A pair with unusual rain can violate
  exchangeability; the pooled model is designed to absorb shared shifts, but not every one.
- The guarantee covers raw scores. The map removes flagged patches under 4 cells, so the rate
  a viewer sees is lower than α (often far lower). Both numbers are reported.

## Known biases in the labels

| Bias | Effect | Applies to |
| --- | --- | --- |
| Unburned islands inside the perimeter | Caps recall below 100% | All detectors equally |
| Real change outside the perimeter (harvest, other fires, prescribed burns, crops) | Counted as false alarms | All detectors equally |
| Perimeter is an outline, not burn severity | Low-severity burn counts as positive though it may not change backscatter | All detectors equally |
| Other 2026 perimeters in the window are excluded from labels, but not from calibration | Real change in a no-event pair (e.g. Olive Butte fire, July 2026) inflates thresholds; conservative | All detectors equally |

## Observed on McCully after the freeze (exploratory, not part of the test)

Mean change from the last pre-fire date (2026-08-18), inside the perimeter versus control
land, and the share of cells changing by more than 1.5 dB:

| To date | HH inside | HH control | HV inside | HV control |
| --- | --- | --- | --- | --- |
| 2026-09-23 (frozen event date) | +0.66 dB (12.3%) | +0.05 dB (4.7%) | +0.45 dB (6.9%) | -0.07 dB (2.1%) |
| 2026-10-17 | -0.59 dB (12.8%) | -0.58 dB (10.6%) | -0.77 dB (17.5%) | -0.54 dB (6.5%) |
| 2026-10-29 | -0.98 dB (30.6%) | -0.63 dB (18.1%) | -1.74 dB (57.5%) | -0.91 dB (22.2%) |

- The burn signal at 100 m on the frozen event pair is modest. HV rose slightly instead of
  falling as the specification assumed; all detectors score absolute change, so this does not
  bias the comparison.
- The burn scar keeps changing after the fire, and more strongly than control land, as the
  autumn season sets in. The post-fire "no-event" pairs therefore contain real change inside
  the scar, which raises thresholds for every detector (conservative).
- A later event date would show a stronger contrast, but choosing it after looking would be a
  post-hoc choice; the frozen event pair is the test.

## Expected failure modes

| Failure | Symptom | Check |
| --- | --- | --- |
| Event too large for the scene | Model learns the burn as normal; recall collapses | McCully is about 55 km² in a window of several thousand km² |
| Rain between dates over part of the scene | Broad, low-contrast alarm regions | Not tested: rainfall covariate (H2) not implemented |
| Steep terrain | Alarms along ridges | Terrain factor outside the 1st-99th percentile is masked |
| Radio interference or processing artifacts | Stripes along the flight direction | Visual check of each date |
| Weak fire signal at 100 m | All detectors near chance | Reported as measured |

## Calibration choices that differ from the plan

- **B2 is scored only on pairs with two same-regime history dates.** The first pre-fire pair
  and the first post-fire pair are not scored by B2, so B2 calibrates on fewer pairs.
- **The spatial fallback uses labels.** If fewer than two no-event pairs existed, calibration
  would use control cells from the event pair, which lets the perimeter reach the threshold.
  It is reported as `calibration_mode: spatial`. McCully has enough no-event pairs, so it is
  not used there.

## What was not tested

- H2 (rainfall), external land cover (WorldCover), optical comparison (OPERA DIST-ALERT).
- Sensitivity to the aggregation factor (it would need the scenes streamed again). Ablations
  and sensitivity to tile size and folds were run after the test, as exploratory analysis
  (`docs/EVALUATION.md`): the result depends on tile size, and added noise of 0.5 dB removes
  most of every detector's skill.
- More sites. Two fires, one of them a pre-registered replication, are a small sample, not a
  general result. The replication (Austin) reproduced the calibration result but not the model's
  advantage over the land-type baseline, and detection there was weak for every method.
- The GCOV `mask` layer is not applied; validity is decided from the power values only. A
  sensitivity check found the results unchanged without the affected 3.3% of cells
  (`docs/DATA.md`).
- The false-alarm patches were not checked against optical imagery.
- The water threshold (HV at or below -24 dB on every date, 3,978 cells) was not compared with
  a water map.
- `numberOfLooks` is not read (to halve streaming volume); no look-count quality filter.

## Synthetic data

The synthetic scenes test the plumbing. Their generator includes a brightness-dependent
shared shift, which the pooled model can learn and class stratification cannot, so the
synthetic gap between M and B3 is not evidence about real data. With that term switched off
(`brightness_coupling_sd: 0`), M still leads B3 but by less:

| Synthetic run | AP M | AP B3 | FAR at 50% recall, M | FAR at 50% recall, B3 |
| --- | --- | --- | --- | --- |
| With brightness coupling (shipped config) | 0.399 | 0.082 | 0.014 | 0.098 |
| Without it | 0.228 | 0.089 | 0.042 | 0.107 |

The remaining gap comes from conditioning on the earlier date: a cell that is unusually bright
for its class on date a tends to come back towards the class level on date b (regression to
the mean), which a class median cannot see. That effect exists in any two-date comparison,
but its size on real NISAR data is what the McCully test measures.
