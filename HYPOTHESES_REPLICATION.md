# Replication on a second fire: pre-registration

Committed and pushed on 2026-11-15, before any radar data for this site was processed. No result
for this site existed at that time. This file is never edited after that commit; results are
appended in a separate section below.

## Why

The McCully result (`HYPOTHESES.md`) rests on one fire. This tests the same claims, with the same
frozen pipeline and settings, on a second fire chosen by fixed rules.

## Site, chosen by rules fixed before processing

- Candidates: the four WFIGS fires listed in the original handoff, minus McCully.
- Original rules: most pre-ignition dates on one track, frame and mode, then most dates in total.
  Three groups tied (4 pre-ignition and 11 dates each): Austin track 106 ascending frame 024,
  Austin track 071 descending frame 066, and Timber track 042 descending frame 070.
- Tie-break, declared before processing: the shortest gap between the last pre-ignition
  acquisition and ignition, so the event pair spans the least unrelated time. Austin, track 106
  ascending: 13 days (others 15 and 17).
- If these scenes cannot cover the fire, the next group in the same ranking is used, and that is
  reported.

**Site:** Austin Fire, Clackamas County, Oregon. Discovered 2026-08-09; final perimeter
12,888 acres (WFIGS).

## Event pair, by the McCully rule

The first post-fire date is the first acquisition on the chosen track after the final perimeter
update. WFIGS records no containment date; the perimeter was last updated 2026-09-29. The event
pair is therefore **2026-07-27 to 2026-10-07**. No-event pairs: 06-21 to 07-03, 07-03 to 07-15 and
07-15 to 07-27 (before the fire); 10-07 to 10-19 and 10-19 to 10-31 (after).

## Settings

`config/site_austin.yaml` differs from `config/site.yaml` only in the site name, search box,
track, perimeter file, incident name and fire dates. Every model setting, threshold, mask, label
rule and evaluation setting is identical. No setting is tuned for this site.

## Hypotheses (unchanged from `HYPOTHESES.md`)

| ID | Hypothesis | Passes if |
| --- | --- | --- |
| H1 | At equal recall on the mapped fire, the pooled model raises fewer false alarms than each baseline | False-alarm rate at 50% recall is lower than every baseline (B1, B2, B3), and the 95% block-bootstrap interval of each difference (M minus baseline) lies entirely below zero |
| H3 | The calibrated threshold delivers its stated false-alarm rate on a pair it never saw | For targets of 1% and 5%, the raw realized rate of M on each left-out no-event pair lies between half and double the target |

The McCully result is called replicated for a hypothesis only if that hypothesis passes here as
well. A failure is reported as a result, with the same figures, in the README and the app.

## Known differences from McCully, noted before processing

- The event pair spans 72 days (McCully: 36), so more seasonal change enters it.
- The Grasshopper Fire (about 94,000 acres, discovered 2026-07-24) overlaps the north-east edge of
  the window. It is excluded from the labels, but its change remains in the 07-15 to 07-27
  calibration pair, which raises thresholds for every method.
- The window is in the Cascade Range, with steeper terrain than McCully.
