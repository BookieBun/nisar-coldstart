# Hypotheses

Frozen before the real event pair is scored. This file is never edited after the freeze
commit; results are appended in a separate section below.

- Site: McCully Fire, Baker County, Oregon (`config/site.yaml`)
- Event pair: last track-077 acquisition before ignition (2026-08-18) to the first after the
  final perimeter update (2026-09-23)
- Frozen: 2026-11-14

| ID | Hypothesis | Passes if | Tier |
| --- | --- | --- | --- |
| H1 | At equal recall on the mapped fire, the pooled model raises fewer false alarms than each baseline | False-alarm rate at 50% recall is lower than every baseline (B1, B2, B3), and the 95% block-bootstrap interval of each difference (M minus baseline) lies entirely below zero | Must |
| H3 | The calibrated threshold delivers its stated false-alarm rate on a pair it never saw | For targets of 1% and 5%, the raw realized rate of M on each left-out no-event pair lies between half and double the target | Must |
| H2 | The pooled model's advantage grows when rainfall differs between the two dates | Positive rank correlation across no-event pairs; descriptive only below six pairs | Stretch (not tested) |

A failed hypothesis is reported as a result, with the same figures.

## Fixed before scoring

All thresholds, hyperparameters, masks and label rules are in `config/site.yaml` at the freeze
commit. Nothing is tuned against the perimeter.

## Decisions taken before the freeze

These resolve conflicts between the specification and the handoff. Each was made before any
real-site result existed.

1. **H3 uses the raw realized rate.** The conformal guarantee covers raw scores. The rate after
   the minimum-patch filter is always lower than the raw rate, so judging H3 on it would test
   the filter, not the calibration. Both rates are reported.
2. **B2 uses a fixed history length (2 dates) from the same regime as the pair's first date.**
   With "all earlier dates" as history, B2's estimator differs between calibration pairs and
   the event pair, so its pooled threshold would be calibrated on a different distribution;
   and a post-fire pair's history would mix pre-fire values into the mean, flagging the burn
   itself. Pairs without enough same-regime history are not scored by B2.
3. **`mode: real | synthetic` in the config** tells `run` whether to call `synth` or `grid`.
