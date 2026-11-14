# Methods

## Problem

Given co-registered NISAR L-band backscatter images of one scene before and after an event,
mark the 100 m cells whose surface changed, at a false-alarm rate fixed in advance, using only
the handful of acquisitions NISAR has made since its calibrated record began on 2026-06-17.

## Why pooling replaces history

Backscatter changes between two dates for two reasons. **Shared change** (rain, drying,
season) moves whole kinds of land together. **Local change** (fire, clearing, flooding)
affects a bounded area. A long per-pixel baseline estimates how one pixel varies over time.
Thousands of similar pixels on the same two dates estimate how that kind of pixel varied this
time, which needs only two acquisitions.

## Preprocessing

1. Read HH and HV power (`HHHH`, `HVHV`), `rtcGammaToSigmaFactor` and `mask` for a window of the
   fire's bounding box plus 30 km, streamed from ASF with HTTPS byte ranges.
2. Average k x k native pixels in **linear power** (k = 10 at 10 m spacing); a cell needs 60%
   usable pixels. Averaging decibels would be biased.
3. Convert to dB. Mask cells that are missing on any date, water (HV at or below -24 dB on every
   date) and terrain-factor outliers (outside the 1st-99th percentile).
4. Assign 10 km tiles to five folds with a seeded permutation.

## Features (per date pair a -> b)

| Name | Definition |
| --- | --- |
| `hh0`, `hv0` | Backscatter on date a (dB) |
| `ratio0` | `hv0 - hh0` |
| `tex0` | Standard deviation of `hv0` in a 3 x 3 window |
| `terrain` | Mean `rtcGammaToSigmaFactor` |
| `cls` | k-means (6 classes) on per-cell medians of pre-ignition HH and HV |

No coordinates, tile or fold ids are features. Targets: `d_hh`, `d_hv`, the change in dB.

## Pooled conditional model M

For fold k, polarization p and quantile τ ∈ {0.05, 0.5, 0.95}, a gradient-boosted regressor
minimizes the pinball loss on cells **outside** fold k:

    ρ_τ(u) = u (τ - 1[u < 0])
    q̂_τ,p^(-k) = argmin_f Σ_{i ∉ fold k} ρ_τ(d_ip - f(z_i))

and predicts cells inside fold k, so every cell is scored by models that never saw its tile.
Predicted quantiles are sorted to repair crossing. The score is the residual scaled by the
predicted spread, with the larger polarization taken:

    r_ip = (d_ip - q̂_0.5,p(z_i)) / max(½ (q̂_0.95,p(z_i) - q̂_0.05,p(z_i)), δ),   δ = 0.25 dB
    s_i  = max_p |r_ip|

Hyperparameters are fixed in the config (learning rate 0.1, 200 iterations, 31 leaves, at
least 200 cells per leaf, L2 1.0) and never tuned against the perimeter.

## Baselines

| ID | Score | Represents |
| --- | --- | --- |
| B1 | max_p \|d_p\| | Log-ratio, the standard method |
| B2 | max_p \|x_b - mean(h)\| / max(sd(h), 0.5 dB), h = the 2 most recent same-regime dates | Per-pixel history, forced to run on a short record |
| B3 | max_p \|d_p - median_c\| / max(½ (q95_c - q05_c), 0.25 dB) per land class c | Classical stratified normalization, no learning |

B1 and B3 are special cases of M: a zero median with constant spread gives B1; a class
median with class spread gives B3. The comparison therefore isolates what conditioning adds.
**If M does not beat B3, the learning adds nothing**, and that is reported.

## Conformal calibration

With n calibration scores from no-event pairs, sorted ascending, the threshold for target
false-alarm rate α is

    t_α = S_(⌈(n+1)(1-α)⌉)

and a cell is flagged when s_i > t_α. Pairs: one event pair (last date before ignition to the
first after the final perimeter update) and every consecutive pair on one side of the fire.
Flagged patches under 4 cells (8-connected) are removed for maps, F1 and IoU, and for every
detector alike; PR curves use raw scores.

## Validation

1. **Freeze**: hypotheses and config committed before the event pair is evaluated.
2. **H3**: leave one no-event pair out, set thresholds from the rest, measure the raw realized
   rate on the one left out. Pass if within [α/2, 2α] for α = 1% and 5% on every pair.
3. **H1**: on the event pair, false-alarm rate at 50% recall; resample 5 km blocks 500 times
   (blocks, because neighbouring cells are correlated) and pass if M is lower than every
   baseline and every M-minus-baseline 95% interval lies below zero.

Positives are cells inside the perimeter eroded by one cell; negatives are cells more than
2 km outside; the ring between, masked cells and other 2026 perimeters are excluded.
Precision-recall rather than ROC, because burned cells are about 1% of the window.
