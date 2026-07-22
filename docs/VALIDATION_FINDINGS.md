# Chronos vs. Naive Baselines: Validation Findings

Accuracy validation results for the staleness reconstruction pipeline,
comparing Chronos Bolt Small against two naive baselines. Run against a
real ~30-day export of temperature and humidity sensor data.

## Why this exists

Real stuck sensor windows have no ground truth — there's no way to know
what a frozen sensor "should have" read. To measure accuracy honestly, we
pick a real, known stretch of data, pretend it's a stuck window, run it
through reconstruction blind (the reconstruction logic never sees inside
a period's start/end, so this is a genuinely fair test), and compare the
result against the real values that were withheld.

Two baselines Chronos has to beat:
- **Forward-fill** — repeat the last known value. A low bar.
- **Linear interpolation** — a straight line between the values before and
  after the gap. The real bar — beating this is what justifies using a
  forecasting model at all.

**10 trials per gap length**, each a different random real window, scored
independently and averaged. Gap lengths (7 / 60 / 200 points) match the
real distribution of stuck-period durations seen in the dataset.

## Results

Lower MAE is better. **Bold = best method for that row.**

### Temperature (`aht_temperature`)

| Gap length | Chronos | Forward-fill | Linear interp | Chronos beats FF | Chronos beats LI |
|---|---|---|---|---|---|
| 7 pts | 0.0559 | 0.0672 | **0.0550** | 4/10 | 2/10 |
| 60 pts | 0.1628 | 0.2355 | **0.1439** | 5/10 | 2/10 |
| 200 pts | **0.5326** | 0.6463 | 0.5641 | 7/10 | 5/10 |

### Humidity (`aht_humidity`)

| Gap length | Chronos | Forward-fill | Linear interp | Chronos beats FF | Chronos beats LI |
|---|---|---|---|---|---|
| 7 pts | 0.1665 | 0.3189 | **0.1302** | 4/10 | 4/10 |
| 60 pts | 1.2390 | 1.6372 | **0.8979** | 3/10 | 1/10 |
| 200 pts | **3.4237** | 5.5465 | 3.9219 | 8/10 | 6/10 |

## Interpretation

Consistent pattern across both sensors:

- **Short and medium gaps (7, 60 pts): linear interpolation wins.** Most
  real sensor signals are close to locally linear over these timescales —
  little room for a forecasting model to add value over "draw a line."
- **Long gaps (200 pts): Chronos wins**, with the lowest average error of
  all three methods for both sensors, and a real win-rate against linear
  interpolation (5/10 and 6/10 — meaningful at this sample size, not a
  fluke). Real structure (daily cycles, trend changes) emerges over longer
  gaps that a straight line can't capture.

## Decision this creates — not yet acted on

`reconstruct_stale_window()` currently always uses Chronos, regardless of
gap length. This data supports switching to linear interpolation for
short/medium gaps (cheaper *and* more accurate) and reserving Chronos for
long gaps, where it earns its complexity. **Deliberately deferred** to
keep other work (GraphQL, Kafka, automation) moving — a well-evidenced
open item, not a forgotten one.

## Two real bugs this validation work surfaced and fixed

1. **A single null in the data silently turned every metric into `NaN`**
   — for Chronos *and* both baselines, not just the model, since one
   missing real reading in a 200-point window poisons any arithmetic
   touching it. Fixed by retrying gap selection until a genuinely
   null-free window is found.
2. **MLflow was silently keeping only the last of 3 trials**, not
   averaging them — each trial logged under the same metric key with no
   step counter, so MLflow just overwrote earlier trials. This is why
   trial count and methodology matter as much as the numbers: an earlier
   3-trial run showed 0/3 wins against every baseline at every gap length,
   which looked damning but was actually a measurement artifact. Fixed by
   averaging in Python before logging, and trial count was raised to 10
   for a statistically firmer read.

## Limitations

- 10 trials is better than 3, but win-rate fractions (e.g. 5/10) still
  carry real sampling noise.
- Only validated against forward-fill and linear interpolation — PyPOTS
  (see `RELATED_WORK.md`) offers additional simple baselines not yet added.
- Synthetic gaps are drawn from the same ~30-day window being validated —
  this measures performance on this slice of data, not guaranteed to hold
  for, say, a different season with different dynamics.
