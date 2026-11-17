# ECG perturbation comparison

## Question

How do segmentation and replacement choice affect temporal LIME's local fit and deletion sensitivity for the QTDB T-peak model?

This is a validation-only prototype. The 16 examples were selected with seed 73, one single-T-peak example per eligible record. The script reads only the `validation` partition. It does not load the test partition.

## Methods

The experiment crosses two segmentations with three replacements:

- **Fixed bins:** contiguous 25 ms bins (105 per 2.5 s window).
- **Signal-adaptive regions:** use the `.man` beat anchor and normalized waveform only to estimate a high-slope QRS region and lower-frequency pre-QRS and post-QRS activity boundaries. Long intervals are subdivided into chunks no longer than 100 ms. QTDB `.q1c` fiducials do not define these boundaries.
- **Zero reference:** zero in the per-window z-scored input.
- **Smoothed reference:** a Gaussian-smoothed copy of the same beat window (sigma 12 samples, about 48 ms at 250 Hz).
- **Same-record donor:** the closest normalized beat window from another annotated beat in the same source record, chosen by mean squared error over an anchor-centered interval. Donor anchor separation is at least one full 625-sample window, so the source windows do not overlap. A 20 ms cosine cross-fade is applied at replacement boundaries. This is an observed comparison signal, not a guaranteed physiological counterfactual.
- **LIME:** 256 masks per fit, seed-controlled, weighted Ridge surrogate. Top-ranked and random deletion sets each replace at least 20% of the window samples. Random deletion uses 20 sets per example.
- **Measures:** weighted local R² and absolute output shift (ms) after replacing top-ranked or random regions. These measure local fit and model sensitivity, not causal faithfulness.

## Results

Means over the 16 validation examples:

| Segmentation + replacement | Regions | Local R² | Top-region shift | Random shift | Top > random mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 25 ms + zero | 105 | 0.813 | 36.35 ms | 12.02 ms | 15/16 |
| Fixed 25 ms + smooth | 105 | 0.823 | 21.30 ms | 8.06 ms | 12/16 |
| Fixed 25 ms + donor | 105 | 0.835 | 8.48 ms | 5.87 ms | 6/16 |
| Adaptive + zero | 27.8 | 0.840 | 25.44 ms | 13.53 ms | 11/16 |
| Adaptive + smooth | 27.8 | 0.859 | 15.36 ms | 8.73 ms | 10/16 |
| Adaptive + donor | 27.8 | 0.871 | 12.83 ms | 4.46 ms | 14/16 |

Top and random replacements covered similar average fractions: about **20.2%** for fixed bins and **21.1–21.8%** for adaptive regions. The replacement reference strongly changed the output shifts. Zero replacement gave the largest shifts; same-record donors gave the smallest shifts and highest mean local fits. That pattern is consistent with zero replacement being more disruptive to the input, but the study does not measure physiological plausibility directly. A larger shift alone does not establish a more faithful explanation.

Adaptive regions used about one quarter as many regions as fixed bins and had higher mean local R² under all three replacement choices. Its top-region shifts exceeded the per-example random mean in 10–14 of 16 examples, depending on replacement. The sample is small, random-shift tails overlap top-region shifts, and behavior varies across records; this is not evidence that adaptive segmentation is generally superior.

## Boundary and failure inspection

For each mask/unmask transition, I measured the RMS change in first differences within 20 ms of the transition, relative to the unperturbed input. This is an indicator of perturbation-induced slope change in normalized units per sample; it is not a clinical signal-quality metric.

| Segmentation + replacement | Top-set slope delta RMS | Random-set slope delta RMS |
| --- | ---: | ---: |
| Fixed + zero | 0.446 | 0.336 |
| Fixed + smooth | 0.320 | 0.245 |
| Fixed + donor | 0.330 | 0.286 |
| Adaptive + zero | 0.357 | 0.269 |
| Adaptive + smooth | 0.301 | 0.188 |
| Adaptive + donor | 0.231 | 0.125 |

The adaptive donor condition had the lowest average slope-change score. Its 20 ms cross-fade is part of that result, so the metric supports only that this implementation made gentler local transitions on this cohort. It does not establish that the donor waveform is physiologically plausible.

Top-ranked deletion shifted the prediction less than the per-example random mean in **10/16 fixed-bin donor** cases and **2/16 adaptive donor** cases. The adaptive donor failures were `sele0110` beat 17 (top 1.10 ms; random mean 1.54 ms) and `sel16420` beat 3 (top 5.57 ms; random mean 6.29 ms). The full JSON records their selected intervals and per-example slope deltas. Across the cohort, top adaptive-donor regions overlapped the reference T-peak ±60 ms in 8/16 examples, the anchor ±80 ms in 9/16, and the late window (>800 ms after the anchor) in 15/16. This suggests the model often uses context outside the T-peak neighborhood; it does not explain why. I visually inspected the two failure beats in [adaptive-boundary-failures.png](../outputs/qtdb-perturbation-comparison/adaptive-boundary-failures.png): the selected regions include the T-peak neighborhood in the first beat and a mix of pre-anchor, near-anchor, and late regions in the second.

One notable case is `sel308` beat 20: the model's T-peak error was **60.2 ms**, although adaptive donor deletion shifted its output **12.8 ms** versus **6.0 ms** for the random mean. Sensitivity can be present even when the predicted location is inaccurate.

## Limits and next experiment

The cohort has only 16 records. Adaptive boundaries are heuristic activity regions, not validated P/QRS/T delineations. Each donor window avoids overlap with the explained target window, but donors can overlap one another. A donor comes from another beat in the same record, so it still shares recording and subject context and is not a verified counterfactual. Zero and smoothing can create artificial boundaries.

The follow-up donor-choice sensitivity analysis is documented in [donor-sensitivity.md](donor-sensitivity.md). It found that LIME rankings and output shifts remain donor-dependent. A donor-ensemble consensus comparison across all 552 eligible validation beats is documented in [consensus-aggregation-comparison.md](consensus-aggregation-comparison.md). A zero-shot LUDB transfer and fixed-five aggregation comparison have since been run; the results and lead-domain caveat are documented in [external-validation-plan.md](external-validation-plan.md). The QTDB test partition remains sealed, and additional databases are still needed for broader validation.

## Reproduction

```sh
uv run python scripts/compare_qtdb_perturbations.py \
  --examples 20 --perturbations 256
```

The selected 16-example cohort, per-example boundaries, and visual inspection plot are in ignored local outputs `outputs/qtdb-perturbation-comparison/summary.json`, `comparison.csv`, `adaptive-boundary-inspection.html`, and `adaptive-boundary-failures.png`.
