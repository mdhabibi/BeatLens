# T-peak explanation prototype

## Setup

- Model output: predicted T-peak offset from the selected-beat anchor, in milliseconds.
- Examples: 30 single-T-peak examples selected with seed 42 from validation, at most two per record across 15 records. No test records were read.
- Integrated Gradients: 48 interpolation steps. Temporal LIME: 25 ms time bins, 512 perturbations per fit, four random seeds.
- Baselines: zero in per-window z-scored input (flat per-window mean) and a Gaussian-smoothed waveform (sigma 12 samples, about 48 ms at 250 Hz).
- Stability: IG under small Gaussian input noise (`std=0.03`), LIME across seeds, and overlap between explanations from the two baselines. The top-bin Jaccard uses the 10% highest-ranked time bins.
- Deletion probe: replace the top-ranked 20% of bins with the corresponding baseline and compare prediction shift with 20 random 20% replacements.

## Expanded validation results

| Measure | Zero baseline | Smoothed baseline |
| --- | ---: | ---: |
| LIME weighted local R², mean (range) | 0.856 (0.666–0.920) | 0.887 (0.795–0.953) |
| IG top-20% deletion output shift | 30.1 ms | 19.6 ms |
| IG random-deletion output shift | 10.6 ms | 7.6 ms |
| LIME top-20% deletion output shift | 26.9 ms | 18.1 ms |
| LIME random-deletion output shift | 10.6 ms | 7.6 ms |

IG's top-10-bin Jaccard was **0.974** across small input-noise changes. LIME's seed stability was **0.747** for the zero baseline and **0.753** for the smoothed baseline. However, top-bin overlap across baselines was only **0.447** for IG and **0.371** for LIME. Baseline choice changes which regions the methods identify as important.

Mean IG completeness residual was **0.98 ms** for the zero baseline and **0.018 ms** for the smoothed baseline. The model's mean prediction error on these 30 selected examples was **26.2 ms** (range 2.5–73.4 ms).

The deletion check is a perturbation sensitivity probe, not decisive evidence of faithfulness. The top-ranked deletion shifted the output more than random deletion on average, but on individual examples it did not do so in 8/30 IG-zero, 11/30 IG-smoothed, 7/30 LIME-zero, and 9/30 LIME-smoothed cases. Replacements also alter the waveform and may create artificial transitions. Treat the maps as exploratory model-behavior views, not causal or physiological evidence. The prototype uses zero as its primary reference and retains the smoothed waveform as a sensitivity view; the [baseline inspection](xai-baseline-inspection.md) documents the choice and the cases where rankings change. No test records were used.

## Known-subject-grouped validation rerun

The demo checkpoint and validation explanations were regenerated after the three documented European ST-T same-subject pairs were kept in one split. On 30 selected examples, mean prediction error was **22.6 ms**; mean LIME weighted local R² was **0.846** for the zero reference and **0.875** for the smoothed reference. Top-bin overlap between baselines was **0.547** for IG and **0.393** for LIME, so the baseline sensitivity remains visible. Mean IG noise stability was **0.969**. No test records were read.

This remains exploratory model-sensitivity analysis, not clinical or causal evidence. The reproducible script is `scripts/explain_qtdb_tpeak.py`; explanation functions are in `src/beatlens/explainers/temporal_attribution.py`. Updated generated files are in ignored local outputs: `outputs/qtdb-explanations-known-subject-groups/explanation-summary.json` and `outputs/qtdb-explanations-known-subject-groups/temporal-attributions.csv`.

## ECG-aware perturbation comparison

A separate validation-only study crossed fixed 25 ms bins and waveform-adaptive regions with zero, smoothed, and same-record donor replacements whose windows avoid the explained target window. It used 16 eligible records and sample-matched top/random deletions. Adaptive regions had higher mean LIME local R² across all three replacements, but output shifts changed substantially with replacement choice and per-example effects varied. A transition-slope probe found the lowest average added slope change for adaptive regions with donor replacement; the cross-fade contributes to that result, so it is not a physiological realism claim. A follow-up over five target-separated donors per beat found low top-region Jaccard overlap across donor choices (0.325 fixed, 0.375 adaptive), showing the maps remain sensitive to replacement choice. See the [method and failure inspection](perturbation-comparison.md) and [donor-sensitivity analysis](donor-sensitivity.md), and ignored outputs in `outputs/qtdb-perturbation-comparison/` and `outputs/qtdb-donor-sensitivity/`.

A 48-example validation pilot averaged normalized LIME rankings across five donors separated from each target window. A follow-up compared three aggregation rules on all 552 eligible validation beats; agreement gains were small and T-peak overlap differed by method. See the [donor-consensus report](donor-consensus.md), the [aggregation comparison](consensus-aggregation-comparison.md), and the [external-validation plan](external-validation-plan.md). No QTDB test records were used for these follow-ups.
