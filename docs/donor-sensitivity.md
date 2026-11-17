# Same-record donor sensitivity

## Question

How much do LIME rankings and deletion shifts change when the target-separated donor beat changes?

This validation-only follow-up uses the same 16 single-T-peak examples selected with seed 73 in the [perturbation comparison](perturbation-comparison.md). It does not load the test partition.

## Protocol

- For each explained beat, collect full-window donor beats from the same record whose anchor is at least 625 samples away. This prevents overlap with the explained target window; selected donors may overlap with each other.
- Rank eligible donors by normalized waveform MSE over the anchor-centered interval and select five quantiles from closest to farthest.
- For each donor, run LIME on both fixed 25 ms bins and signal-adaptive regions, using 256 masks and the same mask seed across donor choices for a given beat.
- Measure pairwise Jaccard overlap of the top regions selected to cover at least 20% of the window, weighted local R², and the top-region output-shift range across donor choices. Compare each top deletion with 20 sample-budget-matched random deletions.

This yields 80 donor-specific analyses for each region method. The farthest donor quantile can be substantially less similar to the explained beat than the nearest one; this tests sensitivity to the replacement reference rather than representing one canonical counterfactual.

## Results

| Measure across donor choices | Fixed 25 ms bins | Signal-adaptive regions |
| --- | ---: | ---: |
| Mean pairwise top-region Jaccard | 0.325 | 0.375 |
| Mean local R² | 0.832 | 0.869 |
| Mean within-beat local R² range | 0.060 | 0.044 |
| Mean within-beat top-shift range | 23.1 ms | 18.5 ms |
| Median within-beat top-shift range | 18.7 ms | 16.5 ms |
| Donor cases where top shift exceeded random mean | 52/80 | 60/80 |

The low pairwise Jaccard values show that the highlighted regions changed materially with donor choice for both segmentations. Adaptive regions were somewhat more stable by these averages, but their rankings were still donor-dependent. Output-shift variability was also substantial.

Two examples show the spread:

- `sel44` beat 21, fixed bins: top-region shift ranged from **3.2 to 58.8 ms** across donors; pairwise Jaccard was **0.276**.
- `sel36` beat 24, adaptive regions: top-region shift ranged from **0.1 to 40.4 ms**; pairwise Jaccard was **0.280**.

These are model-sensitivity measurements. Better local fit or a larger output shift does not establish that the explanation is more faithful, and a same-record donor is not a validated physiological counterfactual.

## Next step

The validation pilot tested donor-ensemble consensus; results are in [donor-consensus.md](donor-consensus.md). The follow-up compared equal averaging, similarity-weighted averaging, and component median on all 552 eligible validation beats; see [consensus-aggregation-comparison.md](consensus-aggregation-comparison.md). The test split remains sealed. LUDB audit, zero-shot transfer gate, and aggregation results are documented in [external-validation-plan.md](external-validation-plan.md).

## Reproduction

```sh
uv run python scripts/analyze_qtdb_donor_sensitivity.py \
  --examples 20 --donors-per-example 5 --perturbations 256 --random-trials 20
```

Per-donor results are in the ignored local outputs `outputs/qtdb-donor-sensitivity/summary.json` and `per-donor.csv`.
