# Donor-ensemble explanation consensus

## Question

Can averaging LIME rankings across several target-separated donor beats produce regions that are more stable than a single donor explanation?

This pilot used **48 single-T-peak validation beats across 16 records**, at most three per record, selected with seed 97. Five same-record donor windows were sampled from similarity quantiles (closest to farthest); each donor window was non-overlapping with the explained beat window. Donor windows could overlap one another. The test split was not loaded.

## Method

- Use the waveform-adaptive regions from the earlier perturbation study, capped at 100 ms. No `.q1c` event defines a region.
- Fit temporal LIME with 256 masks per donor. Use the same mask seed for the five donors for a given beat.
- Normalize each donor's absolute LIME coefficients to sum to one, then average those normalized scores to form the consensus ranking.
- Select consensus regions until they cover at least 20% of the window samples. Compare with each donor-specific top set and with 20 sample-budget-matched random deletions per donor.
- Measure pairwise donor top-set Jaccard, consensus-versus-donor Jaccard, local R², and model-output shift. Check consensus-region overlap with the annotated T peak and beat anchor only after selection.

This evaluates explanation sensitivity on validation data; it does not tune or evaluate the model, and the donor mean is not a validated physiological counterfactual.

## Results

| Measure | Result |
| --- | ---: |
| Mean local LIME R² across donors | 0.875 |
| Mean within-beat donor R² range | 0.047 |
| Pairwise donor top-region Jaccard | 0.403 |
| Consensus-versus-donor top-region Jaccard | 0.556 |
| Mean consensus output shift across donors | 12.16 ms |
| Mean within-beat consensus-shift range across donors | 19.85 ms |
| Mean individual donor top-region shift | 12.40 ms |
| Mean matched random-region shift | 5.82 ms |
| Consensus regions within 60 ms of reference T peak | 29/48 |
| Consensus regions within 80 ms of anchor | 39/48 |
| Consensus regions in late window (>800 ms after anchor) | 48/48 |
| Examples where mean consensus shift exceeded mean random shift | 43/48 |

Consensus regions agreed with an individual donor more often than donors agreed with each other, but the mean Jaccard of 0.556 is still moderate. The mean output shift range across donors was about 20 ms, and some beats varied much more. For `sele0112` beat 21, consensus-region deletion shifted the output from **8.2 to 108.2 ms** across donors; selected donor distances ranged from **0.01 to 2.20 MSE**. The farthest donor can strongly change the measured effect even when the consensus mask is held fixed. A matched comparison of equal mean, similarity-weighted mean, and component median is documented in [the aggregation comparison](consensus-aggregation-comparison.md).

The consensus often included late-window regions. That indicates model dependence on distant context under these perturbations; it does not establish that the context is physiologically meaningful. Also, `sel213` beat 25 had **65.5 ms** localization error while its consensus deletion shift averaged 14.9 ms, so sensitivity does not imply accurate localization.

## Interpretation

The ensemble makes the selected mask more representative of donor-specific rankings, but donor dependence remains substantial. This pilot does not support presenting consensus regions as a single definitive explanation. The follow-up aggregation comparison found a small average agreement gain for the component median. A later zero-shot LUDB transfer passed the timing-prior gate and showed a similar small agreement gain on 433 targets; the donor windows were correlated, and the input-lead mapping is domain-shifted. See the [external validation report](external-validation-plan.md); broader multi-database evaluation remains necessary.

## Reproduction

```sh
uv run python scripts/evaluate_qtdb_donor_consensus.py \
  --examples 48 --max-per-record 3 --donors-per-example 5 \
  --perturbations 256 --random-trials 20
```

The per-example and per-donor outputs are in the ignored local directory `outputs/qtdb-donor-consensus/`.
