# Donor-consensus aggregation comparison

## Question

Does equal averaging, similarity weighting, or a component-wise median produce the most stable adaptive-region ranking across donor beats?

The initial pilot used 48 single-T-peak examples (up to three per record) across 16 records. The expanded run below uses **all 552 eligible single-T-peak validation beats** across those same 16 records. Both use five same-record donors per example, with each donor window separated from the explained target window, adaptive regions, 256 LIME masks per donor, and 20 matched random deletions per donor. Donor windows may overlap one another. The pilot selected examples and donors with seed 97; the expanded run uses the same seed for the deterministic record and donor ordering. The test split was not loaded.

## Aggregation rules

- **Equal mean:** arithmetic mean of each donor's absolute LIME coefficients after L1 normalization.
- **Similarity weighted:** weighted mean of the same normalized donor vectors, with weights `softmax(-MSE / median donor MSE)`. This gives closer donors more influence while retaining nonzero weight for all five.
- **Component median:** component-wise median of the normalized donor vectors, then L1 normalization.

For each rule, adaptive regions were selected by descending score until they covered at least 20% of window samples. Output shift and annotation overlap use that selected set. The same individual-donor and random deletion sets are used for all three comparisons.

## Results

| Measure (mean across 48 examples unless stated) | Equal mean | Similarity weighted | Component median |
| --- | ---: | ---: | ---: |
| Consensus-to-donor top-region Jaccard | 0.556 | 0.540 | **0.565** |
| Consensus output-shift mean (ms) | 12.16 | **11.77** | 12.43 |
| Within-example consensus shift range (ms) | 19.85 | **17.85** | 20.89 |
| Matched random output-shift mean (ms) | 5.82 | 5.82 | 5.82 |
| Examples overlapping T peak ±60 ms | 29/48 | 19/48 | 29/48 |
| Examples overlapping anchor ±80 ms | 39/48 | 35/48 | 34/48 |
| Examples with a selected late-window region | 48/48 | 46/48 | 48/48 |

Donor top sets themselves had mean pairwise Jaccard **0.403** across methods. Mean donor local LIME R² was **0.875** (mean within-example donor R² range **0.047**); these are unchanged because the donor fits are shared. The selected masks were not identical between rules: their mean pairwise Jaccard was 0.748 for equal versus weighted, 0.736 for equal versus median, and 0.647 for weighted versus median.

## Expanded validation cohort

The expanded run includes every eligible beat with one annotated T peak and a fully valid input window: **552 beats from 16 records**. The previous cap of three examples per record was removed. Means below are across beats; confidence intervals for paired differences use a 20,000-resample bootstrap that resamples the 16 records as clusters (seed 544), to account for beats from the same record being related.

| Measure | Equal mean | Similarity weighted | Component median |
| --- | ---: | ---: | ---: |
| Consensus-to-donor top-region Jaccard | 0.551 | 0.534 | **0.562** |
| Mean consensus output shift (ms) | 13.51 | 12.52 | 13.45 |
| Mean within-example shift range (ms) | 21.40 | 18.76 | 21.22 |
| Matched random output-shift mean (ms) | 6.37 | 6.37 | 6.37 |
| Examples overlapping T peak ±60 ms | 371/552 | 270/552 | 334/552 |
| Examples overlapping anchor ±80 ms | 444/552 | 401/552 | 433/552 |
| Examples with a selected late-window region | 542/552 | 545/552 | 537/552 |

The component median's paired agreement gain over equal mean was **0.011 Jaccard** (record-cluster bootstrap 95% interval **0.007 to 0.016**). Similarity weighting was **0.016 lower** than equal mean (**−0.024 to −0.009**). The median's T-peak overlap was 6.7 percentage points lower than equal mean (**−11.5 to −2.2 pp**); similarity weighting was 18.3 points lower (**−25.9 to −10.5 pp**). For the mean shift range, similarity weighting was 2.64 ms lower than equal mean, but its cluster-bootstrap interval included no difference (**−7.67 to 0.55 ms**). The output-shift range is descriptive sensitivity, not a measure where lower alone establishes a better explanation.

Mean selected-mask Jaccard was 0.765 for equal versus weighted, 0.763 for equal versus median, and 0.696 for weighted versus median. Across donors, mean pairwise top-set Jaccard was **0.397**, and mean donor local LIME R² was **0.874** (mean within-example donor R² range **0.056**).

**Donor-window overlap clarification:** the implementation ensures each donor window does not overlap its explained target window; it does not enforce pairwise separation among donor windows. In the 552-example cohort, **400/552** examples had at least one overlapping donor pair, with **655/5,520** donor pairs overlapping. Results above describe the implemented five-donor sampling rule and should not be read as five mutually independent temporal windows per example.

## Interpretation

Across the pilot and expanded cohort, the component median had the highest average agreement with donor-specific top sets. On the larger cohort its gain over equal averaging was small in magnitude but consistent under record-cluster resampling. It also overlapped the reference T peak less often than equal mean. Similarity weighting reduced the average consensus shift range descriptively, but the cluster-bootstrap interval did not establish a difference from equal mean, and it had lower donor agreement and T-peak overlap. The output shifts are perturbation sensitivity measures, not a score where larger or smaller alone establishes a better explanation. All methods retained substantial donor disagreement, and selected late-window regions remained common.

These results support the component median as a useful robustness option when agreement with donor-specific rankings is the objective, while equal mean remains a simple reference. The later LUDB transfer and aggregation study found a similarly small median agreement gain on a fixed-five subset; see the [external validation plan and results](external-validation-plan.md). Neither cohort establishes a generally best aggregation rule for explanation quality: T-peak overlap and perturbation shifts differ by method, the QTDB validation set covers only 16 records, and the external checkpoint has a lead-domain mismatch. This comparison does not tune the prediction model or establish physiological or causal explanations.

## Reproduction

```sh
uv run python scripts/compare_qtdb_consensus_aggregations.py \
  --examples 10000 --max-per-record 100 --donors-per-example 5 \
  --perturbations 256 --random-trials 20
```

The expanded run writes `summary.json`, `per-example.csv`, and `per-donor.csv` to the ignored local directory `outputs/qtdb-consensus-aggregations-full-validation/`. The 48-beat pilot is preserved in `outputs/qtdb-consensus-aggregations/`.
