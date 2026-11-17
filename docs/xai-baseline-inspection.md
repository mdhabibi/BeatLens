# T-peak explanation baseline inspection

## Prototype choice

Use the **zero reference in per-window z-scored input** as the primary explanation reference for the current prototype. The signal is centered per lead and per beat, so zero represents the average value of that normalized window. This answers: which deviations from a flat mean-valued input move the predicted T-peak offset?

Keep the **Gaussian-smoothed waveform** as a required sensitivity view. It preserves slow trends while suppressing sharper morphology. It produced a higher average LIME local R² (0.887 vs. 0.856) and lower IG completeness residual (0.018 vs. 0.984 ms), but selected substantially different regions. Neither baseline is a physiologically observed counterfactual, so this choice is a prototype convention, not a claim that zero is the uniquely correct baseline.

## Where the baseline choice changed the map

For each of 30 seed-42 validation examples, the top 10% of 25 ms bins was compared by Jaccard overlap:

| Comparison | Mean overlap |
| --- | ---: |
| IG, zero vs. smoothed baseline | 0.447 |
| LIME, zero vs. smoothed baseline | 0.371 |

Zero-baseline top bins included a bin within 60 ms of the reference T peak in 22/30 IG examples and 27/30 LIME examples. With the smoothed baseline, that was 6/30 and 14/30. Across methods and baselines, top bins also included the QRS-anchor region (±80 ms) in 28–30 of 30 examples and the late window (>800 ms after the anchor) in 28–29 of 30. The model appears to use context outside the T-peak neighborhood; those regions need waveform-level review before interpretation.

Examples with the largest IG rank changes:

| QTDB beat | Reference / prediction | IG overlap | LIME overlap | Inspection note |
| --- | ---: | ---: | ---: | --- |
| `sel39`, beat 19 | 272 / 300 ms | 0.05 | 0.05 | Zero IG emphasizes bins around −568 ms; smooth IG shifts toward −712 ms and near-anchor bins. |
| `sel39`, beat 20 | 308 / 320 ms | 0.16 | 0.10 | Both methods change their leading regions despite similar predictions. |
| `sele0104`, beat 19 | 212 / 238 ms | 0.22 | 0.47 | Both IG baselines emphasize the anchor and late-window region; LIME overlap is higher. |
| `sele0104`, beat 27 | 196 / 224 ms | 0.22 | 0.38 | Same broad anchor/late-window pattern; ranks shift between references. |

Offsets and predictions are milliseconds relative to the selected-beat anchor. These cases show why the attribution should be presented with its baseline and signal window.

## Deletion sensitivity

Replacing the top-ranked 20% of bins shifted the output more than matched random deletions on average for both references. The effect was inconsistent per example: top deletion shifted less than random deletion in 8/30 IG-zero, 11/30 IG-smoothed, 7/30 LIME-zero, and 9/30 LIME-smoothed cases. This supports only a preliminary perturbation-sensitivity result, not a strong faithfulness claim.

All figures above use the validation cohort; no test records were read for this inspection. Full per-example numbers and temporal attribution scores are in the ignored local `outputs/qtdb-explanations/` artifacts.
