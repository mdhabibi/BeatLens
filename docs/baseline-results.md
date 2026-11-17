# QTDB localization baseline: first run

## Setup

- Split: 68 train, 16 validation, 21 test records; source-database stratified, seed 42.
- Examples: 2,327 train, 546 validation, 750 test annotated beats.
- Input: 2.5-second, two-lead windows at 250 Hz; each window/lead was z-scored independently.
- Model: four-block dilated 1D residual CNN with five temporal heatmap outputs.
- Training: masked BCE heatmap loss, positive weight 50, AdamW, learning rate `1e-3`, batch size 64, maximum 30 epochs, patience 7. Best validation checkpoint was epoch 18; early stopping finished at epoch 25. Device: CPU.
- Metric: event localization MAE after choosing the reference number of events per beat and target type (oracle event count). This isolates localization on annotated events; it does not measure event-presence detection.

## Held-out test result

| Method | Input signal used? | Events | MAE | Median | Within 40 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Four-block CNN | Yes | 5,558 | 426.0 ms | 432.0 ms | 31.6% |
| Train-only ordinal timing prior | No | 5,558 | 66.8 ms | 38.5 ms | 52.6% |

The timing prior predicts each event's relative position from the training-set mean for that event type and ordinal slot. It uses no ECG waveform, but like the CNN evaluation it is given the reference event count for each annotated beat. It is a diagnostic baseline for this localization-only protocol, not an inference-ready detector.

| Target | CNN MAE | Timing-prior MAE | Test events |
| --- | ---: | ---: | ---: |
| Generic onset `(` | 407.0 ms | 42.2 ms | 1,832 |
| Generic end `)` | 403.4 ms | 97.9 ms | 2,238 |
| P peak `p` | 478.6 ms | 38.3 ms | 666 |
| T peak `t` | 480.6 ms | 36.8 ms | 750 |
| U peak `u` | 557.9 ms | 303.5 ms | 72 |

The CNN has not shown useful signal-based localization in this run. Do not use its explanations as evidence of meaningful ECG reasoning. The direct offset-regression follow-up is summarized below. The initial CNN/timing-prior test comparison is exploratory; do not use the test split for future model selection.

## Direct offset regression follow-up

A second model predicts ordered event offsets directly with masked Smooth L1 loss. Its best checkpoint was epoch 11 (19 epochs run). Validation MAE was **82.3 ms**, slightly worse than the train-only timing prior at **80.8 ms**. It improved T-peak and U-peak validation MAE slightly, but remained worse on onset, end, and P-peak localization. Because it did not beat the prior overall on validation, the script did not load or evaluate the test split. The validation-only metrics are in `outputs/qtdb-offset-regression/metrics.json`.

The direct regressor also does not justify XAI work yet. Next, test a prior-residual model or narrower target formulation using validation only. The test split was already inspected for the initial CNN/timing-prior comparison, so it is no longer a fully untouched final holdout. Do not use it for further model selection; report its earlier result as exploratory and plan a new locked holdout or external evaluation for the final model.

## Prior-residual regression follow-up

The residual model predicts corrections to train-only, target-and-ordinal timing means. It selected epoch 12 (20 epochs run) and reached **81.0 ms** validation MAE, compared with **80.8 ms** for the prior. It therefore did not pass the validation selection threshold, and its script did not load the test split. The validation-only run is recorded in `outputs/qtdb-prior-residual/metrics.json`.

The direct regressor did beat the timing prior for T-peak localization alone on validation (34.3 ms versus 40.8 ms), while the aggregate across five target types remained worse. This motivated the focused single-target run below.

## Focused T-peak localization

The single-target CNN was trained only on selected beats with exactly one annotated T peak. Training excluded 95 beats without a T marker and 14 with multiple T markers; validation and test had one T marker for every selected beat. The model used the same 2.5-second, two-lead input windows with per-window/per-lead z-scoring, a scalar offset-regression head, masked Smooth L1 training, and AdamW (`3e-4`, batch 128). The best validation checkpoint was epoch 38 of 40.

| Split | Method | Events | MAE | Median | P90 | Within 40 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Validation | CNN | 546 | 29.4 ms | 24.3 ms | 62.9 ms | 73.6% |
| Validation | Train-only timing prior | 546 | 41.2 ms | 36.9 ms | — | 54.6% |
| Test | CNN | 750 | 29.1 ms | 16.9 ms | 64.0 ms | 79.1% |
| Test | Train-only timing prior | 750 | 36.4 ms | 27.1 ms | 60.9 ms | 74.8% |

The model beat the timing prior on validation, so the script evaluated the test split after selecting the epoch. Its test MAE was 7.4 ms lower than the train-only timing prior; its P90 error was 3.2 ms higher. These are exploratory results: the original test records had already been inspected during the earlier multi-target comparison. Do not present this as a final untouched holdout estimate. The model and target are now suitable for a first explanation prototype, with the evaluation limitation stated.

This is the initial record-level run. A later known-subject-grouped run is summarized below and supersedes it for validation/demo use. The reproducible run is `scripts/train_qtdb_tpeak.py`; original outputs are `outputs/qtdb-tpeak/metrics.json` and `outputs/qtdb-tpeak/best.pt`.

## Known-subject-grouped T-peak run

After the split audit, the model was retrained with the three documented European ST-T repeat-subject pairs assigned to a single partition. The grouped split has 67 train and 16 validation excerpts. Best validation MAE was **26.5 ms** (median 22.3 ms, P90 54.1 ms, 76.4% within 40 ms), versus **42.5 ms** for the train-only timing prior. Best checkpoint was epoch 24 (32 epochs run; CPU). The test partition was not evaluated in this run because previous experiments had already inspected it.

This run is still exploratory: subject identity cannot be verified across all six source databases, and the validation set informed model selection. The checkpoint and metrics are `outputs/qtdb-tpeak-known-subject-groups/best.pt` and `outputs/qtdb-tpeak-known-subject-groups/metrics.json`. `scripts/train_qtdb_tpeak.py` evaluates test only when `--evaluate-test` is explicitly supplied.

The generated checkpoint and full per-target metrics are in ignored local outputs: `outputs/qtdb-baseline/best.pt`, `outputs/qtdb-baseline/metrics.json`, `outputs/qtdb-baseline/timing-prior.json`, and `outputs/qtdb-offset-regression/metrics.json`.
