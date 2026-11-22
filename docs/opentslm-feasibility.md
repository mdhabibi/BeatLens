# OpenTSLM QTDB pilot feasibility

**Status: GPU inference and supervised LoRA training passed. The first tuned model returned valid indices on 8 of 8 validation windows, with higher timing error than the train-only baseline.**

## Nebius validation pilot

The published ECG-QA demo completed on an NVIDIA H100 using OpenTSLM revision
`2968f4b` and `OpenTSLM/llama-3.2-1b-ecg-sp` with LoRA enabled. The
[QTDB pilot](../scripts/run_qtdb_opentslm_pilot.py) then ran the same checkpoint on
one eligible window from each of eight validation records. The prompt supplied
both leads, the sampling rate, and the beat anchor. It requested a zero-based
T-peak index with up to 256 generated tokens. Reference indices were used only
for scoring.

The model returned 0 valid indices out of 8, so its timing MAE is undefined.
The train-only mean-offset baseline had an MAE of **51.5 ms** on these eight
windows. This pilot measured zero-shot behavior on a small validation panel.
The raw log is stored locally at `outputs/qtdb-opentslm/beatlens-qtdb-pilot.log`.

## Question

Can the QTDB signal windows and sparse fiducial targets be presented to OpenTSLM for a small
temporal-model experiment without changing the primary PyTorch baseline?

## Proposed narrow task

Use only selected beats with exactly one observed T-peak target. Provide both ECG leads as separate
time-series prompts and ask the model to return the signed T-peak offset from the `.man` beat anchor
in seconds. Keep the label as an observed-only target; do not turn missing marks into negative
examples. Train on the existing training split and choose settings on validation only. Do not use
the already-inspected test partition for model selection.

This converts a 250 Hz localization target into short numeric text. It is a feasibility experiment,
not evidence that generated decimal values are precise or that OpenTSLM supports native event
localization. The CNN remains the primary localization reference.

## Data-to-prompt smoke check

The QTDB adapter produced a training example from `sel100`, beat 0:

- two leads, 625 samples per lead (2.5 seconds at 250 Hz);
- per-segment, per-lead z-scoring with edge padding masked;
- one observed T-peak target at `+0.256 s` from the beat anchor;
- finite numeric input values and a textual target format.

This checks that the existing adapter can supply the inputs and label shape for a custom OpenTSLM
dataset. OpenTSLM itself was not invoked.

## Initial local runtime check

The initial local BeatLens environment reported Python 3.15 and PyTorch 2.15.0, with neither CUDA nor MPS
available. OpenTSLM and `transformers` are not installed, there is no cached OpenTSLM Hugging Face
checkpoint, and no Hugging Face token is configured. The project is therefore not in a state to run
or time a meaningful fine-tuning experiment locally.

OpenTSLM's current README describes time-series question-answering, captioning, and reasoning
workflows, with text answers and time-series prompts; it does not describe a native sparse-event
regression head. Its published setup uses Hugging Face checkpoints, and it warns that MPS can be
incompatible with pretrained checkpoints. The repository currently requires Python 3.12 or newer.
See the [OpenTSLM README](https://github.com/OpenTSLM/OpenTSLM) and
[`pyproject.toml`](https://github.com/OpenTSLM/OpenTSLM/blob/main/pyproject.toml).

## Supervised pilot

On the Nebius H100, the first optimizer step changed a LoRA weight. The run
completed one pass over 32 windows from distinct training records. On the same
eight validation windows, the tuned model returned 8 valid indices and reached
**70.5 ms MAE**, versus **51.5 ms** for the train-only timing prior. It predicted
index 325 on seven windows and index 313 on the remaining window.

## Matched-prompt waveform control

The control compared the pretrained checkpoint with a repeat of the 32-step
LoRA run, using the same integer-only prompt, greedy decoding, and a 16-token
limit for both checkpoints. Each checkpoint was evaluated on the same eight
validation prompts with the original waveforms and with the waveforms rotated
across examples. Reference indices stayed fixed.

| Checkpoint | Original waveforms | Swapped waveforms |
| --- | --- | --- |
| Pretrained | 0/8 valid indices | 0/8 valid indices |
| Fine-tuned | 8/8 valid, 64.5 ms MAE | 8/8 valid, 64.5 ms MAE |

The tuned checkpoint returned index 325 for all eight windows before and after
the swap. The train-only timing prior achieved 51.5 ms MAE. Fine-tuning improved
numeric output coverage under matched decoding, but the swap produced no change
in any predicted index on this small panel. The PyTorch locator remains the
primary model. The control used validation windows only.

## Four-window overfit check

The same LoRA setup trained for eight passes over four training windows with
reference indices 292, 307, 317, and 378. The prompts were identical. After 32
optimizer steps, greedy decoding returned index 292 for all four windows.
Swapping the waveforms across prompts changed zero predictions. Exact matches
were 1/4 in both conditions. Timing MAE was 126.0 ms, versus 96.0 ms for the
train-only timing prior on these selected training windows.

The generated indices failed to distinguish the four training examples.

## First-token waveform sensitivity

In the H100 diagnostic, repeating the same input gave identical first-token
logits in every case. Swapping waveforms with the prompt fixed changed those
logits for all four comparisons. The maximum absolute change per comparison
was 0.125–0.141 before fine-tuning and 0.141–0.188 after repeating the 32-step
run. The decoded one-token output stayed the same within each pair.

The waveform therefore affects the language model's scores. Whether it raises
the probability of the correct T-peak index remains open. A targeted follow-up
would score the full candidate index strings under each waveform and check
whether each waveform favors its own reference over the swapped reference.

## Full-answer candidate ranking

Teacher-forced loss ranked the full answers for indices 292, 307, 317, and
378. The pretrained checkpoint ranked 317 first on every waveform, and the
tuned checkpoint ranked 292 first on every waveform. Both scored 1/4 top-1
with original waveforms and 1/4 with swapped waveforms. Fine-tuning reduced
candidate losses substantially, while the waveform swaps failed to make the
correct reference consistently more competitive. This diagnostic reused the
four training windows and provides no held-out accuracy estimate. The current
OpenTSLM LoRA configuration has no demonstrated localization benefit, and
the PyTorch locator remains the primary BeatLens model.

## Paired T-peak choices

Each training pair presented two ECG waveforms with the same two candidate
locations and opposite correct answers. The four-pair training probe evaluated
both answer orders. Optimizing answer loss plus pairwise ranking loss with
LoRA alone gave a best paired result of 1/8. Adding the SoftPrompt projector
verified a weight update in both parameter groups and reached 2/8 at epoch 9,
then 0/8 at epoch 12. Mean answer loss fell from 4.207 to 0.233, while mean
ranking loss ended at 0.693, close to the equal-score reference of ln(2).
The larger validation run was not performed. These training-set diagnostics
provide no evidence of generalization.

## Frozen OpenTSLM encoder probe

A ridge regression head trained on frozen SoftPrompt waveform-encoder features
from 376 QTDB training windows. On 92 windows from separate validation records,
it achieved 33.7 ms MAE against 38.3 ms for the train-only timing prior and
ranked 28/46 pairs in the correct timing order. Swapping the waveforms within
each pair raised MAE to 47.6 ms.

On the same validation windows, the existing PyTorch locator achieved 25.6 ms
MAE, 45.6 ms after the same swap, and 36/46 correct pair orders. The two models
used different training budgets. The probe shows that OpenTSLM's frozen
waveform representation contains useful timing information for this panel.
The trained component in this experiment was the regression head.

## Initial patch-localization run

The frozen patch head reached 81.4 ms MAE at its best epoch. Fine-tuning
the OpenTSLM encoder changed a verified encoder weight and improved the
best MAE to 41.6 ms, above the 38.3 ms timing prior. At that epoch, swapping
waveforms gave 39.1 ms MAE. The initial unconstrained patch-argmax setup
therefore showed no benefit over the frozen ridge probe. A revised run
uses a train-derived search interval and a timing-aware objective.

## Encoder replay and aligned adaptation

Before optimization, native and FP32 no-gradient replay reproduced the
frozen features. Gradient-enabled replay changed the features and increased
validation MAE from 33.65 ms to 147.23 ms. Disabling PyTorch's MHA fast path
for extraction, training, and evaluation eliminated the pre-update feature
drift on all 92 validation windows.

With that configuration, the frozen ridge head achieved 33.66 ms MAE.
Three encoder-only epochs at learning rate 1e-7 reached a best adapted MAE
of 33.72 ms, rising to 47.66 ms after swapping waveforms. The timing prior
was 38.34 ms. The aligned fine-tuning run preserved waveform dependence,
while the frozen checkpoint remained best on the development panel.

A matched three-epoch run at learning rate 1e-6 reached a best adapted
MAE of 33.81 ms and a waveform-swap MAE of 47.76 ms. Both encoder adaptation
runs beat the timing prior and preserved waveform dependence. Neither
improved on the aligned frozen ridge baseline on this development panel.
The training script saves only a new overall best, so the checkpoints
from these two runs contain the frozen ridge winner. Adapted metrics
come from the corresponding run logs.

## Locked LUDB transfer

The residual adapter reached 33.80 ms on the QTDB development panel, versus
33.66 ms for frozen ridge. The frozen OpenTSLM encoder and QTDB-trained ridge
readout were then locked and evaluated on 1,603 LUDB windows from 200 records
using the fixed II/V5 mapping. LUDB was used for evaluation only in this
experiment. The model reached 33.49 ms MAE and 68.3% within 40 ms, versus
30.92 ms MAE for the QTDB timing prior. The record-bootstrap 95% interval for
the paired model-minus-prior MAE was −0.53 to +5.64 ms. This transfer run did
not establish an advantage over the timing prior. Lead and acquisition domain
shift limit the comparison. See [the LUDB protocol](external-validation-plan.md).

## Grouped preprocessing and coverage comparison

Five folds within the QTDB training partition kept all 63 split groups
intact. All four configurations predicted the same 2,201 out-of-fold
windows. Ridge penalty was fixed at 0.1. The existing QTDB development
panel and LUDB were excluded.

| Scaling | Fit windows | MAE (ms) | Swap MAE (ms) | Timing prior MAE (ms) |
| --- | --- | ---: | ---: | ---: |
| Per-lead z-score | Original subset | 40.82 | 54.67 | 44.87 |
| Per-lead z-score | All eligible | 47.52 | 58.07 | 44.98 |
| Per-lead min-max to [-1, 1] | Original subset | 49.38 | 57.04 | 44.87 |
| Per-lead min-max to [-1, 1] | All eligible | 49.76 | 60.93 | 44.98 |

The subset with z-score scaling was the only configuration to beat its
timing prior. The coverage comparison changes both the training sample
distribution and the effective regularization under an unnormalized
sum-of-squares objective. It does not establish that additional data are
inherently harmful. Compare regularization and record weighting within
training folds before revisiting encoder adaptation.

### Nested ridge penalty and record weighting

With z-score scaling, penalties were chosen in inner grouped folds and scaled
by the number of fit windows. Outer predictions used the same 2,201 windows.

| Fit windows | Weighting | MAE (ms) | Swap MAE (ms) | Timing prior MAE (ms) |
| --- | --- | ---: | ---: | ---: |
| Original subset | Equal per window | 39.54 | 51.03 | 44.87 |
| Original subset | Equal per record | 40.96 | 51.31 | 44.87 |
| All eligible | Equal per window | 41.60 | 51.00 | 44.98 |
| All eligible | Equal per record | 41.45 | 50.69 | 44.98 |

The best subset result improved from 40.82 to 39.54 ms. The best all-window
result improved from 47.52 to 41.45 ms. Four of five outer folds selected
the upper penalty-grid boundary (3.0) for each all-window weighting scheme,
so this grid does not establish the optimum for larger fits. Neither the
QTDB development panel nor LUDB was used in this comparison.

Extending the grid to 6, 10, and 30 did not improve outer-fold accuracy.
The subset with equal window weights stayed at 39.54 ms MAE and 51.03 ms
after swaps. Equal record weighting on the subset reached 41.35 ms.
All-window fits reached 42.04 ms with equal window weights and 42.19 ms
with equal record weights, both worse than their narrower-grid counterparts.
Retain the original-grid, equally weighted subset recipe for a final fit.
This grouped development result is separate from the earlier locked LUDB
evaluation, which used a different ridge penalty.

### Raw waveform and OpenTSLM readouts

Using equally weighted subset fits, the same nested folds, and the original
penalty grid, all readouts predicted the same 2,201 QTDB windows. The local
patch interval was recomputed from fit annotations for every inner and outer
fit, with 40 ms of margin on either side.

| Readout | MAE (ms) | Swap MAE (ms) |
| --- | ---: | ---: |
| Raw waveform ridge | 41.52 | 47.83 |
| Local raw waveform ridge | 41.22 | 52.59 |
| Full frozen OpenTSLM features | 39.54 | 51.03 |
| Local frozen OpenTSLM patches | 36.86 | 54.11 |
| Timing prior | 44.87 | 44.87 |

The local OpenTSLM readout improved on the full OpenTSLM readout in four
of five folds and on the raw waveform readout cropped to the same interval
in three of five folds. The matched-crop improvement was 4.36 ms pooled
MAE. These are grouped development results using a frozen encoder.

The final local readout was fitted on 376 windows from 65 QTDB training
records. Training-fold selection chose a ridge penalty of 0.3 at 300
windows (effective penalty 0.376). The saved checkpoint selects 6,912
features and predicts milliseconds after the anchor. The 36.86 ms score
above comes from cross-validation, before fitting this checkpoint.

Reloading the saved local readout without refitting gave 32.01 ms MAE
on 92 development windows from 16 separate QTDB records, compared with
33.66 ms for the full-feature readout and 38.34 ms for the timing prior.
Swapping waveforms raised the local readout's MAE to 45.43 ms. The existing
PyTorch locator reached 25.57 ms on the same panel with a different training
budget. The QTDB panel had informed earlier development.

### Locked local readout on LUDB

The same saved QTDB checkpoint was evaluated without refitting on 1,603 LUDB
windows from 200 records using the fixed II/V5 transfer protocol. Its MAE
was **26.26 ms**, with **80.2%** within 40 ms. The timing prior reached
**30.92 ms**, and the earlier full-feature OpenTSLM readout reached
**33.49 ms** on the same windows. The record-bootstrap 95% interval for the
paired local-readout-minus-prior MAE was **−7.45 to −1.93 ms**.

The local readout was developed on QTDB, after the earlier full-feature
LUDB result was known. This is cross-database evaluation without LUDB
refitting. A new cohort would provide independent confirmation.

### OpenTSLM and raw-waveform fusion

A fixed 75/25 mix of local OpenTSLM and raw ECG feature blocks, normalized
on each fit set before concatenation, reached **36.09 ms** MAE across the
same 2,201 out-of-fold QTDB windows. Waveform swapping increased MAE to
51.66 ms. The local OpenTSLM-only readout scored 36.86 ms without swaps.

The saved fusion readout was fitted on 376 QTDB windows from 65 records,
using 7,128 input features and an effective ridge penalty of 3.76. Without
LUDB refitting, it reached **21.75 ms** MAE and **86.7%** within 40 ms across
1,603 windows from 200 records. The same-window prior and local OpenTSLM
readout reached 30.92 and 26.26 ms, respectively. The record-bootstrap
95% interval for the paired fusion-minus-prior MAE was **−11.48 to −6.89 ms**.
The existing PyTorch locator reached 21.65 ms under the same transfer
protocol with a different training budget. The fusion result reuses the
LUDB cohort, whose earlier results were known, and provides no ranking on
an external leaderboard.

Selecting the feature mix inside the inner folds increased outer-fold
MAE to 37.57 ms, versus 36.09 ms for the fixed mix. The fixed 75/25 mix
remains the selected readout. On QTDB record `sele0111`, it predicted
275.86 ms versus a 264.0 ms reference. Temporal LIME achieved weighted
R² of 0.890 over 256 waveform perturbations. Integrated Gradients across
the two readout inputs summed to the smoothed-reference prediction
difference within 0.000006 ms.

### Robust fusion objective

A fixed 40 ms Huber threshold improved QTDB out-of-fold MAE from
36.09 to 35.03 ms, with improvements in all five folds. Swapping waveforms
raised the robust result to 51.32 ms. The final robust readout used 376
QTDB windows across 65 records, with an effective ridge penalty of 1.2533.

On the same 1,603 LUDB windows, robust fusion reached 21.79 ms MAE versus
21.75 ms for ordinary fusion. Both placed 86.7% of predictions within 40 ms.
The direct paired difference (robust minus ordinary) was **+0.046 ms**,
with a record-bootstrap 95% interval of **−0.64 to +0.74 ms**. Robust
fusion had lower mean error on 92 records and higher mean error on 108.
The paired comparison showed no clear LUDB advantage for robust fusion.

### Nonlinear fusion readout

The fixed linear-plus-RBF kernel reduced QTDB out-of-fold MAE to 34.46 ms
from 36.09 ms for linear fusion. MAE rose to 52.17 ms with swapped waveforms.
The QTDB-fitted readout reached 19.90 ms MAE on the same 1,603 LUDB windows
from 200 records, with 88.0% within 40 ms. The paired MAE difference versus
linear fusion was −1.85 ms (record-bootstrap 95% interval −2.69 to −0.99 ms).
Nonlinear fusion had lower mean error on 127 records and higher mean error
on 73. LUDB has been used for repeated comparisons during development.

For the joint nonlinear readout, on QTDB record
`sele0111` the prediction was 264.047 ms and the annotation was 264.000 ms.
Waveform LIME had weighted R² of 0.867 over 256 perturbations. Joint
readout-input IG had a completeness residual of 0.000223 ms on this example.

### Separate nonlinear kernels for OpenTSLM and raw ECG

Combining separate linear-plus-RBF kernels at the fixed 75/25 weighting
reduced QTDB out-of-fold MAE from 34.46 to 33.97 ms. On the same 1,603
LUDB windows from 200 records, MAE fell from 19.90 to 19.52 ms. The paired
difference (separate minus joint kernel) was −0.372 ms with a
record-bootstrap 95% interval of −0.530 to −0.217 ms. Separate kernels
had lower mean error on 128 records and higher mean error on 72. Both
readouts had 88.0% of predictions within 40 ms. This is an exploratory
comparison on LUDB, which was used repeatedly during development.

For the separate-kernel readout, on QTDB record `sele0111`,
it predicted 262.348 ms versus the 264.000 ms annotation. Waveform LIME
achieved weighted R² of 0.864 over 256 perturbations. Readout-input IG
had a completeness residual of 0.0000515 ms on this example.

### Interaction kernel

Adding a fixed 25% product of the encoder and raw waveform kernels
reduced QTDB out-of-fold MAE from 33.97 to 31.38 ms across 2,201 windows,
with lower error in all five folds. Waveform swaps raised MAE to 54.69 ms.
On 1,603 LUDB windows from 200 records, the interaction readout reached
18.60 ms MAE with 88.6% within 40 ms. The paired difference versus separate
kernels was −0.92 ms (record-bootstrap 95% interval −1.26 to −0.58 ms),
with lower mean error on 144 records and higher mean error on 56. LUDB was
used repeatedly during development.

The demo features the interaction readout. On QTDB record `sele0111`, it
predicted 256.964 ms versus the 264.000 ms annotation. Waveform LIME
achieved weighted R² of 0.863 over 256 perturbations. Readout-input IG
had a completeness residual of 0.0000234 ms on this example.

### Full product kernel

The inner searches selected a 100% product weight in all five QTDB folds.
The resulting out-of-fold MAE was 30.77 ms, compared with 31.38 ms for
the fixed 25% interaction readout. The full product improved four of
five folds, and waveform swapping raised its MAE to 54.34 ms.

On 1,603 LUDB windows across 200 records, the full product reached
17.94 ms MAE and 88.8% within 40 ms. The fixed 25% interaction readout
reached 18.60 ms and 88.6%, respectively. The paired difference
(full product minus fixed interaction) was −0.66 ms, with a
record-bootstrap 95% interval of −1.33 to +0.08 ms. The full product
had lower mean error on 111 records and higher mean error on 89.
The fixed interaction readout remains featured in the demo. LUDB has
been used repeatedly during development, and the full product is a
candidate for comparison on a fresh cohort.

### ECG-ID single-lead stress test

The two QTDB-trained checkpoints were evaluated without refitting on 2,835
windows from 310 recordings and 90 participants. Raw lead I was duplicated
to match the models' two-channel input. ECG-ID's T-peak references are
unaudited automated annotations
([PhysioNet ECG-ID](https://physionet.org/content/ecgiddb/1.0.0/)).

| Model | MAE (ms) | Within 40 ms |
| --- | ---: | ---: |
| Fixed 25% interaction | 17.38 | 94.0% |
| Full product | 16.29 | 93.4% |

The paired MAE difference (full product minus fixed interaction) was
−1.09 ms, with a participant-bootstrap 95% interval of −1.86 to −0.27 ms.
The full product had lower mean error on 185 recordings and higher mean
error on 125. The timing-only prior reached 32.98 ms MAE. The full
product now has the lowest observed MAE on both LUDB and ECG-ID.
The fixed interaction readout remains featured in the demo while the
trade-off between average error and threshold accuracy is evaluated
against expert annotations on a fresh cohort.

### Paired error tails

| Cohort | Readout | Median (ms) | P95 (ms) | P99 (ms) | Windows over 80 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| LUDB | Fixed interaction | 10.89 | 62.23 | 123.42 | 52 |
| LUDB | Full product | 10.36 | 58.30 | 121.46 | 48 |
| ECG-ID | Fixed interaction | 12.02 | 43.37 | 99.94 | 36 |
| ECG-ID | Full product | 10.29 | 47.73 | 93.89 | 40 |

At the 40 ms threshold, the full product gained 33 windows and lost 29
on LUDB (+0.25 percentage points, record-bootstrap 95% interval −1.36 to
+1.73). On ECG-ID it gained 37 and lost 54 (−0.60 percentage points,
participant-bootstrap 95% interval −1.73 to +0.68). The threshold
intervals include zero on both cohorts. The observed ECG-ID tail was
mixed, with a lower P99 but a higher P95 and more errors over 80 ms.
These quantiles and counts are descriptive.

### Fixed product-weight comparison on QTDB

Each weight used ridge penalties selected in the inner folds and was scored
on the same 2,201 held-out windows.

| Product weight | MAE (ms) | Within 40 ms | P95 (ms) | P99 (ms) | Windows over 80 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25% | 31.38 | 75.65% | 95.77 | 193.90 | 164 |
| 50% | 30.95 | 77.10% | 97.51 | 190.87 | 170 |
| 75% | 30.65 | 77.10% | 98.43 | 189.69 | 175 |
| 100% | 30.77 | 76.42% | 99.08 | 198.43 | 165 |

The 75% weight had the lowest mean and P99 errors. Its MAE advantage over
the full product was 0.12 ms, and it had ten more errors over 80 ms.
The fixed 25% interaction had the lowest P95 and fewest errors over 80 ms.
These are descriptive comparisons on reused QTDB folds.

On the reused LUDB cohort (1,603 windows, 200 records), the QTDB-fitted
75% model reached 18.15 ms MAE versus 17.94 ms for the full product.
The paired difference (75% minus full product) was +0.21 ms with a
record-bootstrap 95% interval of −0.33 to +0.71 ms. The 75% model had
higher P95 error (62.64 versus 58.30 ms) and two more misses over
80 ms (50 versus 48). It gained 32 windows and lost 30 at the 40 ms
threshold (+0.12 percentage points, 95% interval −1.42 to +1.83).
The full product remains the candidate with the lowest measured
average error on LUDB and ECG-ID.

### Source-record audit for RPT15

The five-subject [RPT15 demonstration archive](https://zenodo.org/records/21652811)
contains manually annotated T peaks. The downloaded ZIP matched Zenodo's
published checksum. Comparing its provenance with the local QTDB record
names and headers found the following source-record matches.

| RPT15 subject | Source recording | QTDB counterpart |
| --- | --- | --- |
| 0001 | MIT-BIH Arrhythmia 100 | sel100 |
| 0009 | MIT-BIH Normal Sinus Rhythm 16265 | sel16265 |
| 0020 | MIT-BIH ST sudden death 33 | sel33 |
| 0021 | MIT-BIH Supraventricular Arrhythmia 803 | sel803 |
| 0027 | European ST-T e0104 | sele0104 |

The demo subset was excluded from the independent benchmark because all
five subjects have source recordings represented in QTDB. This audit
concerns source-record identity, irrespective of segment boundaries or
which QTDB windows were selected for fitting.

### MIT-BIH expert annotation access

[Elgendi et al. (2015)](https://doi.org/10.3390/s150717693) report manual
channel-one P- and T-wave annotation across the 48 MIT-BIH Arrhythmia
records. Their cited annotation download returned HTTP 404. Comparing
MIT-BIH record IDs with the local QTDB headers flagged 15 matches:
100, 102, 103, 104, 114, 116, 117, 123, 213, 221, 223, 230, 231, 232,
and 233. The other 33 are candidates pending label access, confirmation
of subject independence, and review of the annotation convention for
biphasic waves. No checkpoint was scored on this cohort.

### MIT-BIH manual-annotation pilot

The recovered Elgendi archive supplied ECG signals and manual T-peak
annotations for all 48 recordings. Excluding 15 QTDB source-record matches
left 33 records. The pilot sampled 64 eligible beats per record, for 2,112
windows. The bootstrap used 32 participant groups, with records 201 and
202 grouped together. The MATLAB indices were treated as one-based.
Cross-dataset subject independence and the indexing convention were not
independently verified.

| Readout | MAE (ms) | Within 40 ms | P95 (ms) | Windows over 80 ms |
| --- | ---: | ---: | ---: | ---: |
| Fixed 25% interaction | 51.54 | 52.89% | 151.26 | 453 |
| Full product | 51.93 | 51.14% | 146.28 | 443 |
| Timing prior | 56.42 | — | — | — |

Both checkpoints were evaluated without refitting. The paired MAE
difference (full product minus fixed interaction) was +0.39 ms
(95% interval −0.55 to +1.36 ms). The difference in the share within
40 ms was −1.75 percentage points (95% interval −3.79 to +0.19).
Fixed interaction minus timing prior was −4.88 ms (95% interval
−11.42 to +1.47). All intervals use a paired participant-group
bootstrap. The fixed interaction remains the demo default.

For record 101, subtracting one from the MATLAB R-anchor indices produced
1,714 exact matches out of 1,866 against the original MIT-BIH WFDB
annotations, versus 144 without subtraction. This supports the one-based
conversion for that record. It does not independently verify every T label.

Of 510 timing-prior misses over 80 ms, 278 (54.5%) came from six
recordings (219, 124, 222, 118, 215 and 217). Their median reference
times ranged from 182 to 396 ms after the anchor, compared with the
fixed 264 ms QTDB timing prior. Record-level model errors are needed
to determine whether the readouts fail on the same recordings.

Both readouts shared the same six records with the most misses over
80 ms: 124, 219, 214, 109, 222 and 118. These accounted for 287 of 453
misses (63.4%) for the 25% readout and 283 of 443 (63.9%) for the full
product. Four records (124, 219, 222 and 118) also ranked in the timing
prior's top six. On 214 and 109, the 25% readout had 52 and 43 large
misses, respectively, and the full product had 52 and 41. On record
215, the prior had 36 large misses versus four and three for the
readouts. The aggregate record summaries identify where to review
waveforms manually but cannot establish the causes of the misses.

### Beat-level audit on MIT-BIH records 109 and 214

Across the 128 windows selected for review, every model miss over 80 ms
was early. Median signed errors (prediction minus manual label) were
−86.1 and −85.9 ms on 109 for the 25% and full-product readouts, and
−119.7 and −117.1 ms on 214. Median manual reference times were 336.9
and 342.7 ms, respectively. Median predictions from the 25% readout
were 253.7 and 226.1 ms. The two readouts reached 95.41 and 93.67 ms
MAE on these records, versus 75.44 ms for the timing prior.

In the three largest misses on record 214 by the 25% readout, the
first channel was negative at the prediction (226–234 ms), with a
larger-magnitude deflection at 248–256 ms in the 100–500 ms interval.
The manual labels were later (380–399 ms). This describes timing and
waveform proximity without identifying which feature drives the model.
The two records were selected for high error, so their results are a
failure analysis rather than an independent model comparison.

The 376 selected QTDB training targets spanned 164–504 ms (median
260 ms, 95th percentile 345 ms). With the fitted crop rule, the raw
samples spanned 120–548 ms. All 128 reference peaks on 109 and 214 fell
within both the training target range and this crop. Later reference
times were much more common in these deliberately selected records:
48 of 128 targets exceeded 345 ms, versus 19 of 376 in training.
Both models predicted over 80 ms early on all 48 of those windows.
Among the 14 windows at or below 300 ms, neither model had a miss over
80 ms. This establishes an association with later targets on these
records; the effect of crop width or training target balance has not
been isolated.

### Late-target weighting on QTDB

Tripling the training weight for reference peaks after 345 ms reduced
MAE in that held-out group from 107.19 to 105.17 ms and reduced misses
over 80 ms from 114 to 102 across 173 windows. On the other 2,028
windows, MAE rose from 24.92 to 26.39 ms and large misses rose from
50 to 67. Overall MAE rose from 31.38 to 32.58 ms, with higher MAE
in all five outer folds and five more large misses in total. Both
variants selected their ridge penalties in the inner folds. The
uniformly weighted readout remains the demo default.

### Huber interaction on QTDB

With a 40 ms Huber threshold, the 25% interaction kernel's held-out MAE
fell from 31.38 to 31.02 ms, with improvements in three of five folds.
For the 2,028 windows through 345 ms, MAE fell from 24.92 to 24.29 ms
and the number of misses over 80 ms fell from 50 to 49. For the 173
later windows, MAE rose from 107.19 to 109.92 ms and the number of
misses over 80 ms rose from 114 to 123. Overall there were 172 large
misses versus 164 with squared loss. The squared-loss checkpoint
remains the demo default. The Huber variant was evaluated only on QTDB.

### Waveform candidate ranking on QTDB

A waveform-only control enumerated positive and negative extrema in both
smoothed channels and ranked them using a shallow gradient-boosted
classifier. The classifier used peak timing, channel, polarity,
prominence and local waveform samples. Each outer fold derived its search
interval and fitted the classifier from the same selected QTDB training
windows used by the kernel comparisons.

Across 2,201 held-out windows the control reached 23.16 ms MAE and had
140 errors over 80 ms, versus 31.38 ms and 164 for the 25% interaction
kernel. For the 173 targets after 345 ms, MAE was 85.30 ms and 75 errors
exceeded 80 ms, versus 107.19 ms and 114 for the interaction kernel.
For the earlier targets, large misses rose from 50 to 65. This exploratory
control uses no OpenTSLM features and has no external evaluation yet.

The same waveform ranker, fitted on the 376 selected QTDB windows,
was scored without refitting on LUDB and the MIT-BIH pilot.

| Panel | Windows | MAE (ms) | Within 40 ms | Errors over 80 ms |
| --- | ---: | ---: | ---: | ---: |
| LUDB | 1,603 | 11.87 | 92.45% | 75 |
| MIT-BIH pilot | 2,112 | 49.76 | 60.32% | 580 |

For reference peaks after 345 ms, the ranker reached 56.82 ms MAE on
61 LUDB windows and 121.27 ms on 428 MIT-BIH windows. Compared with the
25% OpenTSLM interaction model, the waveform ranker reduced MAE on both
panels but produced more errors over 80 ms (75 versus 52 on LUDB;
580 versus 453 on MIT-BIH). These are descriptive comparisons on
previously used panels. The waveform ranker is a control for assessing
whether a future candidate scorer benefits from OpenTSLM features.

### OpenTSLM patch features in the candidate ranker

Adding the candidate lead's 128 OpenTSLM patch features yielded
24.30 ms MAE, 86.46% within 40 ms, and 166 errors over 80 ms across
2,201 held-out QTDB windows. For 173 peaks after 345 ms, it yielded
93.78 ms MAE and 81 errors over 80 ms. The earlier waveform-only run
reached 23.16 ms and 140 large errors overall, and 85.30 ms and 75 large
errors on later peaks. Both used the same panel and grouped folds, but
the waveform-only run used the local environment and the augmented run
used Nebius. This direct feature concatenation did not show an
improvement in the observed comparison.

The paired Nebius run controlled for environment as well as candidates
and grouped folds. Waveform-only MAE was 23.66 ms versus 24.30 ms with
OpenTSLM patches (difference +0.64 ms; group-bootstrap 95% interval
−0.36 to +1.76 ms). Large misses rose from 153 to 166. After 345 ms,
MAE was 90.20 versus 93.78 ms (difference +3.58 ms; interval −0.38
to +7.37 ms), with 81 large misses for each model. The same-environment
comparison supersedes the cross-environment comparison for this feature
ablation. It showed no clear MAE benefit from patch concatenation.

### Averaging waveform candidates

A fixed score-weighted average of the top three distinct candidate times
reduced errors over 80 ms versus the single-peak ranker on all three
panels. The average can fall between waveform peaks.

| Panel | Single peak MAE (ms) | Top-three MAE (ms) | Single peak misses over 80 ms | Top-three misses over 80 ms |
| --- | ---: | ---: | ---: | ---: |
| QTDB grouped folds | 23.16 | 22.35 | 140 | 112 |
| LUDB | 11.87 | 12.12 | 75 | 46 |
| MIT-BIH pilot | 49.76 | 51.39 | 580 | 491 |

On MIT-BIH, the share within 40 ms also dropped from 60.32% to 56.44%.
These are descriptive results on previously used panels. Averaging is
an experimental timing estimator; its output can fall between extrema.

A selective version averaged up to three distinct times whose scores
were at least 75% of the highest score. Overall MAE and counts over
80 ms, respectively, were 22.09 ms and 125 on QTDB, 11.55 ms and 62 on
LUDB, and 50.88 ms and 551 on the MIT-BIH pilot. Relative to choosing a
single peak, selective averaging reduced large misses on all three
panels and reduced MAE on QTDB and LUDB. Unrestricted top-three
averaging had fewer large misses on every panel, at higher MAE on
each. These comparisons reuse the same development panels.

A score-weighted median of the top three distinct candidates chooses
an observed extremum whenever candidates exist, using the same timing
prior fallback as the other rankers. Relative to choosing a single peak,
MAE changed from 23.16 to 22.76 ms on QTDB, 11.87 to 11.08 ms on LUDB,
and 49.76 to 50.24 ms on MIT-BIH. Misses over 80 ms changed from 140
to 141, 75 to 62, and 580 to 571, respectively. Late-peak MAE increased
on QTDB and LUDB and was nearly unchanged on MIT-BIH. The LUDB result
is descriptive because the panel was used in earlier experiments.

Paired cluster bootstrap intervals resampled QTDB groups, LUDB records,
and MIT-BIH participant groups. Compared with the single-peak ranker,
unrestricted averaging reduced the error rate over 80 ms by 1.27
percentage points on QTDB (95% interval −2.52 to −0.17), 1.81 on LUDB
(−3.24 to −0.62), and 4.21 on MIT-BIH (−6.63 to −2.07). On the 428
late MIT-BIH windows, averaging reduced MAE by 5.30 ms (interval
1.83 to 8.27 ms improvement), while its overall MIT-BIH MAE difference
was +1.63 ms (−1.88 to +5.46). The weighted median's LUDB MAE difference
was −0.79 ms (−1.59 to −0.11); its overall MAE intervals on QTDB and
MIT-BIH included zero. These exploratory intervals are unadjusted for
the repeated comparisons and do not account for refitting the models.

Snapping the top-three weighted mean to its nearest retained extremum
yielded 22.98 ms MAE and 137 errors over 80 ms on QTDB, 11.24 ms and
60 on LUDB, and 50.75 ms and 559 on MIT-BIH. Compared with the
single-peak ranker, only LUDB's overall large-miss rate interval
excluded zero (−1.81 to −0.19 percentage points). All three overall
MAE intervals included zero. Snapping retained part of the observed
reduction in large misses, compared with 112, 46 and 491 misses for
the unsnapped average.

A disagreement flag marked windows with no candidate or an over-80 ms
span among the top three distinct candidates. With unrestricted averaging,
flagged versus unflagged rates of errors over 80 ms were 4.82% versus
6.37% on QTDB, 2.94% versus 2.59% on LUDB, and 25.24% versus 17.33%
on MIT-BIH. The flag marked 1,824/2,201, 1,294/1,603, and 1,581/2,112
windows, respectively. It was too common and inconsistent across panels
to serve as a general review rule.

Restricting the disagreement flag to candidates scoring at least 75%
of the best score reduced the flagged fractions to 3.1% on QTDB, 2.2%
on LUDB, and 5.4% on MIT-BIH. With selective averaging, rates of errors
over 80 ms among flagged versus unflagged windows were 39.13% versus
4.60% on QTDB, 8.57% versus 3.76% on LUDB, and 16.81% versus 26.61%
on MIT-BIH. The flag captured 27/125, 3/62, and 19/551 large misses,
respectively. The LUDB result rests on three flagged misses, and
the MIT-BIH result reverses the direction observed on QTDB.

Using one minus the highest candidate score as a risk ranking for
errors over 80 ms, with missing candidates assigned maximum risk,
gave AUROC of 0.866 on QTDB, 0.867 on LUDB and 0.685 on MIT-BIH
for selective averaging. Average precision was 0.519, 0.184 and
0.436, compared with prevalences of 0.057, 0.039 and 0.261. This is
threshold-free exploratory evidence; the score was not calibrated
as a probability of a large miss.

The bottom-decile cutoff derived from QTDB out-of-fold top scores was
0.61987. Applying the same cutoff without recalibration to external
scores flagged 95/1,603 LUDB windows and caught 20/62 large misses;
on MIT-BIH it flagged 367/2,112 windows and caught 193/551. On QTDB,
the cutoff flagged 221/2,201 and caught 77/125. QTDB was used to set
the cutoff. These evaluations estimate review workload and miss
capture, with no change to the timing predictions.

Under a fixed per-panel 10% review budget (rounded up), ranking by
ascending top candidate score caught 77/125 QTDB large misses with
221 reviews, 32/62 LUDB misses with 161 reviews, and 113/551 MIT-BIH
misses with 212 reviews. Score ties were resolved by panel order.
The budgeted policy uses the panel's unlabeled score ordering and
requires a batch of windows to rank.

Distributing the same review budget proportionally by record widened
review from 25 to 65 records on QTDB, 45 to 160 on LUDB, and 27 to
33 on MIT-BIH. Miss capture fell from 77 to 11, 32 to 4, and 113 to
75, respectively. Within-record ranking used the same score. For
this experiment, pooled ranking caught more large misses at the same
workload, while proportional allocation covered more recordings.
