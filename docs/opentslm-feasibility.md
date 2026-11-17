# OpenTSLM QTDB pilot feasibility

**Status: data formatting passed; model training is blocked in the current environment.**

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

## Runtime and model gate

The current BeatLens environment reports Python 3.15 and PyTorch 2.15.0, with neither CUDA nor MPS
available. OpenTSLM and `transformers` are not installed, there is no cached OpenTSLM Hugging Face
checkpoint, and no Hugging Face token is configured. The project is therefore not in a state to run
or time a meaningful fine-tuning experiment locally.

OpenTSLM's current README describes time-series question-answering, captioning, and reasoning
workflows, with text answers and time-series prompts; it does not describe a native sparse-event
regression head. Its published setup uses Hugging Face checkpoints, and it warns that MPS can be
incompatible with pretrained checkpoints. The repository currently requires Python 3.12 or newer.
See the [OpenTSLM README](https://github.com/OpenTSLM/OpenTSLM) and
[`pyproject.toml`](https://github.com/OpenTSLM/OpenTSLM/blob/main/pyproject.toml).

## Resume criteria

Before running the model pilot, use an environment with a supported Python version, install a pinned
OpenTSLM revision and dependencies, confirm access to a small supported checkpoint, and verify the
intended device. Then run a tiny train/validation smoke experiment on the single-T-peak task. Report
numeric-output parse rate, offset MAE in milliseconds, and comparison with the train-only timing
prior. Keep test evaluation disabled unless a new untouched holdout is established.
