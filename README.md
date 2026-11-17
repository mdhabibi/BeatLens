# BeatLens

**BeatLens is a research workbench for testing whether ECG model explanations are faithful, stable, and consistent with available signal annotations.**

The first release is scoped to the TimeNet hackathon. It will use public, documented ECG data; train or fine-tune a temporal model; compare it with a PyTorch baseline; and make the model's evidence regions inspectable.

> **Research use only.** BeatLens is not a medical device and does not provide diagnosis or treatment recommendations. Model explanations describe model behavior; they do not establish clinical or causal reasons.

## Architecture

```text
BeatLens/
├── configs/                  # Versioned experiment configuration
├── data/                     # Local only; source data is never committed
├── docs/                     # Architecture, data decisions, and evaluation protocol
├── notebooks/                # Exploration only; reusable logic belongs in src/
├── scripts/                  # Reproducible training and evaluation entry points
└── src/beatlens/
    ├── application/          # Use cases that orchestrate domain ports
    ├── domain/               # Framework-independent records and result types
    ├── evaluation/            # Predictive and explanation metrics
    ├── explainers/            # LIME variants and attribution adapters
    ├── infrastructure/        # PyTorch, TimeNet, and persistence adapters
    └── presentation/         # Streamlit research demo
```

The old LIME project remains a historical reference. BeatLens starts as a separate package, with data access, models, explanation methods, evaluation, and presentation kept behind clear interfaces.

## TimeNet integration

TimeNet standardizes and loads time-series data. BeatLens owns model training, inference, explanation, and evaluation. Dataset connectors belong in the TimeNet connector contribution; this repository will consume TimeNet datasets through an adapter rather than copy its engine or connector code. A QTDB connector prototype, shaped for TimeNet's connector package, is now in [`integrations/timenet-connectors`](integrations/timenet-connectors/README.md).

The connector targets TimeNet 0.12.0 at commit `9bd1613019ee61fbc52e02106a1a45990ead7a54`. QTDB is the distinct waveform-delineation dataset; the prototype preserves the full two-lead excerpts and emits sparse tasks only for observed fiducials. Its frozen split is the provisional known-subject-grouped BeatLens split, not a claim of complete patient-independent partitioning.

## Planned experiment

1. Finish review and upstreaming of the QTDB connector prototype, preserving the provisional split caveat.
2. Build the TimeF QTDB release and connect it through BeatLens's data adapter.
3. Keep the PyTorch CNN as the runnable reference; the OpenTSLM QTDB data-format smoke check passed,
   while model training is blocked on runtime/checkpoint readiness ([feasibility note](docs/opentslm-feasibility.md)).
4. Implement ECG-aware perturbation LIME and compare it with one gradient-based attribution method.
5. Report predictive performance, local explanation fidelity, stability, and annotation overlap where the annotations support that measurement.
6. Add a small demo that displays a held-out public example and its model evidence.

## Development

Python 3.11 or newer is required. The project uses `uv` for a locked environment. TimeNet and OpenTSLM dependencies will be pinned after verifying the selected integration path and model access requirements.

## Run the QTDB T-peak demo

The local research demo needs the downloaded QTDB archive, the generated split, and the trained checkpoint. The split keeps documented same-subject European ST-T excerpts together; cross-database patient independence is still unverified. From the repository root:

```sh
uv sync --extra qtdb --extra demo
uv run streamlit run src/beatlens/presentation/app.py
```

The app compares Integrated Gradients and temporal LIME against the primary zero reference and a smoothed-waveform sensitivity reference. It also lets you replace top-ranked time bins and a matched random set, then inspect both altered waveforms and the model's prediction shifts. It uses validation records only. The test records were inspected during exploratory model comparison, so do not treat the earlier test metrics as an untouched final holdout.

## Data and privacy

- Do not commit downloaded ECG files, checkpoints, patient data, credentials, or local experiment outputs.
- Preserve source attribution, license, access conditions, dataset version, and split methodology with each experiment.
- The hackathon demo uses public validation examples only; it does not accept patient uploads.

## Status

QTDB annotation audit, a known-subject-grouped provisional split, a T-peak baseline, and an initial IG/LIME explanation comparison are in place. The local demo displays both explanation references and exposes baseline sensitivity for inspection. QTDB's TimeNet connector contribution and a genuinely untouched final evaluation cohort remain open work. See [the dataset decision note](docs/dataset-decision.md), [baseline results](docs/baseline-results.md), [XAI prototype](docs/xai-prototype.md), and [Round 1 plan](docs/round1-plan.md).
