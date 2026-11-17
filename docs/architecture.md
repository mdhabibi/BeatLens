# Architecture decisions

## Boundaries

BeatLens uses a small ports-and-adapters structure:

- **Domain** defines records and result types without depending on TimeNet or a model implementation.
- **Application** coordinates a use case through interfaces in `domain.ports`.
- **Infrastructure** will adapt TimeNet records and PyTorch/OpenTSLM models to those interfaces.
- **Explainers** will implement perturbation LIME and comparison methods behind the explainer interface.
- **Evaluation** will own predictive and explanation metrics; it will not be mixed into the demo.
- **Presentation** will call application use cases and render their structured results.

## TimeNet boundary

TimeNet is the data interchange and loading layer. Its connector contribution is maintained separately in the TimeNet repository. BeatLens consumes the resulting TimeF dataset through an adapter and owns model training, inference, explanation, and evaluation.

The TimeNet project is pre-release. Pin the version or commit after the selected connector and local build workflow are verified; do not bind the project to an assumed API version.

## Reproducibility

Each experiment will record the source dataset/version, license, split identifiers, preprocessing configuration, model checkpoint/version, random seeds, and evaluation outputs. Raw data and checkpoints remain local or in approved artifact storage, never in Git by default.

## Safety boundary

The hackathon build is a research demo over public data. It will not accept patient uploads or return diagnosis/treatment recommendations. Explanation output will be described as model evidence and accompanied by fidelity/stability measurements where available.
