# TimeNet Hackathon: ECG XAI Project Plan

**Working title:** ECG Evidence Lab  
**Status:** QTDB is the leading distinct connector candidate. Annotation coverage is audited, and the provisional split keeps documented European ST-T repeat-subject records together. Cross-database patient identity remains unavailable; proceed with that limitation explicit.
**Round 1 deadline:** November 23, 2026, 08:59 CET (November 22, 23:59 PST).

## Goal

Build a reproducible research demo that asks:

> When a temporal model delineates ECG waveforms, do its explanations identify signal regions that influence the predicted boundaries, and are those explanations stable?

This is an explainability and model-behavior study, not a diagnostic product. The old repository is an exploratory LIME prototype. Its CNN, dataset assumptions, and notebook do not constrain the new implementation.

## Round 1 scope

1. Contribute one useful ECG dataset representation to TimeNet.
2. Train a model for a narrowly defined ECG waveform-delineation task using the manually annotated beats.
3. Compare a PyTorch 1D CNN/ResNet baseline with a small temporal-language-model experiment if its output format supports the task.
4. Implement ECG-aware, perturbation-based LIME and compare it with one gradient-based attribution method.
5. Publish a small interactive demo and a reproducible evaluation summary.

Out of scope for Round 1: ECG image digitization, patient uploads, diagnosis or treatment advice, avatar, and clinical deployment.

## Dataset decision

### Dataset contribution candidate: QT Database (QTDB)

QTDB is the current leading contribution candidate. PhysioNet provides 105 two-lead ECG excerpts with manually annotated P/QRS/T/U waveform boundaries on selected beats; the release is ODC-By 1.0. The task would shift from beat-class classification to waveform delineation, which provides direct time-localized targets for the explanation study. See [dataset decision](dataset-decision.md) for the audit and caveats.

Do not submit a duplicate MIT-BIH connector: [TimeNet PR #630](https://github.com/OpenTSLM/TimeNet/pull/630) already adds MIT-BIH beat classification and beat/rhythm localization tasks. QTDB is not in the current PhysioNet connector tree and no matching PR appeared in the GitHub search on 2026-11-12.

Never place downloaded ECG files in Git unless the upstream terms and repository policy explicitly allow it. Preserve source citation, license, and access conditions in the TimeNet dataset card.

TimeNet is currently pre-release and its public API may change before version 1.0. Pin the version/commit used for the connector and experiment, and keep the TimeNet connector contribution isolated from the model/app code until its contract is verified.

## Model and evaluation

- Use PyTorch for the experiment code.
- Keep the model task narrow: predict a selected set of P/QRS/T waveform boundaries from annotated windows.
- Split by source record and known subject, not randomly by beat or window, to reduce leakage between train and test.
- Include a simple PyTorch 1D CNN or ResNet baseline and a small OpenTSLM experiment. Treat the temporal model as an experiment, not as a promised performance improvement.
- Report event-level precision/recall/F1 and boundary error in milliseconds with a declared matching tolerance; include annotator agreement on records with a second expert.
- Keep a fixed held-out evaluation set and record model/data versions and random seeds.

## XAI plan

Retain LIME's local perturbation-and-surrogate idea, but replace equal fixed-width slices and mean replacement with ECG-aware regions and plausible perturbations. Compare against Integrated Gradients or Grad-CAM, depending on the selected model.

Report:

- **Local fidelity:** how well the surrogate approximates the model around the explained example.
- **Faithfulness:** deletion/insertion or another defined perturbation test on highlighted regions.
- **Stability:** consistency across explanation seeds and small non-label-changing signal perturbations.
- **Localization check:** overlap with available expert event annotations, only when the annotations actually support that comparison.

Describe highlighted regions as model evidence. Do not claim they establish a clinical or causal reason for a finding.

## Demo

A compact Streamlit research demo is the fastest Round 1 interface:

1. Select an example from the public held-out set.
2. Show the signal, predicted waveform boundaries, reference annotations, and input quality checks.
3. Overlay the explanation and let the user inspect the perturbed signal and its prediction change.
4. Show a short, templated summary derived only from structured model output and the measured explanation results.

No free-form diagnostic generation or patient-data upload in this round. An LLM or avatar can be considered later as a presentation layer, after the underlying evidence and constraints are established.

## Work sequence

| Window (CET) | Deliverable |
|---|---|
| Nov 12–13 | Confirm 2–4 person team; inspect TimeNet registry/repository; settle dataset and label task |
| Nov 14–16 | Dataset contribution draft; reproducible data loading and split |
| Nov 17–19 | PyTorch baseline, temporal-model experiment, and initial XAI metrics |
| Nov 20–21 | Demo, limitations, and results summary |
| Nov 22–23 | Required public posts, Discord post, screenshots, and final submission before 08:59 CET |

## Dependencies and risks

- Round 1 requires a team of 2–4; solo entry is not eligible.
- The data contribution must be useful and non-duplicative.
- A full fine-tuning run may exceed the time or available compute; keep the baseline and a smaller temporal-model experiment as fallback paths.
- Public X and LinkedIn posts and a Discord post are part of the submission requirements.
- This demo is for research and education. A clinician-facing diagnostic product requires separate clinical validation, privacy/security design, and regulatory assessment.

## Immediate next actions

1. Confirm teammate availability and hackathon registration; Round 1 requires a team of 2–4.
2. Prototype the QTDB connector in TimeNet on a tiny sample and preserve dataset provenance, sparse fiducials, and known-subject grouping.
3. Verify the current OpenTSLM model/checkpoint and fine-tuning path for the localization task; keep the PyTorch baseline as the runnable reference.
4. Plan an untouched external or newly collected evaluation cohort before making final performance claims.

## Inventory update (2026-11-15)

The 105 QTDB excerpts have final `.q1c` annotations, 250 Hz sampling and two leads throughout, and map to 105 source records across six source databases. Official European ST-T documentation identifies repeated-subject record groups; three pairs present in QTDB are now kept within one partition. Universal patient IDs across source databases remain unavailable. See [dataset decision](dataset-decision.md).
