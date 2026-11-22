# BeatLens

BeatLens is a research workbench for locating the **T-wave peak** in a two-channel ECG beat given a known **R-peak anchor**. It overlays the model estimate and the reference annotation, then shows which waveform regions influenced the estimate.

[Experiments](docs/opentslm-feasibility.md) · [TimeNet connector](integrations/timenet-connectors/README.md) · [MIT License](LICENSE)

## Demo


https://github.com/user-attachments/assets/dfc9b926-f1e5-4a6e-9d80-2c60835b7ecd

## Results

All experiments below use supplied R anchors. Mean absolute error (MAE) is measured in milliseconds. The cross-database panels were used during development.

| Model | LUDB MAE (1,603 windows) | MIT-BIH errors over 80 ms (2,112 windows) |
| --- | ---: | ---: |
| Training-set timing prior | 30.92 ms | — |
| OpenTSLM interaction-kernel readout shown in the demo | 18.60 ms | 453 |
| Waveform-only candidate ranker | 11.08 ms | 571 |

The waveform-only ranker had the lowest 

https://github.com/user-attachments/assets/4bee4cf4-bf72-49f4-b424-a186e15c8c93

average error on LUDB. The OpenTSLM readout made fewer large misses on the MIT-BIH pilot. The [beat-level audit](docs/opentslm-feasibility.md#beat-level-audit-on-mit-bih-records-109-and-214) shows examples where both OpenTSLM variants placed late peaks too early.

## Models and explanations

- **Fine-tuning experiment.** We fine-tuned OpenTSLM SoftPrompt (Llama 3.2 1B) with LoRA on Nebius to generate the T-peak sample index. In the matched repeat, MAE stayed at 64.5 ms after swapping waveforms, leaving waveform dependence unverified. [Fine-tuning and waveform-swap control](docs/opentslm-feasibility.md#matched-prompt-waveform-control)
- **Demo readout.** The numerical readout combines frozen OpenTSLM ECG patch features and local raw waveform samples. Separate kernels model similarity within each input, and their product captures interactions. We trained the readout on QTDB beat windows. [Model experiments](docs/opentslm-feasibility.md)
- **Temporal LIME** replaces short ECG segments and fits a local approximation to the resulting changes in predicted time. **Integrated Gradients** traces how inputs contribute to the change in predicted timing from a reference to the original input. The saved OpenTSLM example uses feature and waveform inputs to the readout. [Explanation experiments](docs/xai-prototype.md)

## Run locally

Install Python 3.11 or newer and `uv`. Run these commands from the repository root to open the results view or launch the locally trained PyTorch explanation workflow.

```sh
uv sync --locked --extra qtdb --extra demo
uv run --no-sync streamlit run src/beatlens/presentation/app.py
```

**OpenTSLM results** is available immediately. **OpenTSLM explanation** uses a saved inference artifact. To replay your own run, place the JSON output from `scripts/explain_qtdb_opentslm.py` in `outputs/qtdb-opentslm/beatlens-opentslm-interaction-kernel-explanations.json` for the interaction-kernel checkpoint. The video shows that workflow, and the experiment notes document the research runs. The **ECG explanations** view uses the separate PyTorch locator and can be set up locally using the steps below.

Download and extract [QT Database 1.0.0](https://physionet.org/content/qtdb/1.0.0/) so that `data/qt-database-1.0.0/` contains the record files. Run:

```sh
uv run --no-sync python scripts/audit_qtdb.py
uv run --no-sync python scripts/make_qtdb_split.py
uv run --no-sync python scripts/train_qtdb_tpeak.py --output-dir outputs/qtdb-tpeak-known-subject-groups
```

Open **ECG explanations** in Streamlit to inspect validation beats. The waveform plot marks the supplied R anchor at zero and displays the model and reference T peaks separately.

## Data and repository guide

- **QTDB** supplies training and development windows. The provisional split groups known same-subject excerpts. [Task and split](docs/qtdb-task-and-split.md)
- **LUDB** and **MIT-BIH with separate manual T-peak labels** supply cross-database development panels. [Data decisions](docs/dataset-decision.md) · [Source and pilot details](docs/opentslm-feasibility.md)
- **TimeNet integration** maps QTDB records into sparse localization tasks for observed fiducials. The connector prototype was validated locally with TimeNet 0.12.0. [Setup and validation](integrations/timenet-connectors/README.md#validation-status)
- **Implementation** lives in `src/beatlens`. Training and evaluation entry points live in `scripts`. [Architecture](docs/architecture.md)

**Research use only.** Results describe timing on selected beat windows with known R anchors. Clinical performance and end-to-end ECG delineation remain unvalidated.

Built by **Dr. Mahdi Habibi** and **Dr. Sdjad Sadeghi** for the TimeNet Hackathon 2026.
