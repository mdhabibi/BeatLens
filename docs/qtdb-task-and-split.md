# QTDB first task and split

## First label representation

Treat the final `.q1c` file as a sparse set of timestamped expert fiducials. The initial output channels are:

| `.q1c` symbol | Target |
| --- | --- |
| `(` | generic waveform onset marker |
| `)` | generic waveform end marker |
| `p` | P-wave peak |
| `t` | T-wave peak |
| `u` | U-wave peak |

Keep QRS beat annotations such as `N`, `A`, `B`, `Q`, and `V` as beat anchors/metadata; they are not waveform-boundary targets in this first task. Do not pair `(` and `)` into named P/QRS/T boundaries: the symbols do not identify the wave type, and some boundaries are missing or ambiguous.

The first model task is **sparse fiducial localization on annotated beats**, not dense delineation or event-presence detection. Build beat-centered windows from the selected-beat reference annotations, encode each available target as a sample-index offset from the beat anchor, and train/evaluate only for fiducials that are present. A missing marker is unknown, not a negative. Keep a per-target observation mask; do not penalize the model for predicting an event where the expert did not mark one. Report event localization error in milliseconds for observed targets, and avoid claiming false-positive detection performance until label completeness supports it.

The loader is `beatlens.infrastructure.qtdb.iter_qtdb_examples`. By default it returns 2.5-second windows (1.0 second before and 1.5 seconds after each `.man` anchor), with a `(samples, leads)` float array, per-target offsets in samples relative to the anchor, an observation mask, and a valid-sample mask for edge padding. It assigns `.q1c` fiducials to their nearest selected-beat anchor within 1.25 seconds. The 250 Hz signals therefore produce 625-sample windows. Install the adapter dependency with `uv sync --extra qtdb`.

An audit comparison confirmed that the loader retains every target marker in the chosen windows: 17,554 train, 3,961 validation, and 5,558 test fiducials (27,073 total). This checks annotation-to-anchor assignment and window coverage; it does not establish model performance or label completeness.

Use the final audited first-expert `.q1c` labels for the primary task. Retain `.q2c` as a separate sensitivity analysis on the 11-record subset; do not merge annotators' marks into one target.

## Provisional known-subject-grouped split

`scripts/make_qtdb_split.py` creates a deterministic, source-database-stratified 65/15/20 split from the completed audit CSV (`random_state=42`). It keeps the three known repeat-subject pairs from European ST-T together:

- `e0121` / `e0122`
- `e0124` / `e0126`
- `e0129` / `e0133`

The current assignment is:

| Split | Records |
| --- | ---: |
| Train | 67 |
| Validation | 16 |
| Test | 22 |

The generated assignment is `outputs/qtdb-audit/qtdb-split.csv` (ignored by Git because it is derived metadata from the local dataset). It contains 65 train, 16 validation, and 21 test split groups, corresponding to 67, 16, and 22 excerpts. The manifest includes `split_group_id` and `split_group_basis` to show which groupings come from documented shared subjects and which remain source-record-only. Patient IDs cannot be reconciled across all six source databases, so this is not a fully patient-independent split. The prior record-only manifest is retained locally as `qtdb-split-record-level-pre-audit.csv`.

Recreate it after the audit with:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run --with scikit-learn --no-project \
  python scripts/make_qtdb_split.py
```

Example source-level use:

```python
from pathlib import Path
from beatlens.infrastructure.qtdb import iter_qtdb_examples

examples = iter_qtdb_examples(
    Path("data/qt-database-1.0.0"),
    Path("outputs/qtdb-audit/qtdb-split.csv"),
    "train",
)
first = next(examples)
```
