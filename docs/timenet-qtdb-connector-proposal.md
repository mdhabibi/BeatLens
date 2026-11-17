# TimeNet QTDB connector proposal

## Proposed connector

- **Dataset ID:** `physionet/qtdb`
- **Source:** PhysioNet QT Database v1.0.0, `https://physionet.org/files/qtdb/1.0.0/`
- **License:** Open Data Commons Attribution License v1.0
- **Base:** `BasePhysioNetConnector` (WFDB)
- **Task:** `TemporalLocalizationTask(mode=SPARSE)` for observed fiducials on selected beats.

QTDB has 105 two-lead, fifteen-minute excerpts. Each record has a WFDB header and signal file; selected-beat anchors are in `.man`, and the first expert's second-pass fiducials are in `.q1c`. The initial connector would import `.q1c` marks for `(` onset, `)` offset, `p` P peak, `t` T peak, and `u` U peak as distinct target types. It would not merge `.q2c` from the 11-record subset. This matches the existing BeatLens loader's symbol vocabulary and sparse-label handling.

## TimeF record and task shape

Each TimeF `Record` represents one complete QTDB excerpt, rather than one record per beat. It has two lazy `Signal`s, named from the WFDB header (for example, `V5` and `V2`), each with 225,000 values at 250 Hz for `sel102`. Their `RegularAxis` starts at the excerpt's time zero; the signal values are read in WFDB physical units (mV). The record metadata keeps the QTDB record key, source database and source record key, and the current split-group metadata where available.

Create one sparse temporal-localization task for each selected beat and fiducial type that is observed. Each task uses the complete record as input, a 2.5-second beat scope (1.0 second before and 1.5 seconds after the `.man` anchor), and a prompt naming the fiducial type. The target is a tuple of `TimePoint`s at the `.q1c` samples assigned to that beat and type, on the record timeline. Task split is copied from the local grouped split manifest. A missing fiducial produces no task for that type; it is never encoded as a negative label.

## Concrete example: `sel102`, first selected beat

The checked-in local QTDB files show `sel102` has two signals (`V5`, `V2`), 250 Hz, and 225,000 samples. The local grouped split manifest assigns it to `train`. Its first `.man` anchor is sample `150253`, which is `601.012 s` from record start. The corresponding `.q1c` T-wave peak is sample `150316`, at `601.264 s`.

```python
Record(
    metadata={
        "qtdb_record": "sel102",
        "source_database": "MIT-BIH Arrhythmia Database",
        "source_record_id": "102",
        "split_group_id": "MIT-BIH Arrhythmia Database::source-record::102",
    },
    sources=(Source(name="ECG", signals=(v5, v2)),),
)

TemporalLocalizationTask(
    inputs=(record,),
    prompt="Locate the T-wave peak for this selected beat.",
    split="train",
    mode=LocalizationMode.SPARSE,
    scope=TimeInterval.seconds(600.012, 602.512),
    targets=(TimePoint.seconds(601.264),),
)
```

The 250 Hz cadence and the corresponding per-lead `TimeSeriesSpec` use `float32` values and millivolts. Each target time remains on the full recording timeline, so the scoped beat window does not shift or lose source timing.

## Source integrity and scope

The PhysioNet release includes `SHA256SUMS.txt`; the connector should pin downloaded source files to their published digests. The local BeatLens split manifest is derived from the completed audit and keeps the three documented repeat-subject EDB pairs together. It is not included in the public QTDB download, so the first prototype should accept a supplied manifest (or an explicit split configuration) rather than inventing patient identity. The current local manifest is provisional because cross-database subject identity is not fully reconciled.

The proposed dataset ID, source/base, task type, and example shape were approved before implementation. A TimeNet-shaped prototype now lives in `integrations/timenet-connectors/`; it has not been copied into a TimeNet checkout or committed upstream.
