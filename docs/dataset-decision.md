# Dataset decision: Round 1

**Updated:** 2026-11-15  
**Status:** QTDB is the leading distinct connector candidate. Annotation audit, sparse-fiducial task, and a split that keeps documented same-subject excerpts together are in place. Global patient independence remains unverified.

## What the current catalog shows

The [TimeNet dataset catalog](https://docs.timenet.ai/catalog/datasets.html) lists `physionet/ptb-xl` (release `1.0.0`) and `physionet/ecg-qa-cot` (release `1.2.1`), but does not yet list MIT-BIH. The catalog alone was misleading: the GitHub integration check found [OpenTSLM/TimeNet PR #630](https://github.com/OpenTSLM/TimeNet/pull/630), an **open draft** titled “feat(connectors): add MIT-BIH Arrhythmia Database.”

PR #630 already represents 48 records from 47 subjects, 109,966 beat classification tasks, and beat and rhythm localization tasks. It preserves the annotation set, subject linkage for records 201/202, and the ODC-By 1.0 source provenance. This overlaps directly with our planned dataset contribution. Do not implement or submit a duplicate MIT-BIH connector.

## MIT-BIH: useful for the experiment, unavailable as a novel connector contribution

The [PhysioNet release](https://physionet.org/content/mitdb/1.0.0/) remains a technically strong dataset for BeatLens: it is version `1.0.0`, licensed under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), and contains 48 approximately half-hour, two-channel recordings from 47 subjects with roughly 110,000 expert-reviewed beat annotations. PR #630 is not merged, so the connector is not yet in the public catalog. We may use the dataset if it becomes available through TimeNet, but our hackathon connector contribution must be distinct or materially complementary.

Do not copy source recordings into this repository. Any future TimeNet connector should preserve the upstream citation, version, license, access URL, lead metadata, annotation provenance, and task/split definitions in its dataset card. Keep the connector PR separate from BeatLens model and demo code.

## Leading alternative: QT Database (QTDB)

PhysioNet's [QTDB v1.0.0](https://physionet.org/content/qtdb/1.0.0/) has 105 two-lead ECG excerpts, each about 15 minutes long. Experts manually marked P, QRS, T, and sometimes U waveform boundaries on 30–50 selected beats per record. It is available under ODC-By 1.0. The TimeNet connector tree currently has no QTDB folder, and the GitHub PR searches for `qtdb`, `QT Database`, and `P-wave` found no matching QTDB/P-wave connector PRs (checked 2026-11-12).

QTDB aligns especially well with an explanation study that tests ECG waveform delineation: a model can predict waveform boundaries or intervals, while LIME and a gradient attribution method are assessed for stability and faithfulness around those temporal targets. This would shift the experiment from beat-class classification to waveform delineation. It remains model-behavior research, not a diagnostic task.

The manual annotation files need careful interpretation: `.qt1` and `.qt2` are first-pass annotations from two experts, while `.q1c` and `.q2c` are final audited files; second-expert annotations are available for only 11 records. Use the final audited first-expert file as the main target unless the data audit supports another choice, and retain the second-expert subset for a small inter-annotator sensitivity check.

QTDB excerpts come from multiple source databases, and record names encode the source record. Preserve source database and original record identifiers, and group excerpts from the same source record—and known shared subjects—into one split. The manual boundaries cover selected beats, not every beat in each 15-minute signal. Build windows only from annotated beats and document the sampling rule. Attribution overlap must account for the annotation and window construction baseline; it is not causal or clinical validation.

## Local archive inventory

The downloaded archive is present at `data/qt-database-1.0.0/` (86 MB). The audit decoded all 105 final first-expert `.q1c` files and all 11 second-expert `.q2c` files. Every record has two signals sampled at 250 Hz. The `.q1c` files contain 30,696 annotation symbols in aggregate; `.q2c` contains 2,081 symbols. The source table maps all 105 excerpts to 105 distinct source records across six source databases. This means source-record grouping is straightforward, but subject overlap still needs checking where the source documentation provides subject IDs. The archive and generated audit outputs are excluded from Git.

The `.q1c` symbol totals are: `(` 8,336; `)` 11,180; `p` 3,194; `t` 3,542; `u` 821; plus beat annotations `N` 3,528, `A` 30, `B` 50, `Q` 7, and `V` 8. Treat the waveform symbols as sparse fiducial annotations, not dense per-sample labels. The first task is sparse fiducial localization on annotated beats: represent each marked event as a sample offset from its selected-beat anchor, mask missing marks as unknown, and do not pair generic `(`/`)` markers into named wave boundaries. Keep `.q2c` separate for a limited inter-annotator sensitivity analysis, not a broad agreement claim. See [QTDB task and split](qtdb-task-and-split.md).

The QTDB source table maps all 105 excerpts to source records, but not universal patient IDs. The official European ST-T Database documentation identifies repeated recordings from the same subject, including three pairs present in QTDB: `e0121/e0122`, `e0124/e0126`, and `e0129/e0133`. The original record split placed every pair in different partitions. The updated seed-42 split keeps these known pairs together and contains 67 train, 16 validation, and 22 test excerpts (65, 16, and 21 split groups respectively). The split manifest records each group ID and its evidence basis.

This is a **known-subject-grouped provisional split**, not a fully patient-independent split. Subject IDs cannot be joined across the six source databases, and the test partition was already inspected during earlier exploratory model comparisons. Treat all current test metrics as exploratory; use a new or external cohort for final performance claims. The pre-audit record-level manifest is retained locally as `outputs/qtdb-audit/qtdb-split-record-level-pre-audit.csv`.

The script [audit_qtdb.py](../scripts/audit_qtdb.py) decodes the `.q1c`/`.q2c`/`.man` annotations and reads the source-record table bundled with the dataset. Run from the repo root:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run --with wfdb --no-project \
  python scripts/audit_qtdb.py data/qt-database-1.0.0
```

## Next checks

1. Prototype the QTDB TimeNet connector on a tiny local sample; confirm TimeF representation for signals, provenance, sparse manual annotations, and localization tasks.
2. Verify the current OpenTSLM model/checkpoint and fine-tuning path against the localization task. Keep XAI evaluation gated on an adequate signal-based localization model.
3. The [LUDB external-validation plan](external-validation-plan.md) now records a checksum-verified audit and zero-shot transfer/aggregation study. Its lead-domain mismatch and correlated donor windows limit the conclusion; a fixed lead-compatible model and broader external cohort are needed for stronger claims.
4. The existing QTDB test partition has already informed exploratory model work; keep it sealed for any remaining model or explanation decisions.

See the [first baseline results](baseline-results.md) for metrics and the oracle-count evaluation limitation.

## Sources

- [TimeNet dataset catalog](https://docs.timenet.ai/catalog/datasets.html)
- [TimeNet connector guide](https://docs.timenet.ai/connectors.html)
- [TimeNet repository](https://github.com/OpenTSLM/TimeNet)
- [TimeNet PR #630: MIT-BIH connector](https://github.com/OpenTSLM/TimeNet/pull/630)
- [MIT-BIH Arrhythmia Database, PhysioNet v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
- [QT Database, PhysioNet v1.0.0](https://physionet.org/content/qtdb/1.0.0/)
- [QTDB record and source documentation](https://physionet.org/files/qtdb/1.0.0/doc/node4.html)
- [European ST-T Database subject and record documentation](https://physionet.org/content/edb/1.0.0/)
- [QTDB manual annotation documentation](https://physionet.org/files/qtdb/1.0.0/doc/node5.html)
- [TimeNet Online Hackathon](https://challenge.timenet.ai/)
