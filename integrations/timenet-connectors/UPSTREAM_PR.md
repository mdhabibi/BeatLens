# PR title

`feat(connectors): add QTDB sparse fiducial connector`

# PR description

## Summary

Add a `physionet/qtdb` connector for the QT Database (PhysioNet 1.0.0). It downloads the release's
record list and checksum manifest, verifies each required WFDB file, and converts the 105 excerpts
into two-lead ECG records with sparse waveform-fiducial localization tasks.

The connector uses the `.man` selected-beat annotations as task context and first-expert `.q1c`
fiducials as observed targets. It emits tasks only where a target is annotated; absent fiducials
remain unknown rather than becoming negative examples. The dataset card declares ODC-By-1.0.

## Splits and identity caveat

The checked-in 65/15/20 split keeps the three documented EDB repeat-subject pairs together. QTDB
records can derive from several source databases, and cross-database patient identities have not
been resolved. Treat this assignment as provisional for leakage-sensitive evaluation; it does not
claim globally patient-independent splits.

## Related connector work

PR #630 adds the distinct MIT-BIH Arrhythmia beat-classification connector. QTDB provides waveform
delineation labels and this change adds a separate connector. Possible overlap between source
excerpts across databases has not been ruled out, so the two datasets should not be described as
independent patient cohorts without a separate identity audit.

## Validation

- QTDB fixture test: 1 passed.
- `make test-connectors`: 277 connector tests and connector type checks passed.
- `make check`: schema check, Ruff, and `ty` passed.
- Full local QTDB build: 105 records and 14,750 sparse localization tasks (9,336 train; 2,388
  validation; 3,026 test).

The full TimeNet core suite did not complete because it stalled in a network-bound registry
conformance test. The 1,653-test unit subset had 1,652 passing; its unrelated forked Torch DataLoader
test failed in this environment because launching `torch_shm_manager` was denied and the loader
timed out.
