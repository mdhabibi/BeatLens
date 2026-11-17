# TimeNet QTDB connector prototype

This folder contains a development copy of a connector for `physionet/qtdb`, shaped for the
TimeNet connector API. The BeatLens repository remains the home of model training, explanation,
and evaluation; this connector belongs in TimeNet's `timenet-connectors` package if it is adopted.
The API reference was cloned from TimeNet `main` at commit
`9bd1613019ee61fbc52e02106a1a45990ead7a54` (2026-11-13, TimeNet 0.12.0).

## Applying it to a TimeNet checkout

Copy the `qtdb` package into:

```text
packages/timenet-connectors/src/timenet_connectors/datasets/physionet/qtdb/
```

The existing TimeNet checkout already has the `physionet` namespace package. The package includes
its own dataset card, `wfdb` requirement, and frozen grouped split manifest. Its downloader reads
the pinned PhysioNet `RECORDS` and `SHA256SUMS.txt` files, then downloads and verifies each `.hea`,
`.dat`, `.man`, and `.q1c` file required by the converter.

The split is the BeatLens audit's current known-subject-grouped 65/15/20 assignment. It keeps the
three documented EDB repeat-subject pairs together; it does not claim cross-database patient
identity has been resolved. The connector raises if its split file differs from the pinned digest.

## Validation status

Validated against TimeNet commit `9bd1613019ee61fbc52e02106a1a45990ead7a54`:

- QTDB fixture test: 1 passed.
- `make test-connectors`: all 277 connector tests and connector type checks passed.
- `make check`: schema check, Ruff, and `ty` passed.
- Local full build and readback: 105 records, 14,750 sparse localization tasks; train 9,336,
  validation 2,388, test 3,026.

The full TimeNet core suite did not finish: it stalled in a network-bound registry conformance
test. The 1,653-test unit subset completed with 1,652 passing; its one failure was the unrelated
forked Torch DataLoader test, where the environment denied `torch_shm_manager` and the loader
timed out.

### BeatLens workspace recheck (2026-11-16)

The current workspace was rechecked against the connector's pinned source data and split without
claiming a new TimeNet API run: the `RECORDS`, `SHA256SUMS.txt`, and `split.csv` digests match the
connector constants; all 105 records have the required two leads, 250 Hz rate, and `.hea`, `.dat`,
`.man`, and `.q1c` files. Recomputed sparse-task counts match the previous TimeF build:

| Split | Selected-beat scopes | Sparse fiducial tasks |
| --- | ---: | ---: |
| Train | 2,296 | 9,336 |
| Validation | 566 | 2,388 |
| Test | 761 | 3,026 |

The pinned TimeNet checkout and `timenet` packages are not present in this BeatLens workspace, so
the TimeNet fixture test and full TimeF build listed above were not rerun here. The results above
are a fresh input-data and task-count audit; the earlier TimeNet integration results remain as
previously recorded.

The connector is committed in the local TimeNet checkout on branch `feature/qtdb-connector` at
`b5ec9be` and has not been pushed. The default build CLI also hit this machine's SOCKS-proxy
configuration while resolving its remote registry; the successful full build used TimeNet's local
pipeline, the preloaded QTDB cache, and a local registry. A draft PR description is in
[`UPSTREAM_PR.md`](UPSTREAM_PR.md).
