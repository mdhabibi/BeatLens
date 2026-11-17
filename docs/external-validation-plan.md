# External validation plan: LUDB

## Status

LUDB v1.0.1 has been downloaded into ignored local data storage and audited. All **2,805 files listed in the release checksum manifest match**. The annotation and window audit found **1,603 eligible lead-II T-peak targets across all 200 subjects**. The frozen QTDB checkpoint and the fixed-five-donor aggregation comparison have now been evaluated on LUDB.

The QTDB test partition remains sealed for this work. The prior QTDB split is not a final holdout because test records were inspected during earlier model comparisons.

## Why LUDB fits the target

The official [PhysioNet LUDB v1.0.1 page](https://physionet.org/content/ludb/1.0.1/) describes 200 ten-second, 12-lead ECGs from 200 subjects. Cardiologists annotated P, QRS, and T wave peaks and boundaries independently for each lead. The signals are sampled at 500 Hz. The release is openly accessible under ODC-BY 1.0; preserve the database citation and attribution in any report or redistributed derivative.

LUDB provides an independent source, population, acquisition setup, lead layout, sampling frequency, and annotation process. This makes it useful for a cross-database transfer challenge. It is not a drop-in replication of QTDB: QTDB uses 2-lead 250 Hz excerpts with sparse selected-beat annotations, while LUDB uses short 12-lead 500 Hz records with per-lead delineations.

## Compatibility audit before evaluation

1. **Done:** Downloaded and verified the exact v1.0.1 release in ignored local data storage; no source recordings are tracked.
2. **Done:** Audited all 200 records. All are 5,000 samples at 500 Hz and contain leads II and V5. Lead II has 1,831 `N` QRS anchors and 1,642 `t` T-peak marks; lead V5 has 1,830 `N` anchors and 1,643 `t` marks.
3. **Done:** Locked lead II as the target and QRS-anchor lead and lead V5 as the second input before viewing model errors. Both leads are present in all records.
4. **Done:** Resampled the fixed II/V5 inputs from 500 Hz to 250 Hz with `scipy.signal.resample_poly`, then created 2.5-second windows (1.0 s before and 1.5 s after a lead-II QRS anchor). Applied per-window, per-channel z-score normalization and converted target offsets in milliseconds.
5. **Done:** The QTDB checkpoint's input channels were trained on heterogeneous QTDB lead-name pairs, so this is reported as zero-shot cross-database/domain transfer, not clean lead-matched validation.
6. **Done:** The frozen checkpoint was evaluated on all 1,603 LUDB targets. It reached **21.65 ms MAE**, **14.63 ms median absolute error**, **47.12 ms P90**, and **86.5% within 40 ms**. The QTDB train-only timing prior reached **30.92 ms MAE** and **75.7% within 40 ms**. Paired model-minus-prior MAE was **−9.26 ms** (200-record cluster-bootstrap 95% interval **−11.82 to −6.84 ms**), so the predeclared gate passed. No LUDB data was used for training or model selection.

## Explanation aggregation comparison

The existing implementation requires each donor window to be non-overlapping with the explained target window, but it does **not** require donor windows to be non-overlapping with one another. LUDB has at least five eligible target-separated donor candidates for **433 of 1,603 targets**, covering **147 subjects**, and at least three for 1,400 targets. The aggregation comparison below uses the fixed five-donor subset to match the QTDB donor count. The remaining targets were included in the model localization gate, not mixed into the fixed-five aggregation comparison.

If a future study requires donor windows to also be non-overlapping with each other, this archive supports at most two such donors for the current 2.5-second windows; the three-method median-versus-mean comparison then needs a different predeclared protocol. Do not change donor definitions between the QTDB and LUDB runs without labeling the protocol difference.

### External aggregation results

| Measure | Equal mean | Similarity weighted | Component median |
| --- | ---: | ---: | ---: |
| Consensus-to-donor top-region Jaccard | 0.546 | 0.542 | **0.557** |
| Mean consensus output shift (ms) | 15.97 | 15.60 | 15.25 |
| Mean within-example shift range (ms) | 22.28 | 21.78 | 21.09 |
| Matched random output-shift mean (ms) | 7.64 | 7.64 | 7.64 |
| Examples overlapping T peak ±60 ms | 136/433 | 127/433 | 151/433 |
| Examples overlapping anchor ±80 ms | 218/433 | 198/433 | 226/433 |
| Examples with a selected late-window region | 415/433 | 418/433 | 418/433 |

Donor top sets had mean pairwise Jaccard **0.398**; mean donor local LIME R² was **0.858**. The median's paired increase in consensus-to-donor Jaccard over equal mean was **0.0107** (147-subject cluster-bootstrap 95% interval **0.0061 to 0.0155**). Similarity weighting was **0.0044 lower** than equal mean (**−0.0083 to −0.0007**). The median's T-peak overlap advantage over equal mean was 3.5 percentage points, with an interval spanning no difference (**−0.2 to 7.0 pp**). These measures do not identify a universally best aggregation rule.

The selected donor windows overlap one another in this short-recording cohort: **3,334 of 4,330 donor pairs** overlap, and all 433 targets have at least one overlapping donor pair. Every donor remains non-overlapping with the explained target window. Treat donor fits as correlated views, not independent samples. This caveat also applies to the earlier QTDB aggregation results; 655 of 5,520 donor pairs overlapped across 400 of 552 validation targets.

The 433-target LUDB aggregation run produced `summary.json`, `per-example.csv`, and `per-donor.csv` in ignored `outputs/ludb-consensus-aggregations/`. Frozen-model transfer metrics are in ignored `outputs/ludb-tpeak-transfer/`; the dataset audit is in `outputs/ludb-external-audit/`.

Keep the 200 people grouped by subject. Use all eligible LUDB records only as an external evaluation cohort; do not train, fine-tune, or tune aggregation choices on LUDB. Preserve record-level per-lead annotations and identify the chosen primary lead in every output.

## Screen for a second external cohort

The next proposed step was a lead-compatible evaluation on another independent cohort. A local QTDB header audit found only **8** excerpts with the ordered `MLII/V1` pair. Under the existing known-subject-grouped split, those provide **130** single-T-peak training beats from 3 excerpts, **102** validation beats from 2 excerpts, and **110** test beats from 3 excerpts. The test partition has already informed exploratory model work. The other 57 QTDB excerpts identify their channels only as `ECG1/ECG2`; the remainder use heterogeneous lead pairs. This is too little clean training support for a defensible lead-matched model, and using LUDB for training would invalidate its role as the external cohort in the current protocol.

MIT-BIH is not a direct substitute with its standard annotations: its reference files mark beat locations and classes, not T-wave peaks ([PhysioNet MIT-BIH documentation](https://physionet.org/content/mitdb/1.0.0/)). A paper describes a separate, single-expert P/T reannotation on channel one and points to a download at `elgendi.net`; that URL currently returns 404, and no accessible, versioned annotation package or licensing terms were verified ([Elgendi et al. 2015](https://doi.org/10.3390/s150717693)). Treat that as a lead for future data recovery, not as validated ground truth. The European ST-T database marks ST/T change episodes rather than per-beat T-peak fiducials; moreover, 33 QTDB excerpts originate from that database, so it is not a wholly independent cohort for this study ([QTDB source table](https://physionet.org/physiobank/database/qtdb/doc/node3.html)).

The [ECG-ID database](https://physionet.org/content/ecgiddb/1.0.0/) is an accessible alternative with 310 short recordings from 90 volunteers and ten R/T marks per recording, but PhysioNet explicitly describes those peaks as **unaudited outputs from an automated detector**. Its two channels are raw and filtered versions of the same Lead I signal, not a two-lead pair. It can support an exploratory robustness check against automated marks after a one-channel/raw-filtered input protocol is defined; it cannot serve as an expert-reference accuracy gate for the current model.

### ECG-ID exploratory agreement run

The v1.0.0 archive was checksum-verified (999/999 files). The frozen QTDB checkpoint was applied to the two ECG-ID channels (raw Lead I and filtered Lead I), after 500→250 Hz resampling and the same per-window/per-channel normalization. Windows were anchored to the annotated `N` positions; each following `t` mark was paired to its preceding `N` within 1.25 s. Of 3,100 pairs, **2,835** had complete 2.5-second windows, spanning all **90 people** and **310 recordings**. No model training or selection used ECG-ID, and the QTDB test split was not loaded.

| Agreement with automated marks | Frozen QTDB model | QTDB train-only timing prior |
| --- | ---: | ---: |
| Annotated windows | 2,835 | 2,835 |
| Mean absolute difference | 20.86 ms | 32.98 ms |
| Median absolute difference | 14.64 ms | 30.07 ms |
| P90 absolute difference | 40.18 ms | 60.07 ms |
| Within 40 ms | 89.8% | 66.5% |

The paired model-minus-prior difference in mean absolute disagreement was **−12.13 ms** (person-cluster bootstrap 95% interval **−15.26 to −8.89 ms**, 20,000 resamples, seed 647). This indicates stronger agreement with this automated annotation stream than with a constant timing prior. It does **not** establish T-peak localization accuracy: the marks are explicitly unaudited automated detections, and both ECG-ID input channels represent one lead. Treat this as an input/domain stress test only.

Reproduce the run with:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run --extra qtdb \
  python scripts/evaluate_ecgiddb_tpeak_stress.py
```

The script writes `summary.json` and `per-beat.csv` under ignored `outputs/ecgiddb-tpeak-stress/`.

**Decision:** no second cohort currently meets the label, lead, provenance, and independence requirements for a confirmatory T-peak evaluation. Keep the completed LUDB results labeled as zero-shot domain-shifted transfer. Resume a second-cohort gate only after obtaining an independently sourced, versioned dataset with per-beat T-peak references and a defensible channel mapping; alternatively, report a QTDB channel-stratified analysis explicitly as exploratory and keep its previously inspected test records out of confirmatory claims.

Reproduce the local audit with:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run python scripts/audit_ludb_external.py
```

The script writes ignored `summary.json`, `per-record.csv`, and `targets.csv` files to `outputs/ludb-external-audit/`.

## Stop conditions

- Local release files or checksums do not match PhysioNet v1.0.1.
- The primary lead or T-peak annotation mapping is absent or ambiguous.
- Too few complete windows remain after resampling and beat anchoring.
- The model input channels cannot be defended as a fixed mapping; in that case characterize the run strictly as domain-shifted stress testing.
- The model does not pass the predeclared localization gate against the train-only prior; do not make XAI faithfulness or physiological claims.

## Source

- [PhysioNet LUDB v1.0.1: dataset, annotation, access, and license](https://physionet.org/content/ludb/1.0.1/)
- [PhysioNet LUDB README: lead-specific WFDB annotation files](https://physionet.org/content/ludb/1.0.1/README)
- [PhysioNet standard annotation codes](https://physionet.org/static/lightwave/doc/annotations.html)

Reproduce the frozen transfer gate with:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run python scripts/evaluate_ludb_tpeak_transfer.py
```

Reproduce the gated aggregation comparison with:

```sh
UV_CACHE_DIR=/tmp/beatlens-uv-cache uv run python scripts/compare_ludb_consensus_aggregations.py
```
