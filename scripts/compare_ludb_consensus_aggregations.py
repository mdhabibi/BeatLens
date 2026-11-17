"""Compare donor-score aggregation rules on the gated LUDB external cohort."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_qtdb_donor_sensitivity import donor_candidates, quantile_donors  # noqa: E402
from compare_qtdb_consensus_aggregations import analyze_example  # noqa: E402
from beatlens.infrastructure.qtdb import QTDBBeatExample, TARGET_NAMES  # noqa: E402
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor  # noqa: E402

FS_RAW = 500.0
FS_MODEL = 250.0
LEFT_SECONDS = 1.0
RIGHT_SECONDS = 1.5
AGGREGATIONS = ("equal_mean", "similarity_weighted", "component_median")


def load_release_root(data_dir: Path) -> Path:
    manifests = list(data_dir.rglob("RECORDS"))
    if len(manifests) != 1:
        raise SystemExit(f"Expected one LUDB RECORDS file under {data_dir}")
    return manifests[0].parent


def make_record_examples(release_root: Path, record_id: str) -> list[QTDBBeatExample]:
    record_name = str(release_root / record_id)
    record = wfdb.rdrecord(record_name, physical=True)
    if float(record.fs) != FS_RAW:
        raise SystemExit(f"Expected {FS_RAW} Hz for {record_id}; found {record.fs}")
    lead_names = [name.strip().lower() for name in (record.sig_name or [])]
    if "ii" not in lead_names or "v5" not in lead_names:
        raise SystemExit(f"Missing fixed II/V5 inputs in {record_id}")
    lead_indexes = [lead_names.index("ii"), lead_names.index("v5")]
    two_leads_250 = resample_poly(
        np.asarray(record.p_signal[:, lead_indexes], dtype=np.float32), up=1, down=2, axis=0
    ).astype(np.float32)
    annotation = wfdb.rdann(record_name, "ii")
    anchors = sorted(
        int(sample)
        for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True)
        if symbol == "N"
    )
    peaks = sorted(
        int(sample)
        for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True)
        if symbol == "t"
    )
    assigned: dict[int, list[int]] = defaultdict(list)
    for peak in peaks:
        preceding = [anchor for anchor in anchors if anchor <= peak]
        if preceding and peak - preceding[-1] <= round(1.25 * FS_RAW):
            assigned[preceding[-1]].append(peak)

    left = round(LEFT_SECONDS * FS_MODEL)
    right = round(RIGHT_SECONDS * FS_MODEL)
    window_samples = left + right
    examples = []
    for beat_index, anchor_raw in enumerate(anchors):
        anchor_model = round(anchor_raw / 2.0)
        start = anchor_model - left
        end = anchor_model + right
        if start < 0 or end > two_leads_250.shape[0]:
            continue
        peak_offsets = tuple((peak - anchor_raw) / 2.0 for peak in assigned.get(anchor_raw, []))
        events = {name: () for name in TARGET_NAMES}
        events["t_peak"] = peak_offsets
        examples.append(
            QTDBBeatExample(
                record_id=record_id,
                source_database="LUDB",
                source_record_id=record_id,
                split="external",
                beat_index=beat_index,
                anchor_sample=anchor_model,
                sampling_rate_hz=FS_MODEL,
                lead_names=("II", "V5"),
                signal=two_leads_250[start:end].copy(),
                valid_samples=np.ones(window_samples, dtype=bool),
                event_offsets=events,
                observed_targets={name: bool(events[name]) for name in TARGET_NAMES},
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ludb-1.0.1"))
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/ludb-external-audit"))
    parser.add_argument("--transfer-dir", type=Path, default=Path("outputs/ludb-tpeak-transfer"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ludb-consensus-aggregations"))
    parser.add_argument("--donors-per-example", type=int, default=5)
    parser.add_argument("--perturbations", type=int, default=256)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    transfer = json.loads((args.transfer_dir / "summary.json").read_text(encoding="utf-8"))
    if not transfer["localization_gate_passed"]:
        raise SystemExit("The frozen model failed the external localization gate; stopping XAI analysis")
    audit = json.loads((args.audit_dir / "summary.json").read_text(encoding="utf-8"))
    if audit["checksum_mismatches"]:
        raise SystemExit("LUDB checksum audit failed")
    with (args.audit_dir / "targets.csv").open(newline="", encoding="utf-8") as stream:
        target_rows = list(csv.DictReader(stream))
    required_target_donors = args.donors_per_example
    selected_rows = [
        row
        for row in target_rows
        if int(row["target_nonoverlap_donor_candidates"]) >= required_target_donors
    ]
    if not selected_rows:
        raise SystemExit("No target beats have the requested donor count")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QTDBTpeakRegressor(input_leads=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))
    release_root = load_release_root(args.data_dir)
    grouped: dict[str, list[QTDBBeatExample]] = {}
    result_by_key: dict[tuple[str, int], QTDBBeatExample] = {}
    for record_id in sorted({row["record_id"] for row in selected_rows}):
        examples = make_record_examples(release_root, record_id)
        grouped[record_id] = examples
        result_by_key.update(
            {(example.record_id, example.anchor_sample): example for example in examples}
        )

    results: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    donor_pair_count = 0
    overlapping_donor_pair_count = 0
    examples_with_overlapping_donor_pair = 0
    for index, row in enumerate(selected_rows):
        record_id = row["record_id"]
        anchor_model = round(int(row["anchor_sample_500hz"]) / 2.0)
        example = result_by_key.get((record_id, anchor_model))
        if example is None or len(example.event_offsets["t_peak"]) != 1:
            raise SystemExit(f"Audit target mismatch for {record_id} anchor {anchor_model}")
        pool = donor_candidates(example, grouped[record_id])
        donors = quantile_donors(pool, args.donors_per_example)
        if len(donors) != args.donors_per_example:
            raise SystemExit(f"Donor count mismatch for {record_id} anchor {anchor_model}")
        overlap_count = sum(
            abs(left[1].anchor_sample - right[1].anchor_sample) < len(example.signal)
            for left, right in combinations(donors, 2)
        )
        donor_pair_count += args.donors_per_example * (args.donors_per_example - 1) // 2
        overlapping_donor_pair_count += overlap_count
        examples_with_overlapping_donor_pair += overlap_count > 0
        result, rows = analyze_example(
            model,
            example,
            donors,
            perturbations=args.perturbations,
            random_trials=args.random_trials,
            seed=args.seed + index,
        )
        results.append(result)
        donor_rows.extend(rows)
        print(f"[{index + 1}/{len(selected_rows)}] {record_id} anchor {anchor_model}", flush=True)

    aggregate: dict[str, Any] = {
        "records": len({result["record_id"] for result in results}),
        "subjects": len({result["record_id"] for result in results}),
        "examples": len(results),
        "mean_donor_local_r2": float(np.mean([result["mean_donor_local_r2"] for result in results])),
        "mean_donor_local_r2_range": float(np.mean([result["donor_local_r2_range"] for result in results])),
        "mean_pairwise_donor_top_jaccard": float(
            np.mean([result["pairwise_donor_top_jaccard"] for result in results])
        ),
        "by_aggregation": {},
    }
    example_rows = []
    for result in results:
        for name in AGGREGATIONS:
            metrics = result["aggregation_results"][name]
            example_rows.append(
                {
                    "record_id": result["record_id"],
                    "beat_index": result["beat_index"],
                    "aggregation": name,
                    "reference_t_peak_ms": result["reference_t_peak_ms"],
                    "prediction_ms": result["prediction_ms"],
                    "absolute_error_ms": result["absolute_error_ms"],
                    "donor_count": result["donor_count"],
                    "region_count": result["region_count"],
                    "mean_donor_local_r2": result["mean_donor_local_r2"],
                    "donor_local_r2_range": result["donor_local_r2_range"],
                    "pairwise_donor_top_jaccard": result["pairwise_donor_top_jaccard"],
                    **metrics,
                    "selected_intervals_samples": json.dumps(metrics["selected_intervals_samples"]),
                }
            )
    for name in AGGREGATIONS:
        rows = [result["aggregation_results"][name] for result in results]
        aggregate["by_aggregation"][name] = {
            "mean_consensus_vs_donor_top_jaccard": float(
                np.mean([row["mean_vs_donor_top_jaccard"] for row in rows])
            ),
            "mean_consensus_top_shift_ms": float(np.mean([row["consensus_top_shift_mean_ms"] for row in rows])),
            "mean_consensus_top_shift_range_ms": float(
                np.mean([row["consensus_top_shift_range_ms"] for row in rows])
            ),
            "mean_random_shift_ms": float(np.mean([row["random_shift_mean_ms"] for row in rows])),
            "mean_random_shift_p90_ms": float(np.mean([row["random_shift_p90_ms"] for row in rows])),
            "examples_overlapping_t_peak_plusminus_60ms": int(
                sum(row["overlaps_t_peak_plusminus_60ms"] for row in rows)
            ),
            "examples_overlapping_anchor_plusminus_80ms": int(
                sum(row["overlaps_anchor_plusminus_80ms"] for row in rows)
            ),
            "examples_overlapping_late_window": int(sum(row["overlaps_late_window"] for row in rows)),
            "examples_consensus_shift_exceeds_random_mean": int(
                sum(row["consensus_shift_exceeds_random_mean"] for row in rows)
            ),
        }

    summary = {
        "split": "external",
        "dataset": "LUDB v1.0.1",
        "license": "Open Data Commons Attribution License v1.0",
        "input_leads": ["II", "V5"],
        "target_and_anchor_lead": "II",
        "sampling_rate_hz": FS_MODEL,
        "window_seconds": {"before_anchor": LEFT_SECONDS, "after_anchor": RIGHT_SECONDS},
        "resampling": "resample_poly from 500 Hz to 250 Hz, ratio 1/2",
        "normalization": "per-window, per-lead z-score",
        "transfer_localization_gate_passed": True,
        "transfer_summary": str(args.transfer_dir / "summary.json"),
        "donor_selection": "same-record MSE quantiles; each donor avoids overlap with the target window, donor windows may overlap one another",
        "donors_per_example": args.donors_per_example,
        "donor_window_overlap": {
            "definition": "each donor excludes overlap with the target; donors may overlap one another",
            "overlapping_donor_pairs": overlapping_donor_pair_count,
            "total_donor_pairs": donor_pair_count,
            "examples_with_any_overlapping_donor_pair": examples_with_overlapping_donor_pair,
        },
        "eligible_external_targets": len(target_rows),
        "targets_with_requested_donor_count": len(selected_rows),
        "lime_perturbations": args.perturbations,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "aggregation_definitions": {
            "equal_mean": "arithmetic mean of donor-wise L1-normalized absolute LIME coefficient vectors",
            "similarity_weighted": "weighted mean of donor-wise L1-normalized absolute LIME coefficient vectors; weights are softmax(-MSE / median donor MSE)",
            "component_median": "component-wise median of donor-wise L1-normalized absolute LIME coefficient vectors, then L1-normalized",
        },
        "aggregate": aggregate,
        "per_example": results,
        "per_example_aggregation": example_rows,
        "per_donor": donor_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for filename, rows in (("per-example.csv", example_rows), ("per-donor.csv", donor_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
