"""Audit LUDB files and T-peak window/donor coverage for external validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import wfdb


def find_release_root(data_dir: Path) -> Path:
    manifests = list(data_dir.rglob("RECORDS"))
    if len(manifests) != 1:
        raise SystemExit(f"Expected one LUDB RECORDS file under {data_dir}, found {len(manifests)}")
    return manifests[0].parent


def verify_checksums(root: Path) -> tuple[int, list[str]]:
    manifest = root / "SHA256SUMS.txt"
    mismatches = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checked += 1
        if digest != expected:
            mismatches.append(relative.strip())
    return checked, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ludb-1.0.1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ludb-external-audit"))
    parser.add_argument("--before-seconds", type=float, default=1.0)
    parser.add_argument("--after-seconds", type=float, default=1.5)
    parser.add_argument("--max-tpeak-gap-seconds", type=float, default=1.25)
    args = parser.parse_args()

    root = find_release_root(args.data_dir)
    checked, mismatches = verify_checksums(root)
    records = [line.strip() for line in (root / "RECORDS").read_text().splitlines() if line.strip()]
    metadata = list(csv.DictReader((root / "ludb.csv").open(newline="", encoding="utf-8")))
    if mismatches:
        raise SystemExit(f"Checksum mismatch in {len(mismatches)} files: {mismatches[:10]}")

    per_record = []
    targets = []
    fs_counts: Counter[str] = Counter()
    lead_pairs: Counter[str] = Counter()
    ii_symbols: Counter[str] = Counter()
    v5_symbols: Counter[str] = Counter()

    for record_id in records:
        record_name = str(root / record_id)
        record = wfdb.rdrecord(record_name, physical=True)
        fs = float(record.fs)
        fs_counts[str(fs)] += 1
        lead_names = [name.strip().lower() for name in (record.sig_name or [])]
        if "ii" not in lead_names or "v5" not in lead_names:
            raise SystemExit(f"{record_id} is missing required leads II/V5: {lead_names}")
        lead_pairs["/".join(lead_names)] += 1

        ii = wfdb.rdann(record_name, "ii")
        v5 = wfdb.rdann(record_name, "v5")
        ii_symbols.update(ii.symbol)
        v5_symbols.update(v5.symbol)
        anchors = sorted(
            int(sample)
            for sample, symbol in zip(ii.sample, ii.symbol, strict=True)
            if symbol == "N"
        )
        t_peaks = sorted(
            int(sample)
            for sample, symbol in zip(ii.sample, ii.symbol, strict=True)
            if symbol == "t"
        )
        assigned: dict[int, list[int]] = defaultdict(list)
        for t_peak in t_peaks:
            preceding = [anchor for anchor in anchors if anchor <= t_peak]
            if preceding and t_peak - preceding[-1] <= round(args.max_tpeak_gap_seconds * fs):
                assigned[preceding[-1]].append(t_peak)

        left = round(args.before_seconds * fs)
        right = round(args.after_seconds * fs)
        valid_anchors = [
            anchor
            for anchor in anchors
            if anchor >= left and anchor + right <= record.p_signal.shape[0]
        ]
        record_target_rows = []
        for anchor, peaks in sorted(assigned.items()):
            if len(peaks) != 1 or anchor not in valid_anchors:
                continue
            donor_anchors = sorted(
                other
                for other in valid_anchors
                if other != anchor and abs(other - anchor) >= left + right
            )
            # Maximum interval packing when donor windows must also avoid each other.
            packed: list[int] = []
            for donor_anchor in donor_anchors:
                if not packed or donor_anchor - packed[-1] >= left + right:
                    packed.append(donor_anchor)
            row = {
                "record_id": record_id,
                "anchor_sample_500hz": anchor,
                "t_peak_sample_500hz": peaks[0],
                "t_peak_offset_ms": (peaks[0] - anchor) / fs * 1000.0,
                "target_nonoverlap_donor_candidates": len(donor_anchors),
                "mutually_nonoverlap_donor_capacity": len(packed),
            }
            targets.append(row)
            record_target_rows.append(row)

        per_record.append(
            {
                "record_id": record_id,
                "sampling_rate_hz": fs,
                "n_samples": int(record.p_signal.shape[0]),
                "lead_count": len(lead_names),
                "lead_ii_n_anchors": len(anchors),
                "lead_ii_t_peaks": len(t_peaks),
                "valid_full_windows": len(valid_anchors),
                "eligible_single_t_peak_targets": len(record_target_rows),
                "targets_with_3_or_more_target_nonoverlap_donors": sum(
                    row["target_nonoverlap_donor_candidates"] >= 3 for row in record_target_rows
                ),
                "targets_with_5_or_more_target_nonoverlap_donors": sum(
                    row["target_nonoverlap_donor_candidates"] >= 5 for row in record_target_rows
                ),
                "targets_with_2_or_more_mutually_nonoverlap_donors": sum(
                    row["mutually_nonoverlap_donor_capacity"] >= 2 for row in record_target_rows
                ),
            }
        )

    donor_candidate_counts = [row["target_nonoverlap_donor_candidates"] for row in targets]
    mutual_capacities = [row["mutually_nonoverlap_donor_capacity"] for row in targets]
    summary = {
        "dataset": "Lobachevsky University Electrocardiography Database",
        "version": "1.0.1",
        "license": "Open Data Commons Attribution License v1.0",
        "records": len(records),
        "metadata_rows": len(metadata),
        "metadata_subjects": len({row["ID"].strip() for row in metadata}),
        "checksum_files_verified": checked,
        "checksum_mismatches": mismatches,
        "sampling_rates_hz": dict(fs_counts),
        "lead_pair_inventory": dict(lead_pairs),
        "lead_ii_annotation_symbols": dict(ii_symbols),
        "lead_v5_annotation_symbols": dict(v5_symbols),
        "primary_leads": ["ii", "v5"],
        "anchor_symbol": "N in lead II",
        "target_symbol": "t in lead II, assigned to the nearest preceding N within the configured gap",
        "window_seconds": {"before_anchor": args.before_seconds, "after_anchor": args.after_seconds},
        "max_t_peak_anchor_gap_seconds": args.max_tpeak_gap_seconds,
        "eligible_single_t_peak_targets": len(targets),
        "records_with_eligible_targets": sum(row["eligible_single_t_peak_targets"] > 0 for row in per_record),
        "targets_with_at_least_3_donors_each_nonoverlapping_with_target": sum(
            count >= 3 for count in donor_candidate_counts
        ),
        "targets_with_at_least_5_donors_each_nonoverlapping_with_target": sum(
            count >= 5 for count in donor_candidate_counts
        ),
        "targets_with_at_least_2_mutually_nonoverlapping_donors": sum(
            count >= 2 for count in mutual_capacities
        ),
        "maximum_mutually_nonoverlapping_donors": max(mutual_capacities, default=0),
        "donor_window_definition": "each donor window is non-overlapping with the explained target window; donor windows may overlap one another, matching the existing QTDB implementation",
        "per_record": per_record,
        "targets": targets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per-record.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_record[0]))
        writer.writeheader()
        writer.writerows(per_record)
    with (args.output_dir / "targets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_record", "targets"}}, indent=2))


if __name__ == "__main__":
    main()
