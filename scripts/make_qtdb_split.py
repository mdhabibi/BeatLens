"""Create a deterministic QTDB split that keeps documented shared subjects together.

Run after scripts/audit_qtdb.py. The generated split stays in ignored outputs/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


# PhysioNet documents these repeated-record groups in the European ST-T
# Database. Group every member even when only some excerpts occur in QTDB.
EUROPEAN_STT_SHARED_SUBJECTS = (
    {"e0118", "e0119", "e0120", "e0121", "e0122"},
    {"e0123", "e0124", "e0125", "e0126"},
    {"e0129", "e0133"},
    {"e0136", "e0139"},
    {"e0147", "e0148"},
    {"e0154", "e0155"},
    {"e0162", "e0163"},
)


def _split_group(row: dict[str, str]) -> tuple[str, str]:
    source_database = row["source_database"]
    source_record_id = row["source_record_id"]
    if source_database == "European ST-T Database":
        for subject_records in EUROPEAN_STT_SHARED_SUBJECTS:
            if source_record_id in subject_records:
                group_label = "-".join(sorted(subject_records))
                return (
                    f"{source_database}::documented-subject::{group_label}",
                    "documented_shared_subject",
                )
    return (
        f"{source_database}::source-record::{source_record_id}",
        "source_record_only_global_subject_unverified",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/qtdb-audit/qtdb-record-audit.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/qtdb-audit/qtdb-split.csv"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.audit_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"No audit rows found in {args.audit_csv}; run audit_qtdb.py first")

    rows_by_group: dict[str, list[dict[str, str]]] = {}
    source_by_group: dict[str, str] = {}
    for row in rows:
        group_id, _ = _split_group(row)
        rows_by_group.setdefault(group_id, []).append(row)
        source_by_group[group_id] = row["source_database"]

    group_ids = list(rows_by_group)
    strata = [source_by_group[group_id] for group_id in group_ids]
    train_validation_groups, test_groups, train_validation_strata, _ = train_test_split(
        group_ids,
        strata,
        test_size=0.20,
        random_state=args.seed,
        stratify=strata,
    )
    train_groups, validation_groups = train_test_split(
        train_validation_groups,
        test_size=0.1875,  # 16 of the remaining 84 records: 65/15/20 overall.
        random_state=args.seed,
        stratify=train_validation_strata,
    )

    split_by_group = {
        group_id: split
        for split, split_groups in (
            ("train", train_groups),
            ("validation", validation_groups),
            ("test", test_groups),
        )
        for group_id in split_groups
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qtdb_record",
        "source_database",
        "source_record_id",
        "split_group_id",
        "split_group_basis",
        "split",
        "sampling_rate_hz",
        "signal_names",
        "q1c_annotation_count",
        "q1c_symbol_counts",
        "has_q2c",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["qtdb_record"]):
            group_id, group_basis = _split_group(row)
            writer.writerow(
                {
                    **{
                        field: row.get(field, "")
                        for field in fields
                        if field not in {"split_group_id", "split_group_basis", "split", "has_q2c"}
                    },
                    "split_group_id": group_id,
                    "split_group_basis": group_basis,
                    "split": split_by_group[group_id],
                    "has_q2c": str(row["q2c_annotation_count"] != "missing").lower(),
                }
            )

    print(f"Wrote {args.output_csv}")
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if split_by_group[_split_group(row)[0]] == split]
        group_count = len({_split_group(row)[0] for row in selected})
        print(f"{split}: {len(selected)} records, {group_count} split groups")


if __name__ == "__main__":
    main()
