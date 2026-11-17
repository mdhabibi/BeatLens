"""Audit QTDB annotation coverage and source-record grouping.

Run with:
    uv run --with wfdb --no-project python scripts/audit_qtdb.py [QTDB_DIR]

The script reads the original downloaded files and writes only summary CSVs; it never copies ECG
signals or annotations into the output directory.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


class SourceTableParser(HTMLParser):
    """Read QTDB's source-record and annotated-beat table from its bundled documentation."""

    def __init__(self, record_ids: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.record_ids = record_ids
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.cell: list[str] = []
        self.row: list[str] = []
        self.source_database: str | None = None
        self.records: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.row = []
        elif self.in_row and tag in {"td", "th"}:
            self.in_cell = True
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.in_cell and tag in {"td", "th"}:
            self.row.append(" ".join("".join(self.cell).split()))
            self.in_cell = False
            self.cell = []
        elif self.in_row and tag == "tr":
            self._finish_row()
            self.in_row = False
            self.row = []
        elif tag == "table":
            self.in_table = False

    def _finish_row(self) -> None:
        if not self.row:
            return
        heading = re.match(r"Table 2\.\s*[A-Z]: Records from (.+)", self.row[0])
        if heading:
            self.source_database = heading.group(1).strip()
            return
        record_id = self.row[0]
        if record_id not in self.record_ids or len(self.row) < 3:
            return
        source_record_id = record_id[3:] if record_id.startswith("sel") else record_id
        if record_id.startswith("sele"):
            source_record_id = record_id[3:]
        self.records[record_id] = {
            "source_database": self.source_database or "unknown",
            "source_record_id": source_record_id,
            "source_interval": self.row[1],
            "manually_annotated_beats_in_table": self.row[2],
        }


def _count_annotations(wfdb, record_base: Path, extension: str) -> Counter[str] | None:
    annotation_path = record_base.with_suffix(f".{extension}")
    if not annotation_path.is_file():
        return None
    annotation = wfdb.rdann(str(record_base), extension)
    symbols = annotation.symbol if annotation.symbol is not None else []
    return Counter(symbols)


def audit(qtdb_dir: Path, output_dir: Path) -> None:
    try:
        import wfdb
    except ImportError as error:
        raise SystemExit(
            "WFDB is required. Run: uv run --with wfdb --no-project "
            "python scripts/audit_qtdb.py [QTDB_DIR]"
        ) from error

    records_file = qtdb_dir / "RECORDS"
    source_doc = qtdb_dir / "doc" / "node4.html"
    if not records_file.is_file() or not source_doc.is_file():
        raise SystemExit(f"Expected RECORDS and doc/node4.html under {qtdb_dir}")

    record_ids = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
    parser = SourceTableParser(set(record_ids))
    parser.feed(source_doc.read_text(errors="replace"))

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    aggregate_symbols: Counter[tuple[str, str]] = Counter()
    missing_q1c: list[str] = []

    for record_id in record_ids:
        record_base = qtdb_dir / record_id
        source = parser.records.get(record_id, {})
        header = wfdb.rdheader(str(record_base))
        q1c = _count_annotations(wfdb, record_base, "q1c")
        q2c = _count_annotations(wfdb, record_base, "q2c")
        man = _count_annotations(wfdb, record_base, "man")
        if q1c is None:
            missing_q1c.append(record_id)
        for annotator, counts in (("q1c", q1c), ("q2c", q2c)):
            if counts is not None:
                for symbol, count in counts.items():
                    aggregate_symbols[(annotator, symbol)] += count

        rows.append(
            {
                "qtdb_record": record_id,
                "source_database": source.get("source_database", "unknown"),
                "source_record_id": source.get("source_record_id", "unknown"),
                "source_interval": source.get("source_interval", "unknown"),
                "annotated_beats_in_source_table": source.get(
                    "manually_annotated_beats_in_table", "unknown"
                ),
                "sampling_rate_hz": header.fs,
                "signal_names": "|".join(header.sig_name or []),
                "signal_count": header.n_sig,
                "q1c_annotation_count": sum(q1c.values()) if q1c is not None else "missing",
                "q1c_symbol_counts": ";".join(
                    f"{symbol}:{count}" for symbol, count in sorted((q1c or {}).items())
                ),
                "q2c_annotation_count": sum(q2c.values()) if q2c is not None else "missing",
                "man_annotation_count": sum(man.values()) if man is not None else "missing",
                "has_atr": str(record_base.with_suffix(".atr").is_file()).lower(),
            }
        )

    details_path = output_dir / "qtdb-record-audit.csv"
    with details_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    symbol_path = output_dir / "qtdb-annotation-symbol-counts.csv"
    with symbol_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["annotation_file", "symbol", "count"])
        for (annotator, symbol), count in sorted(aggregate_symbols.items()):
            writer.writerow([annotator, symbol, count])

    groups = Counter(
        f"{row['source_database']}::{row['source_record_id']}" for row in rows
    )
    q1c_records = len(record_ids) - len(missing_q1c)
    print(f"QTDB records listed: {len(record_ids)}")
    print(f"Records with final audited q1c annotations: {q1c_records}")
    print(f"Records with second-expert q2c annotations: {sum(row['q2c_annotation_count'] != 'missing' for row in rows)}")
    print(f"Source-record groups: {len(groups)}")
    print(f"Records without a source-table mapping: {sum(row['source_database'] == 'unknown' for row in rows)}")
    if missing_q1c:
        print("Records missing q1c: " + ", ".join(missing_q1c))
    print(f"Wrote {details_path}")
    print(f"Wrote {symbol_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "qtdb_dir",
        nargs="?",
        type=Path,
        default=Path("data/qt-database-1.0.0"),
        help="extracted QTDB v1.0.0 directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qtdb-audit"),
        help="directory for the summary CSV files",
    )
    args = parser.parse_args()
    audit(args.qtdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
