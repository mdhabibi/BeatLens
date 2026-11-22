"""Export the audited LUDB II/V5 windows for locked OpenTSLM evaluation."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly


def prepare(output, data_dir, audit_dir):
    manifests = list(data_dir.rglob("RECORDS"))
    if len(manifests) != 1:
        raise ValueError("Expected one LUDB RECORDS manifest")
    summary = json.loads((audit_dir / "summary.json").read_text())
    if summary["checksum_mismatches"]:
        raise ValueError("LUDB audit reported checksum mismatches")
    records = {}
    windows = []
    with (audit_dir / "targets.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            record_id = row["record_id"]
            if record_id not in records:
                record = wfdb.rdrecord(str(manifests[0].parent / record_id), physical=True)
                leads = [name.strip().lower() for name in record.sig_name]
                if record.fs != 500:
                    raise ValueError(f"Unexpected sampling rate for {record_id}")
                records[record_id] = record.p_signal[:, [leads.index("ii"), leads.index("v5")]]
            anchor = int(row["anchor_sample_500hz"])
            raw = records[record_id][anchor - 500:anchor + 750].astype(np.float32)
            if raw.shape != (1250, 2):
                raise ValueError(f"Invalid audited window for {record_id}")
            waveform = resample_poly(raw, 1, 2, axis=0).astype(np.float32)
            normalized = (waveform - waveform.mean(axis=0)) / waveform.std(axis=0).clip(min=1e-5)
            windows.append({
                "record": record_id, "anchor": 250, "fs": 250,
                "reference": 250 + float(row["t_peak_offset_ms"]) / 4,
                "signal": normalized.T.tolist(),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as stream:
        json.dump(windows, stream, allow_nan=False)
    print(f"Exported {len(windows)} audited LUDB windows from {len(records)} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ludb-1.0.1"))
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/ludb-external-audit"))
    args = parser.parse_args()
    prepare(args.output, args.data_dir, args.audit_dir)
