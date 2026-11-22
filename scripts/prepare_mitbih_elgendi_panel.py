"""Prepare a bounded MIT-BIH pilot from Elgendi et al.'s MATLAB archive."""

import argparse
import gzip
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

QTDB_OVERLAP = set("100 102 103 104 114 116 117 123 213 221 223 230 231 232 233".split())


def prepare(archive, output):
    windows = []
    with ZipFile(archive) as source:
        members = sorted(name for name in source.namelist() if name.endswith(".mat"))
        records = [Path(name).stem.split("_")[-1] for name in members]
        if len(records) != 48 or len(set(records)) != 48 or not QTDB_OVERLAP <= set(records):
            raise ValueError("Unexpected MIT-BIH archive inventory")
        for record, name in zip(records, members, strict=True):
            if record in QTDB_OVERLAP:
                continue
            mat = loadmat(BytesIO(source.read(name)))
            waveform = np.column_stack([mat[key].ravel() for key in ("ECG_1", "ECG_2")])
            waveform = resample_poly(waveform.astype(np.float32), 25, 36, axis=0)
            # MATLAB sample indices are treated as one-based.
            anchors = np.unique(mat["R_Peaks"].ravel().astype(np.int64)) - 1
            peaks = np.unique(mat["T_Peaks_Annot"].ravel().astype(np.int64)) - 1
            eligible = []
            for anchor, next_anchor in zip(anchors, anchors[1:]):
                lo, hi = np.searchsorted(peaks, [anchor + 1, next_anchor])
                center = round(anchor * 25 / 36)
                if hi != lo + 1 or center < 250 or center + 375 > len(waveform):
                    continue
                reference = 250 + peaks[lo] * 25 / 36 - center
                if reference < 625:
                    eligible.append((center, reference))
            if not eligible:
                raise ValueError(f"No eligible windows for record {record}")
            # Even coverage across the recording, with a bounded pilot workload.
            indices = np.linspace(0, len(eligible) - 1, min(64, len(eligible)), dtype=int)
            for index in indices:
                center, reference = eligible[index]
                signal = waveform[center - 250:center + 375]
                scale = signal.std(axis=0).clip(min=1e-5)
                signal = (signal - signal.mean(axis=0)) / scale
                windows.append({
                    "record": record, "subject": "201" if record == "202" else record,
                    "anchor": 250, "fs": 250, "reference": reference,
                    "signal": signal.T.tolist(),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as stream:
        json.dump(windows, stream, allow_nan=False)
    print(f"Exported {len(windows)} windows from {len({w['record'] for w in windows})} records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output)
