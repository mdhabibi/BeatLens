"""Export ECG-ID windows for a locked single-lead domain-shift comparison."""

import argparse
import gzip
import json
from pathlib import Path, PurePosixPath

import numpy as np
import wfdb
from scipy.signal import resample_poly

DATABASE = "ecgiddb/1.0.0"


def record_windows(record_id, raw, samples, symbols):
    subject = PurePosixPath(record_id).parent.name
    peaks = list(zip(samples.tolist(), symbols, strict=True))
    for (anchor, beat), (peak, wave) in zip(peaks, peaks[1:]):
        if beat != "N" or wave != "t":
            continue
        if not (500 <= anchor <= len(raw) - 750 and anchor < peak < anchor + 750):
            continue
        signal = resample_poly(raw[anchor - 500:anchor + 750], 1, 2).astype(np.float32)
        if not np.isfinite(signal).all() or signal.std() < 1e-5:
            continue
        signal = ((signal - signal.mean()) / signal.std()).tolist()
        yield {
            "record": record_id, "subject": subject, "anchor": 250, "fs": 250,
            "reference": 250 + (peak - anchor) / 2, "signal": [signal, signal],
        }


def prepare(output):
    windows = []
    records = wfdb.get_record_list(DATABASE)
    if len(records) != 310:
        raise ValueError(f"Expected 310 ECG-ID records, found {len(records)}")
    for number, record_id in enumerate(records, start=1):
        path = PurePosixPath(record_id)
        directory = f"{DATABASE}/{path.parent}"
        record = wfdb.rdrecord(path.name, pn_dir=directory, channels=[0])
        if record.fs != 500 or record.sig_name[0].strip().lower() != "ecg i":
            raise ValueError(f"Unexpected ECG-ID signal format for {record_id}: {record.sig_name}")
        annotation = wfdb.rdann(path.name, "atr", pn_dir=directory)
        windows.extend(record_windows(record_id, record.p_signal[:, 0],
                                      annotation.sample, annotation.symbol))
        if number % 25 == 0:
            print(f"Read {number}/{len(records)} records", flush=True)
    if not windows:
        raise ValueError("No eligible ECG-ID windows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as stream:
        json.dump(windows, stream, allow_nan=False)
    print(f"Exported {len(windows)} windows from {len({w['record'] for w in windows})} records "
          f"and {len({w['subject'] for w in windows})} participants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
