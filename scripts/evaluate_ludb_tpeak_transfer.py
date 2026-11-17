"""Run a frozen QTDB T-peak checkpoint on the audited LUDB external cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_qtdb_tpeak import prepare_split  # noqa: E402
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor

LEFT_SECONDS = 1.0
TARGET_RATE_HZ = 250.0
EXPECTED_WINDOW_SAMPLES = 625


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(errors: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(errors.size),
        "mae_ms": float(np.mean(np.abs(errors))),
        "median_absolute_error_ms": float(np.median(np.abs(errors))),
        "p90_absolute_error_ms": float(np.quantile(np.abs(errors), 0.9)),
        "within_40ms": float(np.mean(np.abs(errors) <= 40.0)),
        "mean_signed_error_ms": float(np.mean(errors)),
    }


def cluster_bootstrap_mean_ci(
    values_by_record: dict[str, np.ndarray], *, seed: int, resamples: int
) -> tuple[float, float]:
    record_ids = sorted(values_by_record)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.choice(record_ids, size=len(record_ids), replace=True)
        bootstrap_means[index] = np.mean(np.concatenate([values_by_record[key] for key in chosen]))
    return tuple(float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/ludb-1.0.1")
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("outputs/ludb-external-audit")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument(
        "--train-split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv")
    )
    parser.add_argument(
        "--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ludb-tpeak-transfer")
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=544)
    args = parser.parse_args()

    release_manifests = list(args.data_dir.rglob("RECORDS"))
    if len(release_manifests) != 1:
        raise SystemExit(f"Expected one LUDB RECORDS file under {args.data_dir}")
    release_root = release_manifests[0].parent
    audit_summary = json.loads((args.audit_dir / "summary.json").read_text(encoding="utf-8"))
    if audit_summary["checksum_mismatches"]:
        raise SystemExit("The LUDB audit reported checksum mismatches")
    targets_path = args.audit_dir / "targets.csv"
    with targets_path.open(newline="", encoding="utf-8") as stream:
        target_rows = list(csv.DictReader(stream))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if float(checkpoint.get("sampling_rate_hz", TARGET_RATE_HZ)) != TARGET_RATE_HZ:
        raise SystemExit("Checkpoint input rate is not 250 Hz")
    if int(checkpoint.get("window_samples", EXPECTED_WINDOW_SAMPLES)) != EXPECTED_WINDOW_SAMPLES:
        raise SystemExit("Checkpoint window is not 625 samples")

    model = QTDBTpeakRegressor(input_leads=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    # Reuse the exact training-only eligible examples and prior definition from training.
    _, _, _, prior_offset_samples, _ = prepare_split(
        args.qtdb_dir, args.train_split_csv, "train"
    )
    timing_prior_ms = float(prior_offset_samples / TARGET_RATE_HZ * 1000.0)

    record_cache: dict[str, Any] = {}
    per_beat = []
    left_samples_500 = 500
    right_samples_500 = 750
    for row in target_rows:
        record_id = row["record_id"]
        if record_id not in record_cache:
            record = wfdb.rdrecord(str(release_root / record_id), physical=True)
            lead_names = [name.strip().lower() for name in (record.sig_name or [])]
            if float(record.fs) != 500.0:
                raise SystemExit(f"Unexpected sample rate for {record_id}: {record.fs}")
            if "ii" not in lead_names or "v5" not in lead_names:
                raise SystemExit(f"Expected lead II and V5 in {record_id}")
            record_cache[record_id] = (record, [lead_names.index("ii"), lead_names.index("v5")])
        record, lead_indexes = record_cache[record_id]
        anchor = int(row["anchor_sample_500hz"])
        start = anchor - left_samples_500
        end = anchor + right_samples_500
        if start < 0 or end > record.p_signal.shape[0]:
            raise SystemExit(f"Audit window is not valid for {record_id} anchor {anchor}")
        window_500 = np.asarray(record.p_signal[start:end, lead_indexes], dtype=np.float32)
        window_250 = resample_poly(window_500, up=1, down=2, axis=0).astype(np.float32)
        if window_250.shape != (EXPECTED_WINDOW_SAMPLES, 2):
            raise SystemExit(f"Unexpected resampled shape for {record_id}: {window_250.shape}")
        means = window_250.mean(axis=0, keepdims=True)
        scales = window_250.std(axis=0, keepdims=True).clip(min=1e-5)
        normalized = ((window_250 - means) / scales).T[None].astype(np.float32)
        with torch.no_grad():
            normalized_location = float(model(torch.from_numpy(normalized))[0])
        predicted_sample = (normalized_location + 1.0) * (EXPECTED_WINDOW_SAMPLES - 1) / 2.0
        prediction_ms = (predicted_sample - round(TARGET_RATE_HZ * LEFT_SECONDS)) / TARGET_RATE_HZ * 1000.0
        reference_ms = float(row["t_peak_offset_ms"])
        per_beat.append(
            {
                "record_id": record_id,
                "anchor_sample_500hz": anchor,
                "reference_t_peak_offset_ms": reference_ms,
                "prediction_ms": prediction_ms,
                "signed_error_ms": prediction_ms - reference_ms,
                "absolute_error_ms": abs(prediction_ms - reference_ms),
                "train_prior_prediction_ms": timing_prior_ms,
                "train_prior_signed_error_ms": timing_prior_ms - reference_ms,
                "train_prior_absolute_error_ms": abs(timing_prior_ms - reference_ms),
                "target_nonoverlap_donor_candidates": int(row["target_nonoverlap_donor_candidates"]),
            }
        )

    errors = np.asarray([row["signed_error_ms"] for row in per_beat], dtype=np.float64)
    prior_errors = np.asarray([row["train_prior_signed_error_ms"] for row in per_beat], dtype=np.float64)
    errors_by_record: dict[str, list[float]] = defaultdict(list)
    for row in per_beat:
        errors_by_record[row["record_id"]].append(
            row["absolute_error_ms"] - row["train_prior_absolute_error_ms"]
        )
    paired_differences = {key: np.asarray(value) for key, value in errors_by_record.items()}
    # The gate is based on paired absolute-error difference, not signed bias.
    difference = float(
        np.mean([row["absolute_error_ms"] - row["train_prior_absolute_error_ms"] for row in per_beat])
    )
    difference_ci = cluster_bootstrap_mean_ci(
        paired_differences, seed=args.seed, resamples=args.bootstrap_resamples
    )
    records = sorted({row["record_id"] for row in per_beat})
    per_record = []
    for record_id in records:
        rows = [row for row in per_beat if row["record_id"] == record_id]
        model_record = metrics(np.asarray([row["signed_error_ms"] for row in rows]))
        prior_record = metrics(np.asarray([row["train_prior_signed_error_ms"] for row in rows]))
        per_record.append(
            {
                "record_id": record_id,
                "n_targets": len(rows),
                "model_mae_ms": model_record["mae_ms"],
                "train_prior_mae_ms": prior_record["mae_ms"],
                "model_minus_prior_mae_ms": model_record["mae_ms"] - prior_record["mae_ms"],
            }
        )

    summary = {
        "evaluation": "zero-shot QTDB checkpoint transfer to LUDB",
        "dataset": "LUDB v1.0.1",
        "records": len(records),
        "subjects": len(records),
        "targets": len(per_beat),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checksum(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_leads": ["II", "V5"],
        "target_and_anchor_lead": "II",
        "input_resampling": "scipy.signal.resample_poly, 500 Hz to 250 Hz, ratio 1/2",
        "window_seconds": {"before_anchor": 1.0, "after_anchor": 1.5},
        "normalization": "per-window, per-lead z-score",
        "train_only_timing_prior_ms": timing_prior_ms,
        "model": metrics(errors),
        "train_only_timing_prior": metrics(prior_errors),
        "paired_model_minus_prior_mae_ms": difference,
        "paired_model_minus_prior_record_cluster_bootstrap_95ci_ms": list(difference_ci),
        "localization_gate_passed": difference < 0 and difference_ci[1] < 0,
        "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "limitation": "QTDB training records have heterogeneous channel names; this fixed II/V5 transfer is a domain-shift challenge, not a clean lead-matched validation.",
        "per_record": per_record,
        "per_beat": per_beat,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for filename, rows in (("per-beat.csv", per_beat), ("per-record.csv", per_record)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"per_record", "per_beat"}},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
