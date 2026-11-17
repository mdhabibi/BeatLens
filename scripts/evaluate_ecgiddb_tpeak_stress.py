"""Compare the frozen QTDB locator with ECG-ID's unaudited automated T-peak marks.

This is an exploratory agreement stress test, not an expert-reference accuracy
evaluation. ECG-ID provides raw and filtered versions of one lead as its two
channels, and labels its R/T annotations as unaudited automated detections.
"""

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
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor  # noqa: E402

TARGET_RATE_HZ = 250.0
SOURCE_RATE_HZ = 500.0
LEFT_SECONDS = 1.0
RIGHT_SECONDS = 1.5
WINDOW_SAMPLES = 625


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> tuple[int, list[str]]:
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise SystemExit(f"Missing checksum manifest: {manifest}")
    checked = 0
    mismatches: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            mismatches.append(relative)
        checked += 1
    return checked, mismatches


def cluster_bootstrap_ci(
    differences_by_person: dict[str, np.ndarray], *, seed: int, resamples: int
) -> tuple[float, float]:
    people = sorted(differences_by_person)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = rng.choice(people, size=len(people), replace=True)
        values[index] = np.mean(
            np.concatenate([differences_by_person[person] for person in selected])
        )
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def metrics(errors: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(errors)
    return {
        "n": int(errors.size),
        "mae_ms": float(absolute.mean()),
        "median_absolute_error_ms": float(np.median(absolute)),
        "p90_absolute_error_ms": float(np.quantile(absolute, 0.9)),
        "within_40ms": float(np.mean(absolute <= 40.0)),
        "mean_signed_error_ms": float(errors.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ecgiddb-1.0.0"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt"))
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--train-split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ecgiddb-tpeak-stress"))
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=647)
    args = parser.parse_args()

    record_lists = list(args.data_dir.rglob("RECORDS"))
    if len(record_lists) != 1:
        raise SystemExit(f"Expected one ECG-ID RECORDS file under {args.data_dir}")
    root = record_lists[0].parent
    checked_files, checksum_mismatches = verify_manifest(root)
    if checksum_mismatches:
        raise SystemExit(f"ECG-ID checksum mismatches: {checksum_mismatches[:10]}")
    if "unaudited" not in (root / "ANNOTATORS").read_text(encoding="utf-8").lower():
        raise SystemExit("Expected ECG-ID ANNOTATORS to disclose unaudited automated marks")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if float(checkpoint.get("sampling_rate_hz", TARGET_RATE_HZ)) != TARGET_RATE_HZ:
        raise SystemExit("Checkpoint input rate is not 250 Hz")
    if int(checkpoint.get("window_samples", WINDOW_SAMPLES)) != WINDOW_SAMPLES:
        raise SystemExit(f"Checkpoint window is not {WINDOW_SAMPLES} samples")
    model = QTDBTpeakRegressor(input_leads=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(min(8, torch.get_num_threads()))

    # Reuse the QTDB training-only prior; the QTDB test split is never loaded.
    _, _, _, prior_offset_samples, _ = prepare_split(
        args.qtdb_dir, args.train_split_csv, "train"
    )
    prior_ms = float(prior_offset_samples / TARGET_RATE_HZ * 1000.0)

    with (root / "RECORDS").open(encoding="utf-8") as stream:
        record_ids = [line.strip() for line in stream if line.strip()]
    per_beat: list[dict[str, Any]] = []
    excluded = defaultdict(int)
    for record_id in record_ids:
        record_path = root / record_id
        record = wfdb.rdrecord(str(record_path), physical=True)
        annotation = wfdb.rdann(str(record_path), "atr")
        if float(record.fs) != SOURCE_RATE_HZ:
            raise SystemExit(f"Unexpected sampling rate for {record_id}: {record.fs}")
        if list(record.sig_name or []) != ["ECG I", "ECG I filtered"]:
            raise SystemExit(f"Unexpected channel mapping for {record_id}: {record.sig_name}")

        anchors = np.asarray(
            [sample for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol == "N"],
            dtype=np.int64,
        )
        targets = np.asarray(
            [sample for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol == "t"],
            dtype=np.int64,
        )
        if len(anchors) != 10 or len(targets) != 10:
            excluded["records_not_exactly_10_N_and_10_t"] += 1
            continue
        target_by_anchor: dict[int, list[int]] = defaultdict(list)
        for target in targets:
            preceding = anchors[anchors < target]
            if not len(preceding):
                excluded["target_without_preceding_N"] += 1
                continue
            anchor = int(preceding[-1])
            if (target - anchor) / SOURCE_RATE_HZ > 1.25:
                excluded["target_N_gap_over_1_25s"] += 1
                continue
            target_by_anchor[anchor].append(int(target))

        for anchor, assigned_targets in target_by_anchor.items():
            if len(assigned_targets) != 1:
                excluded["anchor_without_unique_t_peak"] += 1
                continue
            start = anchor - round(SOURCE_RATE_HZ * LEFT_SECONDS)
            end = anchor + round(SOURCE_RATE_HZ * RIGHT_SECONDS)
            if start < 0 or end > record.sig_len:
                excluded["incomplete_2_5s_window"] += 1
                continue
            window_500 = np.asarray(record.p_signal[start:end, :2], dtype=np.float32)
            window_250 = resample_poly(window_500, up=1, down=2, axis=0).astype(np.float32)
            if window_250.shape != (WINDOW_SAMPLES, 2):
                raise SystemExit(f"Unexpected window shape for {record_id}: {window_250.shape}")
            means = window_250.mean(axis=0, keepdims=True)
            scales = window_250.std(axis=0, keepdims=True).clip(min=1e-5)
            model_input = ((window_250 - means) / scales).T[None].astype(np.float32)
            with torch.no_grad():
                normalized_location = float(model(torch.from_numpy(model_input))[0])
            predicted_sample = (normalized_location + 1.0) * (WINDOW_SAMPLES - 1) / 2.0
            prediction_ms = (
                (predicted_sample - round(TARGET_RATE_HZ * LEFT_SECONDS))
                / TARGET_RATE_HZ
                * 1000.0
            )
            target_sample = assigned_targets[0]
            pseudo_reference_ms = (target_sample - anchor) / SOURCE_RATE_HZ * 1000.0
            per_beat.append(
                {
                    "person_id": record_path.parent.name,
                    "record_id": record_id,
                    "anchor_sample_500hz": anchor,
                    "pseudo_t_peak_sample_500hz": target_sample,
                    "pseudo_t_peak_offset_ms": pseudo_reference_ms,
                    "model_prediction_ms": prediction_ms,
                    "model_signed_difference_ms": prediction_ms - pseudo_reference_ms,
                    "model_absolute_difference_ms": abs(prediction_ms - pseudo_reference_ms),
                    "qtdb_train_prior_ms": prior_ms,
                    "prior_signed_difference_ms": prior_ms - pseudo_reference_ms,
                    "prior_absolute_difference_ms": abs(prior_ms - pseudo_reference_ms),
                }
            )

    if not per_beat:
        raise SystemExit("No complete ECG-ID windows were eligible")
    differences_by_person: dict[str, list[float]] = defaultdict(list)
    for row in per_beat:
        differences_by_person[row["person_id"]].append(
            row["model_absolute_difference_ms"] - row["prior_absolute_difference_ms"]
        )
    person_differences = {
        person: np.asarray(values, dtype=np.float64)
        for person, values in differences_by_person.items()
    }
    model_errors = np.asarray([row["model_signed_difference_ms"] for row in per_beat])
    prior_errors = np.asarray([row["prior_signed_difference_ms"] for row in per_beat])
    paired_mae_difference = float(
        np.mean(
            [row["model_absolute_difference_ms"] - row["prior_absolute_difference_ms"] for row in per_beat]
        )
    )
    paired_ci = cluster_bootstrap_ci(
        person_differences, seed=args.seed, resamples=args.bootstrap_resamples
    )
    records_per_person: defaultdict[str, set[str]] = defaultdict(set)
    for row in per_beat:
        records_per_person[row["person_id"]].add(row["record_id"])

    summary = {
        "evaluation": "exploratory agreement with ECG-ID automated T-peak marks",
        "dataset": "ECG-ID v1.0.0",
        "license": "Open Data Commons Attribution License v1.0",
        "manifest_files_verified": checked_files,
        "checksum_mismatches": checksum_mismatches,
        "records_in_release": len(record_ids),
        "persons_in_release": len({record_id.split("/")[0] for record_id in record_ids}),
        "eligible_annotated_windows": len(per_beat),
        "persons_evaluated": len(person_differences),
        "records_evaluated": len({row["record_id"] for row in per_beat}),
        "excluded": dict(excluded),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_channels": ["ECG I raw", "ECG I filtered"],
        "target_anchor": "N annotation; paired preceding t annotation within 1.25 s",
        "signal_resampling": "scipy.signal.resample_poly, 500 Hz to 250 Hz, ratio 1/2",
        "window_seconds": {"before_anchor": LEFT_SECONDS, "after_anchor": RIGHT_SECONDS},
        "normalization": "per-window, per-channel z-score",
        "reference_provenance": "unaudited R/T peaks from an automated detector; not expert ground truth",
        "qtdb_train_only_timing_prior_ms": prior_ms,
        "model_agreement_with_automated_marks": metrics(model_errors),
        "timing_prior_agreement_with_automated_marks": metrics(prior_errors),
        "paired_model_minus_prior_mae_ms": paired_mae_difference,
        "paired_model_minus_prior_subject_cluster_bootstrap_95ci_ms": list(paired_ci),
        "bootstrap_seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "interpretation": "Agreement with ECG-ID's automated marks only; this is not T-peak accuracy or clinical validation.",
        "records_per_person_evaluated": {
            person: len(record_ids) for person, record_ids in sorted(records_per_person.items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for filename, rows in (("per-beat.csv", per_beat),):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
