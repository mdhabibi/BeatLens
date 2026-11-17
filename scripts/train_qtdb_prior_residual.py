"""Fit signal-based residuals around train-only QTDB event-timing priors."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beatlens.infrastructure.qtdb import QTDBBeatExample, TARGET_NAMES, iter_qtdb_examples
from beatlens.models.qtdb_offset_regressor import QTDBOffsetRegressor

MAX_SLOTS = 8
LEFT_SECONDS = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_train_prior(
    examples: list[QTDBBeatExample], max_slots: int
) -> tuple[np.ndarray, dict[str, Any]]:
    offsets_by_slot: dict[str, dict[int, list[int]]] = {
        name: defaultdict(list) for name in TARGET_NAMES
    }
    all_offsets: dict[str, list[int]] = {name: [] for name in TARGET_NAMES}
    for example in examples:
        for name in TARGET_NAMES:
            for slot, offset in enumerate(example.event_offsets[name]):
                offsets_by_slot[name][slot].append(offset)
                all_offsets[name].append(offset)

    prior = np.zeros((len(TARGET_NAMES), max_slots), dtype=np.float32)
    summary: dict[str, Any] = {}
    for target_index, name in enumerate(TARGET_NAMES):
        if not all_offsets[name]:
            raise ValueError(f"No training events for target {name}")
        fallback = float(np.mean(all_offsets[name]))
        summary[name] = {"fallback_mean_offset_samples": fallback, "slot_means": {}}
        for slot in range(max_slots):
            values = offsets_by_slot[name].get(slot, [])
            mean = float(np.mean(values)) if values else fallback
            prior[target_index, slot] = mean
            summary[name]["slot_means"][str(slot)] = {
                "mean_offset_samples": mean,
                "n_events": len(values),
            }
    return prior, summary


def prepare_split(
    qtdb_dir: Path,
    split_csv: Path,
    split: str,
    prior_offsets: np.ndarray,
    max_adjustment_samples: int,
) -> tuple[list[QTDBBeatExample], np.ndarray, np.ndarray, np.ndarray]:
    examples = list(iter_qtdb_examples(qtdb_dir, split_csv, split))
    if not examples:
        raise ValueError(f"The {split} split has no beat examples")
    signals = np.stack([example.signal for example in examples]).astype(np.float32)
    valid = np.stack([example.valid_samples for example in examples])
    residual_targets = np.zeros((len(examples), len(TARGET_NAMES), MAX_SLOTS), dtype=np.float32)
    observed = np.zeros_like(residual_targets, dtype=bool)
    for example_index, example in enumerate(examples):
        for target_index, name in enumerate(TARGET_NAMES):
            offsets = example.event_offsets[name]
            if len(offsets) > MAX_SLOTS:
                raise ValueError(f"More than {MAX_SLOTS} markers of target {name} in {example.record_id}")
            for slot, offset in enumerate(offsets):
                residual = (offset - prior_offsets[target_index, slot]) / max_adjustment_samples
                residual_targets[example_index, target_index, slot] = float(
                    np.clip(residual, -1.0, 1.0)
                )
                observed[example_index, target_index, slot] = True

    valid_float = valid[..., None].astype(np.float32)
    counts = np.maximum(valid_float.sum(axis=1, keepdims=True), 1.0)
    means = (signals * valid_float).sum(axis=1, keepdims=True) / counts
    variances = (((signals - means) ** 2) * valid_float).sum(axis=1, keepdims=True) / counts
    scales = np.sqrt(variances).clip(min=1e-5)
    normalized = ((signals - means) / scales) * valid_float
    return examples, normalized.transpose(0, 2, 1), residual_targets, observed


def masked_huber_loss(
    prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    losses = nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=0.02)
    mask = observed.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def localization_metrics(
    model: nn.Module,
    signals: np.ndarray,
    examples: list[QTDBBeatExample],
    prior_offsets: np.ndarray,
    max_adjustment_samples: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    normalized_residuals: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device)
            normalized_residuals.extend(model(batch).cpu().numpy())

    errors_by_target: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    for residual_output, example in zip(normalized_residuals, examples, strict=True):
        fs = example.sampling_rate_hz
        left = round(fs * LEFT_SECONDS)
        window_size = example.signal.shape[0]
        for target_index, name in enumerate(TARGET_NAMES):
            truth_offsets = example.event_offsets[name]
            count = len(truth_offsets)
            if not count:
                continue
            predicted_offsets = (
                prior_offsets[target_index, :count]
                + residual_output[target_index, :count] * max_adjustment_samples
            )
            predicted_positions = left + predicted_offsets
            truth_positions = np.asarray([left + offset for offset in truth_offsets])
            predicted_positions = np.clip(predicted_positions, 0, window_size - 1)
            row_indices, column_indices = linear_sum_assignment(
                np.abs(truth_positions[:, None] - predicted_positions[None, :])
            )
            errors_ms = (
                np.abs(truth_positions[row_indices] - predicted_positions[column_indices])
                / fs
                * 1000.0
            )
            errors_by_target[name].extend(errors_ms.tolist())

    def summarize(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n_events": 0, "mae_ms": None, "median_ms": None, "within_40ms": None}
        array = np.asarray(values)
        return {
            "n_events": int(array.size),
            "mae_ms": float(array.mean()),
            "median_ms": float(np.median(array)),
            "within_40ms": float(np.mean(array <= 40.0)),
        }

    all_errors = [value for values in errors_by_target.values() for value in values]
    return {
        "protocol": "oracle event count per annotated beat/target; train-prior residual localization",
        "overall": summarize(all_errors),
        "by_target": {name: summarize(values) for name, values in errors_by_target.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--prior-json", type=Path, default=Path("outputs/qtdb-baseline/timing-prior.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-prior-residual"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-adjustment-ms", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    train_examples = list(iter_qtdb_examples(args.qtdb_dir, args.split_csv, "train"))
    prior_offsets, prior_summary = build_train_prior(train_examples, MAX_SLOTS)
    fs = train_examples[0].sampling_rate_hz
    max_adjustment_samples = round(args.max_adjustment_ms * fs / 1000.0)
    if max_adjustment_samples < 1:
        raise ValueError("max-adjustment-ms must be positive")
    train_examples, train_x, train_y, train_mask = prepare_split(
        args.qtdb_dir,
        args.split_csv,
        "train",
        prior_offsets,
        max_adjustment_samples,
    )
    validation_examples, validation_x, validation_y, validation_mask = prepare_split(
        args.qtdb_dir,
        args.split_csv,
        "validation",
        prior_offsets,
        max_adjustment_samples,
    )
    prior_validation_mae = float(
        json.loads(args.prior_json.read_text(encoding="utf-8"))["validation"]["overall"]["mae_ms"]
    )

    train_data = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_mask)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = QTDBOffsetRegressor(input_leads=train_x.shape[1], max_slots=MAX_SLOTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y, batch_mask = (
                batch_x.to(device),
                batch_y.to(device),
                batch_mask.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_huber_loss(model(batch_x), batch_y, batch_mask)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        validation_metrics = localization_metrics(
            model,
            validation_x,
            validation_examples,
            prior_offsets,
            max_adjustment_samples,
            device,
            args.batch_size,
        )
        validation_mae = validation_metrics["overall"]["mae_ms"]
        if validation_mae is None:
            raise RuntimeError("No observed markers found in validation split")
        epoch_result: dict[str, float | int] = {
            "epoch": epoch,
            "train_huber_loss": total_loss / max(len(train_loader), 1),
            "validation_localization_mae_ms": float(validation_mae),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)

        if float(validation_mae) < best_validation_mae:
            best_validation_mae = float(validation_mae)
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "target_names": TARGET_NAMES,
                    "max_slots": MAX_SLOTS,
                    "max_adjustment_samples": max_adjustment_samples,
                    "prior_offsets": prior_offsets,
                    "normalization": "per-window, per-lead z-score",
                    "sampling_rate_hz": fs,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = localization_metrics(
        model,
        validation_x,
        validation_examples,
        prior_offsets,
        max_adjustment_samples,
        device,
        args.batch_size,
    )
    selected = float(validation_metrics["overall"]["mae_ms"]) < prior_validation_mae
    result: dict[str, Any] = {
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_records": len({example.record_id for example in train_examples}),
        "validation_records": len({example.record_id for example in validation_examples}),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "max_adjustment_ms": args.max_adjustment_ms,
        "validation": validation_metrics,
        "validation_timing_prior_mae_ms": prior_validation_mae,
        "beats_validation_prior": selected,
        "train_prior": prior_summary,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "test": None,
    }

    # In this run, load the test split only if validation beats the timing prior.
    if selected:
        test_examples, test_x, _, _ = prepare_split(
            args.qtdb_dir,
            args.split_csv,
            "test",
            prior_offsets,
            max_adjustment_samples,
        )
        result["test_records"] = len({example.record_id for example in test_examples})
        result["test_examples"] = len(test_examples)
        result["test"] = localization_metrics(
            model,
            test_x,
            test_examples,
            prior_offsets,
            max_adjustment_samples,
            device,
            args.batch_size,
        )

    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "validation_mae_ms": validation_metrics["overall"]["mae_ms"],
                "validation_prior_mae_ms": prior_validation_mae,
                "beats_validation_prior": selected,
                "test_mae_ms": (
                    result["test"]["overall"]["mae_ms"] if result["test"] else None
                ),
                "metrics_file": str(result_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
