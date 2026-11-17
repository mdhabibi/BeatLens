"""Train and evaluate the first masked QTDB fiducial-localization CNN."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beatlens.infrastructure.qtdb import TARGET_NAMES, iter_qtdb_examples
from beatlens.models.qtdb_cnn import QTDBLocalizationCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(
    qtdb_dir: Path, split_csv: Path, split: str
) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    examples = list(iter_qtdb_examples(qtdb_dir, split_csv, split))
    if not examples:
        raise ValueError(f"The {split} split has no beat examples")
    signals = np.stack([example.signal for example in examples]).astype(np.float32)
    valid = np.stack([example.valid_samples for example in examples])
    observed = np.asarray(
        [[example.observed_targets[name] for name in TARGET_NAMES] for example in examples],
        dtype=bool,
    )
    targets = np.zeros(
        (len(examples), len(TARGET_NAMES), signals.shape[1]), dtype=np.float32
    )
    left_samples = round(examples[0].sampling_rate_hz)
    sigma_samples = max(1.0, examples[0].sampling_rate_hz * 0.006)
    radius = int(np.ceil(3 * sigma_samples))
    for example_index, example in enumerate(examples):
        for target_index, target_name in enumerate(TARGET_NAMES):
            for offset in example.event_offsets[target_name]:
                center = left_samples + offset
                low = max(0, center - radius)
                high = min(signals.shape[1], center + radius + 1)
                positions = np.arange(low, high)
                gaussian = np.exp(-0.5 * ((positions - center) / sigma_samples) ** 2)
                targets[example_index, target_index, low:high] = np.maximum(
                    targets[example_index, target_index, low:high], gaussian
                )

    # Normalize each beat and lead independently; edge padding is excluded from the moments.
    valid_float = valid[..., None].astype(np.float32)
    counts = np.maximum(valid_float.sum(axis=1, keepdims=True), 1.0)
    means = (signals * valid_float).sum(axis=1, keepdims=True) / counts
    variances = (((signals - means) ** 2) * valid_float).sum(axis=1, keepdims=True) / counts
    scales = np.sqrt(variances).clip(min=1e-5)
    signals = ((signals - means) / scales) * valid_float
    return examples, signals.transpose(0, 2, 1), targets, observed, valid


def masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
    positive_weight: float,
) -> torch.Tensor:
    weights = torch.full((len(TARGET_NAMES),), positive_weight, device=logits.device)
    element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=weights[None, :, None],
    )
    supervision_mask = observed[:, :, None].to(logits.dtype) * valid[:, None, :].to(logits.dtype)
    return (element_loss * supervision_mask).sum() / supervision_mask.sum().clamp_min(1.0)


def localization_metrics(
    model: nn.Module,
    signals: np.ndarray,
    examples: list[Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    errors_by_target: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device)
            probabilities.extend(torch.sigmoid(model(batch)).cpu().numpy())

    for probability, example in zip(probabilities, examples, strict=True):
        samples_per_second = example.sampling_rate_hz
        left_samples = round(samples_per_second)
        separation = max(1, round(samples_per_second * 0.04))
        for target_index, target_name in enumerate(TARGET_NAMES):
            offsets = example.event_offsets[target_name]
            if not offsets:
                continue
            count = len(offsets)
            scores = probability[target_index].copy()
            scores[~example.valid_samples] = -np.inf
            selected: list[int] = []
            for _ in range(count):
                ranked = np.argsort(scores)[::-1]
                candidate = next(
                    (
                        int(index)
                        for index in ranked
                        if all(abs(int(index) - point) >= separation for point in selected)
                    ),
                    None,
                )
                if candidate is None:
                    break
                selected.append(candidate)
            if len(selected) < count:
                remaining = [
                    index for index in np.argsort(scores)[::-1] if int(index) not in selected
                ]
                selected.extend(int(index) for index in remaining[: count - len(selected)])
            truth = np.asarray([left_samples + offset for offset in offsets], dtype=np.int64)
            prediction = np.asarray(selected[:count], dtype=np.int64)
            row_indices, column_indices = linear_sum_assignment(
                np.abs(truth[:, None] - prediction[None, :])
            )
            error_ms = (
                np.abs(truth[row_indices] - prediction[column_indices])
                / samples_per_second
                * 1000.0
            )
            errors_by_target[target_name].extend(error_ms.tolist())

    def summarize(errors: list[float]) -> dict[str, float | int | None]:
        if not errors:
            return {"n_events": 0, "mae_ms": None, "median_ms": None, "p90_ms": None, "within_40ms": None}
        values = np.asarray(errors)
        return {
            "n_events": int(values.size),
            "mae_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.quantile(values, 0.9)),
            "within_40ms": float(np.mean(values <= 40.0)),
        }

    combined = [error for errors in errors_by_target.values() for error in errors]
    return {
        "protocol": "oracle event count per annotated beat and target; localization only",
        "overall": summarize(combined),
        "by_target": {name: summarize(errors) for name, errors in errors_by_target.items()},
    }


def evaluate_loss(
    model: nn.Module,
    signals: np.ndarray,
    targets: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    batch_size: int,
    positive_weight: float,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            end = start + batch_size
            batch = torch.from_numpy(signals[start:end]).to(device)
            batch_targets = torch.from_numpy(targets[start:end]).to(device)
            batch_observed = torch.from_numpy(observed[start:end]).to(device)
            batch_valid = torch.from_numpy(valid[start:end]).to(device)
            total_loss += float(
                masked_loss(
                    model(batch),
                    batch_targets,
                    batch_observed,
                    batch_valid,
                    positive_weight,
                ).item()
            )
            batches += 1
    return total_loss / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable in this PyTorch installation")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    loaded = {
        split: load_arrays(args.qtdb_dir, args.split_csv, split)
        for split in ("train", "validation", "test")
    }
    train_examples, train_x, train_y, train_observed, train_valid = loaded["train"]
    validation_examples, validation_x, validation_y, validation_observed, validation_valid = loaded[
        "validation"
    ]
    test_examples, test_x, test_y, test_observed, test_valid = loaded["test"]

    train_data = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        torch.from_numpy(train_observed),
        torch.from_numpy(train_valid),
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )
    model = QTDBLocalizationCNN(input_leads=train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y, batch_observed, batch_valid in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_observed = batch_observed.to(device)
            batch_valid = batch_valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(
                model(batch_x),
                batch_y,
                batch_observed,
                batch_valid,
                args.positive_weight,
            )
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())

        validation_loss = evaluate_loss(
            model,
            validation_x,
            validation_y,
            validation_observed,
            validation_valid,
            device,
            args.batch_size,
            args.positive_weight,
        )
        validation_metrics = localization_metrics(
            model, validation_x, validation_examples, device, args.batch_size
        )
        validation_mae = validation_metrics["overall"]["mae_ms"]
        if validation_mae is None:
            raise RuntimeError("No observed fiducials found in the validation split")
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_train_loss / max(len(train_loader), 1),
            "validation_loss": validation_loss,
            "validation_localization_mae_ms": float(validation_mae),
        }
        history.append(row)
        print(json.dumps(row))

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
                    "input_leads": train_x.shape[1],
                    "window_samples": train_x.shape[-1],
                    "sampling_rate_hz": train_examples[0].sampling_rate_hz,
                    "positive_weight": args.positive_weight,
                    "normalization": "per-window, per-lead z-score",
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
        model, validation_x, validation_examples, device, args.batch_size
    )
    test_metrics = localization_metrics(model, test_x, test_examples, device, args.batch_size)
    results = {
        "seed": args.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "records": {
            split: len({example.record_id for example in values[0]})
            for split, values in loaded.items()
        },
        "beat_examples": {split: len(values[0]) for split, values in loaded.items()},
        "validation": validation_metrics,
        "test": test_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: results[key]
                for key in (
                    "device",
                    "best_epoch",
                    "records",
                    "beat_examples",
                    "validation",
                    "test",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
