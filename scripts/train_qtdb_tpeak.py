"""Train a single-target T-peak locator and compare it with a timing prior."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor

LEFT_SECONDS = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_split(
    qtdb_dir: Path, split_csv: Path, split: str
) -> tuple[list[QTDBBeatExample], np.ndarray, np.ndarray, float, dict[str, int]]:
    all_examples = list(iter_qtdb_examples(qtdb_dir, split_csv, split))
    eligible = [example for example in all_examples if len(example.event_offsets["t_peak"]) == 1]
    excluded = {
        "no_t_peak": sum(not example.event_offsets["t_peak"] for example in all_examples),
        "multiple_t_peaks": sum(len(example.event_offsets["t_peak"]) > 1 for example in all_examples),
    }
    if not eligible:
        raise ValueError(f"No single-T-peak examples in {split} split")

    signals = np.stack([example.signal for example in eligible]).astype(np.float32)
    valid = np.stack([example.valid_samples for example in eligible])
    targets = np.zeros(len(eligible), dtype=np.float32)
    offsets: list[int] = []
    for index, example in enumerate(eligible):
        offset = example.event_offsets["t_peak"][0]
        offsets.append(offset)
        left = round(example.sampling_rate_hz * LEFT_SECONDS)
        targets[index] = 2.0 * (left + offset) / (example.signal.shape[0] - 1) - 1.0

    valid_float = valid[..., None].astype(np.float32)
    counts = np.maximum(valid_float.sum(axis=1, keepdims=True), 1.0)
    means = (signals * valid_float).sum(axis=1, keepdims=True) / counts
    variances = (((signals - means) ** 2) * valid_float).sum(axis=1, keepdims=True) / counts
    scales = np.sqrt(variances).clip(min=1e-5)
    normalized = ((signals - means) / scales) * valid_float
    return eligible, normalized.transpose(0, 2, 1), targets, float(np.mean(offsets)), excluded


def metrics(prediction: np.ndarray, examples: list[QTDBBeatExample]) -> dict[str, float | int]:
    fs = examples[0].sampling_rate_hz
    predicted_samples = (prediction + 1.0) * (examples[0].signal.shape[0] - 1) / 2.0
    left = round(fs * LEFT_SECONDS)
    target_samples = np.asarray(
        [left + example.event_offsets["t_peak"][0] for example in examples], dtype=np.float64
    )
    errors_ms = np.abs(predicted_samples - target_samples) / fs * 1000.0
    return {
        "n_events": int(errors_ms.size),
        "mae_ms": float(errors_ms.mean()),
        "median_ms": float(np.median(errors_ms)),
        "p90_ms": float(np.quantile(errors_ms, 0.9)),
        "within_40ms": float(np.mean(errors_ms <= 40.0)),
    }


def timing_prior_metrics(
    prior_offset: float, examples: list[QTDBBeatExample]
) -> dict[str, float | int]:
    fs = examples[0].sampling_rate_hz
    errors_ms = np.asarray([
        abs(example.event_offsets["t_peak"][0] - prior_offset) / fs * 1000.0
        for example in examples
    ])
    return {
        "n_events": int(errors_ms.size),
        "mae_ms": float(errors_ms.mean()),
        "median_ms": float(np.median(errors_ms)),
        "p90_ms": float(np.quantile(errors_ms, 0.9)),
        "within_40ms": float(np.mean(errors_ms <= 40.0)),
    }


def predict(model: nn.Module, signals: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).to(device)
            outputs.extend(model(batch).cpu().numpy())
    return np.asarray(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-tpeak"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the test partition after validation selection; exploratory if it was previously inspected",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    train_examples, train_x, train_y, prior_offset, train_excluded = prepare_split(
        args.qtdb_dir, args.split_csv, "train"
    )
    validation_examples, validation_x, validation_y, _, validation_excluded = prepare_split(
        args.qtdb_dir, args.split_csv, "validation"
    )
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = QTDBTpeakRegressor(input_leads=train_x.shape[1]).to(device)
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
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(batch_x), batch_y, beta=0.02)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        validation_predictions = predict(model, validation_x, device, args.batch_size)
        validation_metrics = metrics(validation_predictions, validation_examples)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_huber_loss": total_loss / max(len(train_loader), 1),
            "validation_mae_ms": validation_metrics["mae_ms"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_metrics["mae_ms"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae_ms"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "normalization": "per-window, per-lead z-score",
                    "sampling_rate_hz": train_examples[0].sampling_rate_hz,
                    "window_samples": train_x.shape[-1],
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_predictions = predict(model, validation_x, device, args.batch_size)
    validation_metrics = metrics(validation_predictions, validation_examples)
    fs = train_examples[0].sampling_rate_hz
    prior_validation = timing_prior_metrics(prior_offset, validation_examples)
    selected = validation_metrics["mae_ms"] < prior_validation["mae_ms"]
    result: dict[str, Any] = {
        "target": "single annotated T-wave peak per selected beat",
        "excluded_beats": {"train": train_excluded, "validation": validation_excluded},
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_records": len({example.record_id for example in train_examples}),
        "validation_records": len({example.record_id for example in validation_examples}),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "validation": validation_metrics,
        "validation_timing_prior": prior_validation,
        "beats_validation_prior": selected,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "test": None,
    }

    # In this run, load the test split only if validation beats the timing prior.
    if selected and args.evaluate_test:
        test_examples, test_x, _, _, test_excluded = prepare_split(
            args.qtdb_dir, args.split_csv, "test"
        )
        test_predictions = predict(model, test_x, device, args.batch_size)
        result["test_examples"] = len(test_examples)
        result["test_records"] = len({example.record_id for example in test_examples})
        result["test_excluded_beats"] = test_excluded
        result["test"] = metrics(test_predictions, test_examples)
        result["test_timing_prior"] = timing_prior_metrics(prior_offset, test_examples)

    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "validation_mae_ms": validation_metrics["mae_ms"],
                "validation_prior_mae_ms": prior_validation["mae_ms"],
                "beats_validation_prior": selected,
                "test_mae_ms": result["test"]["mae_ms"] if result["test"] else None,
                "metrics_file": str(result_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
