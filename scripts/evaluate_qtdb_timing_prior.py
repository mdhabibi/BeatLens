"""Evaluate an ECG-free, train-only event-timing prior for comparison."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from beatlens.infrastructure.qtdb import TARGET_NAMES, iter_qtdb_examples


def evaluate(train_examples: list[Any], examples: list[Any]) -> dict[str, Any]:
    offsets_by_slot: dict[str, dict[int, list[int]]] = {
        name: defaultdict(list) for name in TARGET_NAMES
    }
    all_offsets: dict[str, list[int]] = {name: [] for name in TARGET_NAMES}
    for example in train_examples:
        for name in TARGET_NAMES:
            for slot, offset in enumerate(example.event_offsets[name]):
                offsets_by_slot[name][slot].append(offset)
                all_offsets[name].append(offset)

    errors_by_target: dict[str, list[float]] = {name: [] for name in TARGET_NAMES}
    for example in examples:
        for name in TARGET_NAMES:
            true_offsets = example.event_offsets[name]
            if not true_offsets:
                continue
            fallback = float(np.mean(all_offsets[name]))
            predictions = np.asarray(
                [
                    float(np.mean(offsets_by_slot[name][slot]))
                    if offsets_by_slot[name].get(slot)
                    else fallback
                    for slot in range(len(true_offsets))
                ]
            )
            truth = np.asarray(true_offsets, dtype=np.float64)
            row_indices, column_indices = linear_sum_assignment(
                np.abs(truth[:, None] - predictions[None, :])
            )
            errors_ms = (
                np.abs(truth[row_indices] - predictions[column_indices])
                / example.sampling_rate_hz
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
        "protocol": "ECG-free train-only ordinal timing prior; oracle event count per beat/target",
        "overall": summarize(all_errors),
        "by_target": {name: summarize(values) for name, values in errors_by_target.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qtdb-baseline/timing-prior.json"))
    args = parser.parse_args()

    train = list(iter_qtdb_examples(args.qtdb_dir, args.split_csv, "train"))
    results = {
        "train_examples": len(train),
        "validation": evaluate(
            train,
            list(iter_qtdb_examples(args.qtdb_dir, args.split_csv, "validation")),
        ),
        "test": evaluate(
            train,
            list(iter_qtdb_examples(args.qtdb_dir, args.split_csv, "test")),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
