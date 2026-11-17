"""Measure how same-record donor choice changes LIME rankings and output shifts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_qtdb_perturbations import (  # noqa: E402
    fixed_regions,
    lime_scores,
    morphology_regions,
    normalized_input,
    perturb,
    predict_ms,
)
from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples  # noqa: E402
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor  # noqa: E402


def donor_candidates(
    example: QTDBBeatExample,
    candidates: list[QTDBBeatExample],
) -> list[tuple[float, QTDBBeatExample]]:
    """Return same-record windows that do not overlap the explained target window.

    Candidate donor windows are not required to be pairwise non-overlapping with one another.
    """
    target = normalized_input(example)
    fs = example.sampling_rate_hz
    left = round(fs * 0.70)
    right = min(target.shape[-1], round(fs * 1.55))
    available = []
    for candidate in candidates:
        if candidate.beat_index == example.beat_index:
            continue
        if abs(candidate.anchor_sample - example.anchor_sample) < len(example.signal):
            continue
        if not candidate.valid_samples.all():
            continue
        donor = normalized_input(candidate)
        if donor.shape != target.shape:
            continue
        distance = float(np.mean((target[:, left:right] - donor[:, left:right]) ** 2))
        available.append((distance, candidate))
    return sorted(available, key=lambda item: item[0])


def quantile_donors(
    candidates: list[tuple[float, QTDBBeatExample]],
    count: int,
) -> list[tuple[float, QTDBBeatExample]]:
    if not candidates:
        return []
    count = min(count, len(candidates))
    indexes = np.rint(np.linspace(0, len(candidates) - 1, count)).astype(int)
    return [candidates[index] for index in dict.fromkeys(indexes.tolist())]


def top_regions_for_budget(
    scores: np.ndarray,
    regions: list[tuple[int, int, str]],
    sample_budget: int,
) -> list[int]:
    lengths = np.asarray([end - start for start, end, _ in regions])
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    used = 0
    for index in order:
        selected.append(int(index))
        used += int(lengths[index])
        if used >= sample_budget:
            break
    return selected


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / max(len(left | right), 1)


def analyze_example(
    model: torch.nn.Module,
    example: QTDBBeatExample,
    donor_pool: list[tuple[float, QTDBBeatExample]],
    *,
    donors_per_example: int,
    perturbations: int,
    random_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    inputs = normalized_input(example)
    fs = example.sampling_rate_hz
    predict = lambda batch: predict_ms(model, batch, fs)
    original_prediction = float(predict(inputs[None])[0])
    reference_ms = float(example.event_offsets["t_peak"][0] / fs * 1000.0)
    sample_budget = int(np.ceil(inputs.shape[-1] * 0.20))
    outputs = []
    for variant, regions in (
        ("fixed_25ms", fixed_regions(inputs.shape[-1], fs)),
        ("adaptive", morphology_regions(inputs, fs)),
    ):
        donor_rankings: list[set[int]] = []
        for donor_rank, (distance, donor_example) in enumerate(
            quantile_donors(donor_pool, donors_per_example)
        ):
            donor = normalized_input(donor_example)
            scores, local_r2 = lime_scores(
                inputs,
                donor,
                regions,
                predict,
                perturbations=perturbations,
                seed=seed,
                fade_samples=round(fs * 0.02),
            )
            selected = top_regions_for_budget(scores, regions, sample_budget)
            donor_rankings.append(set(selected))
            top_mask = np.ones(len(regions), dtype=bool)
            top_mask[selected] = False
            top_prediction = float(
                predict(
                    perturb(inputs, donor, regions, top_mask, fade_samples=round(fs * 0.02))[None]
                )[0]
            )
            top_shift = abs(top_prediction - original_prediction)

            rng = np.random.default_rng(seed + 10_000)
            random_shifts = []
            region_lengths = np.asarray([end - start for start, end, _ in regions])
            for _ in range(random_trials):
                mask = np.ones(len(regions), dtype=bool)
                replaced = 0
                for region_index in rng.permutation(len(regions)):
                    mask[region_index] = False
                    replaced += int(region_lengths[region_index])
                    if replaced >= sample_budget:
                        break
                prediction = float(
                    predict(perturb(inputs, donor, regions, mask, fade_samples=round(fs * 0.02))[None])[0]
                )
                random_shifts.append(abs(prediction - original_prediction))

            outputs.append(
                {
                    "record_id": example.record_id,
                    "beat_index": example.beat_index,
                    "variant": variant,
                    "donor_rank": donor_rank,
                    "donor_beat_index": donor_example.beat_index,
                    "donor_anchor_gap_samples": abs(
                        donor_example.anchor_sample - example.anchor_sample
                    ),
                    "donor_distance_mse": distance,
                    "reference_t_peak_ms": reference_ms,
                    "original_prediction_ms": original_prediction,
                    "absolute_error_ms": abs(original_prediction - reference_ms),
                    "region_count": len(regions),
                    "top_region_count": len(selected),
                    "weighted_local_r2": local_r2,
                    "top_region_prediction_shift_ms": top_shift,
                    "random_region_prediction_shift_mean_ms": float(np.mean(random_shifts)),
                    "random_region_prediction_shift_p90_ms": float(
                        np.percentile(random_shifts, 90)
                    ),
                    "top_shift_exceeds_random_mean": top_shift > float(np.mean(random_shifts)),
                    "selected_intervals_samples": [list(regions[index]) for index in selected],
                }
            )
        variant_rows = [row for row in outputs if row["record_id"] == example.record_id and row["variant"] == variant]
        pairwise = [
            jaccard(donor_rankings[left], donor_rankings[right])
            for left, right in combinations(range(len(donor_rankings)), 2)
        ]
        for row in variant_rows:
            row["within_beat_top_region_jaccard_mean"] = (
                float(np.mean(pairwise)) if pairwise else None
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-donor-sensitivity"))
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--donors-per-example", type=int, default=5)
    parser.add_argument("--perturbations", type=int, default=256)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QTDBTpeakRegressor(input_leads=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grouped: dict[str, list[QTDBBeatExample]] = defaultdict(list)
    for example in iter_qtdb_examples(args.qtdb_dir, args.split_csv, "validation"):
        if example.valid_samples.all():
            grouped[example.record_id].append(example)
    eligible = [
        example
        for record_id in sorted(grouped)
        for example in grouped[record_id]
        if len(example.event_offsets["t_peak"]) == 1 and len(grouped[record_id]) > 1
    ]
    rng = np.random.default_rng(args.seed)
    record_ids = sorted({example.record_id for example in eligible})
    rng.shuffle(record_ids)
    selected = []
    for record_id in record_ids:
        options = [example for example in eligible if example.record_id == record_id]
        selected.append(options[int(rng.integers(len(options)))])
        if len(selected) >= args.examples:
            break
    if not selected:
        raise SystemExit("No eligible validation beats found")

    rows = []
    for example_index, example in enumerate(selected):
        pool = donor_candidates(example, grouped[example.record_id])
        if not pool:
            raise SystemExit(f"No target-separated donor found for {example.record_id} beat {example.beat_index}")
        result = analyze_example(
            model,
            example,
            pool,
            donors_per_example=args.donors_per_example,
            perturbations=args.perturbations,
            random_trials=args.random_trials,
            seed=args.seed + example_index,
        )
        rows.extend(result)
        print(
            f"[{example_index + 1}/{len(selected)}] {example.record_id} beat {example.beat_index}: "
            f"{min(args.donors_per_example, len(pool))} target-separated donors",
            flush=True,
        )

    variant_names = ("fixed_25ms", "adaptive")
    aggregates: dict[str, Any] = {}
    for variant in variant_names:
        variant_rows = [row for row in rows if row["variant"] == variant]
        by_beat: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in variant_rows:
            by_beat[(row["record_id"], row["beat_index"])].append(row)
        r2_ranges = [max(row["weighted_local_r2"] for row in values) - min(row["weighted_local_r2"] for row in values) for values in by_beat.values()]
        top_shift_ranges = [max(row["top_region_prediction_shift_ms"] for row in values) - min(row["top_region_prediction_shift_ms"] for row in values) for values in by_beat.values()]
        aggregates[variant] = {
            "donor_rows": len(variant_rows),
            "mean_region_count": float(np.mean([row["region_count"] for row in variant_rows])),
            "mean_weighted_local_r2": float(np.mean([row["weighted_local_r2"] for row in variant_rows])),
            "mean_within_beat_local_r2_range": float(np.mean(r2_ranges)),
            "mean_within_beat_top_shift_range_ms": float(np.mean(top_shift_ranges)),
            "mean_pairwise_top_region_jaccard_across_donors": float(
                np.mean([row["within_beat_top_region_jaccard_mean"] for row in variant_rows])
            ),
            "mean_top_region_shift_ms": float(
                np.mean([row["top_region_prediction_shift_ms"] for row in variant_rows])
            ),
            "mean_random_region_shift_ms": float(
                np.mean([row["random_region_prediction_shift_mean_ms"] for row in variant_rows])
            ),
            "examples_top_shift_exceeds_random_mean": sum(
                any(
                    row["top_shift_exceeds_random_mean"]
                    for row in values
                )
                for values in by_beat.values()
            ),
            "total_examples": len(by_beat),
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": "validation",
        "examples": len(selected),
        "donors_per_example_requested": args.donors_per_example,
        "donor_selection": "quantiles of donor MSE from closest to farthest, excluding overlapping windows",
        "selection_seed": args.seed,
        "lime_perturbations": args.perturbations,
        "random_trials": args.random_trials,
        "regions": "fixed 25 ms bins and waveform-adaptive regions capped at 100 ms",
        "deletion_budget": "top-ranked and random sets each replace at least 20% of window samples",
        "aggregates": aggregates,
        "per_donor_results": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-donor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[key for key in rows[0] if key != "selected_intervals_samples"],
        )
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "selected_intervals_samples"}
            for row in rows
        )
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    main()
