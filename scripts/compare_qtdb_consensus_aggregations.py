"""Compare donor-score aggregation rules for adaptive-region temporal LIME."""

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
from analyze_qtdb_donor_sensitivity import donor_candidates, quantile_donors  # noqa: E402
from compare_qtdb_perturbations import (  # noqa: E402
    lime_scores,
    morphology_regions,
    normalized_input,
    perturb,
    predict_ms,
)
from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples  # noqa: E402
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor  # noqa: E402

LEFT_SECONDS = 1.0


def top_regions(scores: np.ndarray, regions: list[tuple[int, int, str]], budget: int) -> list[int]:
    lengths = np.asarray([end - start for start, end, _ in regions])
    selected: list[int] = []
    covered = 0
    for index in np.argsort(scores)[::-1]:
        selected.append(int(index))
        covered += int(lengths[index])
        if covered >= budget:
            break
    return selected


def pairwise_jaccard(region_sets: list[set[int]]) -> float | None:
    if len(region_sets) < 2:
        return None
    scores = [
        len(left & right) / max(len(left | right), 1)
        for left, right in combinations(region_sets, 2)
    ]
    return float(np.mean(scores))


def choose_validation_examples(
    grouped: dict[str, list[QTDBBeatExample]],
    *,
    count: int,
    max_per_record: int,
    seed: int,
) -> list[QTDBBeatExample]:
    rng = np.random.default_rng(seed)
    record_ids = sorted(grouped)
    rng.shuffle(record_ids)
    selected: list[QTDBBeatExample] = []
    for record_id in record_ids:
        candidates = [
            example
            for example in grouped[record_id]
            if len(example.event_offsets["t_peak"]) == 1
        ]
        if not candidates:
            continue
        order = rng.permutation(len(candidates))
        selected.extend(candidates[index] for index in order[:max_per_record])
        if len(selected) >= count:
            return selected[:count]
    return selected


def analyze_example(
    model: torch.nn.Module,
    example: QTDBBeatExample,
    donors: list[tuple[float, QTDBBeatExample]],
    *,
    perturbations: int,
    random_trials: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = normalized_input(example)
    fs = example.sampling_rate_hz
    regions = morphology_regions(inputs, fs)
    predict = lambda batch: predict_ms(model, batch, fs)
    with torch.no_grad():
        original_prediction = float(predict(inputs[None])[0])
    reference_ms = float(example.event_offsets["t_peak"][0] / fs * 1000.0)
    anchor_index = round(fs * LEFT_SECONDS)
    target_index = anchor_index + round(example.event_offsets["t_peak"][0])
    sample_budget = int(np.ceil(inputs.shape[-1] * 0.20))

    donor_scores: list[np.ndarray] = []
    donor_local_r2: list[float] = []
    donor_descriptors: list[dict[str, Any]] = []
    for donor_index, (distance, donor_example) in enumerate(donors):
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
        normalized_scores = scores / max(float(scores.sum()), 1e-12)
        donor_scores.append(normalized_scores)
        donor_local_r2.append(local_r2)
        individual_selected = top_regions(normalized_scores, regions, sample_budget)
        donor_descriptors.append(
            {
                "donor_rank": donor_index,
                "donor_beat_index": donor_example.beat_index,
                "donor_distance_mse": distance,
                "donor_anchor_gap_samples": abs(donor_example.anchor_sample - example.anchor_sample),
                "local_r2": local_r2,
                "individual_top_regions": individual_selected,
            }
        )

    individual_sets = [set(top_regions(scores, regions, sample_budget)) for scores in donor_scores]
    distances = np.asarray([distance for distance, _ in donors], dtype=np.float64)
    scale = max(float(np.median(distances)), 1e-12)
    logits = -distances / scale
    similarity_weights = np.exp(logits - np.max(logits))
    similarity_weights /= similarity_weights.sum()
    score_matrix = np.stack(donor_scores)
    median_scores = np.median(score_matrix, axis=0)
    median_scores /= max(float(median_scores.sum()), 1e-12)
    aggregation_scores = {
        "equal_mean": np.mean(score_matrix, axis=0),
        "similarity_weighted": np.average(score_matrix, axis=0, weights=similarity_weights),
        "component_median": median_scores,
    }
    aggregation_results: dict[str, Any] = {}
    donor_rows = []
    individual_shifts = []
    random_shifts_by_donor: list[list[float]] = []
    donors_normalized = [normalized_input(donor_example) for _, donor_example in donors]
    for donor_index, ((distance, donor_example), donor) in enumerate(zip(donors, donors_normalized, strict=True)):
        individual_mask = np.ones(len(regions), dtype=bool)
        individual_mask[list(individual_sets[donor_index])] = False
        individual_input = perturb(inputs, donor, regions, individual_mask, fade_samples=round(fs * 0.02))
        individual_prediction = float(predict(individual_input[None])[0])
        individual_shifts.append(abs(individual_prediction - original_prediction))

        region_lengths = np.asarray([end - start for start, end, _ in regions])
        rng = np.random.default_rng(seed + 20_000 + donor_index)
        donor_random_shifts = []
        for _ in range(random_trials):
            random_mask = np.ones(len(regions), dtype=bool)
            covered = 0
            for region_index in rng.permutation(len(regions)):
                random_mask[region_index] = False
                covered += int(region_lengths[region_index])
                if covered >= sample_budget:
                    break
            random_input = perturb(inputs, donor, regions, random_mask, fade_samples=round(fs * 0.02))
            random_prediction = float(predict(random_input[None])[0])
            donor_random_shifts.append(abs(random_prediction - original_prediction))
        random_shifts_by_donor.append(donor_random_shifts)

    for aggregation_name, scores in aggregation_scores.items():
        selected = top_regions(scores, regions, sample_budget)
        selected_set = set(selected)
        consensus_jaccards = [
            len(selected_set & donor_set) / max(len(selected_set | donor_set), 1)
            for donor_set in individual_sets
        ]
        selected_samples = sum(regions[index][1] - regions[index][0] for index in selected)
        intervals = [list(regions[index]) for index in selected]
        overlaps_t_peak = any(
            max(regions[index][0], target_index - round(fs * 0.06))
            < min(regions[index][1], target_index + round(fs * 0.06))
            for index in selected
        )
        overlaps_anchor = any(
            max(regions[index][0], anchor_index - round(fs * 0.08))
            < min(regions[index][1], anchor_index + round(fs * 0.08))
            for index in selected
        )
        overlaps_late = any(regions[index][1] > anchor_index + round(fs * 0.8) for index in selected)
        consensus_shifts = []
        for donor_index, donor in enumerate(donors_normalized):
            consensus_mask = np.ones(len(regions), dtype=bool)
            consensus_mask[selected] = False
            consensus_input = perturb(inputs, donor, regions, consensus_mask, fade_samples=round(fs * 0.02))
            consensus_prediction = float(predict(consensus_input[None])[0])
            shift = abs(consensus_prediction - original_prediction)
            consensus_shifts.append(shift)
            donor_rows.append(
                {
                    "record_id": example.record_id,
                    "beat_index": example.beat_index,
                    "aggregation": aggregation_name,
                    "donor_rank": donor_index,
                    "donor_beat_index": donors[donor_index][1].beat_index,
                    "donor_distance_mse": float(distances[donor_index]),
                    "similarity_weight": float(similarity_weights[donor_index]),
                    "local_r2": donor_local_r2[donor_index],
                    "individual_top_jaccard_with_aggregation": consensus_jaccards[donor_index],
                    "consensus_top_region_prediction_shift_ms": shift,
                    "individual_top_region_prediction_shift_ms": individual_shifts[donor_index],
                    "random_region_prediction_shift_mean_ms": float(
                        np.mean(random_shifts_by_donor[donor_index])
                    ),
                }
            )
        random_shifts = [value for donor_values in random_shifts_by_donor for value in donor_values]
        random_mean = float(np.mean(random_shifts))
        aggregation_results[aggregation_name] = {
            "selected_region_count": len(selected),
            "replaced_sample_fraction": selected_samples / inputs.shape[-1],
            "mean_vs_donor_top_jaccard": float(np.mean(consensus_jaccards)),
            "pairwise_donor_top_jaccard": pairwise_jaccard(individual_sets),
            "consensus_top_shift_mean_ms": float(np.mean(consensus_shifts)),
            "consensus_top_shift_range_ms": float(max(consensus_shifts) - min(consensus_shifts)),
            "individual_top_shift_mean_ms": float(np.mean(individual_shifts)),
            "random_shift_mean_ms": random_mean,
            "random_shift_p90_ms": float(np.percentile(random_shifts, 90)),
            "consensus_shift_exceeds_random_mean": float(np.mean(consensus_shifts)) > random_mean,
            "overlaps_t_peak_plusminus_60ms": overlaps_t_peak,
            "overlaps_anchor_plusminus_80ms": overlaps_anchor,
            "overlaps_late_window": overlaps_late,
            "selected_intervals_samples": intervals,
        }

    example_result = {
        "record_id": example.record_id,
        "beat_index": example.beat_index,
        "reference_t_peak_ms": reference_ms,
        "prediction_ms": original_prediction,
        "absolute_error_ms": abs(original_prediction - reference_ms),
        "donor_count": len(donors),
        "region_count": len(regions),
        "mean_donor_local_r2": float(np.mean(donor_local_r2)),
        "donor_local_r2_range": float(max(donor_local_r2) - min(donor_local_r2)),
        "pairwise_donor_top_jaccard": pairwise_jaccard(individual_sets),
        "aggregation_results": aggregation_results,
        "donor_results": donor_descriptors,
    }
    return example_result, donor_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-consensus-aggregations"))
    parser.add_argument("--examples", type=int, default=48)
    parser.add_argument("--max-per-record", type=int, default=3)
    parser.add_argument("--donors-per-example", type=int, default=5)
    parser.add_argument("--perturbations", type=int, default=256)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = QTDBTpeakRegressor(input_leads=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grouped: dict[str, list[QTDBBeatExample]] = defaultdict(list)
    for example in iter_qtdb_examples(args.qtdb_dir, args.split_csv, "validation"):
        if example.valid_samples.all():
            grouped[example.record_id].append(example)
    selected = choose_validation_examples(
        grouped,
        count=args.examples,
        max_per_record=args.max_per_record,
        seed=args.seed,
    )
    if not selected:
        raise SystemExit("No eligible validation examples found")

    results: list[dict[str, Any]] = []
    donor_rows: list[dict[str, Any]] = []
    for index, example in enumerate(selected):
        pool = donor_candidates(example, grouped[example.record_id])
        donors = quantile_donors(pool, args.donors_per_example)
        if len(donors) < args.donors_per_example:
            raise SystemExit(
                f"Only {len(donors)} target-separated donors for {example.record_id} beat {example.beat_index}"
            )
        result, rows = analyze_example(
            model,
            example,
            donors,
            perturbations=args.perturbations,
            random_trials=args.random_trials,
            seed=args.seed + index,
        )
        results.append(result)
        donor_rows.extend(rows)
        print(f"[{index + 1}/{len(selected)}] {example.record_id} beat {example.beat_index}", flush=True)

    aggregation_names = ("equal_mean", "similarity_weighted", "component_median")
    aggregate: dict[str, Any] = {
        "records": len({result["record_id"] for result in results}),
        "examples": len(results),
        "mean_donor_local_r2": float(np.mean([result["mean_donor_local_r2"] for result in results])),
        "mean_donor_local_r2_range": float(np.mean([result["donor_local_r2_range"] for result in results])),
        "mean_pairwise_donor_top_jaccard": float(
            np.mean([result["pairwise_donor_top_jaccard"] for result in results])
        ),
        "by_aggregation": {},
    }
    example_rows = []
    for result in results:
        for aggregation_name in aggregation_names:
            metrics = result["aggregation_results"][aggregation_name]
            example_rows.append(
                {
                    "record_id": result["record_id"],
                    "beat_index": result["beat_index"],
                    "aggregation": aggregation_name,
                    "reference_t_peak_ms": result["reference_t_peak_ms"],
                    "prediction_ms": result["prediction_ms"],
                    "absolute_error_ms": result["absolute_error_ms"],
                    "donor_count": result["donor_count"],
                    "region_count": result["region_count"],
                    "mean_donor_local_r2": result["mean_donor_local_r2"],
                    "donor_local_r2_range": result["donor_local_r2_range"],
                    "pairwise_donor_top_jaccard": result["pairwise_donor_top_jaccard"],
                    **metrics,
                    "selected_intervals_samples": json.dumps(metrics["selected_intervals_samples"]),
                }
            )
    for aggregation_name in aggregation_names:
        rows = [result["aggregation_results"][aggregation_name] for result in results]
        aggregate["by_aggregation"][aggregation_name] = {
            "mean_selected_region_count": float(np.mean([row["selected_region_count"] for row in rows])),
            "mean_replaced_sample_fraction": float(np.mean([row["replaced_sample_fraction"] for row in rows])),
            "mean_consensus_vs_donor_top_jaccard": float(
                np.mean([row["mean_vs_donor_top_jaccard"] for row in rows])
            ),
            "mean_consensus_top_shift_ms": float(np.mean([row["consensus_top_shift_mean_ms"] for row in rows])),
            "mean_consensus_top_shift_range_ms": float(
                np.mean([row["consensus_top_shift_range_ms"] for row in rows])
            ),
            "mean_individual_top_shift_ms": float(np.mean([row["individual_top_shift_mean_ms"] for row in rows])),
            "mean_random_shift_ms": float(np.mean([row["random_shift_mean_ms"] for row in rows])),
            "mean_random_shift_p90_ms": float(np.mean([row["random_shift_p90_ms"] for row in rows])),
            "examples_overlapping_t_peak_plusminus_60ms": int(
                sum(row["overlaps_t_peak_plusminus_60ms"] for row in rows)
            ),
            "examples_overlapping_anchor_plusminus_80ms": int(
                sum(row["overlaps_anchor_plusminus_80ms"] for row in rows)
            ),
            "examples_overlapping_late_window": int(sum(row["overlaps_late_window"] for row in rows)),
            "examples_consensus_shift_exceeds_random_mean": int(
                sum(row["consensus_shift_exceeds_random_mean"] for row in rows)
            ),
        }
    summary = {
        "split": "validation",
        "examples": len(results),
        "records": aggregate["records"],
        "donors_per_example": args.donors_per_example,
        "donor_selection": "MSE quantiles from same-record windows; each donor avoids overlap with the target window, but donor windows may overlap one another",
        "regions": "signal-adaptive regions, capped at 100 ms",
        "aggregation_definitions": {
            "equal_mean": "arithmetic mean of donor-wise L1-normalized absolute LIME coefficient vectors",
            "similarity_weighted": "weighted mean of donor-wise L1-normalized absolute LIME coefficient vectors; weights are softmax(-MSE / median donor MSE)",
            "component_median": "component-wise median of donor-wise L1-normalized absolute LIME coefficient vectors, then L1-normalized",
        },
        "lime_perturbations": args.perturbations,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "aggregate": aggregate,
        "per_example": results,
        "per_example_aggregation": example_rows,
        "per_donor": donor_rows,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-example.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(example_rows[0]),
        )
        writer.writeheader()
        writer.writerows(
            example_rows
        )
    with (output_dir / "per-donor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(donor_rows[0]),
        )
        writer.writeheader()
        writer.writerows(
            donor_rows
        )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
