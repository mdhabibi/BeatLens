"""Evaluate adaptive-region LIME consensus across target-separated donor beats."""

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

    consensus_scores = np.mean(np.stack(donor_scores), axis=0)
    consensus_selected = top_regions(consensus_scores, regions, sample_budget)
    consensus_set = set(consensus_selected)
    individual_sets = [set(top_regions(scores, regions, sample_budget)) for scores in donor_scores]
    consensus_jaccards = [
        len(consensus_set & selected) / max(len(consensus_set | selected), 1)
        for selected in individual_sets
    ]
    selected_samples = sum(regions[index][1] - regions[index][0] for index in consensus_selected)
    consensus_intervals = [list(regions[index]) for index in consensus_selected]
    overlaps_t_peak = any(
        max(start, target_index - round(fs * 0.06))
        < min(end, target_index + round(fs * 0.06))
        for index in consensus_selected
        for start, end, _ in [regions[index]]
    )
    overlaps_anchor = any(
        max(regions[index][0], anchor_index - round(fs * 0.08))
        < min(regions[index][1], anchor_index + round(fs * 0.08))
        for index in consensus_selected
    )
    overlaps_late = any(regions[index][1] > anchor_index + round(fs * 0.8) for index in consensus_selected)

    consensus_shifts = []
    individual_shifts = []
    random_shifts = []
    for donor_index, (_, donor_example) in enumerate(donors):
        donor = normalized_input(donor_example)
        consensus_mask = np.ones(len(regions), dtype=bool)
        consensus_mask[consensus_selected] = False
        consensus_input = perturb(
            inputs, donor, regions, consensus_mask, fade_samples=round(fs * 0.02)
        )
        consensus_prediction = float(predict(consensus_input[None])[0])
        consensus_shifts.append(abs(consensus_prediction - original_prediction))

        individual_mask = np.ones(len(regions), dtype=bool)
        individual_mask[list(individual_sets[donor_index])] = False
        individual_input = perturb(
            inputs, donor, regions, individual_mask, fade_samples=round(fs * 0.02)
        )
        individual_prediction = float(predict(individual_input[None])[0])
        individual_shifts.append(abs(individual_prediction - original_prediction))

        region_lengths = np.asarray([end - start for start, end, _ in regions])
        rng = np.random.default_rng(seed + 20_000 + donor_index)
        for _ in range(random_trials):
            random_mask = np.ones(len(regions), dtype=bool)
            covered = 0
            for region_index in rng.permutation(len(regions)):
                random_mask[region_index] = False
                covered += int(region_lengths[region_index])
                if covered >= sample_budget:
                    break
            random_input = perturb(
                inputs, donor, regions, random_mask, fade_samples=round(fs * 0.02)
            )
            random_prediction = float(predict(random_input[None])[0])
            random_shifts.append(abs(random_prediction - original_prediction))

    random_mean = float(np.mean(random_shifts))
    example_result = {
        "record_id": example.record_id,
        "beat_index": example.beat_index,
        "reference_t_peak_ms": reference_ms,
        "prediction_ms": original_prediction,
        "absolute_error_ms": abs(original_prediction - reference_ms),
        "donor_count": len(donors),
        "region_count": len(regions),
        "consensus_selected_region_count": len(consensus_selected),
        "consensus_replaced_sample_fraction": selected_samples / inputs.shape[-1],
        "mean_donor_local_r2": float(np.mean(donor_local_r2)),
        "donor_local_r2_range": float(max(donor_local_r2) - min(donor_local_r2)),
        "mean_consensus_vs_donor_top_jaccard": float(np.mean(consensus_jaccards)),
        "pairwise_donor_top_jaccard": pairwise_jaccard(individual_sets),
        "consensus_top_region_prediction_shift_mean_ms": float(np.mean(consensus_shifts)),
        "consensus_top_region_prediction_shift_range_ms": float(max(consensus_shifts) - min(consensus_shifts)),
        "individual_top_region_prediction_shift_mean_ms": float(np.mean(individual_shifts)),
        "random_region_prediction_shift_mean_ms": random_mean,
        "random_region_prediction_shift_p90_ms": float(np.percentile(random_shifts, 90)),
        "consensus_shift_exceeds_random_mean": float(np.mean(consensus_shifts)) > random_mean,
        "consensus_overlaps_t_peak_plusminus_60ms": overlaps_t_peak,
        "consensus_overlaps_anchor_plusminus_80ms": overlaps_anchor,
        "consensus_overlaps_late_window": overlaps_late,
        "consensus_intervals_samples": consensus_intervals,
        "donor_results": donor_descriptors,
    }
    donor_rows = [
        {
            "record_id": example.record_id,
            "beat_index": example.beat_index,
            "donor_rank": descriptor["donor_rank"],
            "donor_beat_index": descriptor["donor_beat_index"],
            "donor_distance_mse": descriptor["donor_distance_mse"],
            "local_r2": descriptor["local_r2"],
            "individual_top_jaccard_with_consensus": consensus_jaccards[index],
            "individual_top_regions": descriptor["individual_top_regions"],
            "consensus_top_region_prediction_shift_ms": consensus_shifts[index],
            "individual_top_region_prediction_shift_ms": individual_shifts[index],
        }
        for index, descriptor in enumerate(donor_descriptors)
    ]
    return example_result, donor_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-donor-consensus"))
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

    aggregate = {
        "records": len({result["record_id"] for result in results}),
        "examples": len(results),
        "mean_donor_local_r2": float(np.mean([result["mean_donor_local_r2"] for result in results])),
        "mean_donor_local_r2_range": float(np.mean([result["donor_local_r2_range"] for result in results])),
        "mean_consensus_vs_donor_top_jaccard": float(
            np.mean([result["mean_consensus_vs_donor_top_jaccard"] for result in results])
        ),
        "mean_pairwise_donor_top_jaccard": float(
            np.mean([result["pairwise_donor_top_jaccard"] for result in results])
        ),
        "mean_consensus_top_shift_ms": float(
            np.mean([result["consensus_top_region_prediction_shift_mean_ms"] for result in results])
        ),
        "mean_consensus_top_shift_range_ms": float(
            np.mean([result["consensus_top_region_prediction_shift_range_ms"] for result in results])
        ),
        "mean_individual_top_shift_ms": float(
            np.mean([result["individual_top_region_prediction_shift_mean_ms"] for result in results])
        ),
        "mean_random_shift_ms": float(
            np.mean([result["random_region_prediction_shift_mean_ms"] for result in results])
        ),
        "consensus_examples_overlapping_t_peak": sum(
            result["consensus_overlaps_t_peak_plusminus_60ms"] for result in results
        ),
        "consensus_examples_overlapping_anchor": sum(
            result["consensus_overlaps_anchor_plusminus_80ms"] for result in results
        ),
        "consensus_examples_overlapping_late_window": sum(
            result["consensus_overlaps_late_window"] for result in results
        ),
        "examples_consensus_shift_exceeds_random_mean": sum(
            result["consensus_shift_exceeds_random_mean"] for result in results
        ),
    }
    summary = {
        "split": "validation",
        "examples": len(results),
        "records": aggregate["records"],
        "donors_per_example": args.donors_per_example,
        "donor_selection": "MSE quantiles from same-record windows; each donor avoids overlap with the target window, but donor windows may overlap one another",
        "regions": "signal-adaptive regions, capped at 100 ms",
        "consensus": "mean of per-donor LIME absolute coefficients normalized to sum to one",
        "lime_perturbations": args.perturbations,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "aggregate": aggregate,
        "per_example": results,
        "per_donor": donor_rows,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per-example.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[key for key, value in results[0].items() if not isinstance(value, (list, dict))],
        )
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in result.items() if not isinstance(value, (list, dict))}
            for result in results
        )
    with (output_dir / "per-donor.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[key for key in donor_rows[0] if key != "individual_top_regions"],
        )
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "individual_top_regions"}
            for row in donor_rows
        )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
