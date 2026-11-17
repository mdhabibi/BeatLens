"""Prototype Integrated Gradients and temporal LIME on validation T-peak predictions."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from beatlens.explainers.temporal_attribution import integrated_gradients, temporal_lime
from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor

LEFT_SECONDS = 1.0
LIME_SEEDS = (19, 41, 73, 101)
IG_NOISE_SEEDS = (211, 307)


def normalized_input(example: QTDBBeatExample) -> np.ndarray:
    signal = example.signal.astype(np.float32)
    valid = example.valid_samples[:, None].astype(np.float32)
    count = max(float(valid.sum()), 1.0)
    means = (signal * valid).sum(axis=0, keepdims=True) / count
    variance = (((signal - means) ** 2) * valid).sum(axis=0, keepdims=True) / count
    scale = np.sqrt(variance).clip(min=1e-5)
    return (((signal - means) / scale) * valid).T.astype(np.float32)


def output_ms(model: torch.nn.Module, inputs: torch.Tensor, fs: float) -> torch.Tensor:
    normalized_sample = model(inputs)
    absolute_sample = (normalized_sample + 1.0) * (inputs.shape[-1] - 1) / 2.0
    offset_samples = absolute_sample - round(fs * LEFT_SECONDS)
    return offset_samples * (1000.0 / fs)


def predict_ms(model: torch.nn.Module, batch: np.ndarray, fs: float, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            inputs = torch.from_numpy(batch[start : start + batch_size])
            outputs.extend(output_ms(model, inputs, fs).cpu().numpy())
    return np.asarray(outputs)


def segment_sums(scores: np.ndarray, segment_samples: int) -> np.ndarray:
    return np.asarray(
        [
            np.sum(scores[start : min(start + segment_samples, len(scores))])
            for start in range(0, len(scores), segment_samples)
        ]
    )


def top_segments(scores: np.ndarray, fraction: float) -> list[int]:
    count = max(1, int(np.ceil(len(scores) * fraction)))
    return np.argsort(np.abs(scores))[-count:][::-1].astype(int).tolist()


def pairwise_jaccard(rankings: list[list[int]]) -> float | None:
    if len(rankings) < 2:
        return None
    values = []
    for left, right in combinations(rankings, 2):
        a, b = set(left), set(right)
        values.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(values))


def masked_segments(
    inputs: np.ndarray,
    segments: list[int],
    segment_samples: int,
    replacement: np.ndarray,
) -> np.ndarray:
    perturbed = inputs.copy()
    for segment in segments:
        start = segment * segment_samples
        end = min(start + segment_samples, inputs.shape[-1])
        perturbed[:, start:end] = replacement[:, start:end]
    return perturbed


def faithfulness(
    model: torch.nn.Module,
    inputs: np.ndarray,
    fs: float,
    reference_offset_ms: float,
    scores: np.ndarray,
    segment_samples: int,
    replacement: np.ndarray,
    seed: int,
    random_trials: int = 20,
) -> dict[str, float]:
    n_segments = int(np.ceil(inputs.shape[-1] / segment_samples))
    selected = top_segments(scores, 0.20)
    original_prediction = float(predict_ms(model, inputs[None], fs)[0])
    deleted_prediction = float(
        predict_ms(model, masked_segments(inputs, selected, segment_samples, replacement)[None], fs)[0]
    )
    rng = np.random.default_rng(seed)
    random_changes: list[float] = []
    random_error_changes: list[float] = []
    original_error = abs(original_prediction - reference_offset_ms)
    for _ in range(random_trials):
        random_selected = rng.choice(n_segments, size=len(selected), replace=False).tolist()
        prediction = float(
            predict_ms(
                model,
                masked_segments(inputs, random_selected, segment_samples, replacement)[None],
                fs,
            )[0]
        )
        random_changes.append(abs(prediction - original_prediction))
        random_error_changes.append(abs(prediction - reference_offset_ms) - original_error)
    return {
        "deleted_fraction": len(selected) / n_segments,
        "top_deletion_output_shift_ms": abs(deleted_prediction - original_prediction),
        "random_deletion_output_shift_mean_ms": float(np.mean(random_changes)),
        "top_deletion_error_change_ms": abs(deleted_prediction - reference_offset_ms) - original_error,
        "random_deletion_error_change_mean_ms": float(np.mean(random_error_changes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-explanations"))
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--max-per-record", type=int, default=2)
    parser.add_argument("--lime-perturbations", type=int, default=512)
    parser.add_argument("--ig-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = QTDBTpeakRegressor(input_leads=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidates = [
        example
        for example in iter_qtdb_examples(args.qtdb_dir, args.split_csv, "validation")
        if len(example.event_offsets["t_peak"]) == 1
    ]
    by_record: dict[str, list[QTDBBeatExample]] = {}
    for example in candidates:
        by_record.setdefault(example.record_id, []).append(example)
    selection_rng = np.random.default_rng(args.seed)
    record_ids = sorted(by_record)
    selection_rng.shuffle(record_ids)
    selected: list[QTDBBeatExample] = []
    for record_id in record_ids:
        record_examples = by_record[record_id]
        order = selection_rng.permutation(len(record_examples))
        selected.extend(
            record_examples[index]
            for index in order[: args.max_per_record]
        )
        if len(selected) >= args.examples:
            selected = selected[: args.examples]
            break
    if not selected:
        raise SystemExit("No eligible single-T-peak validation examples found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_samples = max(1, round(selected[0].sampling_rate_hz * 0.025))
    summaries: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []

    for example_index, example in enumerate(selected):
        fs = example.sampling_rate_hz
        inputs_np = normalized_input(example)
        inputs = torch.from_numpy(inputs_np[None]).to(device)
        output_fn = lambda normalized_output, sampling_rate=fs: (
            (normalized_output + 1.0) * (inputs.shape[-1] - 1) / 2.0
            - round(sampling_rate * LEFT_SECONDS)
        ) * (1000.0 / sampling_rate)
        with torch.no_grad():
            prediction_ms = float(output_fn(model(inputs)).item())
        reference_ms = float(example.event_offsets["t_peak"][0] / fs * 1000.0)

        # Compare a flat mean-signal reference with a smooth local waveform reference.
        smooth_baseline_np = gaussian_filter1d(inputs_np, sigma=12.0, axis=-1, mode="nearest")
        zero_baseline = torch.zeros_like(inputs)
        smooth_baseline = torch.from_numpy(smooth_baseline_np[None].astype(np.float32))
        ig_zero, completeness_zero = integrated_gradients(
            model, inputs, output_fn, baseline=zero_baseline, steps=args.ig_steps
        )
        ig_smooth, completeness_smooth = integrated_gradients(
            model, inputs, output_fn, baseline=smooth_baseline, steps=args.ig_steps
        )
        ig_zero_signed = ig_zero[0].sum(dim=0).cpu().numpy()
        ig_zero_magnitude = ig_zero[0].abs().sum(dim=0).cpu().numpy()
        ig_smooth_signed = ig_smooth[0].sum(dim=0).cpu().numpy()
        ig_smooth_magnitude = ig_smooth[0].abs().sum(dim=0).cpu().numpy()
        ig_zero_segments = segment_sums(ig_zero_magnitude, segment_samples)
        ig_smooth_segments = segment_sums(ig_smooth_magnitude, segment_samples)

        predictor = lambda batch, sampling_rate=fs: predict_ms(model, batch, sampling_rate)
        lime_zero_results = [
            temporal_lime(
                inputs_np,
                predictor,
                segment_samples=segment_samples,
                replacement=np.zeros_like(inputs_np),
                perturbations=args.lime_perturbations,
                seed=seed,
            )
            for seed in LIME_SEEDS
        ]
        lime_smooth_results = [
            temporal_lime(
                inputs_np,
                predictor,
                segment_samples=segment_samples,
                replacement=smooth_baseline_np,
                perturbations=args.lime_perturbations,
                seed=seed,
            )
            for seed in LIME_SEEDS
        ]

        def lime_scores(results: list[Any]) -> tuple[np.ndarray, np.ndarray]:
            coefficients = np.stack([result.coefficients for result in results])
            return coefficients.mean(axis=0), np.abs(coefficients).mean(axis=0)

        lime_zero_signed, lime_zero_magnitude = lime_scores(lime_zero_results)
        lime_smooth_signed, lime_smooth_magnitude = lime_scores(lime_smooth_results)

        # Explanation stability under mild input noise and baseline/seed changes.
        ig_noise_rankings: list[list[int]] = [top_segments(ig_zero_segments, 0.10)]
        for seed in IG_NOISE_SEEDS:
            generator = torch.Generator().manual_seed(seed + example_index)
            noise = torch.randn(inputs.shape, generator=generator) * 0.03
            noisy = inputs + noise
            noisy_ig, _ = integrated_gradients(
                model, noisy, output_fn, baseline=zero_baseline, steps=args.ig_steps
            )
            noisy_magnitude = noisy_ig[0].abs().sum(dim=0).cpu().numpy()
            ig_noise_rankings.append(
                top_segments(segment_sums(noisy_magnitude, segment_samples), 0.10)
            )
        lime_zero_rankings = [
            top_segments(np.abs(result.coefficients), 0.10) for result in lime_zero_results
        ]
        lime_smooth_rankings = [
            top_segments(np.abs(result.coefficients), 0.10) for result in lime_smooth_results
        ]

        ig_zero_deletion = faithfulness(
            model,
            inputs_np,
            fs,
            reference_ms,
            ig_zero_segments,
            segment_samples,
            np.zeros_like(inputs_np),
            seed=500 + example_index,
        )
        ig_smooth_deletion = faithfulness(
            model,
            inputs_np,
            fs,
            reference_ms,
            ig_smooth_segments,
            segment_samples,
            smooth_baseline_np,
            seed=500 + example_index,
        )
        lime_zero_deletion = faithfulness(
            model,
            inputs_np,
            fs,
            reference_ms,
            lime_zero_magnitude,
            segment_samples,
            np.zeros_like(inputs_np),
            seed=500 + example_index,
        )
        lime_smooth_deletion = faithfulness(
            model,
            inputs_np,
            fs,
            reference_ms,
            lime_smooth_magnitude,
            segment_samples,
            smooth_baseline_np,
            seed=500 + example_index,
        )

        predicted_segment_scores = [
            {
                "ig_zero_signed_ms": float(ig_zero_signed[start : min(start + segment_samples, len(ig_zero_signed))].sum()),
                "ig_zero_absolute": float(ig_zero_segments[segment]),
                "ig_smooth_signed_ms": float(ig_smooth_signed[start : min(start + segment_samples, len(ig_smooth_signed))].sum()),
                "ig_smooth_absolute": float(ig_smooth_segments[segment]),
                "lime_zero_mean_coefficient_ms": float(lime_zero_signed[segment]),
                "lime_zero_mean_absolute_coefficient_ms": float(lime_zero_magnitude[segment]),
                "lime_smooth_mean_coefficient_ms": float(lime_smooth_signed[segment]),
                "lime_smooth_mean_absolute_coefficient_ms": float(lime_smooth_magnitude[segment]),
            }
            for segment, start in enumerate(range(0, inputs_np.shape[-1], segment_samples))
        ]
        for segment, score in enumerate(predicted_segment_scores):
            start_sample = segment * segment_samples
            end_sample = min(start_sample + segment_samples, inputs_np.shape[-1])
            attribution_rows.append(
                {
                    "record_id": example.record_id,
                    "beat_index": example.beat_index,
                    "segment": segment,
                    "start_ms_from_anchor": (start_sample - round(fs * LEFT_SECONDS)) / fs * 1000.0,
                    "end_ms_from_anchor": (end_sample - round(fs * LEFT_SECONDS)) / fs * 1000.0,
                    **score,
                }
            )

        summaries.append(
            {
                "record_id": example.record_id,
                "beat_index": example.beat_index,
                "reference_t_peak_offset_ms": reference_ms,
                "predicted_t_peak_offset_ms": prediction_ms,
                "absolute_error_ms": abs(prediction_ms - reference_ms),
                "ig_zero_completeness_residual_ms": completeness_zero,
                "ig_smooth_completeness_residual_ms": completeness_smooth,
                "lime_zero_weighted_local_r2": [result.weighted_r2 for result in lime_zero_results],
                "lime_smooth_weighted_local_r2": [result.weighted_r2 for result in lime_smooth_results],
                "ig_top10_jaccard_under_noise": pairwise_jaccard(ig_noise_rankings),
                "ig_top10_jaccard_across_baselines": pairwise_jaccard(
                    [top_segments(ig_zero_segments, 0.10), top_segments(ig_smooth_segments, 0.10)]
                ),
                "lime_zero_top10_jaccard_across_seeds": pairwise_jaccard(lime_zero_rankings),
                "lime_smooth_top10_jaccard_across_seeds": pairwise_jaccard(lime_smooth_rankings),
                "lime_top10_jaccard_across_baselines": pairwise_jaccard(
                    [top_segments(lime_zero_magnitude, 0.10), top_segments(lime_smooth_magnitude, 0.10)]
                ),
                "ig_zero_deletion": ig_zero_deletion,
                "ig_smooth_deletion": ig_smooth_deletion,
                "lime_zero_deletion": lime_zero_deletion,
                "lime_smooth_deletion": lime_smooth_deletion,
            }
        )

    summary = {
        "model_checkpoint": str(args.checkpoint),
        "split": "validation",
        "examples": len(selected),
        "max_examples_per_record": args.max_per_record,
        "selection_seed": args.seed,
        "segment_duration_ms": segment_samples / selected[0].sampling_rate_hz * 1000.0,
        "lime_perturbations_per_seed": args.lime_perturbations,
        "lime_seeds": list(LIME_SEEDS),
        "ig_steps": args.ig_steps,
        "ig_noise_seeds": list(IG_NOISE_SEEDS),
        "primary_baseline": "zero in per-window z-scored input",
        "primary_baseline_rationale": "input is per-window centered; the smoothed waveform remains a sensitivity comparison",
        "baselines": {
            "zero": "flat mean signal in per-window z-scored input",
            "smooth": "Gaussian-smoothed input, sigma=12 samples (48ms at 250Hz)",
        },
        "lime_perturbation": "replace contiguous fixed 25ms bins with the selected baseline",
        "deletion_protocol": "replace 20% highest-ranked bins, compare with 20 random 20% replacements",
        "example_results": summaries,
        "aggregate": {
            "mean_prediction_error_ms": float(np.mean([row["absolute_error_ms"] for row in summaries])),
            "mean_lime_zero_weighted_r2": float(
                np.mean([np.mean(row["lime_zero_weighted_local_r2"]) for row in summaries])
            ),
            "mean_lime_smooth_weighted_r2": float(
                np.mean([np.mean(row["lime_smooth_weighted_local_r2"]) for row in summaries])
            ),
            "mean_ig_noise_stability_jaccard": float(
                np.mean([row["ig_top10_jaccard_under_noise"] for row in summaries])
            ),
            "mean_ig_baseline_stability_jaccard": float(
                np.mean([row["ig_top10_jaccard_across_baselines"] for row in summaries])
            ),
            "mean_lime_zero_seed_stability_jaccard": float(
                np.mean([row["lime_zero_top10_jaccard_across_seeds"] for row in summaries])
            ),
            "mean_lime_smooth_seed_stability_jaccard": float(
                np.mean([row["lime_smooth_top10_jaccard_across_seeds"] for row in summaries])
            ),
            "mean_lime_baseline_stability_jaccard": float(
                np.mean([row["lime_top10_jaccard_across_baselines"] for row in summaries])
            ),
            "mean_ig_zero_top_deletion_shift_ms": float(np.mean([row["ig_zero_deletion"]["top_deletion_output_shift_ms"] for row in summaries])),
            "mean_ig_zero_random_deletion_shift_ms": float(np.mean([row["ig_zero_deletion"]["random_deletion_output_shift_mean_ms"] for row in summaries])),
            "mean_ig_smooth_top_deletion_shift_ms": float(np.mean([row["ig_smooth_deletion"]["top_deletion_output_shift_ms"] for row in summaries])),
            "mean_ig_smooth_random_deletion_shift_ms": float(np.mean([row["ig_smooth_deletion"]["random_deletion_output_shift_mean_ms"] for row in summaries])),
            "mean_lime_zero_top_deletion_shift_ms": float(np.mean([row["lime_zero_deletion"]["top_deletion_output_shift_ms"] for row in summaries])),
            "mean_lime_zero_random_deletion_shift_ms": float(np.mean([row["lime_zero_deletion"]["random_deletion_output_shift_mean_ms"] for row in summaries])),
            "mean_lime_smooth_top_deletion_shift_ms": float(np.mean([row["lime_smooth_deletion"]["top_deletion_output_shift_ms"] for row in summaries])),
            "mean_lime_smooth_random_deletion_shift_ms": float(np.mean([row["lime_smooth_deletion"]["random_deletion_output_shift_mean_ms"] for row in summaries])),
        },
    }
    (args.output_dir / "explanation-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "temporal-attributions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attribution_rows[0]))
        writer.writeheader()
        writer.writerows(attribution_rows)

    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
