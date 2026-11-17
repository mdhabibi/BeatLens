"""Compare temporal LIME region sizes and replacement references on validation ECG beats."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge

from beatlens.infrastructure.qtdb import QTDBBeatExample, iter_qtdb_examples
from beatlens.models.qtdb_tpeak import QTDBTpeakRegressor

LEFT_SECONDS = 1.0


def normalized_input(example: QTDBBeatExample) -> np.ndarray:
    signal = example.signal.astype(np.float32)
    valid = example.valid_samples[:, None].astype(np.float32)
    count = max(float(valid.sum()), 1.0)
    means = (signal * valid).sum(axis=0, keepdims=True) / count
    variance = (((signal - means) ** 2) * valid).sum(axis=0, keepdims=True) / count
    scale = np.sqrt(variance).clip(min=1e-5)
    return (((signal - means) / scale) * valid).T.astype(np.float32)


def predict_ms(model: torch.nn.Module, inputs: np.ndarray, fs: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(inputs.astype(np.float32))
        normalized_sample = model(tensor)
        absolute_sample = (normalized_sample + 1.0) * (inputs.shape[-1] - 1) / 2.0
        return ((absolute_sample - round(fs * LEFT_SECONDS)) * (1000.0 / fs)).cpu().numpy()


def _half_height_region(envelope: np.ndarray, peak: int, lower: int, upper: int) -> tuple[int, int]:
    threshold = max(float(envelope[peak]) * 0.35, float(np.median(envelope[lower:upper])) + 0.05)
    start = peak
    end = peak + 1
    while start > lower and envelope[start - 1] >= threshold:
        start -= 1
    while end < upper and envelope[end] >= threshold:
        end += 1
    return start, end


def morphology_regions(inputs: np.ndarray, fs: float) -> list[tuple[int, int, str]]:
    """Partition a beat by signal-derived pre-QRS, QRS, and post-QRS activity.

    Boundaries use only the normalized waveform and the known .man beat anchor; no
    QTDB waveform-fiducial target is used to create explanation regions.
    """
    n_samples = inputs.shape[-1]
    anchor = round(fs * LEFT_SECONDS)
    smooth = gaussian_filter1d(inputs, sigma=max(1.0, fs * 0.04), axis=-1, mode="nearest")
    amplitude = np.sqrt(np.mean(smooth**2, axis=0))
    slope = np.sqrt(np.mean(np.diff(inputs, axis=-1, prepend=inputs[:, :1]) ** 2, axis=0))
    slope = gaussian_filter1d(slope, sigma=max(1.0, fs * 0.008), mode="nearest")

    qrs_lo = max(1, anchor - round(fs * 0.12))
    qrs_hi = min(n_samples - 1, anchor + round(fs * 0.12))
    qrs_peak = qrs_lo + int(np.argmax(slope[qrs_lo:qrs_hi]))
    threshold = max(float(slope[qrs_peak]) * 0.20, float(np.median(slope[qrs_lo:qrs_hi])) * 1.5)
    qrs_start = qrs_peak
    qrs_end = qrs_peak + 1
    while qrs_start > qrs_lo and slope[qrs_start - 1] >= threshold:
        qrs_start -= 1
    while qrs_end < qrs_hi and slope[qrs_end] >= threshold:
        qrs_end += 1
    qrs_start = max(qrs_lo, qrs_start - round(fs * 0.012))
    qrs_end = min(qrs_hi, qrs_end + round(fs * 0.012))
    if qrs_end - qrs_start < round(fs * 0.04):
        qrs_start = max(qrs_lo, anchor - round(fs * 0.04))
        qrs_end = min(qrs_hi, anchor + round(fs * 0.04))

    pre_hi = max(1, qrs_start - round(fs * 0.04))
    pre_lo = max(0, min(anchor - round(fs * 0.60), pre_hi - round(fs * 0.18)))
    post_lo = min(n_samples - 1, qrs_end + round(fs * 0.06))
    post_hi = min(n_samples, max(post_lo + 1, anchor + round(fs * 0.72)))
    pre_peak = pre_lo + int(np.argmax(amplitude[pre_lo:pre_hi])) if pre_hi > pre_lo else pre_lo
    post_peak = post_lo + int(np.argmax(amplitude[post_lo:post_hi])) if post_hi > post_lo else post_lo
    pre_start, pre_end = _half_height_region(amplitude, pre_peak, pre_lo, pre_hi) if pre_hi > pre_lo else (pre_lo, pre_hi)
    post_start, post_end = _half_height_region(amplitude, post_peak, post_lo, post_hi) if post_hi > post_lo else (post_lo, post_hi)

    boundaries = sorted(
        {
            0,
            n_samples,
            max(0, pre_start),
            min(n_samples, pre_end),
            max(0, qrs_start),
            min(n_samples, qrs_end),
            max(0, post_start),
            min(n_samples, post_end),
        }
    )
    max_region_samples = max(1, round(fs * 0.10))
    regions: list[tuple[int, int, str]] = []
    for phase_index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        part_index = 0
        for start in range(left, right, max_region_samples):
            end = min(start + max_region_samples, right)
            if end - start >= max(2, round(fs * 0.008)):
                regions.append((start, end, f"adaptive_{phase_index}_{part_index}"))
                part_index += 1
    return regions


def fixed_regions(n_samples: int, fs: float, segment_seconds: float = 0.025) -> list[tuple[int, int, str]]:
    width = max(1, round(fs * segment_seconds))
    return [
        (start, min(start + width, n_samples), f"fixed_{index}")
        for index, start in enumerate(range(0, n_samples, width))
    ]


def donor_for(example: QTDBBeatExample, candidates: list[QTDBBeatExample]) -> QTDBBeatExample:
    """Choose a same-record donor window that does not overlap the target window."""
    window_samples = len(example.signal)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.beat_index != example.beat_index
        and abs(candidate.anchor_sample - example.anchor_sample) >= window_samples
    ]
    if not eligible:
        raise ValueError(f"No target-separated same-record donor beat available for {example.record_id}")
    target = normalized_input(example)
    fs = example.sampling_rate_hz
    center_start = max(0, round(fs * 0.70))
    center_end = min(target.shape[-1], round(fs * 1.55))
    scored = []
    for candidate in eligible:
        donor = normalized_input(candidate)
        if donor.shape != target.shape or not candidate.valid_samples.all():
            continue
        score = float(np.mean((target[:, center_start:center_end] - donor[:, center_start:center_end]) ** 2))
        scored.append((score, candidate))
    if not scored:
        raise ValueError(f"No full-window same-record donor available for {example.record_id}")
    return min(scored, key=lambda item: item[0])[1]


def replace_region(
    inputs: np.ndarray,
    donor: np.ndarray,
    start: int,
    end: int,
    *,
    fade_samples: int,
) -> np.ndarray:
    output = inputs.copy()
    length = end - start
    fade = min(fade_samples, length // 3)
    weights = np.ones(length, dtype=np.float32)
    if fade > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32))
        weights[:fade] = ramp
        weights[-fade:] = ramp[::-1]
    output[:, start:end] = (
        inputs[:, start:end] * (1.0 - weights[None]) + donor[:, start:end] * weights[None]
    )
    return output


def perturb(
    inputs: np.ndarray,
    donor: np.ndarray,
    regions: list[tuple[int, int, str]],
    keep_mask: np.ndarray,
    *,
    fade_samples: int,
) -> np.ndarray:
    output = inputs.copy()
    for index, (start, end, _) in enumerate(regions):
        if not keep_mask[index]:
            output = replace_region(output, donor, start, end, fade_samples=fade_samples)
    return output


def transition_slope_delta_rms(
    inputs: np.ndarray,
    perturbed: np.ndarray,
    regions: list[tuple[int, int, str]],
    keep_mask: np.ndarray,
    *,
    radius_samples: int,
) -> float:
    """Measure added first-difference energy near masked/unmasked transitions."""
    boundaries = [
        regions[index][0]
        for index in range(1, len(regions))
        if keep_mask[index] != keep_mask[index - 1]
    ]
    if not boundaries:
        return 0.0
    original_slope = np.diff(inputs, axis=-1)
    changed_slope = np.diff(perturbed, axis=-1)
    local_deltas = []
    for boundary in boundaries:
        start = max(0, boundary - radius_samples - 1)
        end = min(original_slope.shape[-1], boundary + radius_samples)
        local_deltas.append(changed_slope[:, start:end] - original_slope[:, start:end])
    values = np.concatenate(local_deltas, axis=-1)
    return float(np.sqrt(np.mean(values**2)))


def lime_scores(
    inputs: np.ndarray,
    donor: np.ndarray,
    regions: list[tuple[int, int, str]],
    predict: Any,
    *,
    perturbations: int,
    seed: int,
    fade_samples: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    masks = (rng.random((perturbations, len(regions))) < 0.8).astype(np.float32)
    masks[0] = 1.0
    batch = np.stack(
        [perturb(inputs, donor, regions, mask.astype(bool), fade_samples=fade_samples) for mask in masks]
    )
    outcomes = np.asarray(predict(batch), dtype=np.float64).reshape(-1)
    distances = np.sqrt(1.0 - masks.mean(axis=1))
    weights = np.exp(-((distances / 0.25) ** 2))
    regressor = Ridge(alpha=1.0, fit_intercept=True).fit(masks, outcomes, sample_weight=weights)
    fitted = regressor.predict(masks)
    mean = float(np.average(outcomes, weights=weights))
    residual = float(np.sum(weights * (outcomes - fitted) ** 2))
    total = float(np.sum(weights * (outcomes - mean) ** 2))
    r2 = 1.0 - residual / total if total > 0 else 1.0
    return np.abs(np.asarray(regressor.coef_, dtype=np.float64)), r2


def compare_one(
    model: torch.nn.Module,
    example: QTDBBeatExample,
    donor_example: QTDBBeatExample,
    *,
    perturbations: int,
    seed: int,
) -> dict[str, Any]:
    fs = example.sampling_rate_hz
    inputs = normalized_input(example)
    donor = normalized_input(donor_example)
    zero = np.zeros_like(inputs)
    smooth = gaussian_filter1d(inputs, sigma=12.0, axis=-1, mode="nearest").astype(np.float32)
    predict = lambda batch: predict_ms(model, batch, fs)
    original = float(predict(inputs[None])[0])
    true_offset = float(example.event_offsets["t_peak"][0] / fs * 1000.0)
    methods: dict[str, Any] = {}
    for variant_name, regions, replacement, fade_samples in (
        ("fixed_25ms_zero", fixed_regions(inputs.shape[-1], fs), zero, 0),
        ("fixed_25ms_smooth", fixed_regions(inputs.shape[-1], fs), smooth, 0),
        ("fixed_25ms_donor", fixed_regions(inputs.shape[-1], fs), donor, round(fs * 0.02)),
        ("adaptive_zero", morphology_regions(inputs, fs), zero, 0),
        ("adaptive_smooth", morphology_regions(inputs, fs), smooth, 0),
        ("adaptive_donor", morphology_regions(inputs, fs), donor, round(fs * 0.02)),
    ):
        scores, local_r2 = lime_scores(
            inputs,
            replacement,
            regions,
            predict,
            perturbations=perturbations,
            seed=seed,
            fade_samples=fade_samples,
        )
        region_lengths = np.asarray([end - start for start, end, _ in regions])
        target_samples = int(np.ceil(inputs.shape[-1] * 0.20))
        ranked = np.argsort(scores)[::-1]
        top: list[int] = []
        selected_samples = 0
        for region_index in ranked:
            top.append(int(region_index))
            selected_samples += int(region_lengths[region_index])
            if selected_samples >= target_samples:
                break
        top_mask = np.ones(len(regions), dtype=bool)
        top_mask[top] = False
        top_perturbed = perturb(inputs, replacement, regions, top_mask, fade_samples=fade_samples)
        top_shift = abs(float(predict(top_perturbed[None])[0]) - original)
        edge_radius = max(1, round(fs * 0.02))
        top_slope_delta = transition_slope_delta_rms(
            inputs, top_perturbed, regions, top_mask, radius_samples=edge_radius
        )
        rng = np.random.default_rng(seed + 10_000)
        random_shifts = []
        random_sample_fractions = []
        random_slope_deltas = []
        for _ in range(20):
            random_mask = np.ones(len(regions), dtype=bool)
            random_samples = 0
            for region_index in rng.permutation(len(regions)):
                random_mask[region_index] = False
                random_samples += int(region_lengths[region_index])
                if random_samples >= target_samples:
                    break
            random_perturbed = perturb(
                inputs, replacement, regions, random_mask, fade_samples=fade_samples
            )
            prediction = float(predict(random_perturbed[None])[0])
            random_shifts.append(abs(prediction - original))
            random_sample_fractions.append(random_samples / inputs.shape[-1])
            random_slope_deltas.append(
                transition_slope_delta_rms(
                    inputs, random_perturbed, regions, random_mask, radius_samples=edge_radius
                )
            )
        methods[variant_name] = {
            "region_count": len(regions),
            "top_region_count": len(top),
            "top_region_intervals_samples": [list(regions[index]) for index in top],
            "top_replaced_sample_fraction": selected_samples / inputs.shape[-1],
            "random_replaced_sample_fraction_mean": float(np.mean(random_sample_fractions)),
            "weighted_local_r2": local_r2,
            "top_region_prediction_shift_ms": top_shift,
            "random_region_prediction_shift_mean_ms": float(np.mean(random_shifts)),
            "random_region_prediction_shift_p90_ms": float(np.percentile(random_shifts, 90)),
            "top_transition_slope_delta_rms": top_slope_delta,
            "random_transition_slope_delta_rms_mean": float(np.mean(random_slope_deltas)),
            "random_transition_slope_delta_rms_p90": float(np.percentile(random_slope_deltas, 90)),
            "top_shift_exceeds_random_mean": top_shift > float(np.mean(random_shifts)),
            "region_boundaries_samples": [[start, end, name] for start, end, name in regions],
        }
    return {
        "record_id": example.record_id,
        "beat_index": example.beat_index,
        "donor_beat_index": donor_example.beat_index,
        "reference_t_peak_ms": true_offset,
        "prediction_ms": original,
        "absolute_error_ms": abs(original - true_offset),
        **methods,
    }


def write_boundary_inspection(
    results: list[dict[str, Any]],
    examples: list[QTDBBeatExample],
    output_path: Path,
) -> None:
    """Write an offline plot for up to two adaptive-donor failure examples."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    by_key = {(example.record_id, example.beat_index): example for example in examples}
    failures = [
        result
        for result in results
        if not result["adaptive_donor"]["top_shift_exceeds_random_mean"]
    ]
    chosen = failures[:2] or sorted(
        results,
        key=lambda result: result["adaptive_donor"]["top_region_prediction_shift_ms"],
    )[:2]
    figure = make_subplots(
        rows=len(chosen) * 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[
            title
            for result in chosen
            for title in (
                f"{result['record_id']} beat {result['beat_index']} · lead 1",
                f"{result['record_id']} beat {result['beat_index']} · lead 2",
            )
        ],
    )
    for example_index, result in enumerate(chosen):
        example = by_key[(result["record_id"], result["beat_index"])]
        signal = normalized_input(example)
        fs = example.sampling_rate_hz
        times = (np.arange(signal.shape[-1]) - round(fs * LEFT_SECONDS)) * 1000.0 / fs
        regions = morphology_regions(signal, fs)
        selected_intervals = result["adaptive_donor"]["top_region_intervals_samples"]
        first_row = example_index * 2 + 1
        reference_ms = result["reference_t_peak_ms"]
        prediction_ms = result["prediction_ms"]
        for lead in range(min(2, signal.shape[0])):
            row = first_row + lead
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=signal[lead],
                    mode="lines",
                    name=example.lead_names[lead] if lead < len(example.lead_names) else f"lead {lead + 1}",
                    line={"width": 1},
                    showlegend=lead == 0 and example_index == 0,
                ),
                row=row,
                col=1,
            )
            for start, end, _ in regions:
                figure.add_vline(
                    x=(start - round(fs * LEFT_SECONDS)) * 1000.0 / fs,
                    line_color="#94a3b8",
                    line_width=0.6,
                    row=row,
                    col=1,
                )
            for start, end, _ in selected_intervals:
                figure.add_vrect(
                    x0=(start - round(fs * LEFT_SECONDS)) * 1000.0 / fs,
                    x1=(end - round(fs * LEFT_SECONDS)) * 1000.0 / fs,
                    fillcolor="#2563eb",
                    opacity=0.10,
                    line_width=0,
                    row=row,
                    col=1,
                )
            figure.add_vline(x=0, line_color="#64748b", line_dash="dot", row=row, col=1)
            figure.add_vline(
                x=reference_ms, line_color="#16a34a", line_dash="dash", row=row, col=1
            )
            figure.add_vline(
                x=prediction_ms, line_color="#dc2626", line_dash="dashdot", row=row, col=1
            )
            figure.update_yaxes(title_text=example.lead_names[lead], row=row, col=1)
    figure.update_xaxes(title_text="Time relative to beat anchor (ms)", row=len(chosen) * 2, col=1)
    figure.update_layout(
        title="Adaptive ECG regions and top donor-perturbation regions",
        height=max(620, len(chosen) * 460),
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 80, "b": 40},
        legend={"orientation": "h", "y": 1.04},
    )
    figure.add_annotation(
        text="Gray: adaptive boundaries · blue: top-ranked regions · green: reference T peak · red: model prediction · zero: beat anchor",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.01,
        showarrow=False,
        font={"size": 11},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/qtdb-tpeak-known-subject-groups/best.pt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qtdb-perturbation-comparison"))
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--perturbations", type=int, default=256)
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
    candidates = [
        example
        for record_id in sorted(grouped)
        for example in grouped[record_id]
        if len(example.event_offsets["t_peak"]) == 1 and len(grouped[record_id]) > 1
    ]
    rng = np.random.default_rng(args.seed)
    record_ids = sorted({example.record_id for example in candidates})
    rng.shuffle(record_ids)
    selected: list[QTDBBeatExample] = []
    for record_id in record_ids:
        choices = [example for example in candidates if example.record_id == record_id]
        selected.append(choices[int(rng.integers(len(choices)))])
        if len(selected) >= args.examples:
            break
    if not selected:
        raise SystemExit("No eligible validation beats with target-separated same-record donor windows")

    results = []
    for index, example in enumerate(selected):
        donor = donor_for(example, grouped[example.record_id])
        results.append(compare_one(model, example, donor, perturbations=args.perturbations, seed=args.seed + index))
        print(f"[{index + 1}/{len(selected)}] {example.record_id} beat {example.beat_index}", flush=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "split": "validation",
        "examples": len(results),
        "selection_seed": args.seed,
        "lime_perturbations": args.perturbations,
        "replacement_variants": {
            "zero": "zero in per-window z-scored input, direct replacement",
            "smooth": "Gaussian-smoothed target window (sigma 12 samples), direct replacement",
            "donor": "closest same-record window separated from the target window, with 20 ms cosine cross-fades",
        },
        "region_variants": {
            "fixed_25ms": "contiguous 25 ms bins",
            "adaptive": "waveform-derived pre-QRS, QRS, and post-QRS activity boundaries, subdivided to at most 100 ms; no q1c target used to define regions",
        },
        "results": results,
        "aggregate": {},
    }
    variant_names = (
        "fixed_25ms_zero",
        "fixed_25ms_smooth",
        "fixed_25ms_donor",
        "adaptive_zero",
        "adaptive_smooth",
        "adaptive_donor",
    )
    for variant_name in variant_names:
        summary["aggregate"][variant_name] = {
            metric: float(np.mean([row[variant_name][metric] for row in results]))
            for metric in (
                "region_count",
                "weighted_local_r2",
                "top_region_prediction_shift_ms",
                "random_region_prediction_shift_mean_ms",
                "random_region_prediction_shift_p90_ms",
                "top_transition_slope_delta_rms",
                "random_transition_slope_delta_rms_mean",
                "random_transition_slope_delta_rms_p90",
            )
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = []
    for result in results:
        for variant_name in variant_names:
            rows.append(
                {
                    "record_id": result["record_id"],
                    "beat_index": result["beat_index"],
                    "donor_beat_index": result["donor_beat_index"],
                    "method": variant_name,
                    "absolute_error_ms": result["absolute_error_ms"],
                    **{
                        key: value
                        for key, value in result[variant_name].items()
                        if key not in {"region_boundaries_samples", "top_region_intervals_samples"}
                    },
                }
            )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_boundary_inspection(
        results,
        selected,
        output_dir / "adaptive-boundary-inspection.html",
    )
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
