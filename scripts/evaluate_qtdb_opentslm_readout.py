"""Evaluate the locked local OpenTSLM readout on QTDB or LUDB."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import torch

from compare_qtdb_opentslm_ridge import block_fusion_kernel, fusion_kernel
from probe_qtdb_opentslm_encoder import make_encoder_extractor


def predict(features, checkpoint, waveforms=None):
    if features.ndim != 2 or features.shape[1] != checkpoint["feature_count"]:
        raise ValueError("Feature shape differs from the locked checkpoint")
    cropped = features.index_select(1, checkpoint["feature_columns"])
    if "raw_columns" in checkpoint:
        if waveforms is None:
            raise ValueError("Fusion checkpoint requires the original normalized waveforms")
        raw = waveforms.flatten(1).index_select(1, checkpoint["raw_columns"])
        cropped = torch.cat((
            0.75 ** 0.5 * (cropped - checkpoint["encoder_block_center"]) / checkpoint["encoder_block_scale"],
            0.5 * (raw - checkpoint["raw_block_center"]) / checkpoint["raw_block_scale"],
        ), dim=1)
    normalized = (cropped - checkpoint["center"]) / checkpoint["scale"]
    if "kernel_split" in checkpoint:
        normalized = block_fusion_kernel(
            normalized, checkpoint["kernel_support"], checkpoint["kernel_split"],
            interaction=checkpoint.get("kernel_interaction", False),
            interaction_weight=checkpoint.get("kernel_interaction_weight", 0.25),
        )
    elif "kernel_support" in checkpoint:
        normalized = fusion_kernel(normalized, checkpoint["kernel_support"])
    return normalized @ checkpoint["weight"] + checkpoint["bias"]


def paired_interval(windows, differences):
    grouped = defaultdict(list)
    for window, difference in zip(windows, differences.tolist(), strict=True):
        grouped[window.get("subject", window["record"])].append(difference)
    totals = torch.tensor([sum(values) for values in grouped.values()])
    counts = torch.tensor([len(values) for values in grouped.values()])
    sample = torch.randint(
        len(totals), (20_000, len(totals)), generator=torch.Generator().manual_seed(544)
    )
    means = totals[sample].sum(1) / counts[sample].sum(1)
    return torch.quantile(means, torch.tensor([0.025, 0.975])).tolist()


def error_tail_metrics(errors):
    quantiles = torch.quantile(errors, errors.new_tensor([0.50, 0.90, 0.95, 0.99]))
    excess = (errors - 40).clamp_min(0)
    return {
        **dict(zip(("median_ms", "p90_ms", "p95_ms", "p99_ms"), quantiles.tolist())),
        "mean_excess_over_40ms": excess.mean().item(),
        "over_80ms": (errors > 80).sum().item(),
    }


def compare_errors(windows, primary, comparison):
    differences = comparison - primary
    primary_hit = primary <= 40
    comparison_hit = comparison <= 40
    hit_change_pp = 100 * (comparison_hit.float() - primary_hit.float())
    by_record = defaultdict(list)
    for window, difference in zip(windows, differences.tolist(), strict=True):
        by_record[window["record"]].append(difference)
    record_means = {
        record: sum(values) / len(values) for record, values in by_record.items()
    }
    return {
        "comparison_mae_ms": comparison.mean().item(),
        "comparison_within_40ms": comparison_hit.float().mean().item(),
        "primary_error_tail": error_tail_metrics(primary),
        "comparison_error_tail": error_tail_metrics(comparison),
        "comparison_minus_primary_within_40ms_pp": hit_change_pp.mean().item(),
        "comparison_minus_primary_within_40ms_pp_ci95": paired_interval(windows, hit_change_pp),
        "windows_gained_within_40ms": (comparison_hit & ~primary_hit).sum().item(),
        "windows_lost_within_40ms": (primary_hit & ~comparison_hit).sum().item(),
        "comparison_minus_primary_mae_ms": differences.mean().item(),
        "comparison_minus_primary_mae_ms_ci95": paired_interval(windows, differences),
        "records_comparison_better": sum(value < -1e-6 for value in record_means.values()),
        "records_comparison_worse": sum(value > 1e-6 for value in record_means.values()),
        "records_tied": sum(abs(value) <= 1e-6 for value in record_means.values()),
        "comparison_minus_primary_mae_ms_by_record": record_means,
    }


def record_error_summary(windows, reference, **errors_by_model):
    indices = defaultdict(list)
    for index, window in enumerate(windows):
        indices[window["record"]].append(index)
    return {
        record: {
            "windows": len(rows),
            "reference_median_ms": torch.quantile(reference[rows], 0.5).item(),
            **{
                model: {
                    "mae_ms": errors[rows].mean().item(),
                    "over_80ms": (errors[rows] > 80).sum().item(),
                }
                for model, errors in errors_by_model.items()
            },
        }
        for record, rows in indices.items()
    }


def run(panel_path, checkpoint_path, output, external_panel=None, comparison_path=None,
        records=None):
    from transformers.utils import logging

    logging.set_verbosity_error()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    comparison = None
    if comparison_path is not None:
        comparison = torch.load(comparison_path, map_location="cpu", weights_only=True)
        for key in ("attention_fastpath_enabled", "sampling_rate_hz", "anchor",
                    "window_length", "feature_count"):
            if comparison[key] != checkpoint[key]:
                raise ValueError(f"Checkpoints differ in {key}")
    torch.backends.mha.set_fastpath_enabled(checkpoint["attention_fastpath_enabled"])
    panel = json.loads(panel_path.read_text())
    if external_panel:
        with gzip.open(external_panel, "rt") as stream:
            windows = json.load(stream)
    else:
        windows = [window for pair in panel["validation"] for window in pair]
        if {w["record"] for w in windows} & {w["record"] for pair in panel["train"] for w in pair}:
            raise ValueError("Development records overlap training records")
    if records:
        missing = set(records) - {w["record"] for w in windows}
        if missing:
            raise ValueError(f"Unknown external panel records: {sorted(missing)}")
        windows = [w for w in windows if w["record"] in records]
    if any(
        w["fs"] != checkpoint["sampling_rate_hz"] or w["anchor"] != checkpoint["anchor"]
        or len(w["signal"][0]) != checkpoint["window_length"] for w in windows
    ):
        raise ValueError("Development window geometry differs from the checkpoint")
    features = make_encoder_extractor()([w["signal"] for w in windows])
    waveforms = None
    if "raw_columns" in checkpoint or (comparison is not None and "raw_columns" in comparison):
        waveforms = torch.tensor([w["signal"] for w in windows])
    estimate = predict(features, checkpoint, waveforms)
    factor = 1000 / checkpoint["sampling_rate_hz"]
    reference = torch.tensor([(w["reference"] - w["anchor"]) * factor for w in windows])
    model_error = (estimate - reference).abs()
    prior_error = (panel["prior_offset"] * factor - reference).abs()
    if not torch.isfinite(model_error).all():
        raise ValueError("Non-finite model errors")
    result = {
        "checkpoint": str(checkpoint_path), "windows": len(windows),
        "records": len({w["record"] for w in windows}),
        "model_mae_ms": model_error.mean().item(),
        "prior_mae_ms": prior_error.mean().item(),
    }
    if external_panel:
        if all("subject" in window for window in windows):
            result["subjects"] = len({window["subject"] for window in windows})
        result["model_within_40ms"] = (model_error <= 40).float().mean().item()
        result["model_minus_prior_mae_ms_ci95"] = paired_interval(windows, model_error - prior_error)
    else:
        swap = estimate.reshape(-1, 2).flip(1).flatten()
        result["swap_mae_ms"] = (swap - reference).abs().mean().item()
    if comparison is not None:
        comparison_estimate = predict(features, comparison, waveforms)
        comparison_error = (comparison_estimate - reference).abs()
        if not torch.isfinite(comparison_error).all():
            raise ValueError("Non-finite comparison errors")
        result["comparison_checkpoint"] = str(comparison_path)
        result.update(compare_errors(windows, model_error, comparison_error))
        result["record_error_summary"] = record_error_summary(
            windows, reference, primary=model_error, comparison=comparison_error, prior=prior_error,
        )
    if records:
        result["window_predictions"] = [
            {
                **window,
                "reference_ms": label,
                "primary_prediction_ms": prediction,
                "comparison_prediction_ms": other,
                "prior_prediction_ms": panel["prior_offset"] * factor,
            }
            for window, label, prediction, other in zip(
                windows, reference.tolist(), estimate.tolist(), comparison_estimate.tolist(),
                strict=True,
            )
        ]
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in {"comparison_minus_primary_mae_ms_by_record", "record_error_summary",
                       "window_predictions"}
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("beatlens-opentslm-local-development.json"))
    parser.add_argument("--external-panel", type=Path, help="Audited LUDB evaluation windows")
    parser.add_argument("--compare-checkpoint", type=Path, help="Second checkpoint for a paired LUDB comparison")
    parser.add_argument("--records", nargs="+", help="Export waveforms and predictions for selected records")
    args = parser.parse_args()
    if args.compare_checkpoint and not args.external_panel:
        parser.error("--compare-checkpoint requires --external-panel")
    if args.records and not args.compare_checkpoint:
        parser.error("--records requires --compare-checkpoint and --external-panel")
    run(args.panel, args.checkpoint, args.output, args.external_panel, args.compare_checkpoint,
        args.records)
