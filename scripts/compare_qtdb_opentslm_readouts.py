"""Compare raw ECG, full OpenTSLM, and T-peak-region OpenTSLM ridge readouts."""

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch

from compare_qtdb_opentslm_ridge import block_fusion_kernel, fusion_kernel, huber_coefficients, ridge_grid

from evaluate_qtdb_opentslm_readout import error_tail_metrics

INTERACTION_WEIGHTS = {
    "opentslm_interaction_kernel_fusion": 0.25,
    "opentslm_interaction_50": 0.50,
    "opentslm_interaction_75": 0.75,
    "opentslm_interaction_100": 1.0,
}

PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0)
FS = 250
ANCHOR = 250
MARGIN_MS = 40


def read_patch_columns(log_path, feature_count, patch_count):
    prefix = "Encoder output shapes: "
    with log_path.open() as log:
        shapes = next((ast.literal_eval(line[len(prefix):]) for line in log
                       if line.startswith(prefix)), None)
    if not shapes or sum(map(math.prod, shapes)) != feature_count:
        raise ValueError("Encoder shapes in log do not match the feature cache")
    columns = []
    offset = 0
    for shape in shapes:
        if len(shape) < 2 or shape[-2] != patch_count:
            raise ValueError(f"Unexpected encoder patch axis: {shape}")
        size = math.prod(shape)
        columns.append(np.arange(offset, offset + size).reshape(-1, patch_count, shape[-1]))
        offset += size
    return np.concatenate(columns, axis=0).transpose(1, 0, 2).reshape(patch_count, -1)


def local_columns(patch_columns, fit_targets_ms, patch_size):
    margin_samples = MARGIN_MS * FS / 1000
    relative_samples = fit_targets_ms * FS / 1000
    lo = max(0, math.floor((ANCHOR + relative_samples.min() - margin_samples) / patch_size))
    hi = min(len(patch_columns), math.floor(
        (ANCHOR + relative_samples.max() + margin_samples) / patch_size
    ) + 1)
    if hi <= lo:
        raise ValueError("Training-derived patch interval is empty")
    return patch_columns[lo:hi].reshape(-1)


def normalize_feature_block(features, fit):
    train = features[fit]
    center = train.mean(dim=0)
    scale = (train - center).square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
    return (features - center) / scale


def compare(raw, encoder, target_ms, folds, groups, selected, records, patch_columns, patch_size,
            *, raw_local_only=False, fusion_only=False, tune_fusion=False, robust_fusion=False,
            nonlinear_fusion=False, block_kernel=False, interaction_kernel=False, tune_interaction=False, interaction_tradeoff=False, late_target_weighting=False, robust_interaction=False):
    target = torch.as_tensor(target_ms, device=raw.device)
    patch_count = math.ceil(625 / patch_size)
    raw_indices = np.arange(1250).reshape(2, 625)
    raw_columns = [
        raw_indices[:, start:start + patch_size].reshape(-1)
        for start in range(0, 625, patch_size)
    ]
    configurations = (("raw_local", raw),) if raw_local_only else (
        ("raw", raw), ("opentslm_full", encoder), ("opentslm_local", encoder),
        ("raw_local", raw),
    )
    if robust_interaction:
        configurations = (("opentslm_interaction_kernel_fusion", encoder),
                          ("opentslm_huber_interaction", encoder))
    elif late_target_weighting:
        configurations = (("opentslm_interaction_kernel_fusion", encoder),
                          ("opentslm_late_weighted", encoder))
    elif interaction_tradeoff:
        configurations = tuple((name, encoder) for name in INTERACTION_WEIGHTS)
    elif nonlinear_fusion:
        configurations = (("opentslm_raw_fusion", encoder), ("opentslm_nonlinear_fusion", encoder))
        if block_kernel:
            configurations += (("opentslm_block_kernel_fusion", encoder),)
        if interaction_kernel:
            configurations += (("opentslm_interaction_kernel_fusion", encoder),)
        if tune_interaction:
            configurations += (("opentslm_tuned_interaction", encoder),)
    elif fusion_only or tune_fusion or robust_fusion:
        configurations = (("opentslm_raw_fusion", encoder),)
    results = []
    for name, matrix in configurations:
        tuned_interaction = name == "opentslm_tuned_interaction"
        late_weight = 3.0 if name == "opentslm_late_weighted" else 1.0
        huber_interaction = name == "opentslm_huber_interaction"
        interaction = name in INTERACTION_WEIGHTS or tuned_interaction or late_weight != 1.0 or huber_interaction
        separate = name == "opentslm_block_kernel_fusion" or interaction
        fusion = separate or name in ("opentslm_raw_fusion", "opentslm_nonlinear_fusion")
        def predict(fit, evaluation, penalties, mix, interaction_weight):
            features = matrix
            kernel_split = None
            if name == "opentslm_local" or fusion:
                columns = local_columns(patch_columns, target_ms[fit], patch_size)
                features = matrix.index_select(1, torch.as_tensor(columns, device=matrix.device))
            if name == "raw_local" or fusion:
                patches = local_columns(np.arange(patch_count), target_ms[fit], patch_size)
                columns = np.concatenate([raw_columns[patch] for patch in patches])
                cropped_raw = raw.index_select(1, torch.as_tensor(columns, device=raw.device))
                if name == "raw_local":
                    features = cropped_raw
                else:
                    if separate:
                        kernel_split = features.shape[1]
                    features = torch.cat((
                        math.sqrt(mix) * normalize_feature_block(features, fit),
                        math.sqrt(1 - mix) * normalize_feature_block(cropped_raw, fit),
                    ), dim=1)
            return ridge_grid(
                features, target, fit, evaluation, records, "window", penalties,
                huber_ms=40.0 if robust_fusion or huber_interaction else None,
                nonlinear=name == "opentslm_nonlinear_fusion", kernel_split=kernel_split,
                interaction=interaction, interaction_weight=interaction_weight, late_weight=late_weight,
            )

        totals = np.zeros(3)
        penalties_by_fold = []
        mix_by_fold = []
        mixes = (0.5, 0.75, 0.9, 1.0) if tune_fusion else (0.75,)
        fixed_weight = INTERACTION_WEIGHTS.get(name, 0.25)
        weights = (0.25, 0.0, 0.5, 0.75, 1.0) if tuned_interaction else (fixed_weight,)
        settings = [(mix, weight) for mix in mixes for weight in weights]
        weights_by_fold = []
        held_out_errors = []
        held_out_late = []
        for outer in range(5):
            inner_errors = np.zeros((len(settings), len(PENALTIES)))
            for inner in range(5):
                if inner == outer:
                    continue
                fit = np.flatnonzero(selected & (folds != outer) & (folds != inner))
                evaluation = np.flatnonzero(folds == inner)
                for index, (mix, weight) in enumerate(settings):
                    scores = predict(fit, evaluation, PENALTIES, mix, weight)
                    inner_errors[index] += (scores - target[evaluation]).abs().sum(dim=1).cpu().numpy()
            mix_index, penalty_index = np.unravel_index(inner_errors.argmin(), inner_errors.shape)
            penalty = PENALTIES[penalty_index]
            mix, weight = settings[mix_index]
            weights_by_fold.append(weight)
            penalties_by_fold.append(penalty)
            mix_by_fold.append(mix)
            fit = np.flatnonzero(selected & (folds != outer))
            evaluation = np.flatnonzero(folds == outer)
            evaluation = evaluation[np.argsort(groups[evaluation], kind="stable")]
            shift = int(np.unique(groups[evaluation], return_counts=True)[1].max())
            if np.any(groups[evaluation] == np.roll(groups[evaluation], shift)):
                raise ValueError("Could not create a cross-group waveform swap")
            expected = target[evaluation]
            predicted = predict(fit, evaluation, (penalty,), mix, weight)[0]
            errors = torch.stack([
                (predicted - expected).abs(),
                (predicted.roll(shift) - expected).abs(),
                (target[fit].mean() - expected).abs(),
            ])
            if interaction_tradeoff or late_target_weighting or robust_interaction:
                held_out_errors.append(errors[0].cpu())
            if late_target_weighting or robust_interaction:
                held_out_late.append((expected > 345).cpu())
            totals += errors.sum(dim=1).cpu().numpy()
            print(json.dumps({"readout": name, "fold": outer, "mae_ms": errors[0].mean().item()}),
                  flush=True)
        result = {
            "readout": name, "windows": len(folds), "penalties_at_300_windows": penalties_by_fold,
            **dict(zip(("mae_ms", "swap_mae_ms", "prior_mae_ms"), (totals / len(folds)).tolist())),
        }
        if interaction_tradeoff:
            absolute_errors = torch.cat(held_out_errors)
            result.update(error_tail_metrics(absolute_errors))
            result["within_40ms"] = (absolute_errors <= 40).float().mean().item()
            result["interaction_weight"] = fixed_weight
        if late_target_weighting or robust_interaction:
            absolute_errors = torch.cat(held_out_errors)
            late = torch.cat(held_out_late)
            result["late_target_weight"] = late_weight
            for label, mask in (("through_345ms", ~late), ("after_345ms", late)):
                errors = absolute_errors[mask]
                result[label] = {
                    "windows": len(errors),
                    "mae_ms": errors.mean().item() if len(errors) else None,
                    "over_80ms": (errors > 80).sum().item(),
                }
        if tuned_interaction:
            result["interaction_weights_by_fold"] = weights_by_fold
        if tune_fusion:
            result["opentslm_mix_by_fold"] = mix_by_fold
        if robust_fusion or huber_interaction:
            result["huber_threshold_ms"] = 40.0
        results.append(result)
        print(json.dumps({"summary": result}), flush=True)
    return results


def fit_final(encoder, targets_ms, folds, selected, records, patch_columns, patch_size, output,
              *, raw=None, huber_ms=None, nonlinear=False, block_kernel=False, interaction_kernel=False,
              interaction_weight=0.25):
    target = torch.as_tensor(targets_ms, device=encoder.device)

    def design(fit):
        cols = torch.as_tensor(local_columns(patch_columns, targets_ms[fit], patch_size),
                               device=encoder.device)
        features = encoder.index_select(1, cols)
        metadata = {"feature_columns": cols.cpu()}
        if raw is not None:
            patches = local_columns(np.arange(len(patch_columns)), targets_ms[fit], patch_size)
            start, stop = patches[0] * patch_size, min((patches[-1] + 1) * patch_size, 625)
            indices = np.arange(1250).reshape(2, 625)
            raw_cols = np.concatenate([
                indices[:, i:min(i + patch_size, stop)].reshape(-1)
                for i in range(start, stop, patch_size)
            ])
            raw_cols = torch.as_tensor(raw_cols, device=encoder.device)
            samples = raw.index_select(1, raw_cols)
            if block_kernel:
                metadata["kernel_split"] = features.shape[1]
            blocks = []
            for name, block, fraction in (
                ("encoder", features, 0.75), ("raw", samples, 0.25),
            ):
                center = block[fit].mean(dim=0)
                centered = block - center
                scale = centered[fit].square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
                blocks.append(math.sqrt(fraction) * centered / scale)
                metadata[f"{name}_block_center"] = center.cpu()
                metadata[f"{name}_block_scale"] = scale.cpu()
            metadata["raw_columns"] = raw_cols.cpu()
            features = torch.cat(blocks, dim=1)
        return features, metadata

    errors = np.zeros(len(PENALTIES))
    for fold in range(5):
        fit = np.flatnonzero(selected & (folds != fold))
        evaluation = np.flatnonzero(folds == fold)
        features, fold_metadata = design(fit)
        prediction = ridge_grid(features, target, fit, evaluation, records, "window", PENALTIES,
                                huber_ms=huber_ms, nonlinear=nonlinear,
                                kernel_split=fold_metadata.get("kernel_split"),
                                interaction=interaction_kernel, interaction_weight=interaction_weight)
        errors += (prediction - target[evaluation]).abs().sum(dim=1).cpu().numpy()
    penalty = PENALTIES[int(errors.argmin())]
    fit = np.flatnonzero(selected)
    features, metadata = design(fit)
    train = features[fit]
    center = train.mean(dim=0)
    train = train - center
    scale = train.square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
    train = train / scale
    mean = target[fit].mean()
    effective_penalty = penalty * len(fit) / 300
    if nonlinear:
        if block_kernel:
            kernel = block_fusion_kernel(
                train, train, metadata["kernel_split"], interaction=interaction_kernel,
                interaction_weight=interaction_weight,
            )
            if interaction_kernel:
                metadata["kernel_interaction"] = True
                metadata["kernel_interaction_weight"] = interaction_weight
        else:
            kernel = fusion_kernel(train, train)
        kernel.diagonal().add_(effective_penalty)
        weight = torch.linalg.solve(kernel, target[fit] - mean)
        metadata["kernel_support"] = train.cpu()
    elif huber_ms is None:
        kernel = train @ train.T
        kernel.diagonal().add_(effective_penalty)
        weight = train.T @ torch.linalg.solve(kernel, target[fit] - mean)
    else:
        train, y = train.double(), target[fit].double()
        dual, mean = huber_coefficients(train @ train.T, y, torch.ones_like(y),
                                        effective_penalty, huber_ms)
        weight = (train.T @ dual).float()
        mean = mean.float()
    if not torch.isfinite(weight).all():
        raise RuntimeError("Final ridge weights are non-finite")
    torch.save({
        "checkpoint": "OpenTSLM/llama-3.2-1b-ecg-sp",
        "opentslm_revision": "2968f4b891baab4307f7e9d0043e87677b593a30",
        "scaling": "per_lead_zscore",
        "attention_fastpath_enabled": False,
        "feature_layout": "flatten_encoder_outputs_in_call_order",
        "feature_count": encoder.shape[1],
        **metadata, "center": center.cpu(), "scale": scale.cpu(),
        "weight": weight.cpu(), "bias": mean.cpu(),
        "output_units": "ms_after_anchor", "sampling_rate_hz": FS, "anchor": ANCHOR,
        "patch_size": patch_size, "margin_ms": MARGIN_MS, "window_length": 625,
        "penalty_at_300_windows": penalty, "effective_penalty": effective_penalty,
        "huber_threshold_ms": huber_ms,
        "train_windows": len(fit), "train_records": len(set(records[fit])),
    }, output)
    print(json.dumps({
        "checkpoint_path": str(output), "train_windows": len(fit),
        "train_records": len(set(records[fit])), "selected_features": features.shape[1],
        "penalty_at_300_windows": penalty, "effective_penalty": effective_penalty,
    }), flush=True)


def run(panel, cache, log, output, raw_local_patch_size=None, *, final_checkpoint=None,
        fusion_only=False, tune_fusion=False, robust_fusion=False, nonlinear_fusion=False, block_kernel=False, interaction_kernel=False, tune_interaction=False, final_interaction_weight=0.25,
        interaction_tradeoff=False, late_target_weighting=False, robust_interaction=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(panel, allow_pickle=False) as data:
        signals = data["signals"]
        if signals.shape[1:] != (2, 625):
            raise ValueError("Expected the QTDB two-lead, 625-sample panel")
        encoder = columns = None
        patch_size = raw_local_patch_size
        if patch_size is None:
            from opentslm.model_config import PATCH_SIZE

            patch_size = PATCH_SIZE
            encoder = torch.load(cache, map_location="cpu", weights_only=True)
            if encoder.ndim != 2 or len(encoder) != len(signals) or not torch.isfinite(encoder).all():
                raise ValueError("Encoder feature cache does not match the panel")
            columns = read_patch_columns(log, encoder.shape[1], math.ceil(signals.shape[-1] / patch_size))
            encoder = encoder.to(device)
        if final_checkpoint:
            fit_final(
                encoder, data["targets"], data["folds"], data["selected"], data["records"],
                columns, patch_size, final_checkpoint,
                raw=torch.as_tensor(signals, device=device).flatten(1) if fusion_only or robust_fusion or nonlinear_fusion else None,
                huber_ms=40.0 if robust_fusion else None,
                nonlinear=nonlinear_fusion, block_kernel=block_kernel,
                interaction_kernel=interaction_kernel, interaction_weight=final_interaction_weight,
            )
            return
        results = compare(
            torch.as_tensor(signals, device=device).flatten(1), encoder,
            data["targets"], data["folds"], data["groups"], data["selected"], data["records"],
            columns, patch_size, raw_local_only=raw_local_patch_size is not None,
            fusion_only=fusion_only, tune_fusion=tune_fusion, robust_fusion=robust_fusion,
            nonlinear_fusion=nonlinear_fusion, block_kernel=block_kernel,
            interaction_kernel=interaction_kernel, tune_interaction=tune_interaction,
            interaction_tradeoff=interaction_tradeoff, late_target_weighting=late_target_weighting, robust_interaction=robust_interaction,
        )
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("cache", type=Path, nargs="?")
    parser.add_argument("extraction_log", type=Path, nargs="?")
    parser.add_argument("--raw-local-patch-size", type=int)
    parser.add_argument("--final-checkpoint", type=Path, help="Fit and save the selected local readout")
    parser.add_argument("--fusion-only", action="store_true", help="Compare a fixed 75/25 local feature mix")
    parser.add_argument("--tune-fusion", action="store_true", help="Select the mix in inner folds")
    parser.add_argument("--robust-fusion", action="store_true", help="Evaluate fixed fusion with Huber loss")
    parser.add_argument("--nonlinear-fusion", action="store_true",
                        help="Compare linear fusion with a linear-plus-RBF kernel")
    parser.add_argument("--block-kernel", action="store_true",
                        help="Also compare separate encoder and waveform kernels")
    parser.add_argument("--interaction-kernel", action="store_true",
                        help="Compare an additive kernel with an encoder-waveform product term")
    parser.add_argument("--tune-interaction", action="store_true",
                        help="Select the product kernel weight inside each training fold")
    parser.add_argument("--final-interaction-weight", type=float, default=None,
                        help="Fixed product weight when fitting the final checkpoint")
    parser.add_argument("--robust-interaction", action="store_true",
                        help="Compare squared and Huber loss on the 25% interaction kernel")
    parser.add_argument("--late-target-weighting", action="store_true",
                        help="Compare uniform and 3x weighting after 345 ms on QTDB")
    parser.add_argument("--interaction-tradeoff", action="store_true",
                        help="Compare fixed product weights on MAE and error tails")
    parser.add_argument("--output", type=Path, default=Path("beatlens-qtdb-readout-cv.json"))
    args = parser.parse_args()
    if args.raw_local_patch_size is None and (not args.cache or not args.extraction_log):
        parser.error("Provide the feature cache and extraction log, or --raw-local-patch-size")
    if args.raw_local_patch_size is not None and args.raw_local_patch_size < 1:
        parser.error("--raw-local-patch-size must be positive")
    if args.final_checkpoint and args.raw_local_patch_size is not None:
        parser.error("--final-checkpoint requires OpenTSLM features")
    if args.fusion_only and args.raw_local_patch_size is not None:
        parser.error("--fusion-only requires cached features")
    if args.tune_fusion and (args.raw_local_patch_size is not None or args.final_checkpoint):
        parser.error("--tune-fusion runs cross-validation using cached features")
    if args.robust_fusion and (args.raw_local_patch_size is not None or args.tune_fusion):
        parser.error("--robust-fusion requires cached features and the fixed mix")
    if args.nonlinear_fusion and (
        args.raw_local_patch_size is not None
        or args.tune_fusion or args.robust_fusion
    ):
        parser.error("--nonlinear-fusion compares fixed-mix readouts using cached features")
    if args.block_kernel and not args.nonlinear_fusion:
        parser.error("--block-kernel requires --nonlinear-fusion")
    if args.interaction_kernel and not args.block_kernel:
        parser.error("--interaction-kernel requires --block-kernel")
    if args.tune_interaction and (not args.interaction_kernel or args.final_checkpoint):
        parser.error("--tune-interaction requires --interaction-kernel in cross-validation mode")
    if args.final_interaction_weight is not None and (
        not args.final_checkpoint or not args.interaction_kernel
        or not 0 <= args.final_interaction_weight <= 1
    ):
        parser.error("--final-interaction-weight requires final interaction fitting and a value in [0, 1]")
    if args.interaction_tradeoff and (
        args.raw_local_patch_size is not None or args.final_checkpoint or args.robust_fusion
        or args.tune_fusion or args.tune_interaction
    ):
        parser.error("--interaction-tradeoff runs fixed-weight cross-validation on cached features")
    if args.late_target_weighting and (
        args.raw_local_patch_size is not None or args.final_checkpoint or args.robust_fusion
        or args.tune_fusion or args.tune_interaction or args.interaction_tradeoff
    ):
        parser.error("--late-target-weighting runs fixed interaction cross-validation on cached features")
    if args.robust_interaction and (
        args.raw_local_patch_size is not None or args.final_checkpoint or args.robust_fusion
        or args.tune_fusion or args.tune_interaction or args.interaction_tradeoff or args.late_target_weighting
    ):
        parser.error("--robust-interaction runs cross-validation on cached features")
    run(args.panel, args.cache, args.extraction_log, args.output, args.raw_local_patch_size,
        final_checkpoint=args.final_checkpoint, fusion_only=args.fusion_only,
        tune_fusion=args.tune_fusion, robust_fusion=args.robust_fusion,
        nonlinear_fusion=args.nonlinear_fusion, block_kernel=args.block_kernel,
        interaction_kernel=args.interaction_kernel, tune_interaction=args.tune_interaction,
        interaction_tradeoff=args.interaction_tradeoff, late_target_weighting=args.late_target_weighting,
        robust_interaction=args.robust_interaction,
        final_interaction_weight=(0.25 if args.final_interaction_weight is None
                                  else args.final_interaction_weight))
