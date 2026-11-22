"""Select ridge penalties inside grouped QTDB folds and score outer folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_qtdb_opentslm_encoder import make_encoder_extractor

PENALTIES = (0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 10.0, 30.0)


def fusion_kernel(left, right, linear=None):
    if linear is None:
        linear = left @ right.T
    distance = left.square().sum(1)[:, None] + right.square().sum(1) - 2 * linear
    return 0.5 * (linear + (-0.5 * distance.clamp_min(0)).exp())


def block_fusion_kernel(left, right, split, *, interaction=False, interaction_weight=0.25):
    """Combine kernels after undoing the fixed block weights."""
    encoder = fusion_kernel(left[:, :split] / (0.75 ** 0.5),
                            right[:, :split] / (0.75 ** 0.5))
    raw = fusion_kernel(left[:, split:] / 0.5, right[:, split:] / 0.5)
    additive = 0.75 * encoder + 0.25 * raw
    if interaction:
        return (1 - interaction_weight) * additive + interaction_weight * encoder * raw
    return additive


def ridge_grid(features, targets, fit, evaluation, records, weighting, penalties, *,
               huber_ms=None, nonlinear=False, kernel_split=None, interaction=False,
               interaction_weight=0.25, late_weight=1.0):
    train, test = features[fit], features[evaluation]
    y = targets[fit]
    weights = np.ones(len(fit), dtype=np.float32)
    if weighting == "record":
        _, inverse, counts = np.unique(records[fit], return_inverse=True, return_counts=True)
        weights = 1 / counts[inverse]
        weights *= len(fit) / weights.sum()
    w = torch.as_tensor(weights, dtype=train.dtype, device=features.device)
    if late_weight != 1.0:
        w = w * torch.where(y > 345, late_weight, 1.0)
        w = w * (len(fit) / w.sum())
    center, mean = (train.T @ w) / w.sum(), (y @ w) / w.sum()
    train, test = train - center, test - center
    scale = ((train.square().sum(dim=1) @ w) / w.sum()).clamp_min(1e-12).sqrt()
    root = w.sqrt()
    if huber_ms is not None:
        return huber_grid(train / scale, test / scale, y, w, penalties, huber_ms,
                          kernel_split=kernel_split, interaction=interaction)
    train, test = train / scale, test / scale
    kernel = train @ train.T
    cross = test @ train.T
    if kernel_split is not None:
        kernel = block_fusion_kernel(train, train, kernel_split, interaction=interaction,
                                     interaction_weight=interaction_weight)
        cross = block_fusion_kernel(test, train, kernel_split, interaction=interaction,
                                     interaction_weight=interaction_weight)
    elif nonlinear:
        kernel = fusion_kernel(train, train, kernel)
        cross = fusion_kernel(test, train, cross)
    kernel = root[:, None] * kernel * root
    cross = cross * root
    rhs = (y - mean) * root
    predictions = []
    for penalty in penalties:
        system = kernel.clone()
        system.diagonal().add_(penalty * len(fit) / 300)
        predictions.append(mean + cross @ torch.linalg.solve(system, rhs))
    return torch.stack(predictions)


def huber_grid(train, test, targets, base_weights, penalties, threshold, *,
               kernel_split=None, interaction=False):
    train = train.double()
    test = test.double()
    if kernel_split is None:
        kernel, cross = train @ train.T, test @ train.T
    else:
        kernel = block_fusion_kernel(train, train, kernel_split, interaction=interaction)
        cross = block_fusion_kernel(test, train, kernel_split, interaction=interaction)
    targets = targets.double()
    base_weights = base_weights.double()
    predictions = []
    for penalty in penalties:
        dual, intercept = huber_coefficients(
            kernel, targets, base_weights, penalty * len(train) / 300, threshold
        )
        predictions.append(cross @ dual + intercept)
    return torch.stack(predictions)


def huber_coefficients(kernel, targets, base_weights, penalty, threshold):
    ones = torch.ones_like(targets)
    rhs = torch.stack((targets, ones), dim=1)
    weights = base_weights
    for _ in range(8):
        system = kernel.clone()
        system.diagonal().add_(penalty / weights)
        solved_y, solved_one = torch.linalg.solve(system, rhs).T
        intercept = solved_y.sum() / solved_one.sum()
        dual = solved_y - solved_one * intercept
        residual = (kernel @ dual + intercept - targets).abs()
        weights = base_weights * (threshold / residual.clamp_min(threshold))
    return dual, intercept


def compare(features, targets, folds, groups, selected, records, output):
    results = []
    for coverage, available in (("subset", selected), ("all", np.ones_like(selected))):
        for weighting in ("window", "record"):
            totals = np.zeros(3)
            outer_penalties = []
            for outer in range(5):
                inner_errors = np.zeros(len(PENALTIES))
                for inner in range(5):
                    if inner == outer:
                        continue
                    fit = np.flatnonzero((folds != outer) & (folds != inner) & available)
                    evaluate = np.flatnonzero(folds == inner)
                    prediction = ridge_grid(features, targets, fit, evaluate, records, weighting, PENALTIES)
                    inner_errors += (prediction - targets[evaluate]).abs().sum(dim=1).cpu().numpy()
                penalty = PENALTIES[int(inner_errors.argmin())]
                outer_penalties.append(penalty)
                fit = np.flatnonzero((folds != outer) & available)
                evaluate = np.flatnonzero(folds == outer)
                evaluate = evaluate[np.argsort(groups[evaluate], kind="stable")]
                offset = int(np.unique(groups[evaluate], return_counts=True)[1].max())
                if np.any(groups[evaluate] == np.roll(groups[evaluate], offset)):
                    raise ValueError("Could not create a cross-group swap")
                prediction = ridge_grid(features, targets, fit, evaluate, records, weighting, [penalty])[0]
                expected = targets[evaluate]
                errors = torch.stack([
                    (prediction - expected).abs(),
                    (prediction.roll(offset) - expected).abs(),
                    (targets[fit].mean() - expected).abs(),
                ])
                totals += errors.sum(dim=1).cpu().numpy()
                print(json.dumps({
                    "coverage": coverage, "weighting": weighting, "fold": outer,
                    "penalty_at_300_windows": penalty, "mae_ms": errors[0].mean().item(),
                }), flush=True)
            result = {
                "coverage": coverage, "weighting": weighting, "evaluation_windows": len(folds),
                "penalties_at_300_windows": outer_penalties,
                **dict(zip(("mae_ms", "swap_mae_ms", "prior_mae_ms"), (totals / len(folds)).tolist())),
            }
            results.append(result)
            print(json.dumps({"summary": result}), flush=True)
            output.write_text(json.dumps(results, indent=2) + "\n")


def run(panel, output, feature_output, feature_input):
    with np.load(panel, allow_pickle=False) as data:
        if feature_input:
            features = torch.load(feature_input, map_location="cpu", weights_only=True)
        else:
            from transformers.utils import logging

            logging.set_verbosity_error()
            torch.backends.mha.set_fastpath_enabled(False)
            features = make_encoder_extractor()(data["signals"])
        if (
            not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[0] != len(data["targets"])
            or not features.is_floating_point() or not torch.isfinite(features).all()
        ):
            raise ValueError("Expected one finite encoder feature vector per panel window")
        if feature_output:
            torch.save(features, feature_output)
        compare(
            features.cuda(), torch.as_tensor(data["targets"], device="cuda"),
            data["folds"], data["groups"], data["selected"], data["records"], output,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("beatlens-qtdb-ridge-cv.json"))
    parser.add_argument("--feature-output", type=Path)
    parser.add_argument("--feature-input", type=Path, help="Reuse features from the same panel")
    args = parser.parse_args()
    run(args.panel, args.output, args.feature_output, args.feature_input)
