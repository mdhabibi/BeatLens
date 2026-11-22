"""Explain the saved OpenTSLM locator with waveform LIME and patch-feature IG."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_qtdb_opentslm_readout import predict
from probe_qtdb_opentslm_encoder import make_encoder_extractor


def patch_attribution(observed, baseline, checkpoint, shapes):
    """Exact integrated gradients of the linear readout in encoder-feature space."""
    columns = checkpoint["feature_columns"]
    contribution = torch.zeros_like(observed)
    encoder_weight = checkpoint["weight"][:len(columns)] / checkpoint["scale"]
    if "raw_columns" in checkpoint:
        encoder_weight = encoder_weight * math.sqrt(0.75) / checkpoint["encoder_block_scale"]
    contribution[columns] = (
        (observed[columns] - baseline[columns]) * encoder_weight
    )
    return aggregate_patches(contribution, shapes)


def aggregate_patches(contribution, shapes):
    offset = 0
    patches = []
    for shape in shapes:
        size = math.prod(shape)
        values = contribution[offset:offset + size].reshape(-1, shape[-2], shape[-1])
        patches.append(values)
        offset += size
    if offset != contribution.numel():
        raise ValueError("Encoder shapes do not match the checkpoint feature layout")
    joined = torch.cat(patches)
    return joined.sum(dim=(0, 2)), joined.abs().sum(dim=(0, 2))


def nonlinear_attribution(features, waveforms, checkpoint, shapes, steps=128):
    """Integrate gradients along the joint encoder-feature and waveform path."""
    features, waveforms = features.double(), waveforms.double()
    state = {
        key: value.double() if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in checkpoint.items()
    }
    alpha = torch.linspace(0, 1, steps + 1, dtype=features.dtype)
    feature_delta, raw_delta = features[0] - features[1], waveforms[0] - waveforms[1]
    feature_path = (features[1] + alpha[:, None] * feature_delta).requires_grad_()
    raw_path = (waveforms[1] + alpha[:, None, None] * raw_delta).requires_grad_()
    scores = predict(feature_path, state, raw_path)
    feature_grad, raw_grad = torch.autograd.grad(scores.sum(), (feature_path, raw_path))
    feature_ig = feature_delta * torch.trapezoid(feature_grad, dx=1 / steps, dim=0)
    raw_ig = raw_delta * torch.trapezoid(raw_grad, dx=1 / steps, dim=0)
    signed, magnitude = aggregate_patches(feature_ig, shapes)
    raw_signed = torch.zeros_like(signed)
    columns = state["raw_columns"]
    patches = (columns % waveforms.shape[-1]) // state["patch_size"]
    raw_signed.scatter_add_(0, patches, raw_ig.flatten()[columns])
    return signed, magnitude, raw_signed


def run(args):
    from scipy.ndimage import gaussian_filter1d
    from temporal_attribution import temporal_lime
    from transformers.utils import logging

    logging.set_verbosity_error()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.backends.mha.set_fastpath_enabled(state["attention_fastpath_enabled"])
    panel = json.loads(args.panel.read_text())
    windows = [w for pair in panel["validation"] for w in pair]
    window = windows[args.window]
    waveform = np.asarray(window["signal"], dtype=np.float32)
    if waveform.shape != (2, state["window_length"]) or window["fs"] != state["sampling_rate_hz"]:
        raise ValueError("Window geometry does not match the saved readout")
    smooth = gaussian_filter1d(waveform, sigma=12, axis=-1, mode="nearest")
    extract = make_encoder_extractor()
    inputs = np.stack([waveform, smooth])
    features, shapes = extract(inputs, return_shapes=True)
    observed, baseline = features
    prediction, reference_prediction = predict(features, state, torch.from_numpy(inputs)).tolist()
    fusion = "raw_columns" in state
    nonlinear = "kernel_support" in state
    if nonlinear:
        signed, magnitude, raw_signed = nonlinear_attribution(
            features, torch.from_numpy(inputs), state, shapes
        )
    else:
        signed, magnitude = patch_attribution(observed, baseline, state, shapes)
        raw_signed = torch.zeros_like(signed)
        if fusion:
            columns = state["raw_columns"]
            raw_difference = torch.from_numpy((waveform - smooth).reshape(-1))[columns]
            raw_contribution = (
                raw_difference * state["weight"][len(state["feature_columns"]):] * 0.5
                / (state["scale"] * state["raw_block_scale"])
            )
            raw_patches = (columns % waveform.shape[-1]) // state["patch_size"]
            raw_signed.scatter_add_(0, raw_patches, raw_contribution)

    def predictor(inputs):
        return predict(extract(inputs), state, torch.as_tensor(inputs)).numpy()

    segment_samples = max(1, round(window["fs"] * 0.025))
    lime = temporal_lime(
        waveform, predictor, segment_samples=segment_samples, replacement=smooth,
        perturbations=args.perturbations, seed=19,
    )
    kernel_label = "linear_rbf" if nonlinear else "linear"
    if "kernel_split" in state:
        kernel_label = "separate_linear_rbf"
    if state.get("kernel_interaction", False):
        kernel_label = "interaction_linear_rbf"
    result = {
        "model": str(args.checkpoint), "record": window["record"], "beat": window["beat"],
        "readout": "fusion" if fusion else "local_patches",
        "kernel": kernel_label,
        "fs": window["fs"], "anchor": window["anchor"], "waveform": waveform.tolist(),
        "prediction_ms": prediction, "baseline_prediction_ms": reference_prediction,
        "reference_ms": (window["reference"] - window["anchor"]) * 1000 / window["fs"],
        "baseline": "48 ms Gaussian-smoothed waveform",
        "patch_ig": {
            "input_space": "encoder_patch_features", "patch_samples": state["patch_size"],
            "signed_ms": signed.tolist(), "magnitude_ms": magnitude.tolist(),
            "completeness_residual_ms": abs(
                (signed + raw_signed).sum().item() - (prediction - reference_prediction)
            ),
        },
        "waveform_lime": {
            "segment_samples": segment_samples, "signed_ms": lime.coefficients.tolist(),
            "weighted_r2": lime.weighted_r2, "perturbations": args.perturbations,
        },
    }
    if fusion:
        result["patch_ig"]["raw_signed_ms"] = raw_signed.tolist()
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({
        "record": window["record"], "prediction_ms": prediction,
        "reference_ms": result["reference_ms"], "lime_weighted_r2": lime.weighted_r2,
        "patch_ig_completeness_residual_ms": result["patch_ig"]["completeness_residual_ms"],
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--perturbations", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("beatlens-opentslm-explanations.json"))
    run(parser.parse_args())
