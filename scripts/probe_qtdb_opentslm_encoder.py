"""Fit a ridge regression probe on frozen OpenTSLM waveform features."""

import argparse
import json
from pathlib import Path


def ridge_predictions(train, targets, validation, penalty):
    import torch

    center = train.mean(dim=0)
    train = train - center
    validation = validation - center
    scale = train.square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
    train, validation = train / scale, validation / scale
    kernel = train @ train.T
    kernel.diagonal().add_(penalty)
    coefficients = torch.linalg.solve(kernel, targets - targets.mean())
    return targets.mean() + (validation @ train.T) @ coefficients


def make_encoder_extractor():
    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate as collate,
    )

    model = OpenTSLM.load_pretrained(
        "OpenTSLM/llama-3.2-1b-ecg-sp", enable_lora=True, device="cuda"
    )
    model.eval()
    model.requires_grad_(False)
    captured = []

    def capture(_module, _args, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected encoder output type: {type(output).__name__}")
        captured.append(output.detach().float().cpu())

    def extract(windows, *, return_shapes=False):
        features = []
        shapes = None
        handle = model.encoder.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                for index, signal in enumerate(windows, start=1):
                    captured.clear()
                    sample = {
                        "pre_prompt": "Two channels from an ECG recording.",
                        "time_series": signal.tolist() if hasattr(signal, "tolist") else signal,
                        "time_series_text": ["ECG channel 1.", "ECG channel 2."],
                        "post_prompt": "Answer:",
                        "answer": "",
                    }
                    batch = collate([sample], patch_size=PATCH_SIZE)
                    model.generate(batch, max_new_tokens=1, do_sample=False)
                    if not captured:
                        raise RuntimeError("The waveform encoder hook did not capture any features")
                    features.append(torch.cat([value.reshape(-1) for value in captured]))
                    if index == 1:
                        shapes = [tuple(v.shape) for v in captured]
                        print("Encoder output shapes:", shapes, flush=True)
                    if index % 50 == 0:
                        print(f"Extracted {index}/{len(windows)} windows.", flush=True)
        finally:
            handle.remove()
        matrix = torch.stack(features)
        if not torch.isfinite(matrix).all():
            raise RuntimeError("Encoder produced non-finite features")
        return (matrix, shapes) if return_shapes else matrix

    return extract


def run(panel_path, output_path):
    import torch

    panel = json.loads(panel_path.read_text())
    train = [window for pair in panel["train"] for window in pair]
    validation = [window for pair in panel["validation"] for window in pair]
    if {w["record"] for w in train} & {w["record"] for w in validation}:
        raise ValueError("Training and validation records overlap")
    if len({window["fs"] for window in train + validation}) != 1:
        raise ValueError("The timing prior requires a common sampling rate")
    extract = make_encoder_extractor()
    train_features = extract([window["signal"] for window in train])
    validation_features = extract([window["signal"] for window in validation])
    if torch.equal(validation_features, validation_features.roll(1, dims=0)):
        raise RuntimeError("Frozen encoder features are identical across validation windows")

    def targets(windows):
        return torch.tensor([
            (window["reference"] - window["anchor"]) * 1000 / window["fs"]
            for window in windows
        ])

    expected = targets(validation)
    train_targets = targets(train)
    prediction = ridge_predictions(train_features, train_targets, validation_features, 0.1)
    prior_ms = panel["prior_offset"] * 1000 / validation[0]["fs"]
    paired = prediction.reshape(-1, 2)
    expected_pairs = expected.reshape(-1, 2)
    difference = paired[:, 0] - paired[:, 1]
    actual_difference = expected_pairs[:, 0] - expected_pairs[:, 1]
    result = {
        "train_windows": len(train),
        "validation_windows": len(validation),
        "features_per_window": train_features.shape[1],
        "encoder_ridge_mae_ms": (prediction - expected).abs().mean().item(),
        "waveform_swap_mae_ms": (paired.flip(1).flatten() - expected).abs().mean().item(),
        "timing_prior_mae_ms": (prior_ms - expected).abs().mean().item(),
        "pair_order_correct": int((difference * actual_difference > 0).sum().item()),
        "pair_order_total": len(paired),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("beatlens-encoder-probe.json"))
    args = parser.parse_args()
    run(args.panel, args.output)
