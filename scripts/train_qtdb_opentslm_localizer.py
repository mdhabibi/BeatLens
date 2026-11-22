"""Adapt OpenTSLM's waveform encoder for patch-level QTDB localization."""

import argparse
import gzip
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


def patch_features(outputs, patch_count):
    traces = []
    for output in outputs:
        if not isinstance(output, torch.Tensor) or output.ndim < 2:
            raise ValueError(f"Unexpected encoder output: {type(output).__name__}")
        if output.shape[-2] != patch_count:
            raise ValueError(
                f"Expected {patch_count} time patches, got encoder shape {tuple(output.shape)}"
            )
        traces.append(output.reshape(-1, patch_count, output.shape[-1]))
    channels = torch.cat(traces, dim=0)
    return channels.permute(1, 0, 2).reshape(patch_count, -1).float()


class PatchLocator(nn.Module):
    def __init__(self, channels, patch_count, anchor_patch):
        super().__init__()
        relative = (torch.arange(patch_count) - anchor_patch) / patch_count
        self.register_buffer("relative", relative.view(1, 1, -1))
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Sequential(
            nn.Conv1d(channels + 1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, features):
        features = self.norm(features).transpose(1, 2)
        position = self.relative.expand(features.shape[0], -1, -1)
        return self.head(torch.cat((features, position), dim=1)).squeeze(1)


def timing_prediction(logits, centers):
    probabilities = logits.softmax(dim=-1)
    return (probabilities * centers).sum(dim=-1)


class RidgeLocator(nn.Module):
    """Initialize a differentiable timing head from the frozen ridge solution."""

    def __init__(self, features, targets):
        super().__init__()
        flat = features.flatten(1)
        center = flat.mean(dim=0)
        centered = flat - center
        scale = centered.square().sum(dim=1).mean().clamp_min(1e-12).sqrt()
        normalized = centered / scale
        kernel = normalized @ normalized.T
        kernel.diagonal().add_(0.1)
        dual = torch.linalg.solve(kernel, targets - targets.mean())
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.weight = nn.Parameter(normalized.T @ dual)
        self.bias = nn.Parameter(targets.mean())

    def forward(self, features):
        return ((features.flatten(1) - self.center) / self.scale) @ self.weight + self.bias


class ResidualAdapter(nn.Module):
    """Learn a bounded timing correction from ordered encoder patches."""

    def __init__(self, ridge, channels, lo, hi, max_correction):
        super().__init__()
        self.ridge = ridge.requires_grad_(False)
        self.lo, self.hi = lo, hi
        self.max_correction = max_correction
        self.adapter = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 16),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear((hi - lo) * 16, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def correction(self, features):
        return self.max_correction * self.adapter(features[:, self.lo:self.hi]).squeeze(1).tanh()

    def forward(self, features):
        return self.ridge(features) + self.correction(features)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    return value


def on_device(value, device, *, fp32=False):
    if isinstance(value, torch.Tensor):
        if fp32 and value.is_floating_point():
            return value.to(device=device, dtype=torch.float32)
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(on_device(item, device, fp32=fp32) for item in value)
    if isinstance(value, dict):
        return {key: on_device(item, device, fp32=fp32) for key, item in value.items()}
    return value


def run(args):
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate as collate,
    )

    if args.align_attention:
        torch.backends.mha.set_fastpath_enabled(False)
        print("PyTorch MHA fast path disabled for extraction, training, and evaluation.", flush=True)
    torch.manual_seed(42)
    rng = random.Random(42)
    panel = json.loads(args.panel.read_text())
    train = [window for pair in panel["train"] for window in pair]
    if args.external_panel:
        with gzip.open(args.external_panel, "rt") as stream:
            validation = json.load(stream)
    else:
        validation = [window for pair in panel["validation"] for window in pair]
    if {w["record"] for w in train} & {w["record"] for w in validation}:
        raise ValueError("Training and validation records overlap")
    geometry = {(len(w["signal"][0]), w["anchor"], w["fs"]) for w in train + validation}
    if len(geometry) != 1:
        raise ValueError("Expected a common window geometry")
    length, anchor, fs = geometry.pop()
    count = math.ceil(length / PATCH_SIZE)
    device = "cuda"
    checkpoint = "OpenTSLM/llama-3.2-1b-ecg-sp"
    model = OpenTSLM.load_pretrained(checkpoint, enable_lora=True, device=device)
    model.eval().requires_grad_(False)
    encoder = model.encoder
    calls = []

    def capture(_module, inputs, kwargs, output):
        calls.append((cpu_copy(inputs), cpu_copy(kwargs), cpu_copy(output)))

    def extract(windows):
        entries = []
        for index, window in enumerate(windows, start=1):
            calls.clear()
            sample = {
                "pre_prompt": "Two channels from an ECG recording.",
                "time_series": window["signal"],
                "time_series_text": ["ECG channel 1.", "ECG channel 2."],
                "post_prompt": "Answer:",
                "answer": "",
            }
            with torch.no_grad():
                model.generate(collate([sample], patch_size=PATCH_SIZE),
                               max_new_tokens=1, do_sample=False)
            if not calls:
                raise RuntimeError("The encoder was not called")
            if index == 1:
                print("Encoder shapes:", [tuple(out.shape) for _, _, out in calls], flush=True)
            features = patch_features([out for _, _, out in calls], count)
            entries.append(([(x, kw) for x, kw, _ in calls], features))
            if index % 50 == 0:
                print(f"Extracted {index}/{len(windows)} windows", flush=True)
        return entries

    handle = encoder.register_forward_hook(capture, with_kwargs=True)
    try:
        train_entries = [] if args.external_panel else extract(train)
        validation_entries = extract(validation)
    finally:
        handle.remove()

    if args.external_panel:
        saved = torch.load(args.external_checkpoint, map_location=device, weights_only=True)
        if (
            saved["head_type"] != "ridge" or saved["encoder"] is not None
            or saved["checkpoint"] != checkpoint or saved["patch_size"] != PATCH_SIZE
            or saved["window_length"] != length or saved["anchor"] != anchor
            or saved["attention_fastpath_enabled"] != torch.backends.mha.get_fastpath_enabled()
        ):
            raise ValueError("External checkpoint does not match the locked frozen ridge protocol")
        features = torch.stack([entry[1] for entry in validation_entries]).to(device)
        if tuple(features.shape[1:]) != tuple(saved["feature_shape"]):
            raise ValueError("External feature shape does not match the checkpoint")
        state = saved["head"]
        prediction = ((features.flatten(1) - state["center"]) / state["scale"]) @ state["weight"]
        prediction = prediction + state["bias"]
        truth = torch.tensor([w["reference"] for w in validation], device=device)
        prior = anchor + panel["prior_offset"]
        model_error = (prediction - truth).abs() * 1000 / fs
        prior_error = (prior - truth).abs() * 1000 / fs
        if not torch.isfinite(model_error).all():
            raise RuntimeError("External predictions contain non-finite values")
        from collections import defaultdict
        by_record = defaultdict(list)
        for window, delta in zip(validation, (model_error - prior_error).cpu().tolist(), strict=True):
            by_record[window["record"]].append(delta)
        totals = torch.tensor([sum(values) for values in by_record.values()])
        counts = torch.tensor([len(values) for values in by_record.values()])
        draws = torch.randint(len(totals), (20000, len(totals)), generator=torch.Generator().manual_seed(544))
        deltas = totals[draws].sum(dim=1) / counts[draws].sum(dim=1)
        report = {
            "windows": len(validation), "records": len({w["record"] for w in validation}),
            "model_mae_ms": model_error.mean().item(),
            "prior_mae_ms": prior_error.mean().item(),
            "model_within_40ms": (model_error <= 40).float().mean().item(),
            "model_minus_prior_mae_ms_ci95": torch.quantile(
                deltas, torch.tensor([0.025, 0.975])
            ).tolist(),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return

    def replay(entry, *, fp32=False):
        inputs, _ = entry
        return patch_features([
            encoder(*on_device(values, device, fp32=fp32), **on_device(kwargs, device, fp32=fp32))
            for values, kwargs in inputs
        ], count)

    with torch.no_grad():
        expected = train_entries[0][1].to(device)
        if not torch.allclose(replay(train_entries[0]), expected, atol=1e-4, rtol=1e-4):
            raise RuntimeError("Direct encoder replay differed from the OpenTSLM feature path")
    head = PatchLocator(expected.shape[-1], count, anchor / PATCH_SIZE).to(device)
    train_targets = torch.tensor([w["reference"] for w in train], device=device)
    validation_targets = torch.tensor([w["reference"] for w in validation], device=device)
    # Use training annotations only, with 40 ms of headroom at each end.
    margin = round(0.040 * fs)
    lo = max(0, (int(train_targets.min()) - margin) // PATCH_SIZE)
    hi = min(count, (int(train_targets.max()) + margin) // PATCH_SIZE + 1)
    centers = (
        torch.arange(lo, hi, device=device, dtype=torch.float32) * PATCH_SIZE
        + (PATCH_SIZE - 1) / 2
    ).clamp_max(length - 1)
    print(f"Train-derived search interval: samples {lo * PATCH_SIZE}–{min(hi * PATCH_SIZE, length) - 1}",
          flush=True)
    train_features = torch.stack([entry[1] for entry in train_entries]).to(device)
    validation_features = torch.stack([entry[1] for entry in validation_entries]).to(device)
    if args.ridge_init or args.residual_adapter:
        head = RidgeLocator(train_features, train_targets.float())
    best_mae = float("inf")

    def evaluate(label, epoch, features, *, save=True):
        nonlocal best_mae
        outputs = torch.cat([head(batch) for batch in features.split(32)])
        predictions = (
            outputs if args.ridge_init or args.residual_adapter
            else timing_prediction(outputs[:, lo:hi], centers)
        )
        swapped = predictions.reshape(-1, 2).flip(1).reshape(-1)
        error = lambda values: (values - validation_targets).abs().mean().item() * 1000 / fs
        mae = error(predictions)
        print(json.dumps({
            "stage": label, "epoch": epoch, "validation_windows": len(validation),
            "mae_ms": mae, "swap_mae_ms": error(swapped),
            "timing_prior_mae_ms": error(anchor + panel["prior_offset"]),
        }), flush=True)
        if save and mae < best_mae:
            best_mae = mae
            torch.save({
                "checkpoint": checkpoint, "patch_size": PATCH_SIZE, "stage": label,
                "epoch": epoch, "validation_mae_ms": mae, "head": head.state_dict(),
                "search_patch_bounds": (lo, hi), "window_length": length, "anchor": anchor,
                "head_type": (
                    "ridge_residual_adapter" if label == "residual_adapter"
                    else "ridge" if args.ridge_init or args.residual_adapter else "patch"
                ),
                "feature_shape": tuple(train_features.shape[1:]),
                "attention_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
                "ridge_encoder_lr": args.ridge_encoder_lr if args.ridge_init else None,
                "encoder": encoder.state_dict() if label == "adapted_encoder" else None,
            }, args.output)

    if args.residual_adapter:
        head.eval()
        with torch.no_grad():
            evaluate("frozen_ridge", 0, validation_features)
            records = sorted({window["record"] for window in train})
            rng.shuffle(records)
            record_fold = {record: index % 5 for index, record in enumerate(records)}
            folds = torch.tensor([record_fold[window["record"]] for window in train], device=device)
            residual_targets = torch.empty_like(train_targets, dtype=torch.float32)
            for fold in range(5):
                held_out = folds == fold
                fold_ridge = RidgeLocator(train_features[~held_out], train_targets[~held_out].float())
                residual_targets[held_out] = (
                    train_targets[held_out] - fold_ridge(train_features[held_out])
                )
        head = ResidualAdapter(head, expected.shape[-1], lo, hi, 0.040 * fs).to(device)
        optimizer = torch.optim.AdamW(head.adapter.parameters(), lr=3e-4)
        for epoch in range(1, args.head_epochs + 1):
            head.train()
            order = list(range(len(train)))
            rng.shuffle(order)
            for start in range(0, len(order), 32):
                ids = order[start:start + 32]
                features = train_features[ids]
                correction = head.correction(features)
                residual = residual_targets[ids]
                loss = nn.functional.smooth_l1_loss(correction, residual, beta=2.0)
                loss = loss + 0.1 * correction.square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite residual adapter loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(head.adapter.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            head.eval()
            with torch.no_grad():
                evaluate("residual_adapter", epoch, validation_features)
        return

    if args.ridge_init:
        head.eval()
        with torch.no_grad():
            evaluate("frozen_ridge", 0, validation_features)
            if args.replay_diagnostic:
                for label, fp32 in (("native_replay", False), ("fp32_replay", True)):
                    if fp32:
                        encoder.float()
                    replayed = torch.stack([
                        replay(entry, fp32=fp32) for entry in validation_entries
                    ])
                    print(json.dumps({
                        "stage": label,
                        "mean_feature_drift": (
                            (replayed - validation_features).square().sum(dim=(1, 2))
                            / head.scale.square()
                        ).mean().item(),
                    }), flush=True)
                    evaluate(label, 0, replayed, save=False)
                encoder.requires_grad_(True)
                with torch.enable_grad():
                    grad_features = torch.stack([
                        replay(entry, fp32=True).detach() for entry in validation_entries
                    ])
                encoder.requires_grad_(False)
                print(json.dumps({
                    "stage": "grad_enabled_replay",
                    "mean_feature_drift": (
                        (grad_features - validation_features).square().sum(dim=(1, 2))
                        / head.scale.square()
                    ).mean().item(),
                }), flush=True)
                evaluate("grad_enabled_replay", 0, grad_features, save=False)
                return
    head_epochs = 0 if args.ridge_init else args.head_epochs
    for label, epochs in (("frozen_encoder", head_epochs), ("adapted_encoder", args.encoder_epochs)):
        if not epochs:
            continue
        adapting = label == "adapted_encoder"
        if adapting:
            best_head = torch.load(args.output, map_location=device, weights_only=True)["head"]
            head.load_state_dict(best_head)
        stable_adaptation = args.ridge_init and adapting
        head_lr = 3e-4 if adapting else 1e-3
        groups = [{"params": head.parameters(), "lr": head_lr}]
        if adapting:
            encoder.requires_grad_(True)
            if stable_adaptation:
                head.requires_grad_(False)
                encoder.float()
                if args.align_attention:
                    with torch.enable_grad():
                        features = torch.stack([
                            replay(entry, fp32=True).detach() for entry in validation_entries
                        ])
                    drift = (
                        (features - validation_features).square().sum(dim=(1, 2))
                        / head.scale.square()
                    ).mean().item()
                    print(json.dumps({
                        "stage": "aligned_pre_training", "mean_feature_drift": drift,
                    }), flush=True)
                    if drift > 1e-4:
                        raise RuntimeError("Encoder paths still disagree before training")
                groups = [{
                    "params": encoder.parameters(),
                    "lr": args.ridge_encoder_lr,
                    "weight_decay": 0.0,
                }]
            else:
                groups.append({"params": encoder.parameters(), "lr": 1e-5})
        optimizer = torch.optim.AdamW(groups)
        for epoch in range(1, epochs + 1):
            head.train(not stable_adaptation)
            # Evaluation mode keeps encoder stochastic layers disabled but permits gradients.
            encoder.train(adapting and not stable_adaptation)
            order = list(range(len(train)))
            rng.shuffle(order)
            step_size = 8 if adapting else 32
            drift_total = 0.0
            for start in range(0, len(order), step_size):
                ids = order[start:start + step_size]
                features = (
                    torch.stack([replay(train_entries[i], fp32=stable_adaptation) for i in ids])
                    if adapting else train_features[ids]
                )
                target = train_targets[ids]
                outputs = head(features)
                prediction = outputs if args.ridge_init else timing_prediction(outputs[:, lo:hi], centers)
                location_loss = nn.functional.smooth_l1_loss(
                    prediction / PATCH_SIZE, target.float() / PATCH_SIZE, beta=2.0
                )
                loss = location_loss
                if stable_adaptation:
                    frozen_features = train_features[ids]
                    drift = (
                        (features - frozen_features).square().sum(dim=(1, 2))
                        / head.scale.square()
                    ).mean()
                    loss = loss + 10.0 * drift
                    drift_total += drift.detach().item() * len(ids)
                if not args.ridge_init:
                    patch_loss = nn.functional.cross_entropy(outputs[:, lo:hi], target // PATCH_SIZE - lo)
                    loss = loss + 0.25 * patch_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite localization loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if adapting and epoch == 1 and start == 0:
                    updates = [
                        (p, p.detach().clone()) for p in encoder.parameters()
                        if p.grad is not None and p.grad.count_nonzero().item()
                    ]
                    if not updates:
                        raise RuntimeError("No gradient reached the waveform encoder")
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    1.0, error_if_nonfinite=True,
                )
                optimizer.step()
                if adapting and epoch == 1 and start == 0:
                    if all(torch.equal(p.detach(), before) for p, before in updates):
                        raise RuntimeError("The first update did not change the checked encoder weight")
                    print("Verified first encoder weight update.", flush=True)
                    del updates
            if stable_adaptation:
                print(json.dumps({"epoch": epoch, "mean_feature_drift": drift_total / len(train)}),
                      flush=True)
            head.eval()
            encoder.eval()
            with torch.no_grad():
                features = (
                    torch.stack([replay(entry, fp32=stable_adaptation) for entry in validation_entries])
                    if adapting else validation_features
                )
                evaluate(label, epoch, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("beatlens-opentslm-localizer.pt"))
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--encoder-epochs", type=int, default=3)
    parser.add_argument("--ridge-init", action="store_true")
    parser.add_argument("--residual-adapter", action="store_true")
    parser.add_argument("--ridge-encoder-lr", type=float, default=1e-7)
    parser.add_argument("--replay-diagnostic", action="store_true")
    parser.add_argument("--align-attention", action="store_true")
    parser.add_argument("--external-panel", type=Path)
    parser.add_argument("--external-checkpoint", type=Path)
    args = parser.parse_args()
    if args.head_epochs < 1 or args.encoder_epochs < 0:
        parser.error("Use at least one head epoch and zero or more encoder epochs")
    if args.replay_diagnostic and not args.ridge_init:
        parser.error("--replay-diagnostic requires --ridge-init")
    if args.residual_adapter and args.ridge_init:
        parser.error("--residual-adapter initializes its own ridge head")
    if not math.isfinite(args.ridge_encoder_lr) or args.ridge_encoder_lr <= 0:
        parser.error("--ridge-encoder-lr must be finite and positive")
    if bool(args.external_panel) != bool(args.external_checkpoint):
        parser.error("Provide both --external-panel and --external-checkpoint")
    run(args)
