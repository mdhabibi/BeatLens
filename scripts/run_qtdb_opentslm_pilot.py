"""Prepare, train, and score the QTDB pilot with OpenTSLM."""

import argparse
import json
import math
import re
from pathlib import Path


def prepare(
    panel_path: Path, qtdb_dir: Path, split_csv: Path, *, supervised: bool = False
) -> None:
    from train_qtdb_tpeak import LEFT_SECONDS, prepare_split

    def select(examples, signals, count):
        windows = []
        seen_records = set()
        for example, signal in zip(examples, signals, strict=True):
            if example.record_id in seen_records or not example.valid_samples.all():
                continue
            seen_records.add(example.record_id)
            anchor = round(LEFT_SECONDS * example.sampling_rate_hz)
            windows.append({
                "record": example.record_id,
                "beat": example.beat_index,
                "fs": example.sampling_rate_hz,
                "anchor": anchor,
                "reference": anchor + example.event_offsets["t_peak"][0],
                "lead_names": example.lead_names,
                "signal": signal.tolist(),
            })
            if len(windows) == count:
                return windows
        raise ValueError(f"Expected {count} distinct records with eligible windows")

    train, train_signals, _, prior_offset, _ = prepare_split(qtdb_dir, split_csv, "train")
    validation, signals, _, _, _ = prepare_split(qtdb_dir, split_csv, "validation")
    panel = {"prior_offset": prior_offset, "windows": select(validation, signals, 8)}
    if supervised:
        panel["training_windows"] = select(train, train_signals, 32)
        print("Prepared 32 supervised training windows.")
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_text(json.dumps(panel))
    print(f"Prepared 8 validation windows. Train-only prior: {prior_offset:.2f} samples.")


def select_overfit_windows(windows):
    by_index = {window["reference"]: window for window in windows}
    indices = sorted(by_index)
    if len(indices) < 4:
        raise ValueError("Need four distinct training targets for the overfit check")
    return [by_index[indices[round(i * (len(indices) - 1) / 3)]] for i in range(4)]


def score(
    panel_path: Path, *, fine_tune: bool = False, controls: bool = False,
    overfit: bool = False, logits: bool = False, candidates: bool = False,
) -> None:
    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    panel = json.loads(panel_path.read_text())
    if fine_tune and not panel.get("training_windows"):
        raise ValueError("Prepare a panel with --supervised before training")
    print(torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(42)
    model = OpenTSLM.load_pretrained(
        "OpenTSLM/llama-3.2-1b-ecg-sp", enable_lora=True, device="cuda"
    )

    def batch_for(window, *, training=False):
        fs, anchor = window["fs"], window["anchor"]
        length = len(window["signal"][0])
        post_prompt = (
            f"The beat anchor is at zero-based sample {anchor}. Locate the T-wave peak "
            f"of this beat. Briefly explain your choice, then end on a line containing "
            f"Answer: followed by one integer from 0 to {length - 1}."
        )
        if fine_tune or controls:
            post_prompt = (
                f"The beat anchor is at zero-based sample {anchor}. Return the T-wave peak "
                f"index as one integer from 0 to {length - 1}."
            )
        sample = {
            "pre_prompt": f"These ECG leads show the same window sampled at {fs:g} Hz.",
            "time_series": window["signal"],
            "time_series_text": [
                f"ECG channel {index}." if controls or overfit else f"ECG lead {lead}."
                for index, lead in enumerate(window["lead_names"], start=1)
            ],
            "post_prompt": post_prompt,
            "answer": f"{window['reference']}{model.get_eos_token()}" if training else "",
        }
        return extend_time_series_to_match_patch_size_and_aggregate(
            [sample], patch_size=PATCH_SIZE
        )

    if overfit:
        windows = select_overfit_windows(panel["training_windows"])
        donors = windows[2:] + windows[:2]
        if len({(w["anchor"], w["fs"], len(w["signal"][0])) for w in windows}) != 1:
            raise ValueError("Overfit windows must share an anchor, sampling rate, and length")
        swapped = [
            dict(window, signal=donor["signal"], reference=donor["reference"])
            for window, donor in zip(windows, donors, strict=True)
        ]
        print("Overfit training indices:", [window["reference"] for window in windows])
        if logits:
            compare_first_token_logits(model, windows, swapped, batch_for, "pretrained")
        if candidates:
            rank_candidate_indices(model, windows, swapped, batch_for, "pretrained")
        train_lora(model, windows, batch_for, epochs=8)
        if candidates:
            rank_candidate_indices(model, windows, swapped, batch_for, "tuned")
            return
        if logits:
            compare_first_token_logits(model, windows, swapped, batch_for, "tuned")
            return
        original = evaluate(
            model, windows, batch_for, panel["prior_offset"],
            max_new_tokens=16, label="overfit/original_waveforms",
        )
        predictions = evaluate(
            model, swapped, batch_for, panel["prior_offset"],
            max_new_tokens=16, label="overfit/swapped_waveforms",
        )
        print(json.dumps({
            "exact_original": sum(p == w["reference"] for p, w in zip(original, windows, strict=True)),
            "exact_after_swap": sum(p == w["reference"] for p, w in zip(predictions, swapped, strict=True)),
            "changed_predictions": sum(a != b for a, b in zip(original, predictions, strict=True)),
            "total": len(windows),
        }), flush=True)
        return

    if controls:
        windows = panel["windows"]
        shuffled = [
            dict(window, signal=donor["signal"])
            for window, donor in zip(windows, windows[1:] + windows[:1], strict=True)
        ]
        for checkpoint in ("original", "tuned"):
            if checkpoint == "tuned":
                train_lora(model, panel["training_windows"], batch_for)
            original = evaluate(
                model, windows, batch_for, panel["prior_offset"],
                max_new_tokens=16, label=f"{checkpoint}/original_waveforms",
            )
            swapped = evaluate(
                model, shuffled, batch_for, panel["prior_offset"],
                max_new_tokens=16, label=f"{checkpoint}/swapped_waveforms",
            )
            paired = [
                (left, right) for left, right in zip(original, swapped, strict=True)
                if left is not None and right is not None
            ]
            print(json.dumps({
                "checkpoint": checkpoint,
                "changed_validity": sum(
                    (a is None) != (b is None) for a, b in zip(original, swapped, strict=True)
                ),
                "changed_valid_indices": sum(a != b for a, b in paired),
                "valid_in_both": len(paired),
                "total": len(windows),
            }), flush=True)
        return

    if fine_tune:
        train_lora(model, panel["training_windows"], batch_for)
    evaluate(
        model, panel["windows"], batch_for, panel["prior_offset"],
        max_new_tokens=16 if fine_tune else 256,
    )


def rank_candidate_indices(model, windows, swapped, batch_for, checkpoint):
    import torch

    model.eval()
    indices = sorted(window["reference"] for window in windows)
    summaries = {}
    with torch.inference_mode():
        for condition, inputs in (("original", windows), ("swapped", swapped)):
            top1 = 0
            original_margins = []
            for window, original, replacement in zip(inputs, windows, swapped, strict=True):
                scores = {
                    index: -model.compute_loss(
                        batch_for(dict(window, reference=index), training=True)
                    ).item()
                    for index in indices
                }
                if not all(map(math.isfinite, scores.values())):
                    raise RuntimeError("Candidate scoring produced non-finite loss")
                reference_score = scores[window["reference"]]
                unique_top = sum(value >= reference_score for value in scores.values()) == 1
                top1 += unique_top
                margin = scores[original["reference"]] - scores[replacement["reference"]]
                original_margins.append(margin)
                print(json.dumps({
                    "checkpoint": checkpoint, "condition": condition,
                    "record": window["record"], "reference": window["reference"],
                    "candidate_scores": scores, "reference_is_unique_top": unique_top,
                    "original_minus_swap_reference_score": margin,
                }), flush=True)
            summaries[condition] = {"top1": top1, "margins": original_margins}
    print(json.dumps({
        "checkpoint": checkpoint, "total": len(windows),
        "original_top1": summaries["original"]["top1"],
        "swapped_top1": summaries["swapped"]["top1"],
        "margin_shift_toward_swapped_reference": [
            original - replacement for original, replacement in zip(
                summaries["original"]["margins"], summaries["swapped"]["margins"], strict=True
            )
        ],
    }), flush=True)


def compare_first_token_logits(model, windows, swapped, batch_for, checkpoint):
    import torch

    heads = [module for name, module in model.named_modules() if name.endswith(".lm_head")]
    if len(heads) != 1:
        raise RuntimeError(f"Expected one language-model output head, found {len(heads)}")
    model.eval()
    captured = []

    def capture(_module, _inputs, output):
        if not captured:
            captured.append(output[0, -1].detach().float().cpu())

    def probe(batch):
        captured.clear()
        with torch.inference_mode():
            output = model.generate(batch, max_new_tokens=1, do_sample=False)[0]
        if not captured or not torch.isfinite(captured[0]).all():
            raise RuntimeError("No finite first-token logits captured from the output head")
        return captured[0], output

    hook = heads[0].register_forward_hook(capture)
    try:
        for window, donor_input in zip(windows, swapped, strict=True):
            original_batch, swapped_batch = batch_for(window), batch_for(donor_input)
            original_signal = torch.as_tensor(original_batch[0]["time_series"])
            swapped_signal = torch.as_tensor(swapped_batch[0]["time_series"])
            if torch.equal(original_signal, swapped_signal):
                raise RuntimeError("Waveforms became identical during batch preparation")
            original, token = probe(original_batch)
            repeated, _ = probe(original_batch)
            changed, swapped_token = probe(swapped_batch)
            delta = (original - changed).abs()
            print(json.dumps({
                "checkpoint": checkpoint,
                "record": window["record"],
                "repeat_max_logit_delta": (original - repeated).abs().max().item(),
                "swap_max_logit_delta": delta.max().item(),
                "swap_mean_logit_delta": delta.mean().item(),
                "original_first_token": token,
                "swapped_first_token": swapped_token,
            }), flush=True)
    finally:
        hook.remove()


def evaluate(model, windows, batch_for, prior_offset, *, max_new_tokens, label=None):
    from statistics import mean

    import torch

    model.eval()
    if label:
        print(f"Evaluation: {label}", flush=True)
        torch.manual_seed(42)
    predictions = []
    prior_errors, model_errors, paired_prior_errors = [], [], []
    for window in windows:
        fs, anchor, reference = (window[key] for key in ("fs", "anchor", "reference"))
        length = len(window["signal"][0])
        with torch.inference_mode():
            decoding = {"do_sample": False} if label else {}
            output = model.generate(batch_for(window), max_new_tokens=max_new_tokens, **decoding)[0].strip()
        match = re.search(r"(?:^|\bAnswer:\s*)([0-9]{1,6})\.?\s*\Z", output, re.IGNORECASE)
        prediction = int(match[1]) if match and int(match[1]) < length else None
        predictions.append(prediction)
        prior_error = abs(anchor + prior_offset - reference) * 1000 / fs
        prior_errors.append(prior_error)
        model_error = None
        if prediction is not None:
            model_error = abs(prediction - reference) * 1000 / fs
            model_errors.append(model_error)
            paired_prior_errors.append(prior_error)
        print(json.dumps({
            "record": window["record"], "beat": window["beat"], "reference": reference,
            "prediction": prediction, "model_error_ms": model_error,
            "prior_error_ms": prior_error, "output": output,
        }), flush=True)
    print(json.dumps({
        "valid": len(model_errors), "total": len(prior_errors),
        "coverage": len(model_errors) / len(prior_errors),
        "model_mae_ms_on_valid": mean(model_errors) if model_errors else None,
        "prior_mae_ms_on_model_valid": mean(paired_prior_errors) if paired_prior_errors else None,
        "prior_mae_ms_all": mean(prior_errors),
    }, indent=2))
    return predictions


def train_lora(model, windows, batch_for, *, epochs=1):
    import random

    import torch

    adapters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    ]
    if not adapters:
        raise RuntimeError("No trainable LoRA weights found")
    optimizer = torch.optim.AdamW(adapters, lr=1e-4)
    model.train()
    rng = random.Random(42)
    order = []
    for _ in range(epochs):
        epoch = list(windows)
        rng.shuffle(epoch)
        order.extend(epoch)
    for step, window in enumerate(order, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch_for(window, training=True))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters, 1.0, error_if_nonfinite=True)
        if step == 1:
            updated = next(
                (p for p in adapters if p.grad is not None and torch.count_nonzero(p.grad)),
                None,
            )
            if updated is None:
                raise RuntimeError("First training step produced no LoRA gradient")
            before = updated.detach().clone()
        optimizer.step()
        if step == 1:
            if torch.equal(before, updated.detach()):
                raise RuntimeError("First optimizer step did not change the checked LoRA weight")
            print("First optimizer step changed a LoRA weight.", flush=True)
        print(f"Training step {step}/{len(order)} loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "score", "fine-tune", "controls", "overfit", "logits", "candidates")
    )
    parser.add_argument("panel", type=Path)
    parser.add_argument("--qtdb-dir", type=Path, default=Path("data/qt-database-1.0.0"))
    parser.add_argument("--split-csv", type=Path, default=Path("outputs/qtdb-audit/qtdb-split.csv"))
    parser.add_argument("--supervised", action="store_true", help="Include labeled training windows")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.panel, args.qtdb_dir, args.split_csv, supervised=args.supervised)
    else:
        score(
            args.panel,
            fine_tune=args.stage in ("fine-tune", "controls", "overfit", "logits", "candidates"),
            controls=args.stage == "controls",
            overfit=args.stage in ("overfit", "logits", "candidates"),
            logits=args.stage == "logits",
            candidates=args.stage == "candidates",
        )
