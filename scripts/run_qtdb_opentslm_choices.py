"""Train and evaluate waveform-dependent T-peak choices on QTDB."""

import argparse
import json
import math
import random
from pathlib import Path


def prepare(path):
    from train_qtdb_tpeak import LEFT_SECONDS, prepare_split

    def pairs(examples, signals, seed):
        rng = random.Random(seed)
        windows = []
        by_record = {}
        for example, signal in zip(examples, signals, strict=True):
            if example.valid_samples.all():
                by_record.setdefault(example.record_id, []).append((example, signal))
        for record in by_record.values():
            for example, signal in rng.sample(record, min(6, len(record))):
                fs = example.sampling_rate_hz
                anchor = round(LEFT_SECONDS * fs)
                windows.append({
                    "record": example.record_id, "beat": example.beat_index,
                    "fs": fs, "anchor": anchor, "signal": signal.tolist(),
                    "reference": anchor + example.event_offsets["t_peak"][0],
                })
        rng.shuffle(windows)
        result = []
        while windows:
            first = windows.pop()
            gap = rng.randint(6, 25)
            eligible = [
                index for index, other in enumerate(windows)
                if first["record"] != other["record"]
                and first["fs"] == other["fs"]
                and first["anchor"] == other["anchor"]
                and 6 <= abs(first["reference"] - other["reference"]) <= 30
            ]
            if not eligible:
                continue
            index = min(eligible, key=lambda i: abs(abs(
                first["reference"] - windows[i]["reference"]
            ) - gap))
            result.append([first, windows.pop(index)])
        return result

    data = Path("data/qt-database-1.0.0")
    split = Path("outputs/qtdb-audit/qtdb-split.csv")
    train, train_signals, _, prior, _ = prepare_split(data, split, "train")
    val, val_signals, _, _, _ = prepare_split(data, split, "validation")
    panel = {
        "train": pairs(train, train_signals, 42),
        "validation": pairs(val, val_signals, 43),
        "prior_offset": prior,
    }
    train_records = {w["record"] for pair in panel["train"] for w in pair}
    val_records = {w["record"] for pair in panel["validation"] for w in pair}
    if len(panel["train"]) < 4 or not val_records or train_records & val_records:
        raise ValueError("Insufficient pairs or overlapping records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(panel, allow_nan=False))
    print(f"Train: {len(panel['train'])} pairs / {len(train_records)} records")
    print(f"Validation: {len(panel['validation'])} pairs / {len(val_records)} records")


def run(path, epochs, smoke, *, train_projector=False):
    import torch
    from opentslm.model.llm.OpenTSLM import OpenTSLM
    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate as collate,
    )

    panel = json.loads(path.read_text())
    torch.manual_seed(42)
    rng = random.Random(42)
    model = OpenTSLM.load_pretrained(
        "OpenTSLM/llama-3.2-1b-ecg-sp", enable_lora=True, device="cuda"
    )
    adapters = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name.lower()
    }
    if not adapters:
        raise RuntimeError("No trainable LoRA parameters")
    projector = {}
    if train_projector:
        projector_module = model.projector.projector
        projector_module.requires_grad_(True)
        projector_ids = {id(parameter) for parameter in projector_module.parameters()}
        projector = {
            name: parameter for name, parameter in model.named_parameters()
            if id(parameter) in projector_ids
        }
        if not projector or set(projector) & set(adapters):
            raise RuntimeError("Could not identify a separate time-series projector group")
    trainable = adapters | projector
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in trainable)
    groups = {"lora": adapters}
    if projector:
        groups["projector"] = projector
    optimizer = torch.optim.AdamW([
        {"params": list(group.values()), "lr": 1e-4}
        for group in groups.values()
    ])
    print(json.dumps({
        "trainable_parameters": {
            name: sum(parameter.numel() for parameter in group.values())
            for name, group in groups.items()
        }
    }), flush=True)

    def candidate_loss(window, options, answer):
        a, b = options
        sample = {
            "pre_prompt": f"These two ECG channels were sampled at {window['fs']:g} Hz.",
            "time_series": window["signal"],
            "time_series_text": ["ECG channel 1.", "ECG channel 2."],
            "post_prompt": (
                f"The beat anchor is at sample {window['anchor']}. "
                "Which sample is closer to this beat's T-wave peak? "
                f"A: {a}. B: {b}. Answer with A or B. Answer:"
            ),
            "answer": answer + model.get_eos_token(),
        }
        return model.compute_loss(collate([sample], patch_size=PATCH_SIZE))

    def evaluate(pairs, label):
        model.eval()
        right = both = follows_signal = prior_right = 0
        with torch.inference_mode():
            for first, second in pairs:
                for options in (
                    (first["reference"], second["reference"]),
                    (second["reference"], first["reference"]),
                ):
                    correct = []
                    predictions = []
                    for window in (first, second):
                        difference = (
                            candidate_loss(window, options, "B")
                            - candidate_loss(window, options, "A")
                        ).item()
                        if not math.isfinite(difference):
                            raise RuntimeError("Non-finite evaluation margin")
                        prediction = "A" if difference > 0 else "B" if difference < 0 else None
                        target = "A" if window["reference"] == options[0] else "B"
                        predictions.append(prediction)
                        correct.append(prediction == target)
                        prior = window["anchor"] + panel["prior_offset"]
                        prior_choice = min(range(2), key=lambda i: abs(options[i] - prior))
                        prior_right += options[prior_choice] == window["reference"]
                    right += sum(correct)
                    both += all(correct)
                    follows_signal += None not in predictions and predictions[0] != predictions[1]
        result = {
            "stage": label, "correct": right, "total": len(pairs) * 4,
            "both_correct": both, "opposite_predictions": follows_signal,
            "paired_total": len(pairs) * 2, "timing_prior_correct": prior_right,
        }
        print(json.dumps(result), flush=True)
        return both, right

    train = panel["train"][:4] if smoke else panel["train"]
    evaluation = train if smoke else panel["validation"]
    evaluate(evaluation, "pretrained_train_probe" if smoke else "pretrained_validation")
    best = (-1, -1)
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train)
        rng.shuffle(order)
        positive_total = ranking_total = 0.0
        for step, pair in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)
            options = tuple(w["reference"] for w in pair)
            if rng.randrange(2):
                options = options[::-1]
            for window in pair:
                answer = "A" if window["reference"] == options[0] else "B"
                other = "B" if answer == "A" else "A"
                positive = candidate_loss(window, options, answer)
                negative = candidate_loss(window, options, other)
                ranking = torch.nn.functional.softplus(positive - negative)
                loss = positive + ranking
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                (loss / len(pair)).backward()
                positive_total += positive.detach().item()
                ranking_total += ranking.detach().item()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
            first_step = {}
            if epoch == step == 1:
                for name, group in groups.items():
                    parameter = next((
                        p for p in group.values()
                        if p.grad is not None and torch.count_nonzero(p.grad).item()
                    ), None)
                    if parameter is None:
                        raise RuntimeError(f"No gradient reached the {name} parameters")
                    first_step[name] = (parameter, parameter.detach().clone())
            optimizer.step()
            for name, (parameter, before) in first_step.items():
                if torch.equal(parameter.detach(), before):
                    raise RuntimeError(f"First step did not change the checked {name} weight")
                print(f"First optimizer step changed a {name} weight.", flush=True)
            if step % 25 == 0:
                print(f"Epoch {epoch}, pair {step}/{len(order)}", flush=True)
        count = 2 * len(order)
        print(json.dumps({
            "epoch": epoch,
            "mean_answer_loss": positive_total / count,
            "mean_ranking_loss": ranking_total / count,
            "mean_total_loss": (positive_total + ranking_total) / count,
        }), flush=True)
        metric = evaluate(evaluation, f"{'train_probe' if smoke else 'validation'}_epoch_{epoch}")
        if smoke and metric[0] == len(evaluation) * 2:
            break
        if not smoke and metric > best:
            best = metric
            checkpoint = path.with_name(
                "qtdb-choices-best-lora-projector.pt" if projector else "qtdb-choices-best-lora.pt"
            )
            torch.save({
                "adapters": {name: value.detach().cpu() for name, value in adapters.items()},
                "projector": {name: value.detach().cpu() for name, value in projector.items()},
                "epoch": epoch, "validation_both_correct": metric[0],
                "validation_correct": metric[1],
                "base_checkpoint": "OpenTSLM/llama-3.2-1b-ecg-sp",
            }, checkpoint)
            print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "train"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-projector", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.stage == "prepare":
        prepare(args.panel)
    else:
        run(args.panel, args.epochs, args.stage == "smoke", train_projector=args.train_projector)
