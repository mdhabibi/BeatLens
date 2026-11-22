"""Compare OpenTSLM scaling and training coverage in grouped QTDB training folds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def prepare(output):
    from train_qtdb_tpeak import prepare_split

    manifest = Path("outputs/qtdb-audit/qtdb-split.csv")
    examples, signals, _, _, _ = prepare_split(Path("data/qt-database-1.0.0"), manifest, "train")
    with manifest.open(newline="") as stream:
        groups = {row["qtdb_record"]: row["split_group_id"] for row in csv.DictReader(stream)}
    sample = json.loads(Path("outputs/qtdb-opentslm/choices-panel.json").read_text())
    subset = {(w["record"], w["beat"]) for pair in sample["train"] for w in pair}
    rows = [i for i, example in enumerate(examples) if example.valid_samples.all()]
    examples = [examples[i] for i in rows]
    selected = np.array([(e.record_id, e.beat_index) in subset for e in examples])
    if selected.sum() != len(subset):
        raise ValueError("Could not reproduce the original training subset")
    group_ids = np.array([groups[e.record_id] for e in examples])
    unique = np.unique(group_ids)
    np.random.default_rng(42).shuffle(unique)
    group_to_fold = {group: i % 5 for i, group in enumerate(unique)}
    folds = np.array([group_to_fold[group] for group in group_ids])
    targets = np.array([e.event_offsets["t_peak"][0] * 1000 / e.sampling_rate_hz for e in examples],
                       dtype=np.float32)
    np.savez_compressed(output, signals=signals[rows], targets=targets, selected=selected,
                        folds=folds, groups=group_ids, records=np.array([e.record_id for e in examples]))
    print(f"Prepared {len(rows)} complete windows, {selected.sum()} subset windows, "
          f"{len(unique)} groups across five folds.")


def bounded(signals):
    low = signals.min(axis=-1, keepdims=True)
    high = signals.max(axis=-1, keepdims=True)
    span = high - low
    return np.where(span > 1e-8, 2 * (signals - low) / span.clip(min=1e-8) - 1, 0)


def compare(features, targets, folds, groups, selected, scaling):
    import torch
    from probe_qtdb_opentslm_encoder import ridge_predictions

    features = features.to("cuda")
    targets = torch.as_tensor(targets, device="cuda")
    results = []
    for coverage, available in (("subset", selected), ("all", np.ones_like(selected))):
        sums = np.zeros(3)
        total = 0
        for fold in range(5):
            fit = (folds != fold) & available
            evaluation = np.flatnonzero(folds == fold)
            evaluation = evaluation[np.argsort(groups[evaluation], kind="stable")]
            _, counts = np.unique(groups[evaluation], return_counts=True)
            donors = np.roll(evaluation, int(counts.max()))
            if np.any(groups[evaluation] == groups[donors]):
                raise ValueError("Could not create a cross-group swap within this fold")
            evaluation_ids = torch.as_tensor(evaluation, device="cuda")
            fit_ids = torch.as_tensor(np.flatnonzero(fit), device="cuda")
            expected = targets[evaluation_ids]
            y = targets[fit_ids]
            prediction = ridge_predictions(features[fit_ids], y, features[evaluation_ids], 0.1)
            swapped = prediction.roll(int(counts.max()))
            errors = torch.stack([
                (prediction - expected).abs(), (swapped - expected).abs(),
                (y.mean() - expected).abs(),
            ])
            fold_mae = errors.mean(dim=1).cpu().tolist()
            sums += errors.sum(dim=1).cpu().numpy()
            total += len(evaluation)
            print(json.dumps({
                "scaling": scaling, "coverage": coverage, "fold": fold,
                "train_windows": int(fit.sum()), "evaluation_windows": len(evaluation),
                "mae_ms": fold_mae[0], "swap_mae_ms": fold_mae[1], "prior_mae_ms": fold_mae[2],
            }), flush=True)
        result = dict(zip(("mae_ms", "swap_mae_ms", "prior_mae_ms"), (sums / total).tolist()))
        results.append({"scaling": scaling, "coverage": coverage, "evaluation_windows": total, **result})
        print(json.dumps({"summary": results[-1]}), flush=True)
    return results


def run(panel, output):
    import torch
    from transformers.utils import logging
    from probe_qtdb_opentslm_encoder import make_encoder_extractor

    logging.set_verbosity_error()
    torch.backends.mha.set_fastpath_enabled(False)
    extract = make_encoder_extractor()
    results = []
    with np.load(panel, allow_pickle=False) as data:
        signals = data["signals"]
        for scaling, waveforms in (("zscore", signals), ("bounded", bounded(signals))):
            print(f"Extracting {scaling} encoder features.", flush=True)
            features = extract(waveforms)
            results.extend(compare(
                features, data["targets"], data["folds"], data["groups"], data["selected"], scaling,
            ))
            del features
            output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compare"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("beatlens-qtdb-scaling-cv.json"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.panel)
    else:
        run(args.panel, args.output)
