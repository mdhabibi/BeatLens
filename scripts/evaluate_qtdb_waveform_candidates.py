"""Evaluate a waveform-only peak ranker on the existing grouped QTDB folds."""

import argparse
import gzip
import json

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import HistGradientBoostingClassifier


def candidates(signals):
    features, times, windows = [], [], []
    offsets = np.arange(-24, 25, 4)
    for window, signal in enumerate(signals):
        for lead, row in enumerate(signal):
            smoothed = savgol_filter(row, 11, 2)
            for sign in (-1, 1):
                indices, properties = find_peaks(sign * smoothed, prominence=0.1, distance=10)
                for index, prominence in zip(indices, properties["prominences"], strict=True):
                    if 24 <= index < len(row) - 24:
                        time = (index - 250) * 4
                        times.append(time)
                        windows.append(window)
                        features.append([time, lead, sign, prominence, *smoothed[index + offsets]])
    return tuple(map(np.asarray, (features, times, windows)))


def add_patch_features(candidate_data, cache_path, window_count):
    import torch

    features, times, windows = candidate_data
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    # The locked ECG encoder emits one (lead, patch, feature) tensor per window.
    if not isinstance(cache, torch.Tensor) or cache.shape != (window_count, 2 * 157 * 128):
        raise ValueError("Expected flattened OpenTSLM features with shape (windows, 40192)")
    patches = cache.float().numpy().reshape(window_count, 2, 157, 128)
    leads = features[:, 1].astype(int)
    patch = (250 + times // 4) // 4
    local = patches[windows, leads, patch]
    if not np.isfinite(local).all():
        raise ValueError("Nonfinite OpenTSLM patch features")
    return (np.column_stack((features, local)), times, windows)


def score_windows(train_candidates, targets, train, test_candidates, count, *, average_candidates=False, score_ratio=0.0, aggregation="mean", candidate_diagnostics=None):
    features, times, windows = train_candidates
    first = int((250 + (targets[train].min() - 40) / 4) // 4) * 4
    stop = (int((250 + (targets[train].max() + 40) / 4) // 4) + 1) * 4
    lower, upper = (first - 250) * 4, (stop - 250) * 4
    fit = (times >= lower) & (times < upper) & train[windows]
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=0)
    model.fit(features[fit], np.abs(times[fit] - targets[windows[fit]]) <= 20)
    prediction = np.full(count, targets[train].mean(), dtype=float)
    features, times, windows = test_candidates
    indices = np.flatnonzero((times >= lower) & (times < upper))
    if len(indices):
        scores = model.predict_proba(features[indices])[:, 1]
        order = np.argsort(scores, kind="stable")
        if average_candidates:
            # Keep the strongest three distinct peak times per window.
            choices = {}
            for offset in order[::-1]:
                index = indices[offset]
                choice = choices.setdefault(windows[index], {})
                if len(choice) < 3:
                    choice.setdefault(times[index], scores[offset])
            for window, choice in choices.items():
                probabilities = np.asarray(list(choice.values())).clip(min=1e-8)
                close = probabilities >= score_ratio * probabilities[0]
                peak_times, weights = np.asarray(list(choice))[close], probabilities[close]
                if candidate_diagnostics is not None:
                    candidate_diagnostics[window] = (np.ptp(peak_times), next(iter(choice.values())))
                if aggregation == "median":
                    by_time = np.argsort(peak_times)
                    cumulative = np.cumsum(weights[by_time])
                    middle = np.searchsorted(cumulative, cumulative[-1] / 2)
                    prediction[window] = peak_times[by_time[middle]]
                else:
                    average = np.average(peak_times, weights=weights)
                    if aggregation == "snap":
                        # Candidates are ordered by score, so ties favor the stronger peak.
                        average = peak_times[np.argmin(np.abs(peak_times - average))]
                    prediction[window] = average
        else:
            for index in indices[order]:
                prediction[windows[index]] = times[index]
    return prediction


def grouped_predictions(training, targets, folds, selected, *, average_candidates=False, score_ratio=0.0, aggregation="mean", candidate_diagnostics=None):
    prediction = np.empty(len(targets))
    features, times, windows = training
    for fold in np.unique(folds):
        train = (folds != fold) & selected
        test = folds == fold
        indices = np.flatnonzero(test)
        eligible = test[windows]
        testing = features[eligible], times[eligible], np.searchsorted(indices, windows[eligible])
        fold_diagnostics = np.full((len(indices), 2), np.nan) if candidate_diagnostics is not None else None
        prediction[test] = score_windows(
            training, targets, train, testing, len(indices), average_candidates=average_candidates, score_ratio=score_ratio,
            aggregation=aggregation, candidate_diagnostics=fold_diagnostics,
        )
        if candidate_diagnostics is not None:
            candidate_diagnostics[test] = fold_diagnostics
    return prediction


def paired_interval(differences, groups):
    _, group_indices = np.unique(groups, return_inverse=True)
    totals = np.bincount(group_indices, weights=differences)
    counts = np.bincount(group_indices)
    draws = np.random.default_rng(42).integers(len(totals), size=(10000, len(totals)))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def stratified_review_indices(risk_score, records, budget):
    _, record_index, counts = np.unique(records, return_inverse=True, return_counts=True)
    fractions = counts * budget / len(records)
    quotas = np.floor(fractions).astype(int)
    remainder = budget - quotas.sum()
    quotas[np.argsort(quotas - fractions, kind="stable")[:remainder]] += 1
    selected = []
    for record, quota in enumerate(quotas):
        indices = np.flatnonzero(record_index == record)
        order = np.argsort(-risk_score[indices], kind="stable")
        selected.extend(indices[order[:quota]])
    return np.asarray(selected, dtype=int)


def evaluate(panel, external_panel=None, feature_cache=None, *, average_candidates=False, score_ratio=0.0, aggregation="mean", audit_spread=False):
    with np.load(panel, allow_pickle=False) as data:
        training = candidates(data["signals"])
        targets, folds, selected = data["targets"], data["folds"], data["selected"]
        if feature_cache or average_candidates:
            groups = data["groups"]
        if audit_spread:
            review_records = data["records"]
    if external_panel:
        with gzip.open(external_panel, "rt") as stream:
            rows = json.load(stream)
        if any(row["fs"] != 250 or row["anchor"] != 250
               or np.shape(row["signal"]) != (2, 625) for row in rows):
            raise ValueError("Expected 625-sample, 250 Hz, two-channel windows anchored at 250")
        testing = candidates([row["signal"] for row in rows])
        prediction = score_windows(training, targets, selected, testing, len(rows))
        if average_candidates:
            diagnostics = np.full((len(rows), 2), np.nan) if audit_spread else None
            averaged = score_windows(
                training, targets, selected, testing, len(rows), average_candidates=True, score_ratio=score_ratio,
                aggregation=aggregation, candidate_diagnostics=diagnostics,
            )
        targets = np.asarray([(row["reference"] - row["anchor"]) * 4 for row in rows])
        if average_candidates:
            groups = np.asarray([row.get("subject", row["record"]) for row in rows])
        if audit_spread:
            review_records = np.asarray([row["record"] for row in rows])
    else:
        prediction = grouped_predictions(training, targets, folds, selected)
        if average_candidates:
            diagnostics = np.full((len(targets), 2), np.nan) if audit_spread else None
            averaged = grouped_predictions(
                training, targets, folds, selected, average_candidates=True, score_ratio=score_ratio,
                aggregation=aggregation, candidate_diagnostics=diagnostics,
            )
    predictions = {"waveform_only": prediction}
    if average_candidates:
        label = "waveform_selective_average" if score_ratio else "waveform_top3_average"
        if aggregation == "median":
            label = "waveform_weighted_median"
        elif aggregation == "snap":
            label = "waveform_snapped_average"
        predictions[label] = averaged
    if feature_cache:
        augmented = add_patch_features(training, feature_cache, len(targets))
        predictions["waveform_plus_opentslm"] = grouped_predictions(
            augmented, targets, folds, selected,
        )
    for label, mask in (("all", np.ones(len(targets), dtype=bool)), ("after_345ms", targets > 345)):
        if not mask.any():
            continue
        for model, values in predictions.items():
            errors = abs(values[mask] - targets[mask])
            print(json.dumps({"model": model, "group": label, "windows": len(errors),
                              "mae_ms": float(errors.mean()), "over_80ms": int((errors > 80).sum()),
                              "within_40ms": float((errors <= 40).mean())}))
        for model, values in predictions.items():
            if model == "waveform_only":
                continue
            baseline_error = abs(prediction[mask] - targets[mask])
            method_error = abs(values[mask] - targets[mask])
            differences = method_error - baseline_error
            large_miss_delta_pp = 100 * (
                (method_error > 80).astype(float) - (baseline_error > 80)
            )
            print(json.dumps({
                "comparison_model": model,
                "group": label,
                "mae_delta_ms": float(differences.mean()),
                "mae_delta_ms_ci95": paired_interval(differences, groups[mask]),
                "over_80ms_delta": int((method_error > 80).sum() - (baseline_error > 80).sum()),
                "over_80ms_delta_pp_ci95": paired_interval(large_miss_delta_pp, groups[mask]),
            }))

    if audit_spread:
        from sklearn.metrics import average_precision_score, roc_auc_score

        spread, peak_score = diagnostics.T
        # A missing candidate is flagged rather than scored as zero disagreement.
        flagged = ~np.isfinite(spread) | (spread > 80)
        errors = abs(averaged - targets)
        large_miss = errors > 80
        risk_score = 1 - np.nan_to_num(peak_score, nan=0.0)
        if large_miss.any() and not large_miss.all():
            print(json.dumps({
                "risk_signal": "low_top_candidate_score",
                "auroc": roc_auc_score(large_miss, risk_score),
                "average_precision": average_precision_score(large_miss, risk_score),
                "large_miss_prevalence": large_miss.mean().item(),
            }))
        if external_panel:
            calibration = np.full((len(selected), 2), np.nan)
            with np.load(panel, allow_pickle=False) as data:
                grouped_predictions(
                    training, data["targets"], folds, selected, average_candidates=True,
                    score_ratio=score_ratio, aggregation=aggregation,
                    candidate_diagnostics=calibration,
                )
            calibration_scores = np.nan_to_num(calibration[:, 1], nan=0.0)
        else:
            calibration_scores = np.nan_to_num(peak_score, nan=0.0)
        cutoff = float(np.quantile(calibration_scores, 0.10))
        review = np.nan_to_num(peak_score, nan=0.0) <= cutoff
        caught = int((review & large_miss).sum())
        print(json.dumps({
            "review_policy": "qtdb_bottom_decile_top_score",
            "top_score_cutoff": cutoff,
            "review_windows": int(review.sum()),
            "windows": len(review),
            "large_misses_caught": caught,
            "large_misses": int(large_miss.sum()),
        }))
        budget = int(np.ceil(0.10 * len(risk_score)))
        # Break score ties by panel order, independently of reference labels.
        review_indices = np.argsort(-risk_score, kind="stable")[:budget]
        print(json.dumps({
            "review_policy": "fixed_10_percent_budget",
            "review_windows": budget,
            "windows": len(risk_score),
            "large_misses_caught": int(large_miss[review_indices].sum()),
            "large_misses": int(large_miss.sum()),
        }))
        stratified = stratified_review_indices(risk_score, review_records, budget)
        print(json.dumps({
            "review_policy": "proportional_per_record_10_percent_budget",
            "review_windows": len(stratified),
            "records_reviewed": len(np.unique(review_records[stratified])),
            "pooled_records_reviewed": len(np.unique(review_records[review_indices])),
            "large_misses_caught": int(large_miss[stratified].sum()),
            "large_misses": int(large_miss.sum()),
        }))
        for label, mask in (("flagged", flagged), ("unflagged", ~flagged)):
            print(json.dumps({
                "disagreement": label, "windows": int(mask.sum()),
                "large_misses": int((errors[mask] > 80).sum()),
                "large_miss_rate": float((errors[mask] > 80).mean()) if mask.any() else None,
            }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel")
    parser.add_argument("--external-panel")
    parser.add_argument("--feature-cache", help="Aligned flattened QTDB OpenTSLM features")
    parser.add_argument("--average-candidates", action="store_true")
    parser.add_argument("--score-ratio", type=float, default=0.0,
                        help="Average candidates with at least this fraction of the highest score")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidate-median", action="store_true",
                        help="Use the score-weighted median of retained peaks")
    mode.add_argument("--snap-to-candidate", action="store_true",
                      help="Snap the weighted mean to its nearest retained peak")
    parser.add_argument("--audit-spread", action="store_true",
                        help="Measure large misses when candidate times span over 80 ms")
    args = parser.parse_args()
    if args.audit_spread and not args.average_candidates:
        parser.error("--audit-spread requires --average-candidates")
    if (args.candidate_median or args.snap_to_candidate) and not args.average_candidates:
        parser.error("Candidate aggregation options require --average-candidates")
    if not 0 <= args.score_ratio <= 1 or (args.score_ratio and not args.average_candidates):
        parser.error("--score-ratio requires --average-candidates and a value in [0, 1]")
    if args.feature_cache and args.external_panel:
        parser.error("--feature-cache currently supports QTDB grouped evaluation only")
    evaluate(args.panel, args.external_panel, args.feature_cache,
             average_candidates=args.average_candidates, score_ratio=args.score_ratio,
             aggregation="median" if args.candidate_median else "snap" if args.snap_to_candidate else "mean",
             audit_spread=args.audit_spread)
