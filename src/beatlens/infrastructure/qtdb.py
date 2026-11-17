"""QTDB adapter for sparse, beat-centered fiducial localization examples."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TARGET_SYMBOLS: dict[str, str] = {
    "(": "onset",
    ")": "offset",
    "p": "p_peak",
    "t": "t_peak",
    "u": "u_peak",
}
TARGET_NAMES = tuple(TARGET_SYMBOLS.values())


@dataclass(frozen=True)
class QTDBBeatExample:
    """One selected beat and its observed sparse fiducials.

    ``signal`` has shape ``(samples, leads)`` in the physical units decoded by WFDB.
    ``event_offsets`` stores sample offsets relative to the beat anchor (negative is before it).
    A missing target (``observed_targets[name] is False``) is unknown, not a negative label.
    ``valid_samples`` is false only for zero padding at a record edge.
    """

    record_id: str
    source_database: str
    source_record_id: str
    split: str
    beat_index: int
    anchor_sample: int
    sampling_rate_hz: float
    lead_names: tuple[str, ...]
    signal: np.ndarray
    valid_samples: np.ndarray
    event_offsets: dict[str, tuple[int, ...]]
    observed_targets: dict[str, bool]


def iter_qtdb_examples(
    qtdb_dir: Path,
    split_csv: Path,
    split: str,
    *,
    before_anchor_seconds: float = 1.0,
    after_anchor_seconds: float = 1.5,
    max_event_distance_seconds: float = 1.25,
) -> Iterator[QTDBBeatExample]:
    """Yield beat-centered examples for one split.

    The `.man` reference beat annotations define anchors. Fiducial samples from `.q1c` are
    assigned to the nearest selected-beat anchor, but only within `max_event_distance_seconds`.
    This keeps sparse annotated beats separate from the many unannotated beats in each excerpt.
    The function does not normalize, resample, or treat absent markers as negative labels.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")
    if min(before_anchor_seconds, after_anchor_seconds, max_event_distance_seconds) <= 0:
        raise ValueError("window and assignment durations must be positive")
    try:
        import wfdb
    except ImportError as error:
        raise ImportError("Install the QTDB extra with `uv sync --extra qtdb`") from error

    with split_csv.open(newline="", encoding="utf-8") as stream:
        selected = [row for row in csv.DictReader(stream) if row["split"] == split]
    if not selected:
        raise ValueError(f"No records assigned to split {split!r} in {split_csv}")

    left_samples_by_rate: dict[float, int] = {}
    right_samples_by_rate: dict[float, int] = {}
    assignment_radius_by_rate: dict[float, int] = {}

    for row in selected:
        record_id = row["qtdb_record"]
        record_base = qtdb_dir / record_id
        record = wfdb.rdrecord(str(record_base), physical=True)
        rate = float(record.fs)
        left = left_samples_by_rate.setdefault(rate, round(before_anchor_seconds * rate))
        right = right_samples_by_rate.setdefault(rate, round(after_anchor_seconds * rate))
        assignment_radius = assignment_radius_by_rate.setdefault(
            rate, round(max_event_distance_seconds * rate)
        )
        window_size = left + right
        signal = np.asarray(record.p_signal, dtype=np.float32)
        if signal.ndim != 2:
            raise ValueError(f"Expected a two-dimensional ECG signal for {record_id}")

        anchors_annotation = wfdb.rdann(str(record_base), "man")
        anchors = np.asarray(anchors_annotation.sample, dtype=np.int64)
        if anchors.size == 0:
            raise ValueError(f"No selected-beat anchors in {record_id}.man")
        anchors.sort()

        annotation = wfdb.rdann(str(record_base), "q1c")
        assigned: list[dict[str, list[int]]] = [defaultdict(list) for _ in anchors]
        for sample, symbol in zip(annotation.sample, annotation.symbol, strict=True):
            target = TARGET_SYMBOLS.get(symbol)
            if target is None:
                continue
            insertion = int(np.searchsorted(anchors, sample))
            candidates = [
                index for index in (insertion - 1, insertion) if 0 <= index < len(anchors)
            ]
            closest = min(
                candidates,
                key=lambda index: abs(int(sample) - int(anchors[index])),
            )
            distance = abs(int(sample) - int(anchors[closest]))
            if distance > assignment_radius:
                continue
            anchor_offset = int(sample) - int(anchors[closest])
            if -left <= anchor_offset < right:
                assigned[closest][target].append(anchor_offset)

        for beat_index, anchor in enumerate(anchors):
            start = int(anchor) - left
            end = int(anchor) + right
            source_start = max(start, 0)
            source_end = min(end, signal.shape[0])
            destination_start = source_start - start
            destination_end = destination_start + (source_end - source_start)
            window = np.zeros((window_size, signal.shape[1]), dtype=np.float32)
            valid = np.zeros(window_size, dtype=bool)
            window[destination_start:destination_end] = signal[source_start:source_end]
            valid[destination_start:destination_end] = True

            events = {
                target: tuple(sorted(assigned[beat_index].get(target, [])))
                for target in TARGET_NAMES
            }
            yield QTDBBeatExample(
                record_id=record_id,
                source_database=row["source_database"],
                source_record_id=row["source_record_id"],
                split=split,
                beat_index=beat_index,
                anchor_sample=int(anchor),
                sampling_rate_hz=rate,
                lead_names=tuple(record.sig_name or ()),
                signal=window,
                valid_samples=valid,
                event_offsets=events,
                observed_targets={target: bool(events[target]) for target in TARGET_NAMES},
            )
