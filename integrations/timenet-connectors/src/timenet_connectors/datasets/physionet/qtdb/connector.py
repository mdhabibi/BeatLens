"""Convert QTDB excerpts and sparse first-expert fiducials into TimeF."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import RegularAxis
from timenet.errors import TimeNetConnectorError, TimeNetDownloadError
from timenet.types import (
    Annotation,
    LocalizationMode,
    Split,
    TemporalLocalizationTask,
    TimeInterval,
    TimePoint,
    TimeSeriesSpec,
    ureg,
)
from timenet_connectors.bases.physionet import BasePhysioNetConnector, WfdbHeader
from timenet_connectors.download import Artifact, download_files


_SOURCE_URL = "https://physionet.org/files/qtdb/1.0.0"
_SOURCE_ROOT = "qtdb-1.0.0"
_RECORDS_SHA256 = "29aeb9287007953f42ffc4da9419eff17d34f681c1c8e4f34992b94f252f9a71"
_CHECKSUMS_SHA256 = "50ddae2c0515df6f9711a098ba0cb7f4e35b6b25642f905bd423f7f390d18497"
_SPLIT_SHA256 = "b23ed4664958c29732c39e5f84f9374487991a78626621d8a32b4ff1bb61786f"
_SPLIT_PATH = Path(__file__).with_name("split.csv")
_RECORD_ID = re.compile(r"[A-Za-z0-9]+\Z")
_SOURCE_FILES = ("hea", "dat", "man", "q1c")
_CHECKSUM_FIELDS = 2
_LEAD_COUNT = 2
_SAMPLING_RATE_HZ = 250
_TARGET_SYMBOLS = {
    "(": "onset",
    ")": "offset",
    "p": "p_peak",
    "t": "t_peak",
    "u": "u_peak",
}
_TARGET_PROMPTS = {
    "onset": "waveform onset",
    "offset": "waveform offset",
    "p_peak": "P-wave peak",
    "t_peak": "T-wave peak",
    "u_peak": "U-wave peak",
}
_BEFORE_SECONDS = 1.0
_AFTER_SECONDS = 1.5
_ASSIGNMENT_SECONDS = 1.25
_ECG = TimeSeriesSpec(spec_type="ecg", name="ECG", unit_value=ureg.millivolt)


@dataclass(frozen=True)
class QtdbSplit:
    """The approved split and provenance row for one QTDB excerpt."""

    split: Split
    source_database: str
    source_record_id: str
    split_group_id: str
    split_group_basis: str


@dataclass(frozen=True)
class QtdbSource:
    """The downloaded release root and its per-record split metadata."""

    root: Path
    record_ids: tuple[str, ...]
    splits: dict[str, QtdbSplit]


def _read_split_manifest(path: Path) -> dict[str, QtdbSplit]:
    """Read and validate the connector's frozen, grouped record split."""
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise TimeNetConnectorError(f"cannot read QTDB split manifest {path}: {exc}") from exc
    if digest != _SPLIT_SHA256:
        raise TimeNetConnectorError(f"{path} does not match the pinned QTDB split manifest")
    required = {
        "qtdb_record",
        "source_database",
        "source_record_id",
        "split_group_id",
        "split_group_basis",
        "split",
    }
    result: dict[str, QtdbSplit] = {}
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or ()):
                raise TimeNetConnectorError(f"{path} is missing required columns {sorted(required)}")
            for row in reader:
                record_id = row["qtdb_record"]
                if not _RECORD_ID.fullmatch(record_id):
                    raise TimeNetConnectorError(f"invalid QTDB record id in {path}: {record_id!r}")
                if record_id in result:
                    raise TimeNetConnectorError(f"duplicate QTDB record id in {path}: {record_id!r}")
                result[record_id] = QtdbSplit(
                    split=Split(row["split"]),
                    source_database=row["source_database"],
                    source_record_id=row["source_record_id"],
                    split_group_id=row["split_group_id"],
                    split_group_basis=row["split_group_basis"],
                )
    except (OSError, KeyError, ValueError, csv.Error) as exc:
        raise TimeNetConnectorError(f"cannot read QTDB split manifest {path}: {exc}") from exc
    return result


def _read_records(path: Path) -> tuple[str, ...]:
    """Read the release's stable record list and reject malformed or repeated names."""
    try:
        values = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError as exc:
        raise TimeNetDownloadError(f"cannot read QTDB RECORDS file {path}: {exc}") from exc
    if not values or any(not _RECORD_ID.fullmatch(value) for value in values):
        raise TimeNetDownloadError(f"QTDB RECORDS file {path} contains no records or an invalid record id")
    if len(values) != len(set(values)):
        raise TimeNetDownloadError(f"QTDB RECORDS file {path} repeats a record id")
    return values


def _read_checksums(path: Path) -> dict[str, str]:
    """Read the per-file SHA-256 values published with the PhysioNet release."""
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == _CHECKSUM_FIELDS:
                result[fields[1].lstrip("* ")] = fields[0].lower()
    except OSError as exc:
        raise TimeNetDownloadError(f"cannot read QTDB checksum list {path}: {exc}") from exc
    return result


def _closest_anchor(sample: int, anchors: np.ndarray) -> int:
    """Return the index of the closest selected-beat anchor to a fiducial sample."""
    insertion = int(np.searchsorted(anchors, sample))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(anchors)]
    return min(candidates, key=lambda index: abs(sample - int(anchors[index])))


@dataclass(frozen=True)
class QtdbMarkers:
    """Selected-beat annotations and their assigned first-expert fiducials."""

    anchors: np.ndarray
    anchor_annotations: tuple[Annotation, ...]
    assigned: tuple[dict[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class QtdbTaskContext:
    """The settings needed to emit one excerpt's localization tasks."""

    record_id: str
    split: QtdbSplit
    markers: QtdbMarkers
    rate: Fraction
    sig_len: int


def _read_markers(wfdb: Any, base: Path, record: Record, header: WfdbHeader, rate: Fraction) -> QtdbMarkers:
    """Read selected-beat anchors and add available `.q1c` markers to the record."""
    anchors_annotation = wfdb.rdann(str(base), "man")
    anchors = np.asarray(anchors_annotation.sample, dtype=np.int64)
    if anchors.size == 0 or np.any(np.diff(anchors) <= 0):
        raise TimeNetConnectorError(".man anchors are empty, unsorted, or repeated")
    if int(anchors[-1]) >= header.sig_len:
        raise TimeNetConnectorError(".man anchor falls outside the ECG excerpt")

    rate_hz = float(rate)
    anchor_annotations = [
        Annotation(
            key="qtdb.selected_beat",
            value=symbol,
            span=TimePoint.seconds(float(anchor) / rate_hz),
            source=".man",
        )
        for anchor, symbol in zip(anchors, anchors_annotation.symbol, strict=True)
    ]
    assigned: list[dict[str, list[int]]] = [{target: [] for target in _TARGET_PROMPTS} for _ in anchors]
    event_annotations: list[Annotation] = []
    fiducials = wfdb.rdann(str(base), "q1c")
    left = round(_BEFORE_SECONDS * rate_hz)
    right = round(_AFTER_SECONDS * rate_hz)
    assignment_radius = round(_ASSIGNMENT_SECONDS * rate_hz)
    for sample_value, symbol in zip(fiducials.sample, fiducials.symbol, strict=True):
        target_name = _TARGET_SYMBOLS.get(symbol)
        if target_name is None:
            continue
        sample = int(sample_value)
        if sample < 0 or sample >= header.sig_len:
            raise TimeNetConnectorError(f".q1c sample {sample} falls outside the ECG excerpt")
        beat_index = _closest_anchor(sample, anchors)
        offset = sample - int(anchors[beat_index])
        if abs(offset) > assignment_radius or not -left <= offset < right:
            continue
        assigned[beat_index][target_name].append(sample)
        event_annotations.append(
            Annotation(
                key=f"qtdb.fiducial.{target_name}",
                span=TimePoint.seconds(sample / rate_hz),
                source=".q1c first expert, second pass",
            )
        )

    attached = record.add_annotations((*anchor_annotations, *event_annotations))
    return QtdbMarkers(
        anchors=anchors,
        anchor_annotations=attached[: len(anchor_annotations)],
        assigned=tuple({target: tuple(samples) for target, samples in beat.items()} for beat in assigned),
    )


def _add_tasks(
    dataset: TimeFDataset,
    record: Record,
    context: QtdbTaskContext,
) -> None:
    """Add one sparse localization task for each observed fiducial type and selected beat."""
    rate_hz = float(context.rate)
    before = round(_BEFORE_SECONDS * rate_hz)
    after = round(_AFTER_SECONDS * rate_hz)
    for beat_index, anchor_value in enumerate(context.markers.anchors):
        anchor = int(anchor_value)
        scope_start = max(0, anchor - before) / rate_hz
        scope_end = min(context.sig_len, anchor + after) / rate_hz
        if scope_start >= scope_end:
            continue
        for target_name, prompt_name in _TARGET_PROMPTS.items():
            samples = sorted(set(context.markers.assigned[beat_index][target_name]))
            if not samples:
                continue
            targets = tuple(TimePoint.seconds(sample / rate_hz) for sample in samples)
            dataset.add_task(
                task=TemporalLocalizationTask(
                    inputs=(record,),
                    prompt=f"Locate the {prompt_name} for this selected beat.",
                    split=context.split.split,
                    mode=LocalizationMode.SPARSE,
                    scope=TimeInterval.seconds(scope_start, scope_end),
                    input_annotations=(context.markers.anchor_annotations[beat_index],),
                    targets=targets,
                    metadata={
                        "qtdb_record": context.record_id,
                        "selected_beat_index": beat_index,
                        "beat_anchor_sample": anchor,
                        "fiducial_type": target_name,
                    },
                )
            )


class QtdbConnector(BasePhysioNetConnector[QtdbSource]):
    """Import QTDB excerpts with sparse, beat-scoped first-expert fiducial tasks."""

    async def download_async(self, cache_dir: Path) -> list[QtdbSource]:  # ruff: ignore[no-self-use]
        """Fetch checksum-pinned source files and pair them with the frozen split manifest."""
        root = cache_dir / _SOURCE_ROOT
        records_path = root / "RECORDS"
        checksums_path = root / "SHA256SUMS.txt"
        await download_files(
            [
                Artifact(
                    f"{_SOURCE_URL}/RECORDS",
                    records_path,
                    sha256=_RECORDS_SHA256,
                ),
                Artifact(
                    f"{_SOURCE_URL}/SHA256SUMS.txt",
                    checksums_path,
                    sha256=_CHECKSUMS_SHA256,
                ),
            ]
        )
        record_ids = _read_records(records_path)
        checksums = _read_checksums(checksums_path)
        splits = _read_split_manifest(_SPLIT_PATH)
        if set(record_ids) != set(splits):
            missing = sorted(set(record_ids) - set(splits))
            extra = sorted(set(splits) - set(record_ids))
            raise TimeNetConnectorError(
                f"QTDB split manifest does not match the release; missing={missing}, extra={extra}"
            )

        artifacts: list[Artifact] = []
        for record_id in record_ids:
            for extension in _SOURCE_FILES:
                filename = f"{record_id}.{extension}"
                digest = checksums.get(filename)
                if digest is None:
                    raise TimeNetDownloadError(f"QTDB SHA256SUMS.txt has no digest for {filename}")
                artifacts.append(
                    Artifact(
                        f"{_SOURCE_URL}/{filename}",
                        root / filename,
                        sha256=digest,
                    )
                )
        await download_files(artifacts)
        return [QtdbSource(root=root, record_ids=record_ids, splits=splits)]

    def convert(self, raw_refs: list[QtdbSource]) -> TimeFDataset:
        """Build complete-record signals, fiducial annotations, and observed-only tasks."""
        if len(raw_refs) != 1:
            raise TimeNetConnectorError(f"QTDB expects one source reference, got {len(raw_refs)}")
        source_ref = raw_refs[0]
        dataset = TimeFDataset(metadata=self.metadata())

        try:
            wfdb = self._wfdb()
        except ImportError as exc:
            raise TimeNetConnectorError(str(exc)) from exc

        for record_id in source_ref.record_ids:
            self._convert_record(dataset, source_ref, record_id, wfdb)
        return dataset

    def _convert_record(self, dataset: TimeFDataset, source: QtdbSource, record_id: str, wfdb: Any) -> None:
        """Convert one excerpt and add its observed-beat localization tasks."""
        split = source.splits[record_id]
        base = source.root / record_id
        try:
            header = self._read_header(base)
            if len(header.sig_name) != _LEAD_COUNT:
                raise TimeNetConnectorError(f"expected two ECG leads, found {len(header.sig_name)}")
            rate = Fraction(str(header.fs))
            if rate != _SAMPLING_RATE_HZ:
                raise TimeNetConnectorError(f"expected a 250 Hz sampling rate, found {header.fs}")
            signals = tuple(
                Signal.from_loader(
                    spec=_ECG,
                    name=name,
                    time_axis=RegularAxis.from_rate_hz(rate),
                    loader=self._lead_loader(base, header, lead_index),
                    n_values=header.sig_len,
                )
                for lead_index, name in enumerate(header.sig_name)
            )
            record = dataset.add_record(
                record=Record(
                    metadata={
                        "qtdb_record": record_id,
                        "source_database": split.source_database,
                        "source_record_id": split.source_record_id,
                        "split_group_id": split.split_group_id,
                        "split_group_basis": split.split_group_basis,
                        "split": split.split.value,
                        "source_version": "1.0.0",
                    },
                    sources=(Source(name="QTDB ECG", signals=signals),),
                )
            )
            markers = _read_markers(wfdb, base, record, header, rate)
            _add_tasks(
                dataset,
                record,
                QtdbTaskContext(record_id, split, markers, rate, header.sig_len),
            )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            raise TimeNetConnectorError(f"invalid QTDB record {record_id}: {exc}") from exc


CONNECTOR = QtdbConnector
