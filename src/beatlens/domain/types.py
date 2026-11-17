"""Framework-independent types shared by BeatLens use cases and adapters."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ECGRecord:
    """A single ECG record with signal metadata and dataset annotations."""

    record_id: str
    signal: np.ndarray
    sampling_rate_hz: float
    lead_names: tuple[str, ...]
    labels: tuple[str, ...]
    subject_id: str | None = None


@dataclass(frozen=True)
class Prediction:
    """A model output with provenance."""

    record_id: str
    scores: Mapping[str, float]
    predicted_labels: tuple[str, ...]
    model_name: str
    model_version: str


@dataclass(frozen=True)
class Explanation:
    """A local attribution result; attribution values describe model behavior."""

    record_id: str
    target_label: str
    method: str
    values: np.ndarray
    local_fidelity: float | None = None
    random_seed: int | None = None
