"""Interfaces that keep application logic independent of data and ML frameworks."""

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from beatlens.domain.types import ECGRecord, Explanation, Prediction


class ECGDataset(Protocol):
    """Read-only access to ECG records and their task labels."""

    def iter_records(self, split: str) -> Iterable[ECGRecord]: ...


class ECGClassifier(Protocol):
    """Predict class scores for one or more ECG signals."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def predict(self, signals: Sequence[np.ndarray]) -> Sequence[Prediction]: ...


class ECGExplainer(Protocol):
    """Explain one model prediction for a single ECG record."""

    @property
    def name(self) -> str: ...

    def explain(
        self,
        record: ECGRecord,
        classifier: ECGClassifier,
        target_label: str,
        random_seed: int,
    ) -> Explanation: ...
