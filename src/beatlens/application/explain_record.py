"""Application use case for producing and explaining one model prediction."""

from dataclasses import dataclass

from beatlens.domain.ports import ECGClassifier, ECGExplainer
from beatlens.domain.types import ECGRecord, Explanation, Prediction


@dataclass(frozen=True)
class ExplainRecordResult:
    """Prediction and local explanation returned by the explain-record use case."""

    prediction: Prediction
    explanation: Explanation


@dataclass(frozen=True)
class ExplainRecord:
    """Coordinate prediction and XAI without depending on a concrete framework."""

    classifier: ECGClassifier
    explainer: ECGExplainer

    def execute(
        self,
        record: ECGRecord,
        target_label: str | None = None,
        random_seed: int = 0,
    ) -> ExplainRecordResult:
        """Predict one record and explain the selected output class."""
        predictions = self.classifier.predict([record.signal])
        if len(predictions) != 1:
            raise ValueError("Classifier must return exactly one prediction for one ECG record.")

        prediction = predictions[0]
        selected_label = target_label
        if selected_label is None:
            if not prediction.predicted_labels:
                raise ValueError("Prediction has no selected label to explain.")
            selected_label = prediction.predicted_labels[0]

        if selected_label not in prediction.scores:
            raise ValueError(f"Target label {selected_label!r} is absent from prediction scores.")

        explanation = self.explainer.explain(
            record=record,
            classifier=self.classifier,
            target_label=selected_label,
            random_seed=random_seed,
        )
        return ExplainRecordResult(prediction=prediction, explanation=explanation)
