from pathlib import Path

from timenet.types import Split, TemporalLocalizationTask, TimeInterval, TimePoint
from timenet_connectors.datasets.physionet.qtdb.connector import (
    QtdbConnector,
    QtdbSource,
    QtdbSplit,
)


_FIXTURES = Path(__file__).parent / "fixtures"


def test_convert_keeps_full_signals_and_creates_sparse_fiducial_tasks():
    split = QtdbSplit(
        split=Split.TRAIN,
        source_database="fixture database",
        source_record_id="fixture-1",
        split_group_id="fixture-1",
        split_group_basis="fixture",
    )
    dataset = QtdbConnector().convert(
        [QtdbSource(root=_FIXTURES, record_ids=("qtdb_fixture",), splits={"qtdb_fixture": split})]
    )

    assert len(dataset.records) == 1
    record = dataset.records[0]
    assert {signal.name for signal in record.signals} == {"V5", "V2"}
    assert {signal.name: signal.n_values for signal in record.signals} == {"V5": 750, "V2": 750}
    assert record.metadata["split"] == "train"
    assert len(record.annotations) == 7  # two selected beats and five marked fiducials
    assert len(dataset.tasks) == 5

    t_peak_tasks = [task for task in dataset.tasks if task.metadata["fiducial_type"] == "t_peak"]
    assert len(t_peak_tasks) == 2
    first = t_peak_tasks[0]
    assert isinstance(first, TemporalLocalizationTask)
    assert first.split is Split.TRAIN
    assert first.mode.value == "sparse"
    assert first.targets == (TimePoint.seconds(1.2),)
    assert isinstance(first.scope, TimeInterval)
    assert first.scope.start_us == 0
    assert first.scope.end_us == 2_500_000
    assert [len(signal.to_numpy()) for signal in record.signals] == [750, 750]
