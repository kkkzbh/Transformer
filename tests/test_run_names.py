from __future__ import annotations

from datetime import UTC, datetime

from transformer.utils.run_names import descending_timestamp_name


def test_descending_timestamp_name_sorts_newest_first() -> None:
    older = descending_timestamp_name(datetime(2026, 6, 16, 11, 18, 3, tzinfo=UTC))
    newer = descending_timestamp_name(datetime(2026, 6, 16, 11, 20, 3, tzinfo=UTC))

    assert sorted([older, newer]) == [newer, older]
    assert newer.endswith("20260616-112003")
