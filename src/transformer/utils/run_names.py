from __future__ import annotations

from datetime import datetime

MAX_SORT_EPOCH_SECONDS = 9_999_999_999


def descending_timestamp_name(now: datetime | None = None) -> str:
    current = now or datetime.now()
    reverse_key = MAX_SORT_EPOCH_SECONDS - int(current.timestamp())
    if reverse_key < 0:
        raise ValueError("timestamp is outside the supported sortable range.")
    timestamp = current.strftime("%Y%m%d-%H%M%S")
    return f"0-{reverse_key:010d}-{timestamp}"
