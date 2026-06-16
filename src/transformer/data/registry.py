from __future__ import annotations

from transformer.config import TaskConfig
from transformer.data.tasks.base import Seq2SeqTask
from transformer.data.tasks.reverse import ReverseTask


def create_task(config: TaskConfig) -> Seq2SeqTask:
    match config.name:
        case "reverse":
            return ReverseTask(config)
        case unknown:
            raise ValueError(f"Unknown task: {unknown}")
