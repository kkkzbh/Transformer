from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from torch.utils.data import Dataset

from transformer.data.batch import Seq2SeqBatch, Seq2SeqSample
from transformer.data.vocab import Vocab


class Seq2SeqTask(Protocol):
    @property
    def name(self) -> str: ...  # 任务注册名。

    @property
    def vocab(self) -> Vocab: ...  # 任务词表。

    def make_train_dataset(self) -> Dataset[Seq2SeqSample]: ...

    def make_eval_dataset(self) -> Dataset[Seq2SeqSample]: ...

    def make_test_dataset(self) -> Dataset[Seq2SeqSample]: ...

    def make_collate_fn(self) -> Callable[[list[Seq2SeqSample]], Seq2SeqBatch]: ...

    def prepare(self) -> Path: ...

    def encode_source(self, text: str) -> list[int]: ...

    def decode_target(self, ids: list[int]) -> str: ...
