from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from math import isclose
from pathlib import Path

from torch.utils.data import Dataset

from transformer.config import TaskConfig
from transformer.data.batch import Seq2SeqBatch, Seq2SeqSample, collate_seq2seq
from transformer.data.vocab import BOS_TOKEN, EOS_TOKEN, Vocab


@dataclass(frozen=True, slots=True)
class ReverseDataset(Dataset[Seq2SeqSample]):
    size: int
    start_index: int
    min_len: int
    max_len: int
    seed: int
    digit_count: int
    vocab: Vocab
    source_eos: bool = True

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("dataset size must be positive.")
        if self.min_len <= 0 or self.max_len < self.min_len:
            raise ValueError("length bounds must satisfy 0 < min_len <= max_len.")

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Seq2SeqSample:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        rng = random.Random(self.seed + self.start_index + index)
        length = rng.randint(self.min_len, self.max_len)
        tokens = [str(rng.randrange(self.digit_count)) for _ in range(length)]
        reversed_tokens = list(reversed(tokens))

        src_tokens = tokens + ([EOS_TOKEN] if self.source_eos else [])
        tgt_in_tokens = [BOS_TOKEN, *reversed_tokens]
        tgt_out_tokens = [*reversed_tokens, EOS_TOKEN]

        return Seq2SeqSample(
            src=self.vocab.encode(src_tokens),
            tgt_in=self.vocab.encode(tgt_in_tokens),
            tgt_out=self.vocab.encode(tgt_out_tokens),
        )


@dataclass(frozen=True, slots=True)
class ReverseTask:
    config: TaskConfig
    name: str = field(init=False, default="reverse")
    vocab: Vocab = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocab", Vocab.digits(self.config.digit_count))

    def make_train_dataset(self) -> Dataset[Seq2SeqSample]:
        split = self.split_sizes()
        return ReverseDataset(
            size=split.train,
            start_index=0,
            min_len=self.config.min_len,
            max_len=self.config.max_len,
            seed=self.config.seed,
            digit_count=self.config.digit_count,
            vocab=self.vocab,
            source_eos=self.config.source_eos,
        )

    def make_eval_dataset(self) -> Dataset[Seq2SeqSample]:
        split = self.split_sizes()
        return ReverseDataset(
            size=split.val,
            start_index=split.train,
            min_len=self.config.min_len,
            max_len=self.config.max_len,
            seed=self.config.seed,
            digit_count=self.config.digit_count,
            vocab=self.vocab,
            source_eos=self.config.source_eos,
        )

    def make_test_dataset(self) -> Dataset[Seq2SeqSample]:
        split = self.split_sizes()
        return ReverseDataset(
            size=split.test,
            start_index=split.train + split.val,
            min_len=self.config.min_len,
            max_len=self.config.max_len,
            seed=self.config.seed,
            digit_count=self.config.digit_count,
            vocab=self.vocab,
            source_eos=self.config.source_eos,
        )

    def split_sizes(self) -> SplitSizes:
        ratios = (self.config.train_ratio, self.config.val_ratio, self.config.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("train_ratio, val_ratio, and test_ratio must all be positive.")
        if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")
        if self.config.dataset_size < 3:
            raise ValueError("dataset_size must be at least 3 so every split is non-empty.")

        train = int(self.config.dataset_size * self.config.train_ratio)
        val = int(self.config.dataset_size * self.config.val_ratio)
        test = self.config.dataset_size - train - val
        if min(train, val, test) <= 0:
            raise ValueError("dataset_size and split ratios produce an empty split.")
        return SplitSizes(train=train, val=val, test=test)

    def make_collate_fn(self) -> Callable[[list[Seq2SeqSample]], Seq2SeqBatch]:
        return partial(collate_seq2seq, pad_id=self.vocab.pad_id)

    def prepare(self) -> Path:
        task_dir = self.config.data_dir / self.name
        task_dir.mkdir(parents=True, exist_ok=True)
        self.vocab.to_json(task_dir / "vocab.json")
        self._write_samples(task_dir / "samples.jsonl")
        return task_dir

    def encode_source(self, text: str) -> list[int]:
        tokens = text.split()
        self.vocab.require_known(tokens)
        if self.config.source_eos:
            tokens = [*tokens, EOS_TOKEN]
        return self.vocab.encode(tokens)

    def decode_target(self, ids: list[int]) -> str:
        return " ".join(self.vocab.decode(ids, stop_at_eos=True))

    def _write_samples(self, path: Path) -> None:
        dataset = self.make_eval_dataset()
        sample_count = min(self.config.samples_to_export, self.split_sizes().val)
        with path.open("w", encoding="utf-8") as handle:
            for index in range(sample_count):
                sample = dataset[index]
                handle.write(
                    json.dumps(
                        {
                            "src": sample.src,
                            "tgt_in": sample.tgt_in,
                            "tgt_out": sample.tgt_out,
                            "decoded_src": self.vocab.decode(sample.src, stop_at_eos=True),
                            "decoded_tgt": self.vocab.decode(sample.tgt_out, stop_at_eos=True),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train: int
    val: int
    test: int
