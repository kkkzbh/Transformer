from __future__ import annotations

import pytest

from transformer.config import TaskConfig
from transformer.data.tasks.reverse import ReverseDataset, ReverseTask
from transformer.data.vocab import BOS_TOKEN, EOS_TOKEN


def test_reverse_sample_alignment_includes_target_eos() -> None:
    task = ReverseTask(
        TaskConfig(
            min_len=4,
            max_len=4,
            dataset_size=10,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=7,
            source_eos=True,
        )
    )
    sample = task.make_train_dataset()[0]

    src = _tokens(task, sample.src)
    tgt_in = _tokens(task, sample.tgt_in)
    tgt_out = _tokens(task, sample.tgt_out)

    source_digits = src[:-1]
    target_digits = list(reversed(source_digits))

    assert src[-1] == EOS_TOKEN
    assert tgt_in == [BOS_TOKEN, *target_digits]
    assert tgt_out == [*target_digits, EOS_TOKEN]


def test_encode_source_matches_source_eos_policy() -> None:
    task = ReverseTask(TaskConfig(source_eos=True))

    assert task.encode_source("3 7")[-1] == task.vocab.eos_id


def test_split_sizes_and_offsets_are_deterministic() -> None:
    task = ReverseTask(
        TaskConfig(
            dataset_size=100,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
    )

    split = task.split_sizes()
    train = task.make_train_dataset()
    val = task.make_eval_dataset()
    test = task.make_test_dataset()

    assert split.train == 80
    assert split.val == 10
    assert split.test == 10
    assert isinstance(train, ReverseDataset)
    assert isinstance(val, ReverseDataset)
    assert isinstance(test, ReverseDataset)
    assert train.size == 80
    assert val.size == 10
    assert test.size == 10
    assert train.start_index == 0
    assert val.start_index == 80
    assert test.start_index == 90


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        ReverseTask(
            TaskConfig(
                dataset_size=100,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.2,
            )
        )


def _tokens(task: ReverseTask, ids: list[int]) -> list[str]:
    return [task.vocab.tokens[token_id] for token_id in ids]
