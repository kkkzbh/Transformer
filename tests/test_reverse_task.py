from __future__ import annotations

from transformer.config import TaskConfig
from transformer.data.tasks.reverse import ReverseTask
from transformer.data.vocab import BOS_TOKEN, EOS_TOKEN


def test_reverse_sample_alignment_includes_target_eos() -> None:
    task = ReverseTask(
        TaskConfig(
            min_len=4,
            max_len=4,
            train_size=4,
            val_size=4,
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


def _tokens(task: ReverseTask, ids: list[int]) -> list[str]:
    return [task.vocab.tokens[token_id] for token_id in ids]
