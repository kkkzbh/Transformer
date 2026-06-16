from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True, slots=True)
class Seq2SeqSample:
    src: list[int]
    tgt_in: list[int]
    tgt_out: list[int]


@dataclass(frozen=True, slots=True)
class Seq2SeqBatch:
    src: Tensor
    tgt_in: Tensor
    tgt_out: Tensor

    def to(self, device: torch.device) -> Seq2SeqBatch:
        return Seq2SeqBatch(
            src=self.src.to(device),
            tgt_in=self.tgt_in.to(device),
            tgt_out=self.tgt_out.to(device),
        )


def collate_seq2seq(samples: Sequence[Seq2SeqSample], *, pad_id: int) -> Seq2SeqBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    return Seq2SeqBatch(
        src=_pad([sample.src for sample in samples], pad_id=pad_id),
        tgt_in=_pad([sample.tgt_in for sample in samples], pad_id=pad_id),
        tgt_out=_pad([sample.tgt_out for sample in samples], pad_id=pad_id),
    )


def _pad(sequences: Sequence[list[int]], *, pad_id: int) -> Tensor:
    tensors = [torch.tensor(sequence, dtype=torch.long) for sequence in sequences]
    return pad_sequence(tensors, batch_first=True, padding_value=pad_id)
