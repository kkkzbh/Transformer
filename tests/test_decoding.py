from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from transformer.inference.decoding import greedy_decode
from transformer.modeling.transformer import Seq2SeqTransformer


class _FixedNextTokenModel:
    def __init__(self, next_ids: list[int], vocab_size: int) -> None:
        self.next_ids = next_ids      # 强制生成的 token 序列。
        self.vocab_size = vocab_size  # 模拟 logits 词表大小。
        self.calls = 0                # 前向调用次数。

    def eval(self) -> None:
        return None

    def __call__(self, src: Tensor, generated: Tensor) -> Tensor:
        del src
        next_id = self.next_ids[self.calls]
        self.calls += 1
        logits = torch.full((generated.size(0), generated.size(1), self.vocab_size), -1e9)
        logits[:, -1, next_id] = 0.0
        return logits


def test_greedy_decode_stops_after_eos() -> None:
    eos_id = 2
    model = _FixedNextTokenModel(next_ids=[4, eos_id, 5], vocab_size=6)
    src = torch.tensor([[7, 8]], dtype=torch.long)

    output = greedy_decode(
        cast(Seq2SeqTransformer, model),
        src,
        bos_id=1,
        eos_id=eos_id,
        max_len=8,
    )

    assert output == [4, eos_id]
    assert model.calls == 2
