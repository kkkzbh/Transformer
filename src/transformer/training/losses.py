from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def seq2seq_cross_entropy(logits: Tensor, targets: Tensor, *, pad_id: int) -> Tensor:
    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_id,
    )
