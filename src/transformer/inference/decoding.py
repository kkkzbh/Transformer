from __future__ import annotations

import torch
from torch import Tensor

from transformer.modeling.transformer import Seq2SeqTransformer


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: Tensor,
    *,
    bos_id: int,
    eos_id: int,
    max_len: int,
) -> list[int]:
    model.eval()
    if src.dim() != 2 or src.size(0) != 1:
        raise ValueError("greedy_decode expects src with shape [1, src_len].")

    generated = torch.tensor([[bos_id]], dtype=torch.long, device=src.device)
    for _ in range(max_len):
        logits = model(src, generated)
        next_id = logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_id[:, None]], dim=1)
        if next_id.item() == eos_id:
            break
    return generated[0, 1:].tolist()
