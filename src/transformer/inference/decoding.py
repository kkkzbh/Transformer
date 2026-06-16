from __future__ import annotations

import torch
from torch import Tensor

from transformer.modeling.cache import EncoderOutput, PastKeyValues
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
    if max_len <= 0:
        return []

    encoder_outputs: EncoderOutput | None = None
    past_key_values: PastKeyValues | None = None
    current_token = torch.tensor([[bos_id]], dtype=torch.long, device=src.device)
    output_ids: list[int] = []
    for position in range(max_len):
        output = model(
            src if encoder_outputs is None else None,
            current_token,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=position,
        )
        encoder_outputs = output.encoder_outputs
        past_key_values = output.past_key_values
        logits = output.logits
        next_id = logits[:, -1, :].argmax(dim=-1)
        output_ids.append(int(next_id.item()))
        if next_id.item() == eos_id:
            break
        current_token = next_id[:, None]
    return output_ids
