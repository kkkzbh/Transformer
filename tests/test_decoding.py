from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor

from transformer.config import ModelConfig
from transformer.inference.decoding import greedy_decode
from transformer.modeling.cache import EncoderOutput, PastKeyValues, Seq2SeqLMOutput
from transformer.modeling.transformer import Seq2SeqTransformer


class _FixedNextTokenModel:
    def __init__(self, next_ids: list[int], vocab_size: int) -> None:
        self.next_ids = next_ids      # 强制生成的 token 序列。
        self.vocab_size = vocab_size  # 模拟 logits 词表大小。
        self.pad_id = 0               # 模拟 padding token id。
        self.calls = 0                # 模型 forward 调用次数。
        self.src_seen: list[bool] = []  # 每步是否收到原始 src。
        self.past_seen: list[bool] = []  # 每步是否收到已有 K/V 缓存。
        self.seen_tokens: list[int] = []  # 解码器看到的单 token 输入。
        self.seen_positions: list[int] = []  # 解码器收到的位置编号。

    def eval(self) -> None:
        return None

    def __call__(
        self,
        src: Tensor | None,
        tgt_in: Tensor,
        *,
        encoder_outputs: EncoderOutput | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int | None = None,
    ) -> Seq2SeqLMOutput:
        if not use_cache or cache_position is None:
            raise AssertionError("greedy_decode should call the cached model path.")
        if src is None and encoder_outputs is None:
            raise AssertionError("first decode step should provide src or encoder_outputs.")

        self.src_seen.append(src is not None)
        self.past_seen.append(past_key_values is not None)
        self.seen_tokens.append(int(tgt_in.item()))
        self.seen_positions.append(cache_position)

        next_id = self.next_ids[self.calls]
        self.calls += 1
        logits = torch.full((tgt_in.size(0), tgt_in.size(1), self.vocab_size), -1e9)
        logits[:, -1, next_id] = 0.0
        return Seq2SeqLMOutput(
            logits=logits,
            encoder_outputs=encoder_outputs
            if encoder_outputs is not None
            else EncoderOutput(memory=cast(Tensor, src), padding_mask=None),
            past_key_values=past_key_values or PastKeyValues.empty(0),
        )


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
    assert model.src_seen == [True, False]
    assert model.past_seen == [False, True]
    assert model.seen_tokens == [1, 4]
    assert model.seen_positions == [0, 1]


def test_greedy_decode_does_not_reuse_cache_between_calls() -> None:
    eos_id = 2
    model = _FixedNextTokenModel(next_ids=[eos_id, eos_id], vocab_size=6)
    src = torch.tensor([[7, 8]], dtype=torch.long)

    first_output = greedy_decode(
        cast(Seq2SeqTransformer, model),
        src,
        bos_id=1,
        eos_id=eos_id,
        max_len=8,
    )
    second_output = greedy_decode(
        cast(Seq2SeqTransformer, model),
        src,
        bos_id=1,
        eos_id=eos_id,
        max_len=8,
    )

    assert first_output == [eos_id]
    assert second_output == [eos_id]
    assert model.past_seen == [False, False]


def test_use_cache_rejects_multi_token_decode() -> None:
    model = _tiny_model()
    src = torch.tensor([[3, 5, 7]], dtype=torch.long)
    tgt_in = torch.tensor([[1, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="single-token"):
        model(src, tgt_in, use_cache=True, cache_position=0)


def test_cache_position_requires_use_cache() -> None:
    model = _tiny_model()
    src = torch.tensor([[3, 5, 7]], dtype=torch.long)
    tgt_in = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(ValueError, match="cache_position requires use_cache"):
        model(src, tgt_in, cache_position=0)


def test_cache_position_must_match_existing_cache_length() -> None:
    model = _tiny_model()
    src = torch.tensor([[3, 5, 7]], dtype=torch.long)
    tgt_in = torch.tensor([[1]], dtype=torch.long)
    encoder_outputs = model.encode(src, src_padding_mask=src.eq(model.pad_id))
    past_key_values = PastKeyValues.empty(len(model.decoder_layers))

    with pytest.raises(ValueError, match="cache_position must match existing self-attn cache"):
        model(
            None,
            tgt_in,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=4,
        )


def test_cached_decode_matches_full_decode_logits() -> None:
    model = _tiny_model()
    model.eval()
    src = torch.tensor([[3, 5, 7]], dtype=torch.long)
    tgt_in = torch.tensor([[1, 4, 6, 8]], dtype=torch.long)

    full_logits = model(src, tgt_in).logits
    encoder_outputs = None
    past_key_values = None
    step_logits: list[Tensor] = []
    first_cross_key: Tensor | None = None

    for position in range(tgt_in.size(1)):
        output = model(
            src if encoder_outputs is None else None,
            tgt_in[:, position : position + 1],
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=position,
        )
        encoder_outputs = output.encoder_outputs
        past_key_values = output.past_key_values
        step_logits.append(output.logits)

        assert past_key_values is not None
        layer_cache = past_key_values.layers[0]
        assert layer_cache.self_attn.key is not None
        assert layer_cache.self_attn.key.size(2) == position + 1
        assert layer_cache.cross_attn.key is not None
        assert layer_cache.cross_attn.key.size(2) == src.size(1)
        if first_cross_key is None:
            first_cross_key = layer_cache.cross_attn.key
        else:
            assert layer_cache.cross_attn.key is first_cross_key

    cached_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5)


def _tiny_model() -> Seq2SeqTransformer:
    torch.manual_seed(0)
    return Seq2SeqTransformer(
        vocab_size=12,
        pad_id=0,
        config=ModelConfig(
            d_model=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=32,
            dropout=0.0,
            max_positions=16,
        ),
    )
