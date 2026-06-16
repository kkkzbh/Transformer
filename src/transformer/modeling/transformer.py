from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor, nn

from transformer.config import ModelConfig
from transformer.modeling.cache import EncoderOutput, PastKeyValues, Seq2SeqLMOutput
from transformer.modeling.layers import (
    DecoderLayer,
    EncoderLayer,
    SinusoidalPositionalEncoding,
)
from transformer.modeling.mask import make_causal_mask


@dataclass(frozen=True, slots=True)
class _DecodeCacheState:
    past_key_values: PastKeyValues  # 已解析的解码器 K/V 缓存。
    cache_position: int             # 当前单 token 的绝对位置。


class Seq2SeqTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, config: ModelConfig) -> None:
        super().__init__()
        self.config = config  # 模型超参数。
        self.pad_id = pad_id  # 填充词元 id。

        self.src_embedding = nn.Embedding(  # 源 token 嵌入。
            vocab_size, config.d_model, padding_idx=pad_id
        )
        self.tgt_embedding = nn.Embedding(  # 目标 token 嵌入。
            vocab_size, config.d_model, padding_idx=pad_id
        )
        self.position = SinusoidalPositionalEncoding(  # 共享位置编码。
            config.d_model, config.max_positions
        )
        self.dropout = nn.Dropout(config.dropout)  # 嵌入层 dropout。

        self.encoder_layers = nn.ModuleList(  # 编码器层栈。
            [
                EncoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(  # 解码器层栈。
            [
                DecoderLayer(config.d_model, config.num_heads, config.d_ff, config.dropout)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)             # 编码器最终归一化。
        self.decoder_norm = nn.LayerNorm(config.d_model)             # 解码器最终归一化。
        self.output_projection = nn.Linear(config.d_model, vocab_size)  # 词元 logits。

        self._reset_parameters()

    def forward(
        self,
        src: Tensor | None,
        tgt_in: Tensor,
        *,
        encoder_outputs: EncoderOutput | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int | None = None,
    ) -> Seq2SeqLMOutput:
        resolved_encoder_outputs = self._resolve_encoder_outputs(src, encoder_outputs)
        cache_state = self._resolve_decode_cache(
            tgt_in=tgt_in,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        tgt_padding_mask = tgt_in.eq(self.pad_id)
        logits = self.decode(
            tgt_in,
            resolved_encoder_outputs,
            tgt_padding_mask=tgt_padding_mask,
            cache_state=cache_state,
        )
        return Seq2SeqLMOutput(
            logits=logits,
            encoder_outputs=resolved_encoder_outputs,
            past_key_values=None if cache_state is None else cache_state.past_key_values,
        )

    def encode(self, src: Tensor, *, src_padding_mask: Tensor | None = None) -> EncoderOutput:
        x = self.dropout(self.position(self.src_embedding(src) * math.sqrt(self.config.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_padding_mask=src_padding_mask)
        return EncoderOutput(memory=self.encoder_norm(x), padding_mask=src_padding_mask)

    def decode(
        self,
        tgt_in: Tensor,
        encoder_outputs: EncoderOutput,
        *,
        tgt_padding_mask: Tensor | None = None,
        cache_state: _DecodeCacheState | None = None,
    ) -> Tensor:
        use_cache = cache_state is not None
        causal_mask = None if use_cache else make_causal_mask(tgt_in.size(1), device=tgt_in.device)
        self_padding_mask = None if use_cache else tgt_padding_mask
        start_position = 0 if cache_state is None else cache_state.cache_position
        x = self.dropout(
            self.position(
                self.tgt_embedding(tgt_in) * math.sqrt(self.config.d_model),
                start_position=start_position,
            )
        )
        for index, layer in enumerate(self.decoder_layers):
            x = layer(
                x,
                encoder_outputs.memory,
                tgt_causal_mask=causal_mask,
                tgt_padding_mask=self_padding_mask,
                src_padding_mask=encoder_outputs.padding_mask,
                layer_cache=None
                if cache_state is None
                else cache_state.past_key_values.layers[index],
            )
        x = self.decoder_norm(x)
        return self.output_projection(x)

    def _resolve_encoder_outputs(
        self,
        src: Tensor | None,
        encoder_outputs: EncoderOutput | None,
    ) -> EncoderOutput:
        if encoder_outputs is not None:
            return encoder_outputs
        if src is None:
            raise ValueError("src is required when encoder_outputs is not provided.")
        return self.encode(src, src_padding_mask=src.eq(self.pad_id))

    def _resolve_decode_cache(
        self,
        *,
        tgt_in: Tensor,
        past_key_values: PastKeyValues | None,
        use_cache: bool,
        cache_position: int | None,
    ) -> _DecodeCacheState | None:
        if not use_cache:
            if cache_position is not None:
                raise ValueError("cache_position requires use_cache=True.")
            if past_key_values is not None:
                raise ValueError("past_key_values requires use_cache=True.")
            return None

        if tgt_in.size(1) != 1:
            raise ValueError("use_cache=True currently supports single-token decoding only.")
        if cache_position is None:
            raise ValueError("cache_position is required when use_cache=True.")
        if past_key_values is None:
            past_key_values = PastKeyValues.empty(len(self.decoder_layers))
        if len(past_key_values.layers) != len(self.decoder_layers):
            raise ValueError("past_key_values layer count must match decoder layer count.")
        self._validate_cache_position(past_key_values, cache_position)
        return _DecodeCacheState(past_key_values=past_key_values, cache_position=cache_position)

    def _validate_cache_position(
        self,
        past_key_values: PastKeyValues,
        cache_position: int,
    ) -> None:
        for index, layer_cache in enumerate(past_key_values.layers):
            key = layer_cache.self_attn.key
            value = layer_cache.self_attn.value
            if (key is None) != (value is None):
                raise RuntimeError(f"Layer {index} self-attn cache is partially initialized.")
            cached_len = 0 if key is None else key.size(2)
            if cached_len != cache_position:
                raise ValueError(
                    "cache_position must match existing self-attn cache length "
                    f"for layer {index}: expected {cached_len}, got {cache_position}."
                )

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        self._reset_padding_embeddings()

    def _reset_padding_embeddings(self) -> None:
        nn.init.zeros_(self.src_embedding.weight[self.pad_id])
        nn.init.zeros_(self.tgt_embedding.weight[self.pad_id])
